"""
PyTorch-based Python port of the C++ HoQp (hierarchical QP) implementation.

Notes/limitations:
- This implementation focuses on translating the matrix assembly logic to torch
  and keeping tensors on the specified device (GPU when available).
- For the QP solve, a direct dense solve of the quadratic system (H z = -c)
  is used. That mirrors the unconstrained optimum. The original C++ used
  a dedicated QP solver (proxqp) that enforces linear inequalities; here we
  currently DO NOT fully enforce inequalities. This keeps the implementation
  simple and GPU-resident. See comments in `solve_problem` for options to
  improve (projected/ADMM solver or Python bindings to proxqp).

The API follows the C++ class closely: Task, HoQp with similar getters.
"""
from typing import Optional
import torch
from wbc_ik_qp.ik import *


class Task:
    """Lightweight Task container similar to the C++ Task.
    Fields:
      a_: equality constraint matrix (m_eq x n)
      b_: equality rhs (m_eq)
      d_: inequality matrix (m_ineq x n)
      f_: inequality rhs (m_ineq)
    All data are torch tensors (can be empty zero-sized tensors).
    """

    def __init__(self, a: Optional[torch.Tensor] = None, b: Optional[torch.Tensor] = None,
                 d: Optional[torch.Tensor] = None, f: Optional[torch.Tensor] = None,
                 num_decision_vars: Optional[int] = None, device=None, dtype=None):
        if a is None:
            if num_decision_vars is None:
                self.a_ = torch.zeros((0, 0), device=device, dtype=dtype)
            else:
                self.a_ = torch.zeros((0, num_decision_vars), device=device, dtype=dtype)
        else:
            self.a_ = a

        self.b_ = torch.zeros(0, device=device, dtype=dtype) if b is None else b
        if d is None:
            if num_decision_vars is None:
                self.d_ = torch.zeros((0, 0), device=device, dtype=dtype)
            else:
                self.d_ = torch.zeros((0, num_decision_vars), device=device, dtype=dtype)
        else:
            self.d_ = d
        self.f_ = torch.zeros(0, device=device, dtype=dtype) if f is None else f

    def __add__(self, rhs: "Task") -> "Task":
        # Concatenate rows (stack tasks).
        # If the two tasks have differing numbers of columns, pad the smaller
        # matrices with zeros on the right so torch.cat succeeds.
        def pad_to_cols(mat: torch.Tensor, cols: int) -> torch.Tensor:
            if mat.numel() == 0:
                return torch.zeros((0, cols), device=mat.device, dtype=mat.dtype)
            if mat.shape[1] == cols:
                return mat
            new = torch.zeros((mat.shape[0], cols), device=mat.device, dtype=mat.dtype)
            new[:, :mat.shape[1]] = mat
            return new

        # Determine target column counts for 'a' and 'd'
        a_cols = max(self.a_.shape[1] if self.a_.numel() else 0, rhs.a_.shape[1] if rhs.a_.numel() else 0)
        d_cols = max(self.d_.shape[1] if self.d_.numel() else 0, rhs.d_.shape[1] if rhs.d_.numel() else 0)

        a_padded_left = pad_to_cols(self.a_, a_cols)
        a_padded_right = pad_to_cols(rhs.a_, a_cols)
        a = torch.cat([a_padded_left, a_padded_right], dim=0) if a_cols > 0 else torch.zeros((0, 0), device=self.a_.device if self.a_.numel() else rhs.a_.device, dtype=self.a_.dtype if self.a_.numel() else rhs.a_.dtype)

        b = torch.cat([self.b_, rhs.b_], dim=0) if self.b_.numel() and rhs.b_.numel() else (rhs.b_ if self.b_.numel() == 0 else self.b_)

        d_padded_left = pad_to_cols(self.d_, d_cols)
        d_padded_right = pad_to_cols(rhs.d_, d_cols)
        d = torch.cat([d_padded_left, d_padded_right], dim=0) if d_cols > 0 else torch.zeros((0, 0), device=self.d_.device if self.d_.numel() else rhs.d_.device, dtype=self.d_.dtype if self.d_.numel() else rhs.d_.dtype)

        f = torch.cat([self.f_, rhs.f_], dim=0) if self.f_.numel() and rhs.f_.numel() else (rhs.f_ if self.f_.numel() == 0 else self.f_)

        return Task(a=a, b=b, d=d, f=f)

    @staticmethod
    def concatenate_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        if v1.numel() == 0:
            return v2
        if v2.numel() == 0:
            return v1
        return torch.cat([v1, v2], dim=0)


class HoQp:
    """Hierarchical QP similar to C++ implementation, using torch tensors.

    Constructor accepts a Task and an optional higher_problem (HoQp instance).
    """

    def __init__(self, task: Task, higher_problem: Optional["HoQp"] = None, device=None, dtype=torch.float64):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.task_ = task
        self.higher_problem_ = higher_problem

        # Will be set during initialization
        self.init_vars()
        self.formulate_problem()
        self.solve_problem()
        self.build_z_matrix()
        self.stack_slack_solutions()

    # Public getters matching C++ naming
    def getStackedZMatrix(self):
        return self.stacked_z_

    def getStackedTasks(self):
        return self.stacked_tasks_

    def getStackedSlackSolutions(self):
        return self.stacked_slack_vars_

    def getSolutions(self):
        x = self.x_prev_ + self.stacked_z_prev_ @ self.decision_vars_solutions_
        return x

    def getSlackedNumVars(self):
        return self.stacked_tasks_.d_.shape[0]

    # Internal methods
    def init_vars(self):
        t = self.task_
        self.num_slack_vars_ = t.d_.shape[0]
        self.has_eq_constraints_ = t.a_.shape[0] > 0
        self.has_ineq_constraints_ = self.num_slack_vars_ > 0

        if self.higher_problem_ is not None:
            self.stacked_z_prev_ = self.higher_problem_.getStackedZMatrix()
            self.stacked_tasks_prev_ = self.higher_problem_.getStackedTasks()
            self.stacked_slack_solutions_prev_ = self.higher_problem_.getStackedSlackSolutions()
            self.x_prev_ = self.higher_problem_.getSolutions()
            self.num_prev_slack_vars_ = self.higher_problem_.getSlackedNumVars()

            self.num_decision_vars_ = self.stacked_z_prev_.shape[1]
        else:
            # If no higher problem, number of decision vars is max of columns
            ncols = max(t.a_.shape[1] if t.a_.numel() else 0, t.d_.shape[1] if t.d_.numel() else 0)
            self.num_decision_vars_ = ncols
            self.stacked_tasks_prev_ = Task(num_decision_vars=self.num_decision_vars_, device=self.device, dtype=self.dtype)
            self.stacked_z_prev_ = torch.eye(self.num_decision_vars_, device=self.device, dtype=self.dtype)
            self.stacked_slack_solutions_prev_ = torch.zeros(0, device=self.device, dtype=self.dtype)
            self.x_prev_ = torch.zeros(self.num_decision_vars_, device=self.device, dtype=self.dtype)
            self.num_prev_slack_vars_ = 0

        self.stacked_tasks_ = self.task_ + self.stacked_tasks_prev_

        # Convenience matrices
        self.eye_nv_nv_ = torch.eye(self.num_slack_vars_, device=self.device, dtype=self.dtype)
        self.zero_nv_nx_ = torch.zeros((self.num_slack_vars_, self.num_decision_vars_), device=self.device, dtype=self.dtype)

    def formulate_problem(self):
        self.build_h_matrix()
        self.build_c_vector()
        self.build_d_matrix()
        self.build_f_vector()

    def build_h_matrix(self):
        nx = self.num_decision_vars_
        nv = self.num_slack_vars_
        # z is [decision_vars; slack_vars]
        zdim = nx + nv
        h = torch.zeros((zdim, zdim), device=self.device, dtype=self.dtype)

        if self.has_eq_constraints_:
            a_curr_z_prev = self.task_.a_ @ self.stacked_z_prev_
            z_t_a_t_a_z = a_curr_z_prev.T @ a_curr_z_prev + 1e-12 * torch.eye(nx, device=self.device, dtype=self.dtype)
        else:
            z_t_a_t_a_z = torch.zeros((nx, nx), device=self.device, dtype=self.dtype)

        # top-left, top-right, bottom-left, bottom-right
        top_left = z_t_a_t_a_z
        top_right = torch.zeros((nx, nv), device=self.device, dtype=self.dtype)
        bottom_left = torch.zeros((nv, nx), device=self.device, dtype=self.dtype)
        bottom_right = self.eye_nv_nv_

        h[:nx, :nx] = top_left
        h[:nx, nx:] = top_right
        h[nx:, :nx] = bottom_left
        h[nx:, nx:] = bottom_right

        self.h_ = h

    def build_c_vector(self):
        nx = self.num_decision_vars_
        nv = self.num_slack_vars_
        c = torch.zeros((nx + nv,), device=self.device, dtype=self.dtype)

        if self.has_eq_constraints_:
            temp = (self.task_.a_ @ self.stacked_z_prev_).T @ (self.task_.a_ @ self.x_prev_ - self.task_.b_)
        else:
            temp = torch.zeros((nx,), device=self.device, dtype=self.dtype)

        c[:nx] = temp
        # slack part stays zero
        self.c_ = c

    def build_d_matrix(self):
        nx = self.num_decision_vars_
        nv = self.num_slack_vars_

        rows = 2 * nv + self.num_prev_slack_vars_
        cols = nx + nv
        d = torch.zeros((rows, cols), device=self.device, dtype=self.dtype)

        stacked_zero = torch.zeros((self.num_prev_slack_vars_, nv), device=self.device, dtype=self.dtype)

        if self.has_ineq_constraints_:
            d_curr_z = self.task_.d_ @ self.stacked_z_prev_
        else:
            d_curr_z = torch.zeros((0, nx), device=self.device, dtype=self.dtype)

        # Build block matrix like C++ implementation
        # note shapes must align
        # First block row: [ zero_nv_nx_, -I ]
        if nv > 0:
            d[:nv, :nx] = self.zero_nv_nx_
            d[:nv, nx:] = -torch.eye(nv, device=self.device, dtype=self.dtype)

            # middle block rows
            start = nv
            end = nv + self.num_prev_slack_vars_
            if self.num_prev_slack_vars_ > 0:
                d[start:end, :nx] = self.stacked_tasks_prev_.d_ @ self.stacked_z_prev_
                d[start:end, nx:] = stacked_zero

            # bottom block
            start2 = nv + self.num_prev_slack_vars_
            if nv > 0:
                d[start2:start2 + d_curr_z.shape[0], :nx] = d_curr_z
                d[start2:start2 + d_curr_z.shape[0], nx:] = -torch.eye(nv, device=self.device, dtype=self.dtype)

        self.d_ = d

    def build_f_vector(self):
        nv = self.num_slack_vars_
        rows = 2 * nv + self.num_prev_slack_vars_
        f = torch.zeros((rows,), device=self.device, dtype=self.dtype)

        if self.has_ineq_constraints_:
            f_minus_d_x_prev = self.task_.f_ - self.task_.d_ @ self.x_prev_
        else:
            f_minus_d_x_prev = torch.zeros((0,), device=self.device, dtype=self.dtype)

        # second block for previous stacked tasks
        second = torch.zeros((self.num_prev_slack_vars_,), device=self.device, dtype=self.dtype)
        if self.num_prev_slack_vars_ > 0:
            second = self.stacked_tasks_prev_.f_ - self.stacked_tasks_prev_.d_ @ self.x_prev_ + self.stacked_slack_solutions_prev_

        f[:nv] = 0
        if self.num_prev_slack_vars_ > 0:
            f[nv:nv + self.num_prev_slack_vars_] = second
        if f_minus_d_x_prev.numel() > 0:
            f[nv + self.num_prev_slack_vars_: nv + self.num_prev_slack_vars_ + f_minus_d_x_prev.numel()] = f_minus_d_x_prev

        self.f_ = f

    def build_z_matrix(self):
        # Build stacked_z_ similar to C++: if eq constraints exist compute kernel
        if self.has_eq_constraints_:
            assert self.task_.a_.shape[1] > 0
            # compute nullspace of (A * stacked_z_prev_)
            mat = self.task_.a_ @ self.stacked_z_prev_
            # torch doesn't have fullPivLu().kernel(); we compute SVD nullspace
            # compute reduced SVD to make vh shape compatible with s length
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
            tol = 1e-12
            null_mask = s <= tol
            # If no nullspace, stacked_z_ is zeros with appropriate cols
            if null_mask.numel() == 0 or null_mask.sum() == 0:
                self.stacked_z_ = torch.zeros_like(self.stacked_z_prev_)
            else:
                # columns of V^H.T corresponding to small singular values
                nullspace = vh.T[:, null_mask]
                self.stacked_z_ = self.stacked_z_prev_ @ nullspace
        else:
            self.stacked_z_ = self.stacked_z_prev_

    def solve_problem(self):
        """Solve the QP z = [decision_vars; slack_vars].

        Preferred strategy: try to use the reLU-QP solver implemented in the
        project's `reluqp` package. We assemble H, g, A, l, u for the z variable
        and call that solver so the heavy work stays on device (GPU) and we
        avoid unnecessary CPU/GPU transfers. If the import or solve fails we
        fall back to a dense torch solve (previous behaviour).
        """
        H = self.h_
        g = self.c_
        A = self.d_
        f = self.f_

        m = A.shape[0]
        device = self.device
        dtype = self.dtype

        # prepare bounds: represent d z <= f as l <= A z <= u with l = -inf, u = f
        if m > 0:
            l = torch.full((m,), -float('inf'), device=device, dtype=dtype)
            u = f.clone().to(device=device, dtype=dtype)
        else:
            # no inequality constraints
            l = torch.zeros((0,), device=device, dtype=dtype)
            u = torch.zeros((0,), device=device, dtype=dtype)

        used_reluqp = False
        try:
            # import the reluqp solver and try to use it (it expects torch tensors)
            import reluqp.reluqpth as reluqpth

            model = reluqpth.ReLU_QP()
            # call setup with device/precision to ensure tensors remain on device
            model.setup(H, g, A, l, u, device=device, precision=dtype)
            results = model.solve()
            z = results.x
            # Ensure z is a torch tensor on expected device
            if not isinstance(z, torch.Tensor):
                z = torch.as_tensor(z, device=device, dtype=dtype)

            used_reluqp = True
        except Exception:
            # On any failure (missing package, interface mismatch or runtime),
            # fall back to the dense solve below and keep everything on device.
            used_reluqp = False

        if not used_reluqp:
            # regularize for numerical stability
            reg = 1e-9
            try:
                z = torch.linalg.solve(H + reg * torch.eye(H.shape[0], device=device, dtype=dtype), -g)
            except RuntimeError:
                z = -torch.matmul(torch.linalg.pinv(H + reg * torch.eye(H.shape[0], device=device, dtype=dtype)), g)

        nx = self.num_decision_vars_
        nv = self.num_slack_vars_
        self.decision_vars_solutions_ = z[:nx].to(device=device, dtype=dtype)
        self.slack_vars_solutions_ = z[nx: nx + nv].to(device=device, dtype=dtype) if nv > 0 else torch.zeros(0, device=device, dtype=dtype)

        # store for access
        self.qp_solution_ = z.to(device=device, dtype=dtype)

    def stack_slack_solutions(self):
        if self.higher_problem_ is not None:
            self.stacked_slack_vars_ = torch.cat([self.higher_problem_.getStackedSlackSolutions(), self.slack_vars_solutions_], dim=0)
        else:
            self.stacked_slack_vars_ = self.slack_vars_solutions_



if __name__ == '__main__':
    # tiny local smoke test when running module directly
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    # create a small Task
    n = 6
    meq = 2
    mineq = 3
    a = torch.randn((meq, n), device=dev, dtype=dtype)
    b = torch.randn((meq,), device=dev, dtype=dtype)
    d = torch.randn((mineq, n), device=dev, dtype=dtype)
    f = torch.randn((mineq,), device=dev, dtype=dtype)
    t = Task(a=a, b=b, d=d, f=f, device=dev, dtype=dtype)
    h = HoQp(t, device=dev, dtype=dtype)
    print('HoQp solved. solution length:', h.qp_solution_.shape[0])
