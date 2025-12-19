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
                 num_decision_vars: Optional[int] = None, device=None, dtype=None, weight: float = 1.0):
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
        
        # Task weight for hierarchical combining
        self.weight_ = weight

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

        # Combine weights using sum (could also use max, average, etc.)
        combined_weight = self.weight_ + rhs.weight_
        return Task(a=a, b=b, d=d, f=f, weight=combined_weight)

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

    def __init__(self, task: Task, higher_problem: Optional["HoQp"] = None, device=None, dtype=torch.float64, task_weight: float = 1.0):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.task_ = task
        self.higher_problem_ = higher_problem
        # Store the task weight for this level
        self.task_weight_ = task_weight

        # Will be set during initialization
        self.init_vars()
        self.build_z_matrix()  # Build nullspace first to determine correct dimensions
        self.update_decision_vars_count()  # Update num_decision_vars_ based on nullspace
        self.formulate_problem()
        self.solve_problem()
        self.stack_slack_solutions()

    # Public getters matching C++ naming
    def getStackedZMatrix(self):
        return self.stacked_z_

    def getStackedTasks(self):
        return self.stacked_tasks_

    def getStackedSlackSolutions(self):
        return self.stacked_slack_vars_

    def getSolutions(self):
        # Check if stacked_z_ has any columns for proper matrix multiplication
        if self.stacked_z_.shape[1] == 0:
            # No remaining degrees of freedom from nullspace projection
            # In this case, the optimization solved the overconstrained system directly
            # The solution should be applied to achieve the current task as much as possible
            if hasattr(self, 'decision_vars_solutions_') and self.decision_vars_solutions_.numel() > 0:
                # We have a direct solution for the current task
                # Apply it by reconstructing the full solution
                result = self.x_prev_.clone()
                if self.has_eq_constraints_:
                    # Use pseudoinverse to find the least-squares adjustment
                    # that satisfies the current task: A * (x_prev + delta) = b
                    # => A * delta = b - A * x_prev
                    residual = self.task_.b_ - self.task_.a_ @ self.x_prev_
                    try:
                        A_pinv = torch.linalg.pinv(self.task_.a_)
                        delta = A_pinv @ residual
                        result = self.x_prev_ + delta
                    except:
                        # Fall back to previous solution if pseudoinverse fails
                        result = self.x_prev_
                return result
            else:
                return self.x_prev_
        else:
            # Apply the optimization result projected through stacked_z_
            x = self.x_prev_ + self.stacked_z_ @ self.decision_vars_solutions_
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
            # Get information from higher priority problem
            self.stacked_tasks_prev_ = self.higher_problem_.getStackedTasks()
            self.stacked_slack_solutions_prev_ = self.higher_problem_.getStackedSlackSolutions()
            self.x_prev_ = self.higher_problem_.getSolutions()
            self.num_prev_slack_vars_ = self.higher_problem_.getSlackedNumVars()
            
            # For hierarchical QP, we need the nullspace of the stacked higher-priority constraints
            # but we want to preserve DOF index mapping (not compact nullspace representation)
            higher_tasks = self.higher_problem_.getStackedTasks()
            if higher_tasks.a_.numel() > 0 and higher_tasks.a_.shape[0] > 0:
                # Create a nullspace that preserves DOF mapping by zeroing out used DOF
                # instead of using compact QR-based nullspace representation
                try:
                    mat = higher_tasks.a_
                    n = mat.shape[1]  # Total DOF count
                    
                    # Find which DOF are actually used by higher priority tasks
                    # (have non-zero columns in the constraint matrix)
                    used_dof = torch.any(torch.abs(mat) > 1e-10, dim=0)
                    free_dof = ~used_dof
                    num_free_dof = torch.sum(free_dof).item()
                    
                    if num_free_dof > 0:
                        # Create nullspace matrix that preserves DOF indices
                        # Shape: (n, num_free_dof) where each column corresponds to one free DOF
                        self.stacked_z_prev_ = torch.zeros((n, num_free_dof), device=self.device, dtype=self.dtype)
                        free_indices = torch.nonzero(free_dof, as_tuple=True)[0]
                        for i, dof_idx in enumerate(free_indices):
                            self.stacked_z_prev_[dof_idx, i] = 1.0
                        
                        # print(f'init_vars: preserved DOF mapping nullspace, {num_free_dof} free DOF out of {n}')
                        # print(f'Used DOF: {torch.nonzero(used_dof, as_tuple=True)[0].tolist()}')
                        # print(f'Free DOF: {free_indices.tolist()}')
                        
                        # Save the free DOF indices for use in build_z_matrix
                        self.free_dof_from_prev = free_indices.tolist()
                    else:
                        # All DOF are constrained by higher priority
                        self.stacked_z_prev_ = torch.zeros((n, 0), device=self.device, dtype=self.dtype)
                        print(f'init_vars: higher tasks use all DOF, no nullspace')
                except Exception as e:
                    print(f'DOF-preserving nullspace computation failed: {e}, falling back to QR')
                    # Fallback to original QR method
                    mat = higher_tasks.a_
                    Q, R = torch.linalg.qr(mat.T, mode='complete') 
                    rank = torch.linalg.matrix_rank(R)
                    n = mat.shape[1]
                    if rank < n:
                        self.stacked_z_prev_ = Q[:, rank:]
                    else:
                        self.stacked_z_prev_ = torch.zeros((n, 0), device=self.device, dtype=self.dtype)
            else:
                # No constraints from higher problem, use full space
                n = max(t.a_.shape[1] if t.a_.numel() else 0, t.d_.shape[1] if t.d_.numel() else 0)
                self.stacked_z_prev_ = torch.eye(n, device=self.device, dtype=self.dtype)
                print(f'init_vars: no higher constraints, using identity matrix {n}x{n}')

            # num_decision_vars_ should reflect the space we can actually optimize in.
            # If stacked_z_prev_ has zero columns, we'll later use regularization, so
            # we should base this on the original decision space size rather than
            # the nullspace size to ensure the optimization matrices are sized correctly.
            if self.stacked_z_prev_.shape[1] == 0:
                # No nullspace from higher priority, but we'll use regularization
                # so we need the full original space size for proper matrix dimensions
                self.num_decision_vars_ = self.stacked_z_prev_.shape[0]
            else:
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
        
        if self.has_eq_constraints_:
            a_curr_z_prev = self.task_.a_ @ self.stacked_z_prev_
            # Apply task weight to the objective function
            task_weight = self.task_weight_ * self.task_.weight_
            # Handle case where stacked_z_prev_ has zero columns (no nullspace)
            # In this case a_curr_z_prev will be [m, 0], so we use a different formula
            if a_curr_z_prev.shape[1] == 0:
                # No projectable space, use task jacobian directly in regularized form
                # Use the original task jacobian size for this case
                original_size = self.task_.a_.shape[1]
                z_t_a_t_a_z = task_weight * (self.task_.a_.T @ self.task_.a_) + 1e-12 * torch.eye(original_size, device=self.device, dtype=self.dtype)
                # Make sure it matches our expected dimension
                if z_t_a_t_a_z.shape[0] != nx:
                    # This shouldn't happen if our logic is correct, but let's be safe
                    z_t_a_t_a_z = torch.zeros((nx, nx), device=self.device, dtype=self.dtype)
            else:
                # Normal case: use projected jacobian
                # The size should match nx (which is now the nullspace dimension)  
                z_t_a_t_a_z = task_weight * (a_curr_z_prev.T @ a_curr_z_prev) + 1e-12 * torch.eye(a_curr_z_prev.shape[1], device=self.device, dtype=self.dtype)
        else:
            z_t_a_t_a_z = torch.zeros((nx, nx), device=self.device, dtype=self.dtype)

        # Build H matrix with dimensions based on the computed z_t_a_t_a_z
        actual_nx = z_t_a_t_a_z.shape[0]
        zdim = actual_nx + nv
        h = torch.zeros((zdim, zdim), device=self.device, dtype=self.dtype)
        
        # top-left, top-right, bottom-left, bottom-right
        top_left = z_t_a_t_a_z
        top_right = torch.zeros((actual_nx, nv), device=self.device, dtype=self.dtype)
        bottom_left = torch.zeros((nv, actual_nx), device=self.device, dtype=self.dtype)
        bottom_right = self.eye_nv_nv_

        h[:actual_nx, :actual_nx] = top_left
        if nv > 0:
            h[:actual_nx, actual_nx:] = top_right
            h[actual_nx:, :actual_nx] = bottom_left
            h[actual_nx:, actual_nx:] = bottom_right

        self.h_ = h

    def build_c_vector(self):
        nx = self.num_decision_vars_
        nv = self.num_slack_vars_

        if self.has_eq_constraints_:
            a_z_prev = self.task_.a_ @ self.stacked_z_prev_
            # Apply task weight to the gradient
            task_weight = self.task_weight_ * self.task_.weight_
            # Handle case where stacked_z_prev_ has zero columns (no nullspace)
            if a_z_prev.shape[1] == 0:
                # No projectable space, compute gradient directly from task residual
                temp = task_weight * (self.task_.a_.T @ (self.task_.a_ @ self.x_prev_ - self.task_.b_))
            else:
                residual = self.task_.a_ @ self.x_prev_ - self.task_.b_
                temp = task_weight * (a_z_prev.T @ residual)
        else:
            temp = torch.zeros((nx,), device=self.device, dtype=self.dtype)

        # Build c vector with dimensions based on the computed temp
        actual_nx = temp.shape[0] if temp.numel() > 0 else nx
        c = torch.zeros((actual_nx + nv,), device=self.device, dtype=self.dtype)
        
        c[:actual_nx] = temp
        # slack part stays zero (if any)
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
        # Build stacked_z_ matrix for hierarchical QP nullspace projection
        if self.has_eq_constraints_:
            assert self.task_.a_.shape[1] > 0
            
            # Special case: if this is a standalone HoQp (no higher priority tasks), 
            # we don't need nullspace projection - just solve directly
            if self.higher_problem_ is None:
                n = self.task_.a_.shape[1]
                self.stacked_z_ = torch.eye(n, device=self.device, dtype=self.dtype)
                # print(f'build_z_matrix: standalone problem, using full identity matrix {n}x{n}')
                return
            
            # For hierarchical problems, we need to compute the nullspace correctly
            if self.stacked_z_prev_.shape[1] == 0:
                # No free DOF from higher priority - current task is overconstrained
                # But we still need to allow the task to contribute to the solution
                # Use the task's own DOF as the active space
                n = self.task_.a_.shape[1]
                self.stacked_z_ = torch.zeros((n, 0), device=self.device, dtype=self.dtype)
                print(f'build_z_matrix: no nullspace from higher priority, empty nullspace')
                return
                
            # Compute nullspace while preserving DOF structure
            # Key insight: we want to find which DOF are still free after applying current task constraints
            # within the space allowed by higher priority tasks
            
            try:
                # Project current task constraint into previous nullspace
                A_proj = self.task_.a_ @ self.stacked_z_prev_  # Shape: [task_constraints, prev_free_dof]
                
                if A_proj.shape[0] == 0 or A_proj.shape[1] == 0:
                    # No constraints or no previous free DOF
                    self.stacked_z_ = self.stacked_z_prev_
                    print('build_z_matrix: no projected constraints, using previous nullspace')
                    return
                    
                # Use QR decomposition for stable nullspace computation
                Q, R = torch.linalg.qr(A_proj.T)  # QR of transpose to work with columns
                
                # Determine rank from R matrix
                rank = 0
                for i in range(min(R.shape[0], R.shape[1])):
                    if torch.abs(R[i, i]) > 1e-12:
                        rank += 1
                    else:
                        break
                        
                free_dofs = A_proj.shape[1] - rank
                # print(f'build_z_matrix: QR nullspace with {free_dofs} free DOF (rank={rank})')
                
                if free_dofs > 0:
                    # Nullspace basis in the reduced space
                    Z_local = Q[:, rank:]  # Shape: [prev_free_dof, remaining_free_dof]
                    # Map back to full DOF space
                    self.stacked_z_ = self.stacked_z_prev_ @ Z_local
                else:
                    # Current task fully constrains remaining DOF
                    n = self.task_.a_.shape[1]
                    self.stacked_z_ = torch.zeros((n, 0), device=self.device, dtype=self.dtype)
                    print('build_z_matrix: current task fully constrains remaining space')
                    
            except Exception as e:
                print(f'build_z_matrix: QR decomposition failed: {e}, using fallback')
                # Fallback: no nullspace
                n = self.task_.a_.shape[1]
                self.stacked_z_ = torch.zeros((n, 0), device=self.device, dtype=self.dtype)
                    
        else:
            # No constraints, use previous nullspace directly
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

    def update_decision_vars_count(self):
        """Update num_decision_vars_ based on the actual nullspace dimension."""
        if hasattr(self, 'stacked_z_') and self.stacked_z_.numel() > 0:
            self.num_decision_vars_ = self.stacked_z_.shape[1]
        # Update dependent matrices
        self.zero_nv_nx_ = torch.zeros((self.num_slack_vars_, self.num_decision_vars_), device=self.device, dtype=self.dtype)

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
