"""ho_qp.py

这个文件是本项目的 HoQP（分层二次规划）实现。

本次重构目标：对齐你提供的开源参考实现 `hoqp_e.py` 的数学结构：

- 每一层使用 `HoQPLevel`：
  - 公式化：计算 $Z_p$、$x^*$、$v_p^*$，并构造 $H,c,D,f$
  - 求解：解 QP 得到 $z_p, v_p$
  - 后处理：更新 $Z_{p+} = Z_p \cdot \mathcal{N}(A Z_p)$，以及堆叠 slack

- 仍然使用 torch 张量（可在 GPU 上跑），并优先使用 ReLUQP-py 做 QP 求解；失败时回退到一个
  仅用于开发/调试的稠密线性解（注意：该回退不保证满足不等式）。

同时，为了不一次性改爆上层调用，本文件保留：
- `Task`：字段沿用 a_/b_/d_/f_ 命名（兼容现有 wbc/task 代码）。
- `HoQp(task, higher_problem=None)`：作为 `HoQPLevel` 的兼容封装，外部接口保留
  getSolutions/getStackedTasks/getStackedSlackSolutions 等。
- 新增 `HoQP`：更贴近 `hoqp_e` 的“按 priority 添加 task 生成器并 solve”的用法，方便你后续把
  wbc 的多任务组合改得更清晰。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch


def _safe_sqrt_weight(w: float, *, device, dtype) -> torch.Tensor:
    """Return sqrt(max(w, eps)) as tensor."""
    try:
        wf = float(w)
    except Exception:
        wf = 1.0
    wf = max(wf, 0.0)
    # keep positive to avoid killing the objective completely
    eps = 1e-12
    return torch.sqrt(torch.tensor(wf + eps, device=device, dtype=dtype))


def _symmetrize(H: torch.Tensor) -> torch.Tensor:
    # Numerical noise may break symmetry; keep solver happy.
    return 0.5 * (H + H.transpose(0, 1))


def _max_abs(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(x)).detach().cpu())


def _estimate_min_eig_sym(H: torch.Tensor) -> float:
    """Best-effort estimate of minimum eigenvalue for symmetric matrix."""
    if H.numel() == 0:
        return 0.0
    try:
        # eigvalsh is for symmetric/Hermitian matrices
        vals = torch.linalg.eigvalsh(_symmetrize(H))
        return float(torch.min(vals).detach().cpu())
    except Exception:
        return float('nan')


def _estimate_cond_sym(H: torch.Tensor) -> float:
    """Best-effort condition number estimate for symmetric PSD-ish matrix."""
    if H.numel() == 0:
        return 0.0
    try:
        vals = torch.linalg.eigvalsh(_symmetrize(H))
        vmax = torch.max(vals)
        vmin = torch.min(vals)
        # avoid divide-by-zero
        return float((vmax / torch.clamp(vmin, min=1e-18)).detach().cpu())
    except Exception:
        return float('nan')


def _as_tensor(x, *, device, dtype) -> torch.Tensor:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


@dataclass
class Task:
    """一个 HoQP task。

    参考 hoqp_e 定义：
      A x - b = w
      D x - f <= v

    其中 slack (w, v) 由 HoQP 内部隐式最小化。

    这里仍沿用本仓库历史命名：a_/b_/d_/f_。
    """

    a_: torch.Tensor
    b_: torch.Tensor
    d_: torch.Tensor
    f_: torch.Tensor
    weight_: float = 1.0

    # 兼容之前的构造方式：Task(a=..., b=..., d=..., f=..., num_decision_vars=..., device=..., dtype=..., weight=...)
    def __init__(
        self,
        a_: Optional[torch.Tensor] = None,
        b_: Optional[torch.Tensor] = None,
        d_: Optional[torch.Tensor] = None,
        f_: Optional[torch.Tensor] = None,
        weight_: float = 1.0,
        # legacy aliases
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        d: Optional[torch.Tensor] = None,
        f: Optional[torch.Tensor] = None,
        num_decision_vars: Optional[int] = None,
        device=None,
        dtype: Optional[torch.dtype] = None,
        weight: Optional[float] = None,
    ):
        # Important: do NOT default to CUDA automatically.
        # Let the caller decide the device, otherwise tasks created on a CPU
        # code-path can silently end up on cuda:0 and break matmul later.
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float64

        # prefer explicit a_/b_..., fall back to legacy names
        A = a_ if a_ is not None else a
        B = b_ if b_ is not None else b
        D = d_ if d_ is not None else d
        F = f_ if f_ is not None else f

        if weight is not None:
            weight_ = float(weight)
        self.weight_ = float(weight_)

        if A is None:
            if num_decision_vars is None:
                A = torch.zeros((0, 0), device=device, dtype=dtype)
            else:
                A = torch.zeros((0, int(num_decision_vars)), device=device, dtype=dtype)
        if D is None:
            if num_decision_vars is None:
                D = torch.zeros((0, 0), device=device, dtype=dtype)
            else:
                D = torch.zeros((0, int(num_decision_vars)), device=device, dtype=dtype)

        if B is None:
            B = torch.zeros((int(A.shape[0]),), device=device, dtype=dtype)
        if F is None:
            F = torch.zeros((int(D.shape[0]),), device=device, dtype=dtype)

        self.a_ = _as_tensor(A, device=device, dtype=dtype)
        self.b_ = _as_tensor(B, device=device, dtype=dtype)
        self.d_ = _as_tensor(D, device=device, dtype=dtype)
        self.f_ = _as_tensor(F, device=device, dtype=dtype)

    @staticmethod
    def empty(n_des: int, *, device=None, dtype=torch.float64) -> "Task":
        device = device if device is not None else torch.device("cpu")
        return Task(
            a_=torch.zeros((0, n_des), device=device, dtype=dtype),
            b_=torch.zeros((0,), device=device, dtype=dtype),
            d_=torch.zeros((0, n_des), device=device, dtype=dtype),
            f_=torch.zeros((0,), device=device, dtype=dtype),
            weight_=1.0,
        )

    def is_valid(self, n_des: int) -> bool:
        return (
            self.a_.dim() == 2
            and self.d_.dim() == 2
            and self.a_.shape[1] == n_des
            and self.d_.shape[1] == n_des
            and self.b_.shape[0] == self.a_.shape[0]
            and self.f_.shape[0] == self.d_.shape[0]
        )

    def update(self, other: "Task") -> None:
        self.a_ = other.a_
        self.b_ = other.b_
        self.d_ = other.d_
        self.f_ = other.f_
        self.weight_ = other.weight_

    def __add__(self, other: "Task") -> "Task":
        # 行拼接（和 hoqp_e 一致）。
        if other is None:
            return Task(self.a_, self.b_, self.d_, self.f_, self.weight_)
        if self.a_.numel() == 0 and self.b_.numel() == 0 and self.d_.numel() == 0 and self.f_.numel() == 0:
            return Task(other.a_, other.b_, other.d_, other.f_, other.weight_)
        if other.a_.numel() == 0 and other.b_.numel() == 0 and other.d_.numel() == 0 and other.f_.numel() == 0:
            return Task(self.a_, self.b_, self.d_, self.f_, self.weight_)
        # Be robust to mixed device/dtype (common when some tasks come from
        # legacy code paths defaulting to CPU). We align `other` to `self`.
        dev = self.a_.device
        dt = self.a_.dtype
        oa = _as_tensor(other.a_, device=dev, dtype=dt)
        ob = _as_tensor(other.b_, device=dev, dtype=dt)
        od = _as_tensor(other.d_, device=dev, dtype=dt)
        of = _as_tensor(other.f_, device=dev, dtype=dt)
        return Task(
            a_=torch.cat([self.a_, oa], dim=0),
            b_=torch.cat([self.b_, ob], dim=0),
            d_=torch.cat([self.d_, od], dim=0),
            f_=torch.cat([self.f_, of], dim=0),
            weight_=self.weight_ + other.weight_,
            device=dev,
            dtype=dt,
        )


def _null_space(A: torch.Tensor, rtol: float = 1e-12) -> torch.Tensor:
    """返回 A 的零空间基（列向量），对齐 scipy.linalg.null_space 行为。"""
    m, n = A.shape
    if m == 0:
        return torch.eye(n, device=A.device, dtype=A.dtype)
    # SVD: A = U S Vh, nullspace basis is V[:, rank:]
    U, S, Vh = torch.linalg.svd(A, full_matrices=True)
    if S.numel() == 0:
        return torch.eye(n, device=A.device, dtype=A.dtype)
    tol = rtol * torch.max(S)
    rank = int(torch.sum(S > tol).item())
    V = Vh.transpose(0, 1)
    if rank >= n:
        return torch.zeros((n, 0), device=A.device, dtype=A.dtype)
    return V[:, rank:]


def _solve_qp_reluqp_or_fallback(
    *,
    H: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    device,
    dtype,
) -> torch.Tensor:
    """优先用 ReLUQP 求解；失败则回退到稠密线性解（仅供调试）。"""
    try:
        import reluqp.reluqpth as reluqpth

        model = reluqpth.ReLU_QP()
        model.setup(H, g, A, l, u, device=device, precision=dtype)
        results = model.solve()
        z = results.x
        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(z, device=device, dtype=dtype)
        return z.to(device=device, dtype=dtype)
    except Exception:
        # fallback: unconstrained optimum of quadratic
        reg = 1e-9
        I = torch.eye(H.shape[0], device=device, dtype=dtype)
        try:
            return torch.linalg.solve(H + reg * I, -g)
        except Exception:
            return -(torch.linalg.pinv(H + reg * I) @ g)


class HoQPLevel:
    """对齐 hoqp_e.HoQPLevel 的 torch/ReLUQP 版本。"""

    def __init__(
        self,
        task: Task,
        higher_level: Optional["HoQPLevel"] = None,
        *,
        device=None,
        dtype=torch.float64,
        # stability knobs
        use_task_weight: bool = True,
        damping: float = 1e-6,
        damping_auto: bool = True,
        scaling: str = "maxabs",  # 'none' | 'maxabs' | 'diag'
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        # 强制 task 在同一 device/dtype，避免 wbc 里混用 CPU/CUDA
        self.tasks: Task = Task(
            a_=_as_tensor(task.a_, device=self.device, dtype=self.dtype),
            b_=_as_tensor(task.b_, device=self.device, dtype=self.dtype),
            d_=_as_tensor(task.d_, device=self.device, dtype=self.dtype),
            f_=_as_tensor(task.f_, device=self.device, dtype=self.dtype),
            weight_=float(getattr(task, "weight_", 1.0)),
            device=self.device,
            dtype=self.dtype,
        )
        self.higher_level: Optional[HoQPLevel] = higher_level

        self.use_task_weight = bool(use_task_weight)
        self.damping = float(damping)
        self.damping_auto = bool(damping_auto)
        self.scaling = str(scaling)

        # dimensions
        self.nv: int = 0
        self.nz: int = 0

        # previous solution info
        self.v_p_star: torch.Tensor = torch.zeros((0,), device=self.device, dtype=self.dtype)
        self.x_star: torch.Tensor = torch.zeros((0,), device=self.device, dtype=self.dtype)
        self.Z_p: torch.Tensor = torch.zeros((0, 0), device=self.device, dtype=self.dtype)

        # solution
        self.z_p: torch.Tensor = torch.zeros((0,), device=self.device, dtype=self.dtype)
        self.v_p: torch.Tensor = torch.zeros((0,), device=self.device, dtype=self.dtype)

        # stacked tasks
        self.stacked_tasks_prev: Task = Task.empty(0, device=self.device, dtype=self.dtype)
        self.stacked_tasks: Task = Task.empty(0, device=self.device, dtype=self.dtype)

        # QP matrices
        self.H: torch.Tensor = torch.zeros((0, 0), device=self.device, dtype=self.dtype)
        self.c: torch.Tensor = torch.zeros((0,), device=self.device, dtype=self.dtype)
        self.D: torch.Tensor = torch.zeros((0, 0), device=self.device, dtype=self.dtype)
        self.f: torch.Tensor = torch.zeros((0,), device=self.device, dtype=self.dtype)
        self.Z_p_plus: torch.Tensor = torch.zeros((0, 0), device=self.device, dtype=self.dtype)

        # debug
        self._debug_last: Dict[str, float] = {}

    @property
    def has_eq_constraint(self) -> bool:
        return self.tasks.a_.shape[0] > 0

    @property
    def has_ineq_constraint(self) -> bool:
        return self.tasks.d_.shape[0] > 0

    def _formulate(self) -> None:
        self.nv = int(self.tasks.d_.shape[0])
        if self.higher_level is not None:
            # 注意：higher_level 可能在不同 device/dtype 上（例如旧代码构造时没传 device），
            # 必须在这里统一搬移，避免后续 matmul 触发 CPU/CUDA 混用。
            self.Z_p = _as_tensor(self.higher_level.Z_p_plus, device=self.device, dtype=self.dtype)
            self.v_p_star = _as_tensor(self.higher_level.v_p_star, device=self.device, dtype=self.dtype)
            self.x_star = _as_tensor(self.higher_level.get_solution(), device=self.device, dtype=self.dtype)
            self.nz = int(self.Z_p.shape[1])
            self.stacked_tasks_prev = Task(
                a_=_as_tensor(self.higher_level.stacked_tasks.a_, device=self.device, dtype=self.dtype),
                b_=_as_tensor(self.higher_level.stacked_tasks.b_, device=self.device, dtype=self.dtype),
                d_=_as_tensor(self.higher_level.stacked_tasks.d_, device=self.device, dtype=self.dtype),
                f_=_as_tensor(self.higher_level.stacked_tasks.f_, device=self.device, dtype=self.dtype),
                weight_=float(getattr(self.higher_level.stacked_tasks, "weight_", 0.0)),
            )
        else:
            # 第一层：Z_p = I
            self.nz = int(self.tasks.a_.shape[1] if self.tasks.a_.numel() else self.tasks.d_.shape[1])
            self.Z_p = torch.eye(self.nz, device=self.device, dtype=self.dtype)
            self.v_p_star = torch.zeros((0,), device=self.device, dtype=self.dtype)
            self.x_star = torch.zeros((self.nz,), device=self.device, dtype=self.dtype)
            self.stacked_tasks_prev = Task.empty(self.nz, device=self.device, dtype=self.dtype)

        # 兜底：确保递推状态一律在当前 device/dtype 上（防止 higher_level 里某些字段没同步）。
        self.Z_p = _as_tensor(self.Z_p, device=self.device, dtype=self.dtype)
        self.x_star = _as_tensor(self.x_star, device=self.device, dtype=self.dtype)
        self.v_p_star = _as_tensor(self.v_p_star, device=self.device, dtype=self.dtype)
        self.stacked_tasks_prev = Task(
            a_=_as_tensor(self.stacked_tasks_prev.a_, device=self.device, dtype=self.dtype),
            b_=_as_tensor(self.stacked_tasks_prev.b_, device=self.device, dtype=self.dtype),
            d_=_as_tensor(self.stacked_tasks_prev.d_, device=self.device, dtype=self.dtype),
            f_=_as_tensor(self.stacked_tasks_prev.f_, device=self.device, dtype=self.dtype),
            weight_=float(getattr(self.stacked_tasks_prev, "weight_", 0.0)),
        )

        self.stacked_tasks = self.tasks + self.stacked_tasks_prev

        self._build_h()
        self._build_c()
        self._build_d()
        self._build_f()

        # final touches: symmetrize + add damping + scaling for ReLUQP stability
        self._stabilize_qp_matrices()

    def _get_weight_sqrt(self) -> torch.Tensor:
        if not self.use_task_weight:
            return torch.tensor(1.0, device=self.device, dtype=self.dtype)
        # Always create the scalar weight on *this* level's device/dtype.
        return _safe_sqrt_weight(getattr(self.tasks, "weight_", 1.0), device=self.device, dtype=self.dtype)

    def _stabilize_qp_matrices(self) -> None:
        # Symmetrize Hessian (numerical safety)
        self.H = _symmetrize(self.H)

        # Damping only on the z-block, not on slack block (slack already has I).
        # This is the classic Levenberg-Marquardt style regularization.
        if self.nz > 0:
            lam = max(self.damping, 0.0)
            if self.damping_auto:
                # Heuristic: scale damping with average diagonal magnitude.
                try:
                    diag_mean = torch.mean(torch.abs(torch.diag(self.H[: self.nz, : self.nz])))
                    lam = float((lam + 1e-12) * (diag_mean.detach().cpu().item() + 1.0))
                except Exception:
                    pass
            Izz = torch.eye(self.nz, device=self.device, dtype=self.dtype)
            self.H[: self.nz, : self.nz] = self.H[: self.nz, : self.nz] + lam * Izz

        # Scaling: keep operator magnitudes in a reasonable range for first-order solvers.
        # We apply a global scaling alpha to (H, c) only. This doesn't change the minimizer.
        # NOTE: constraints (A,l,u) are not scaled here to keep interpretation simple.
        scale_alpha = 1.0
        if self.scaling.lower() == "maxabs":
            m = _max_abs(self.H)
            if m > 0:
                scale_alpha = 1.0 / m
        elif self.scaling.lower() == "diag":
            # very simple diagonal scaling: normalize average diagonal to 1
            try:
                d = torch.abs(torch.diag(self.H))
                dmean = float(torch.mean(d).detach().cpu())
                if dmean > 0:
                    scale_alpha = 1.0 / dmean
            except Exception:
                pass

        if scale_alpha != 1.0:
            self.H = self.H * scale_alpha
            self.c = self.c * scale_alpha

        # store debug metrics
        try:
            self._debug_last["H_maxabs"] = _max_abs(self.H)
            self._debug_last["c_norm"] = float(torch.linalg.norm(self.c).detach().cpu()) if self.c.numel() else 0.0
            if self.nz > 0:
                Hzz = self.H[: self.nz, : self.nz]
                self._debug_last["Hzz_min_eig"] = _estimate_min_eig_sym(Hzz)
                self._debug_last["Hzz_cond"] = _estimate_cond_sym(Hzz)
            else:
                self._debug_last["Hzz_min_eig"] = 0.0
                self._debug_last["Hzz_cond"] = 0.0
        except Exception:
            pass

    def _build_h(self) -> None:
        if self.has_eq_constraint:
            w = self._get_weight_sqrt()
            # ensure scalar broadcast works consistently
            if w.dim() != 0:
                w = w.reshape(())
            t = (w * self.tasks.a_) @ self.Z_p
            temp = t.transpose(0, 1) @ t
        else:
            temp = torch.zeros((self.nz, self.nz), device=self.device, dtype=self.dtype)

        self.H = torch.block_diag(temp, torch.eye(self.nv, device=self.device, dtype=self.dtype))

    def _build_c(self) -> None:
        if self.has_eq_constraint:
            w = self._get_weight_sqrt()
            if w.dim() != 0:
                w = w.reshape(())
            r = (w * (self.tasks.a_ @ self.x_star - self.tasks.b_))
            AwT = (w * self.tasks.a_).transpose(0, 1)
            grad = self.Z_p.transpose(0, 1) @ AwT @ r
        else:
            grad = torch.zeros((self.nz,), device=self.device, dtype=self.dtype)

        self.c = torch.cat([grad, torch.zeros((self.nv,), device=self.device, dtype=self.dtype)], dim=0)

    def _build_d(self) -> None:
        # 对齐 hoqp_e 的 block：
        # [ 0      -I ]
        # [ D_prev Z_p   0]
        # [ D_curr Z_p  -I]
        prev_D = self.stacked_tasks_prev.d_
        curr_D = self.tasks.d_ if self.has_ineq_constraint else torch.zeros((0, prev_D.shape[1]), device=self.device, dtype=self.dtype)

        rows = self.nv + prev_D.shape[0] + curr_D.shape[0]
        cols = self.nz + self.nv
        self.D = torch.zeros((rows, cols), device=self.device, dtype=self.dtype)

        # top block
        if self.nv > 0:
            self.D[: self.nv, self.nz : self.nz + self.nv] = -torch.eye(self.nv, device=self.device, dtype=self.dtype)

        # middle block (prev)
        r0 = self.nv
        r1 = r0 + prev_D.shape[0]
        if prev_D.shape[0] > 0:
            self.D[r0:r1, : self.nz] = prev_D @ self.Z_p

        # bottom block (curr)
        r2 = r1
        r3 = r2 + curr_D.shape[0]
        if curr_D.shape[0] > 0:
            self.D[r2:r3, : self.nz] = curr_D @ self.Z_p
            # 注意：每层 slack 变量只对应本层 nv，所以 bottom 的 -I 是 nv x nv
            # curr_D.shape[0] == nv 时是方阵；若不等式行数 != nv，本实现仍按 hoqp_e 假设 nv 行。
            if curr_D.shape[0] == self.nv and self.nv > 0:
                self.D[r2:r3, self.nz : self.nz + self.nv] = -torch.eye(self.nv, device=self.device, dtype=self.dtype)
            elif self.nv > 0:
                # 兼容：只填充对角能填的部分
                k = min(curr_D.shape[0], self.nv)
                self.D[r2 : r2 + k, self.nz : self.nz + k] = -torch.eye(k, device=self.device, dtype=self.dtype)

    def _build_f(self) -> None:
        # 参考 hoqp_e：
        # f = [0,
        #      f_prev - D_prev x* + v*_prev,
        #      f_curr - D_curr x*]

        # prev block
        prev_D = self.stacked_tasks_prev.d_
        prev_f = self.stacked_tasks_prev.f_
        if prev_D.shape[0] > 0:
            second = prev_f - prev_D @ self.x_star + self.v_p_star
        else:
            second = torch.zeros((0,), device=self.device, dtype=self.dtype)

        # curr block
        if self.has_ineq_constraint:
            third = self.tasks.f_ - self.tasks.d_ @ self.x_star
        else:
            third = torch.zeros((0,), device=self.device, dtype=self.dtype)

        self.f = torch.cat([
            torch.zeros((self.nv,), device=self.device, dtype=self.dtype),
            second,
            third,
        ], dim=0)

    def _solve(self) -> bool:
        # 将 D z <= f 转成 l <= A z <= u
        A = self.D
        u = self.f
        l = torch.full_like(u, -float("inf"))

        z = _solve_qp_reluqp_or_fallback(H=self.H, g=self.c, A=A, l=l, u=u, device=self.device, dtype=self.dtype)
        if z.numel() == 0:
            return False

        self.z_p = z[: self.nz]
        self.v_p = z[self.nz : self.nz + self.nv] if self.nv > 0 else torch.zeros((0,), device=self.device, dtype=self.dtype)
        return True

    def _post_process(self) -> None:
        if self.has_eq_constraint:
            kernel = _null_space(self.tasks.a_ @ self.Z_p)
            self.Z_p_plus = self.Z_p @ kernel
        else:
            self.Z_p_plus = self.Z_p

        if self.has_ineq_constraint:
            self.v_p_star = torch.cat([self.v_p_star, self.v_p], dim=0)
        else:
            self.v_p_star = self.v_p_star

        # debug
        try:
            self._debug_last["nz"] = float(self.nz)
            self._debug_last["nv"] = float(self.nv)
            self._debug_last["Z_p_plus_dim"] = float(self.Z_p_plus.shape[1]) if self.Z_p_plus.numel() else 0.0
        except Exception:
            pass

    def get_solution(self) -> torch.Tensor:
        return self.x_star + self.Z_p @ self.z_p

    def solve(self) -> bool:
        self._formulate()
        ok = self._solve()
        if ok:
            self._post_process()
        return ok

    def get_last_debug(self) -> Dict[str, float]:
        return dict(self._debug_last)


class HoQP:
    """更贴近 hoqp_e 的管理器：按 priority 收集 task_builder 并逐层求解。"""

    def __init__(self, n_des: int, *, device=None, dtype=torch.float64):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.n_des = int(n_des)
        self.task_levels: List[List[Callable[[], Task]]] = []
        self.levels: List[HoQPLevel] = []
        self.lowest_priority = -1

        # stability knobs (can be adjusted by caller after construction)
        self.use_task_weight: bool = True
        self.damping: float = 1e-6
        self.damping_auto: bool = True
        self.scaling: str = "maxabs"

    def add_task(self, priority: int, task_builder: Callable[[], Task]) -> None:
        if priority < 0 or priority > self.lowest_priority + 1:
            raise ValueError("Priority must be sequential (0..lowest+1).")

        if priority > self.lowest_priority:
            self.lowest_priority = priority
            higher = self.levels[priority - 1] if priority > 0 else None
            self.levels.append(
                HoQPLevel(
                    Task.empty(self.n_des, device=self.device, dtype=self.dtype),
                    higher,
                    device=self.device,
                    dtype=self.dtype,
                    use_task_weight=self.use_task_weight,
                    damping=self.damping,
                    damping_auto=self.damping_auto,
                    scaling=self.scaling,
                )
            )
            self.task_levels.append([])
        self.task_levels[priority].append(task_builder)

    def solve(self) -> Optional[torch.Tensor]:
        failed = False
        for p in range(self.lowest_priority + 1):
            collected = Task.empty(self.n_des, device=self.device, dtype=self.dtype)
            for fn in self.task_levels[p]:
                t = fn()
                if t is None:
                    continue
                # 强制对齐 device/dtype
                t = Task(
                    a_=_as_tensor(t.a_, device=self.device, dtype=self.dtype),
                    b_=_as_tensor(t.b_, device=self.device, dtype=self.dtype),
                    d_=_as_tensor(t.d_, device=self.device, dtype=self.dtype),
                    f_=_as_tensor(t.f_, device=self.device, dtype=self.dtype),
                    weight_=float(getattr(t, "weight_", 1.0)),
                )
                if not t.is_valid(self.n_des):
                    raise ValueError(f"Invalid task shape at priority {p}: A{tuple(t.a_.shape)} b{tuple(t.b_.shape)} D{tuple(t.d_.shape)} f{tuple(t.f_.shape)} n_des={self.n_des}")
                collected = collected + t

            self.levels[p].tasks.update(collected)
            if not self.levels[p].solve():
                failed = True
                break
        return None if failed else self.levels[-1].get_solution()


class HoQp:
    """兼容层：保持旧用法 HoQp(task, higher_problem) 但内部使用 HoQPLevel。"""

    def __init__(
        self,
        task: Task,
        higher_problem: Optional["HoQp"] = None,
        device=None,
        dtype=torch.float64,
        task_weight: float = 1.0,
        # stability knobs
        use_task_weight: bool = True,
        damping: float = 1e-6,
        damping_auto: bool = True,
        scaling: str = "maxabs",
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        # 兼容旧字段
        if not isinstance(task, Task):
            raise TypeError("task must be ho_qp.Task")
        # 应用权重（与历史行为一致：权重只影响目标，不改约束）
        self.task_ = Task(
            a_=_as_tensor(task.a_, device=self.device, dtype=self.dtype),
            b_=_as_tensor(task.b_, device=self.device, dtype=self.dtype),
            d_=_as_tensor(task.d_, device=self.device, dtype=self.dtype),
            f_=_as_tensor(task.f_, device=self.device, dtype=self.dtype),
            weight_=float(task.weight_) * float(task_weight),
        )
        self.higher_problem_ = higher_problem
        higher_level = higher_problem._level if higher_problem is not None else None
        self._level = HoQPLevel(
            self.task_,
            higher_level,
            device=self.device,
            dtype=self.dtype,
            use_task_weight=use_task_weight,
            damping=damping,
            damping_auto=damping_auto,
            scaling=scaling,
        )
        ok = self._level.solve()
        if not ok:
            raise RuntimeError("HoQp level failed to solve")

    def getStackedZMatrix(self):
        return self._level.Z_p_plus

    def getStackedTasks(self):
        return self._level.stacked_tasks

    def getStackedSlackSolutions(self):
        return self._level.v_p_star

    def getSlackedNumVars(self):
        return int(self._level.stacked_tasks.d_.shape[0])

    def getSolutions(self):
        return self._level.get_solution()

    def get_last_debug(self):
        return self._level.get_last_debug()

    def get_debug_chain(self):
        chain = []
        if self.higher_problem_ is not None:
            chain.extend(self.higher_problem_.get_debug_chain())
        chain.append(self.get_last_debug())
        return chain


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    n = 6
    meq = 2
    mineq = 3
    a = torch.randn((meq, n), device=dev, dtype=dtype)
    b = torch.randn((meq,), device=dev, dtype=dtype)
    d = torch.randn((mineq, n), device=dev, dtype=dtype)
    f = torch.randn((mineq,), device=dev, dtype=dtype)
    t = Task(a_=a, b_=b, d_=d, f_=f)
    h = HoQp(t, device=dev, dtype=dtype)
    print("sol shape:", h.getSolutions().shape)
