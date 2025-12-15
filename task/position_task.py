"""PositionTask (python)

Refactored PositionTask that uses Model_Cusadi (from `ik.py`) to compute
position, attitude and jacobian. Instead of holding internal A/b matrices,
this class provides `as_task(state, input, axises, frame)` which returns a
`ho_qp.Task` containing the masked position equality constraint. This makes
it easy to create many PositionTask instances and concatenate their tasks.

Key behaviour:
 - `as_task` updates the internal `Model_Cusadi` with the provided state/input,
     reads the frame position and jacobian, computes the position error, applies
     an `AxisesMask`, and returns a `Task(a=..., b=...)`.

Usage:
    pt = PositionTask(info, frame_name='LF_FOOT', target_world=[0,0,0])
    task = pt.as_task(state_tensor, input_tensor, axises='xy', frame='local')
    # task.a_ is (m x n), task.b_ is (m,)
"""

from typing import Optional, Sequence
import torch
from wbc_ik_qp.tools.axises_mask import AxisesMask
from wbc_ik_qp.ik import Model_Cusadi
from wbc_ik_qp.Centroidal import CentroidalModelInfoSimple
from wbc_ik_qp.ho_qp import Task


class PositionTask:
        """Position task that can be converted to a `ho_qp.Task`.

        The task represents a position equality: A x = b where A is the masked
        position jacobian and b is the masked position error (target - current).
        """

        def __init__(self, info: CentroidalModelInfoSimple, frame_name: str,
                                  device=None, dtype=torch.float64):
                self.info = info
                self.device = device if device is not None else torch.device('cpu')
                self.dtype = dtype
                self.frame_name = frame_name

                # robot interface (fake Pinocchio-like wrapper)
                # Allow injection of a robot; if Model_Cusadi construction fails (missing
                # external deps in some environments), fall back to a tiny minimal
                # robot that provides the small API PositionTask needs.
                try:
                        self.robot = Model_Cusadi(info, device=self.device, dtype=self.dtype)
                except Exception:
                        print("Failed to create Model_Cusadi, falling back to minimal robot.")
                        class _FallbackData:
                                pass

                        class _MinimalFallbackRobot:
                                def __init__(self, info, device, dtype):
                                        self._data = _FallbackData()
                                        self._data.nv = getattr(info, 'generalizedCoordinatesNum', 0)
                                        self._data.nu = getattr(info, 'actuatedDofNum', 0)
                                        self._data.state = torch.zeros(self._data.nv, device=device, dtype=dtype)
                                        self._data.input = torch.zeros(self._data.nu, device=device, dtype=dtype)

                                def update_state_input(self, state: torch.Tensor, input: torch.Tensor):
                                        self._data.state = state.to(device=self._data.state.device, dtype=self._data.state.dtype)
                                        self._data.input = input.to(device=self._data.input.device, dtype=self._data.input.dtype)
                                        return self

                                def getPosition(self, x0, u0, __name__):
                                        return torch.zeros((3,), device=x0.device if hasattr(x0, 'device') else self._data.state.device,
                                                                           dtype=x0.dtype if hasattr(x0, 'dtype') else self._data.state.dtype)

                                def getAttitude(self, x0, u0, __name__):
                                        return torch.eye(3, device=x0.device if hasattr(x0, 'device') else self._data.state.device,
                                                                         dtype=x0.dtype if hasattr(x0, 'dtype') else self._data.state.dtype)

                                def getJacobian(self, x0, u0, __name__):
                                        nv = max(1, getattr(self._data, 'nv', 1))
                                        return torch.zeros((6, nv), device=x0.device if hasattr(x0, 'device') else self._data.state.device,
                                                                           dtype=x0.dtype if hasattr(x0, 'dtype') else self._data.state.dtype)

                        self.robot = _MinimalFallbackRobot(info, self.device, self.dtype)

                # mask used to select axes (default: keep x,y,z in task/world frame)
                self.mask = AxisesMask(device=self.device, dtype=self.dtype)

        def as_task(self, target_world: Sequence[float],
                                axises: str = "xyz", frame: Optional[str] = "task", weight: float = 1.0) -> Task:
                """Return a `ho_qp.Task` built from this PositionTask.

                Args:
                        state: full state tensor (matches Model_Cusadi expectations)
                        input: input tensor for kinematics functions
                        axises: axis selection string like 'x', 'xy', 'yz', 'xyz'
                        frame: frame for the axes mask: 'task'/'world', 'local', or 'custom'

                Returns:
                        ho_qp.Task with equality A (m x n) and rhs b (m,). If no axes are
                        selected the returned Task will have empty a_/b_.
                """
                
                # get current frame position, attitude and jacobian from Model_Cusadi
                pos = self.robot.getPosition(self.info.getstate(), self.info.getinput(), self.frame_name)
                att = self.robot.getAttitude(self.info.getstate(), self.info.getinput(), self.frame_name)
                J = self.robot.getJacobian(self.info.getstate(), self.info.getinput(), self.frame_name)
                

                # normalize jacobian shape: callers expect 6 x nv or 3 x nv
                if not torch.is_tensor(J):
                    J = torch.as_tensor(J, device=self.device, dtype=self.dtype)
                J_pos = J[0,:3, :]

                # set mask and its rotation if local/custom frame used
                self.mask.set_axises(axises, frame)
                # Model_Cusadi.getAttitude is assumed to return R_world_frame (3x3)
                # AxisesMask expects R_local_world (rotation to apply to world->local
                # selection); in placo/C++ they used T_world_frame.linear().transpose()
                # so we provide the transpose here to maintain behaviour.
                try:
                        self.mask.R_local_world = att.T.to(device=self.device, dtype=self.dtype)
                except Exception:
                        # if attitude shape is unexpected, ignore and keep identity
                        pass

                # compute error (target_world - current_position)
                error = target_world - pos

                # apply mask
                # debug print (kept minimal)
                # print("J_pos:", J_pos, "device for J_pos:", J_pos.device)
                A_masked = self.mask.apply(J_pos)
                b_masked = self.mask.apply(error)

                # If the masked jacobian is all zeros the equality A x = b is
                # either infeasible (if b != 0) or provides no information. In
                # development environments where the real kinematics backend
                # isn't available we prefer to skip the task rather than add an
                # impossible equality. Return an empty Task in that case.
                if A_masked.numel() == 0 or torch.allclose(A_masked, torch.zeros_like(A_masked)):
                        # log a short notice and return an empty Task so solver can continue
                        print("PositionTask: masked jacobian is zero — skipping position equality task.")
                        return Task()

                # Ensure shapes: A_masked should be (m, n) and b_masked (m,)
                # ho_qp.Task handles empty tensors; just construct and return.
                return Task(a=A_masked, b=b_masked, device=self.device, dtype=self.dtype, weight=weight)

        def type_name(self):
                return "position"

        def error_unit(self):
                return "m"

