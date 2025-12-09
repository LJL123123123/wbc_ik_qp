"""PositionTask (python)

This module provides a lightweight PositionTask implementation that mirrors
the behaviour of the C++ `placo::kinematics::PositionTask` but uses a
fake pinocchio-like robot interface for kinematics. It's intentionally
minimal: the goal is to provide `update()` which computes `A` and `b` like
the original class and a `mask` attribute using `AxisesMask`.

Usage:
  task = PositionTask(frame_index=0, target_world=[0,0,0], robot=fake_robot)
  task.update()
  A = task.A  # masked jacobian
  b = task.b  # masked error
"""

from typing import Optional, Sequence
import torch
from wbc_ik_qp.tools.axises_mask import AxisesMask


class FakeKinematicRobot:
    """Very small fake robot exposing the methods used by PositionTask.

    - N: number of generalized coordinates
    - get_T_world_frame(frame_index) -> (R: 3x3 torch, t: 3-tensor)
    - frame_jacobian(frame_index, option) -> (3 x N) torch matrix
    """

    def __init__(self, N: int, device=None, dtype=torch.float64):
        self.N = N
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype

    def get_T_world_frame(self, frame_index: int):
        # Return identity rotation and a translation that depends on frame_index
        R = torch.eye(3, device=self.device, dtype=self.dtype)
        t = torch.zeros((3,), device=self.device, dtype=self.dtype)
        # small deterministic offset so tasks differ with frame
        t += 0.1 * (frame_index % 10)
        return R, t

    def frame_jacobian(self, frame_index: int, option=None):
        # Return a simple 3 x N jacobian: identity on first 3 dofs if available,
        # zeros elsewhere. This is intentionally minimal for tests.
        J = torch.zeros((3, self.N), device=self.device, dtype=self.dtype)
        for i in range(min(3, self.N)):
            J[i, i] = 1.0
        return J


class PositionTask:
    """Minimal python PositionTask compatible with the project's task API."""

    def __init__(self, frame_index: int, target_world: Sequence[float], robot: Optional[FakeKinematicRobot] = None,
                 device=None, dtype=torch.float64):
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        self.frame_index = int(frame_index)
        self.target_world = torch.as_tensor(target_world, device=self.device, dtype=self.dtype).view(3)
        self.robot = robot if robot is not None else FakeKinematicRobot(18, device=self.device, dtype=self.dtype)

        # Output matrices (set by update)
        self.A = torch.zeros((0, 0), device=self.device, dtype=self.dtype)
        self.b = torch.zeros((0,), device=self.device, dtype=self.dtype)

        # mask used to select axes
        self.mask = AxisesMask(device=self.device, dtype=self.dtype)

    def update(self):
        """Compute A and b using the robot kinematics and apply the mask.

        A will be the masked top-3 rows of the frame jacobian (shape m x N) and
        b will be the masked position error (length m), where m is the number
        of selected axes in the mask.
        """
        R_world_frame, t_world_frame = self.robot.get_T_world_frame(self.frame_index)

        # mask expects R_local_world = (local frame -> world)?? In the C++ code
        # they use T_world_frame.linear().transpose(), so we copy that behaviour
        # and set mask.R_local_world accordingly.
        # use .T for PyTorch transpose
        self.mask.R_local_world = R_world_frame.T.to(dtype=self.dtype)

        error = self.target_world - t_world_frame

        J = self.robot.frame_jacobian(self.frame_index, option='LOCAL_WORLD_ALIGNED')
        # top 3 rows represent position jacobian
        J_pos = J[0:3, :]

        # Apply mask to rows
        A_masked = self.mask.apply(J_pos)
        b_masked = self.mask.apply(error)

        self.A = A_masked
        self.b = b_masked

    def type_name(self):
        return "position"

    def error_unit(self):
        return "m"

            