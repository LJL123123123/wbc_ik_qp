"""ContactConstraintTask

Level 0 contact constraint for stance feet.

Implements equality constraints of the form:
    J_[i,b] Δx_b + J_[i,q] Δq = 0
where the decision variable x is assumed to be the *velocity-space increment*
(18 DoF): [base_vel(6); joint_vel(12)]

In this repo, foot PositionTask provides a position Jacobian J_pos with shape
(3, 18) (possibly after internal DOF filtering). We simply stack stance feet
linear Jacobians and set RHS to zero.

Note:
- This is a kinematic constraint; it does not require the full dynamics.
- We keep it lightweight and compatible with the current HoQP implementation.
"""

from __future__ import annotations

from typing import Dict, Iterable, List
import torch

from ho_qp import Task
from Centroidal import CentroidalModelInfoSimple
from ik import Model_Cusadi


class ContactConstraintTask:
    def __init__(self, info: CentroidalModelInfoSimple, device=None, dtype=torch.float64):
        self.info = info
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        self.robot = Model_Cusadi(info, device=self.device, dtype=self.dtype)

        # map short leg key -> frame name used by IK model
        self.leg_frame = {
            'LF': 'LF_FOOT',
            'RF': 'RF_FOOT',
            'LH': 'LH_FOOT',
            'RH': 'RH_FOOT',
        }

    def as_task(self, stance_legs: Iterable[str], weight: float = 1.0) -> Task:
        stance_legs = list(stance_legs)
        if len(stance_legs) == 0:
            return Task()

        rows: List[torch.Tensor] = []
        for leg in stance_legs:
            frame = self.leg_frame.get(leg, leg)
            J = self.robot.getJacobian(self.info.getstate(), self.info.getinput(), frame)
            if not torch.is_tensor(J):
                J = torch.as_tensor(J, device=self.device, dtype=self.dtype)
            if len(J.shape) == 3 and J.shape[0] == 1:
                J = J.squeeze(0)
            # linear part
            J_pos = J[:3, :] if J.shape[0] >= 3 else J
            # ensure 3x18
            if J_pos.shape[1] != 18:
                # best-effort: pad/crop to 18
                J_pad = torch.zeros((J_pos.shape[0], 18), device=self.device, dtype=self.dtype)
                cols = min(18, J_pos.shape[1])
                J_pad[:, :cols] = J_pos[:, :cols]
                J_pos = J_pad
            rows.append(J_pos)

        A = torch.cat(rows, dim=0)
        b = torch.zeros((A.shape[0],), device=self.device, dtype=self.dtype)
        return Task(a=A, b=b, device=self.device, dtype=self.dtype, weight=weight)
