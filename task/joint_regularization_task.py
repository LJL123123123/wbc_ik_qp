"""JointRegularizationTask

Level 3: posture regularization / minimum action.

We implement a simple L2 objective as an equality least-squares term:
    w * (Δq - Δq_ref) = 0
which in HoQP's formulation becomes A x = b with:
    A = sqrt(w) * [0_{12x6} I_{12}]
    b = sqrt(w) * Δq_ref

This keeps joints close to a nominal posture (or minimum velocity increment).
"""

from __future__ import annotations

import torch
from ho_qp import Task


class JointRegularizationTask:
    def __init__(self, device=None, dtype=torch.float64):
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype

    def as_task(self, dq_ref: torch.Tensor | None = None, weight: float = 1e-3) -> Task:
        # decision variable is (18,) = [base(6); joints(12)]
        nv = 18
        if dq_ref is None:
            dq_ref = torch.zeros((12,), device=self.device, dtype=self.dtype)
        else:
            dq_ref = dq_ref.to(device=self.device, dtype=self.dtype).reshape(-1)
            if dq_ref.numel() != 12:
                raise ValueError('dq_ref must have 12 elements')

        w = float(weight)
        s = math.sqrt(w) if w > 0 else 0.0
        A = torch.zeros((12, nv), device=self.device, dtype=self.dtype)
        A[:, 6:18] = torch.eye(12, device=self.device, dtype=self.dtype) * s
        b = dq_ref * s
        return Task(a=A, b=b, device=self.device, dtype=self.dtype, weight=1.0)


import math
