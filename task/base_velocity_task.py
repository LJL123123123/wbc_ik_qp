"""BaseVelocityTask

Level 1: base translation (x,y) velocity tracking + yaw/heading rate.

Decision variable x is assumed to be velocity-space increment (18):
  [v_base(6); dq(12)] where v_base = [vx, vy, vz, wx, wy, wz]

We build a simple equality objective on selected components.
"""

from __future__ import annotations

import torch
from ho_qp import Task


class BaseVelocityTask:
    def __init__(self, device=None, dtype=torch.float64):
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype

    def as_task(
        self,
        v_xy_des: torch.Tensor | None = None,
        yaw_rate_des: float | torch.Tensor = 0.0,
        weight_xy: float = 1.0,
        weight_yaw: float = 1.0,
        residual_scale: float | torch.Tensor | None = None,
    ) -> Task:
        nv = 18
        rows = []
        rhs = []

        if v_xy_des is not None:
            v_xy_des = v_xy_des.to(device=self.device, dtype=self.dtype).reshape(-1)
            if v_xy_des.numel() != 2:
                raise ValueError('v_xy_des must have 2 elements')

            s = float(weight_xy) ** 0.5
            Axy = torch.zeros((2, nv), device=self.device, dtype=self.dtype)
            Axy[0, 0] = s  # vx
            Axy[1, 1] = s  # vy
            rows.append(Axy)
            rhs.append(v_xy_des * s)

        # yaw rate is wz (index 5)
        if weight_yaw is not None:
            s = float(weight_yaw) ** 0.5
            Ay = torch.zeros((1, nv), device=self.device, dtype=self.dtype)
            Ay[0, 5] = s
            if torch.is_tensor(yaw_rate_des):
                y = yaw_rate_des.to(device=self.device, dtype=self.dtype).reshape(())
            else:
                y = torch.tensor(float(yaw_rate_des), device=self.device, dtype=self.dtype)
            rows.append(Ay)
            rhs.append(y.reshape(1) * s)

        if not rows:
            return Task()

        A = torch.cat(rows, dim=0)
        b = torch.cat(rhs, dim=0)

        # ---- residual normalization ----
        if residual_scale is None:
            residual_scale = 1.0
        if torch.is_tensor(residual_scale):
            s = residual_scale.to(device=self.device, dtype=self.dtype).reshape(())
            s = torch.clamp(s, min=torch.tensor(1e-12, device=self.device, dtype=self.dtype))
            inv_s = 1.0 / s
        else:
            inv_s = 1.0 / max(float(residual_scale), 1e-12)

        A = A * inv_s
        b = b * inv_s
        return Task(a=A, b=b, device=self.device, dtype=self.dtype, weight=1.0)
