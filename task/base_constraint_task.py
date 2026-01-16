"""
Base constraint task for constraining the floating base to zero motion
"""

import torch
from ho_qp import Task
from Centroidal import CentroidalModelInfoSimple


class BaseConstraintTask:
    """Task that constrains the floating base (first 6 DOF) to zero motion"""
    
    def __init__(self, info: CentroidalModelInfoSimple, device=None, dtype=torch.float64):
        self.info = info
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        
    def as_task(self, weight: float = 1.0, residual_scale: float | torch.Tensor | None = None) -> Task:
        """Create a task that constrains base DOF to zero
        
        Returns:
            Task with A matrix constraining first 6 DOF and b = 0
        """
        # Use 18 DOF for velocity space (6 base velocity + 12 joint velocities)
        nv = 18  # Velocity DOF (not position DOF which is 19)
        
        # Create constraint matrix: only orientation DOF (indices 3,4,5)
        A = torch.zeros((3, nv), device=self.device, dtype=self.dtype)
        A[0, 3] = 1.0  # Roll constraint
        A[1, 4] = 1.0  # Pitch constraint  
        A[2, 5] = 1.0  # Yaw constraint
        
        # Target is zero rotation for base
        b = torch.zeros(3, device=self.device, dtype=self.dtype)
        
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

        return Task(a=A, b=b, weight=weight)