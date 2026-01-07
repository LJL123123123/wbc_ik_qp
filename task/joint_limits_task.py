"""JointLimitsTask

A task that creates joint angle limits as inequality constraints for the HoQP solver.
This prevents joints from exceeding their physical limits.
"""

import torch
from wbc_ik_qp.ho_qp import Task
from wbc_ik_qp.Centroidal import CentroidalModelInfoSimple


class JointLimitsTask:
    """Task to enforce joint angle limits as inequality constraints."""

    def __init__(self, info: CentroidalModelInfoSimple, device=None, dtype=torch.float64):
        self.info = info
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        
        # Default joint limits for ANYmal robot (in radians)
        # These are typical values - adjust based on actual robot specifications
        self.joint_limits = {
            'lower': [-3.14159, -1.57, -3.14159] * 4,  # Hip, Thigh, Shank for each leg
            'upper': [ 3.14159,  4.71,  3.14159] * 4   # More restrictive for some joints
        }

    def set_joint_limits(self, lower_limits, upper_limits):
        """Set custom joint limits.
        
        Args:
            lower_limits: List of 12 lower joint limits (3 per leg: LF, LH, RF, RH)
            upper_limits: List of 12 upper joint limits (3 per leg: LF, LH, RF, RH)
        """
        assert len(lower_limits) == 12 and len(upper_limits) == 12, \
            "Must provide 12 joint limits (3 per leg)"
        
        self.joint_limits['lower'] = lower_limits
        self.joint_limits['upper'] = upper_limits

    def as_task(self, current_state, weight: float = 1.0) -> Task:
        """Create inequality constraints for joint limits.
        
        The constraints are formulated as:
        q_lower <= q_current + dq <= q_upper
        Which becomes:
        -dq <= q_current - q_lower  (lower bound constraint)
         dq <= q_upper - q_current  (upper bound constraint)
        
        Args:
            current_state: Current robot state tensor (19 DOF: 6 base + 12 joints + 1 prismatic)
            weight: Task weight (not used for inequality constraints)
            
        Returns:
            Task with inequality constraints d @ x <= f
        """
        # Use 18 DOF for velocity space (6 base velocity + 12 joint velocities)
        nv = 18  # Velocity DOF (not position DOF which is 19)
        
        # Extract current joint positions (indices 6-17 in position, but we work with velocities)
        # For velocity constraints, we assume we're working with velocity limits
        # or we can use the current position to compute remaining range
        
        if hasattr(current_state, 'shape') and len(current_state.shape) > 0:
            current_positions = current_state[6:18]  # 12 joint positions
        else:
            # If no current state provided, use conservative limits
            current_positions = torch.zeros(12, device=self.device, dtype=self.dtype)
        
        # Create inequality constraint matrices
        # We have 24 constraints: 12 lower bounds + 12 upper bounds
        num_constraints = 24
        
        D = torch.zeros((num_constraints, nv), device=self.device, dtype=self.dtype)
        f = torch.zeros(num_constraints, device=self.device, dtype=self.dtype)
        
        # Convert limits to tensors
        q_lower = torch.tensor(self.joint_limits['lower'], device=self.device, dtype=self.dtype)
        q_upper = torch.tensor(self.joint_limits['upper'], device=self.device, dtype=self.dtype)
        
        # For velocity-based constraints: 
        # We limit joint velocities to prevent exceeding joint limits in one time step
        # Assuming dt = 1.0, this becomes: q_current + dq <= q_upper
        
        # Lower bound constraints: -dq <= q_current - q_lower => dq >= q_lower - q_current
        # Reformulated as: -dq <= -(q_lower - q_current) 
        for i in range(12):
            joint_idx = 6 + i  # Joint DOF indices in velocity vector
            
            # Lower bound: dq >= q_lower[i] - current_positions[i]
            # Reformulated: -dq <= -(q_lower[i] - current_positions[i])
            D[i, joint_idx] = -1.0
            f[i] = -(q_lower[i] - current_positions[i])
            
            # Upper bound: dq <= q_upper[i] - current_positions[i]  
            D[12 + i, joint_idx] = 1.0
            f[12 + i] = q_upper[i] - current_positions[i]
        
        return Task(d=D, f=f, device=self.device, dtype=self.dtype, weight=weight)

    def set_conservative_limits(self):
        """Set conservative joint limits that match placo's ANYmal settings."""
        # Match placo's joint limits exactly:
        # HAA: [-π/2, π/2], HFE: [-π, π], KFE: [-π, π]
        import math
        
        # More conservative limits for better consistency with placo
        self.joint_limits = {
            'lower': [-math.pi/3, -math.pi*0.8, -math.pi*0.8] * 4,  # Slightly tighter HAA, HFE, KFE for each leg  
            'upper': [ math.pi/3,  math.pi*0.8,  math.pi*0.8] * 4   # Prevent extreme configurations
        }

    def type_name(self):
        return "joint_limits"

    def error_unit(self):
        return "rad"