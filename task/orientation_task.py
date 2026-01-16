"""OrientationTask (python)

Refactored OrientationTask that uses Model_Cusadi (from `ik.py`) to compute
attitude and angular jacobian. Instead of holding internal A/b matrices,
this class provides `as_task(state, input, axises, frame)` which returns a
`ho_qp.Task` containing the masked orientation equality constraint. This makes
it easy to create many OrientationTask instances and concatenate their tasks.

Key behaviour:
 - `as_task` updates the internal `Model_Cusadi` with the provided state/input,
     reads the frame attitude and jacobian, computes the orientation error, applies
     an `AxisesMask`, and returns a `Task(a=..., b=...)`.

Usage:
    ot = OrientationTask(info, frame_name='LF_FOOT', target_attitude=torch.eye(3))
    task = ot.as_task(state_tensor, input_tensor, axises='xy', frame='local')
    # task.a_ is (m x n), task.b_ is (m,)
"""

from typing import Optional, Sequence
import torch
import numpy as np

try:
    from tools.axises_mask import AxisesMask
    from Centroidal import CentroidalModelInfoSimple
    from ik import Model_Cusadi  
    from ho_qp import Task
except ImportError:
    # Fallback imports for testing environments
    print("Warning: Some dependencies not available, using fallbacks for OrientationTask")
    
    # Minimal fallback classes
    class AxisesMask:
        def __init__(self, device=None, dtype=torch.float64):
            self.device = device or torch.device('cpu')
            self.dtype = dtype
            self.R_local_world = torch.eye(3, device=self.device, dtype=self.dtype)
        
        def set_axises(self, axises: str, frame: str):
            pass
        
        def apply(self, tensor):
            return tensor
    
    # class CentroidalModelInfoSimple:
    #     def __init__(self, generalizedCoordinatesNum=12, actuatedDofNum=12, numThreeDofContacts=4):
    #         self.generalizedCoordinatesNum = generalizedCoordinatesNum
    #         self.actuatedDofNum = actuatedDofNum
    #         self.numThreeDofContacts = numThreeDofContacts
    #         self.state = torch.zeros(generalizedCoordinatesNum)
    #         self.input = torch.zeros(actuatedDofNum)
        
    #     def getstate(self):
    #         return self.state
        
    #     def getinput(self):
    #         return self.input
    
    class Task:
        def __init__(self, a=None, b=None, device=None, dtype=torch.float64, weight=1.0):
            self.a_ = a
            self.b_ = b
            self.weight_ = weight


class OrientationTask:
        """Orientation task that can be converted to a `ho_qp.Task`.

        The task represents an orientation equality: A x = b where A is the masked
        angular jacobian and b is the masked orientation error (target - current).
        """

        def __init__(self, info: CentroidalModelInfoSimple, frame_name: str,
                                  device=None, dtype=torch.float64):
                self.info = info
                self.device = device if device is not None else torch.device('cpu')
                self.dtype = dtype
                self.frame_name = frame_name
                self.error = torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)

                # robot interface (fake Pinocchio-like wrapper)
                # Allow injection of a robot; if Model_Cusadi construction fails (missing
                # external deps in some environments), fall back to a tiny minimal
                # robot that provides the small API OrientationTask needs.
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

        def as_task(
                self,
                target_attitude: torch.Tensor,
                axises: str = "xyz",
                frame: Optional[str] = "task",
                weight: float = 1.0,
                residual_scale: float | torch.Tensor | None = None,
        ) -> Task:
                """Return a `ho_qp.Task` built from this OrientationTask.

                Args:
                        target_attitude: target rotation matrix (3x3) in world frame
                        axises: axis selection string like 'x', 'xy', 'yz', 'xyz'
                        frame: frame for the axes mask: 'task'/'world', 'local', or 'custom'
                        weight: weight for this task in the hierarchical QP

                Returns:
                        ho_qp.Task with equality A (m x n) and rhs b (m,). If no axes are
                        selected the returned Task will have empty a_/b_.
                """
                
                # get current frame attitude and jacobian from Model_Cusadi
                att = self.robot.getAttitude(self.info.getstate(), self.info.getinput(), self.frame_name)
                J = self.robot.getJacobian(self.info.getstate(), self.info.getinput(), self.frame_name)
                
                # ensure attitude is on correct device/dtype
                if not torch.is_tensor(att):
                    att = torch.as_tensor(att, device=self.device, dtype=self.dtype)
                att = att.to(device=self.device, dtype=self.dtype)
                
                # ensure target_attitude is a tensor on correct device/dtype
                if not torch.is_tensor(target_attitude):
                    target_attitude = torch.as_tensor(target_attitude, device=self.device, dtype=self.dtype)
                target_attitude = target_attitude.to(device=self.device, dtype=self.dtype)

                # normalize jacobian shape: CasADi returns [1, 6, nv], we need [3, nv] for angular part
                if not torch.is_tensor(J):
                    J = torch.as_tensor(J, device=self.device, dtype=self.dtype)
                
                # Handle CasADi output format: remove batch dimension and extract angular part
                if len(J.shape) == 3 and J.shape[0] == 1:
                    J = J.squeeze(0)  # Remove batch dimension: [1, 6, nv] -> [6, nv]
                
                # Extract angular jacobian (last 3 rows for angular velocity)
                if J.shape[0] >= 6:
                    J_ang = J[3:6, :]  # Angular jacobian (3 x nv)
                else:
                    # Fallback: assume it's already angular jacobian
                    J_ang = J[:3, :] if J.shape[0] >= 3 else J
                
                # Apply DOF filtering based on frame type to match placo's behavior
                if self.frame_name in ['LF_FOOT', 'LH_FOOT', 'RF_FOOT', 'RH_FOOT']:
                    # For foot frames, zero out the floating base DOF (first 6 columns)
                    # Only joint DOF affect foot orientations
                    J_ang_modified = J_ang.clone()
                    J_ang_modified[:, :6] = 0.0  # Zero out floating base DOF
                    J_ang = J_ang_modified
                elif self.frame_name == 'com':
                    # For COM frame, only use floating base orientation DOF (columns 3:6)
                    # Zero out floating base position DOF and all joint DOF
                    J_ang_modified = J_ang.clone()
                    J_ang_modified[:, :3] = 0.0   # Zero out position DOF (0:3)
                    J_ang_modified[:, 6:] = 0.0   # Zero out joint DOF (6:)
                    J_ang = J_ang_modified

                # set mask and its rotation if local/custom frame used
                self.mask.set_axises(axises, frame)
                # For orientation, we may want to use current attitude for frame transformation
                try:
                        self.mask.R_local_world = att.T.to(device=self.device, dtype=self.dtype)
                except Exception:
                        # if attitude shape is unexpected, ignore and keep identity
                        pass

                # compute orientation error using rotation matrix logarithm
                # For small angles: error ≈ log(R_target * R_current^T)
                error = self.compute_orientation_error(target_attitude, att)

                # ---- residual normalization ----
                # 让 b 的“典型尺度”接近 1，避免不同 task 单位/量级差异把 QP 条件数拉爆。
                # 这里采用：A x = b，两边同除以 scale -> (A/scale) x = (b/scale)
                # weight 仍然走 HoQPLevel 的 sqrt(weight) 机制。
                if residual_scale is None:
                    # 姿态误差单位是 rad；经验上把 10deg 作为“1”的尺度比较稳。
                    residual_scale = float(torch.deg2rad(torch.tensor(10.0)))
                if torch.is_tensor(residual_scale):
                    s = residual_scale.to(device=self.device, dtype=self.dtype).reshape(())
                    s = torch.clamp(s, min=torch.tensor(1e-12, device=self.device, dtype=self.dtype))
                    inv_s = 1.0 / s
                else:
                    s = max(float(residual_scale), 1e-12)
                    inv_s = 1.0 / s

                # apply mask
                A_masked = self.mask.apply(J_ang)
                b_masked = self.mask.apply(error)

                # apply normalization (broadcast-safe)
                A_masked = A_masked * inv_s
                b_masked = b_masked * inv_s

                # If the masked jacobian is all zeros the equality A x = b is
                # either infeasible (if b != 0) or provides no information. In
                # development environments where the real kinematics backend
                # isn't available we prefer to skip the task rather than add an
                # impossible equality. Return an empty Task in that case.
                if A_masked.numel() == 0 or torch.allclose(A_masked, torch.zeros_like(A_masked)):
                        # log a short notice and return an empty Task so solver can continue
                        print("OrientationTask: masked jacobian is zero — skipping orientation equality task.")
                        return Task()

                # Ensure shapes: A_masked should be (m, n) and b_masked (m,)
                # ho_qp.Task handles empty tensors; just construct and return.
                return Task(a=A_masked, b=b_masked, device=self.device, dtype=self.dtype, weight=weight)

        def compute_orientation_error(self, target_R: torch.Tensor, current_R: torch.Tensor) -> torch.Tensor:
            R_err = target_R @ current_R.transpose(-1, -2)

            # trace(R) -> cos(theta) = (trace - 1)/2
            tr = torch.trace(R_err)
            cos_theta = torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
            theta = torch.acos(cos_theta)  # in [0, pi]

            # vee of skew-symmetric part: vee(R - R^T) = [R32-R23, R13-R31, R21-R12]
            # We'll compute w = vee(R_err - R_err^T)
            skew = R_err - R_err.transpose(-1, -2)
            w = torch.stack([skew[2, 1], skew[0, 2], skew[1, 0]])  # (3,)

            # For small angles, use first-order approximation: log(R) ≈ 0.5 * vee(R - R^T)
            eps = 1e-6
            if theta < eps:
                e = 0.5 * w
                return e

            # For general case: log(R) = (theta / (2 sin theta)) * vee(R - R^T)
            sin_theta = torch.sin(theta)

            # Avoid division blow-up extremely close to pi (rare but can happen)
            # clamp sin_theta to keep numeric stability
            sin_theta = torch.clamp(sin_theta, min=1e-8)

            scale = theta / (2.0 * sin_theta)
            e = scale * w
            return e

        def type_name(self):
                return "orientation"

        def error_unit(self):
                return "rad"
