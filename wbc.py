"""
Lightweight Python port of the Wbc class (a simplified/fake pinocchio interface)
that assembles Tasks for the provided Python HoQp implementation.

This file intentionally uses simplified (placeholder) dynamics quantities so it
can run without a full Pinocchio / ocs2 stack. The goal is to provide a
compatible interface and generate Task objects with correct shapes for the
existing `ho_qp.py` implementation.

Notes:
- All tensors use CPU torch.float64 by default.
- The fake Pinocchio interface provides minimal attributes used by the task
  assembly (mass matrix M, nonlinear effects nle, dAg placeholder, etc.).
"""
from dataclasses import dataclass
from typing import Optional, Sequence
import torch
import math

from pathlib import Path

from wbc_logger import WbcCsvLogger
from gait_manager_cuda import GaitCycleManagerCuda, GaitParamsCuda


try:
    from ho_qp import Task, HoQp
except Exception:
    # If running as a plain script (e.g. `python run_wbc.py`), relative imports
    # fail with "no known parent package". Prefer retrying absolute import after
    # ensuring this directory is on sys.path.
    try:
        from .ho_qp import Task, HoQp
    except Exception:
        import os
        import sys

        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from ho_qp import Task, HoQp

from Centroidal import CentroidalModelInfoSimple
from task.position_task import PositionTask
from task.orientation_task import OrientationTask
from task.base_constraint_task import BaseConstraintTask
from ik import Model_Cusadi

@dataclass
class WbcTask:
    position: PositionTask
    orientation: OrientationTask
    base_constraint: BaseConstraintTask

class FakePinocchioModel:
    def __init__(self, nq: int):
        # provide a names list for parity with C++ code that may inspect it
        self.names = [f"q_{i}" for i in range(nq)]


class FakePinocchioData:
    def __init__(self, nq: int, device, dtype):
        # mass matrix (nq x nq)
        self.M = torch.eye(nq, device=device, dtype=dtype)
        # nonlinear effects vector (nq)
        self.nle = torch.zeros((nq,), device=device, dtype=dtype)
        # placeholder for centroidal momentum time variation matrix
        # in C++ this was Data.dAg (6x18 typically). We'll create a generic
        # small matrix compatible with the tasks (6 x nq).
        self.dAg = torch.zeros((6, nq), device=device, dtype=dtype)


class FakePinocchioInterface:
    """A tiny fake Pinocchio interface exposing the minimal model/data used
    by the simplified Wbc implementation."""

    def __init__(self, info: CentroidalModelInfoSimple, device=None, dtype=torch.float64):
        self.info = info
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        self._model = FakePinocchioModel(info.generalizedCoordinatesNum)
        self._data = FakePinocchioData(info.generalizedCoordinatesNum, self.device, self.dtype)

    def getModel(self):
        return self._model

    def getData(self):
        return self._data

    # The real Pinocchio api runs kinematics and updates internal data.
    # Here we provide no-op implementations (placeholders) to keep the API.
    def forwardKinematics(self, *args, **kwargs):
        return None

    def computeJointJacobians(self, *args, **kwargs):
        return None

    def updateFramePlacements(self, *args, **kwargs):
        return None

    def crba(self, *args, **kwargs):
        return None

    def nonLinearEffects(self, *args, **kwargs):
        return None


class Wbc:
    """Simplified Python Wbc that constructs Tasks compatible with the
    Python HoQp implementation. Uses fake pinocchio data.

    Constructor signature mirrors the C++ class but with simplified args. The
    real dynamics are not computed; instead placeholder matrices are produced
    with correct shapes so the optimizer stack can be exercised.
    """

    def __init__(self, task_file: str, pino_interface: FakePinocchioInterface,
                 info: CentroidalModelInfoSimple, ee_kinematics: Optional[object] = None,
                 verbose: bool = False, device=None, dtype=torch.float64):
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        self.pino = pino_interface
        self.info = info
        self.ee_kinematics = ee_kinematics

        self.verbose = verbose

        # --- gait cycle manager (CUDA-friendly) ---
        self.gait = GaitCycleManagerCuda(
            device=self.device,
            dtype=self.dtype,
            params=GaitParamsCuda(),
        )

        # --- CSV logger (centralized in WBC) ---
        self.logger = WbcCsvLogger(base_dir=Path('./debug'))
        # decision vars: [u_dot (nq) ; contact_forces (3*numContacts) ; tau (act)]
        self.num_decision_vars = (
            info.generalizedCoordinatesNum + 3 * info.numThreeDofContacts + info.actuatedDofNum
        )

        # Measurements placeholders
        nq = info.generalizedCoordinatesNum
        self.measured_q = torch.zeros((nq,), device=self.device, dtype=self.dtype)
        self.measured_v = torch.zeros((nq,), device=self.device, dtype=self.dtype)

        # simple placeholders for contact Jacobians
        self.j_ = torch.zeros((3 * info.numThreeDofContacts, nq), device=self.device, dtype=self.dtype)
        self.dj_ = torch.zeros_like(self.j_)

        # Task parameters (defaults)
        self.torque_limits_ = torch.ones((info.actuatedDofNum,), device=self.device, dtype=self.dtype) * 50.0
        self.friction_coeff_ = 0.6
        self.swing_kp_ = 50.0
        self.swing_kd_ = 1.0
        self.robot = Model_Cusadi(info, device=self.device, dtype=self.dtype)

        try:
            self.com_PositionTask = PositionTask(
                self.info,
                frame_name='com',
                device=self.device,
                dtype=self.dtype,
            )
        except Exception:
            print("Failed to create com_PositionTask, setting to None.")
            self.com_PositionTask = None
        try:
            self.com_OrientationTask = OrientationTask(
                self.info,
                frame_name='com',
                device=self.device,
                dtype=self.dtype,
            )
        except Exception:
            print("Failed to create com_OrientationTask, setting to None.")
            self.com_OrientationTask = None
        try:
            self.LF_PositionTask = PositionTask(
                self.info,
                frame_name='LF_FOOT',
                device=self.device,
                dtype=self.dtype,
            )
        except Exception:
            print("Failed to create LF_PositionTask, setting to None.")
            self.LF_PositionTask = None
        try:        # print('\n--- ho1 (LF standalone) diagnostic ---')

            self.LH_PositionTask = PositionTask(
                self.info,
                frame_name='LH_FOOT',
                device=self.device,
                dtype=self.dtype,
            )
        except Exception:
            print("Failed to create LH_PositionTask, setting to None.")
            self.LH_PositionTask = None
        try:
            self.RF_PositionTask = PositionTask(
                self.info,
                frame_name='RF_FOOT',
                device=self.device,
                dtype=self.dtype,
            )
        except Exception:
            print("Failed to create RF_PositionTask, setting to None.")
            self.RF_PositionTask = None
        try:
            self.RH_PositionTask = PositionTask(
                self.info,
                frame_name='RH_FOOT',
                device=self.device,
                dtype=self.dtype,
            )
        except Exception:
            print("Failed to create RH_PositionTask, setting to None.")
            self.RH_PositionTask = None
        self.target_pos = {
            "com": torch.tensor([0.0, 0., 0.0], device=self.device, dtype=self.dtype),
            "LH": torch.tensor([-0.44, 0.27, -0.55], device=self.device, dtype=self.dtype),
            "LF": torch.tensor([0.44, 0.27, -0.55], device=self.device, dtype=self.dtype),
            "RF": torch.tensor([0.44, -0.27, -0.55], device=self.device, dtype=self.dtype),
            "RH": torch.tensor([-0.44, -0.27, -0.55], device=self.device, dtype=self.dtype),
        }

        self.target_ori = {
            "com": torch.eye(3, device=self.device, dtype=self.dtype),
            "LH": torch.eye(3, device=self.device, dtype=self.dtype),
            "LF": torch.eye(3, device=self.device, dtype=self.dtype),
            "RF": torch.eye(3, device=self.device, dtype=self.dtype),
            "RH": torch.eye(3, device=self.device, dtype=self.dtype),
        }

        try:
            # BaseConstraintTask expects (info, device, dtype)
            # (do not pass `self` here; that caused a signature mismatch)
            self.base_constraint = BaseConstraintTask(
                self.info,
                device=self.device,
                dtype=self.dtype,
            )
        except Exception:
            print("Failed to create base_constraint, setting to None.")
            self.base_constraint = None

        # internal time for gait manager & logging
        self._t = 0.0

        # last planned targets snapshot (for logging)
        self._last_plan = None

        # last solution snapshot
        self._last_sol = None


    # ----------------- Task builders -----------------
    def formulateFloatingBaseEomTask(self) -> Task:
        nq = self.info.generalizedCoordinatesNum
        nd = self.num_decision_vars

        data = self.pino.getData()
        # a: (nq x nd)
        a = torch.zeros((nq, nd), device=self.device, dtype=self.dtype)
        # left block: mass matrix
        a[:, :nq] = data.M
        # middle block: -j_.T (forces)
        a[:, nq:nq + 3 * self.info.numThreeDofContacts] = -self.j_.transpose(0, 1)
        # right block: -s^T (actuated selection)
        s = torch.zeros((self.info.actuatedDofNum, nq), device=self.device, dtype=self.dtype)
        s[:, 6:6 + self.info.actuatedDofNum] = torch.eye(self.info.actuatedDofNum, device=self.device, dtype=self.dtype)
        a[:, nq + 3 * self.info.numThreeDofContacts:] = -s.transpose(0, 1)

        b = -data.nle
        return Task(a=a, b=b)

    def formulateTorqueLimitsTask(self) -> Task:
        na = self.info.actuatedDofNum
        nd = self.num_decision_vars
        d = torch.zeros((2 * na, nd), device=self.device, dtype=self.dtype)
        # torque decision vars reside at the end of x
        tau_start = self.info.generalizedCoordinatesNum + 3 * self.info.numThreeDofContacts
        d[0:na, tau_start:tau_start + na] = torch.eye(na, device=self.device, dtype=self.dtype)
        d[na:2 * na, tau_start:tau_start + na] = -torch.eye(na, device=self.device, dtype=self.dtype)

        # f contains upper bounds and lower bounds
        f = torch.cat([self.torque_limits_, self.torque_limits_], dim=0)
        return Task(d=d, f=f)

    def formulateNoContactMotionTask(self) -> Task:
        # For legs not in contact we would set their accelerations to zero.
        # We'll create an equality on the contact force columns to be zero for
        # non-contact legs (placeholder: none are enforced here). Return empty.
        return Task()

    def formulateFrictionConeTask(self) -> Task:
        # Inequality rows that approximate friction pyramid constraints.
        # We'll produce a simple box constraint on contact forces as placeholder.
        n_contacts = self.info.numThreeDofContacts
        nd = self.num_decision_vars
        if n_contacts == 0:
            return Task()

        # We constrain each contact force component to be within +/- (robotMass*10)
        ub = self.info.robotMass * 10.0
        rows = 3 * n_contacts * 2
        d = torch.zeros((rows, nd), device=self.device, dtype=self.dtype)
        # contact force columns start at nq
        cf_start = self.info.generalizedCoordinatesNum
        for i in range(3 * n_contacts):
            # f_i <= ub ->  1 * f_i <= ub
            d[i, cf_start + i] = 1.0
            # -f_i <= ub -> -1 * f_i <= ub  (lower bound symmetric)
            d[rows // 2 + i, cf_start + i] = -1.0

        f = torch.ones((rows,), device=self.device, dtype=self.dtype) * ub
        return Task(d=d, f=f)

    def formulateBaseAccelTask(self) -> Task:
        # Keep base accel equal to desired centroidal momentum rate
        nd = self.num_decision_vars
        a = torch.zeros((6, nd), device=self.device, dtype=self.dtype)
        a[:, :6] = torch.eye(6, device=self.device, dtype=self.dtype)
        b = torch.zeros((6,), device=self.device, dtype=self.dtype)
        return Task(a=a, b=b)

    def formulateCentroidalMomentumTask(self) -> Task:
        # A_g (6 x nq) block in front
        nq = self.info.generalizedCoordinatesNum
        nd = self.num_decision_vars
        a = torch.zeros((6, nd), device=self.device, dtype=self.dtype)
        # Place a simple centroidal mapping on q-dot part
        a[:, :nq] = torch.zeros((6, nq), device=self.device, dtype=self.dtype)
        b = torch.zeros((6,), device=self.device, dtype=self.dtype)
        return Task(a=a, b=b)

    def formulateSwingLegTask(self) -> Task:
        # Placeholder: no swing leg equality enforced
        return Task()

    def formulateContactForceTask(self) -> Task:
        # Placeholder: no extra equality; real implementation couples contact
        # forces to centroidal momentum. We'll return empty to keep solver happy.
        return Task()

    # ----------------- Public API -----------------
    def update_targets(self, target_pos: dict, target_ori: dict):
        """Update target positions and orientations for tasks.

        Args:
            target_pos: dict mapping frame names to target position tensors (3,)
            target_ori: dict mapping frame names to target orientation tensors (3,3)
        """
        self.target_pos = target_pos
        self.target_ori = target_ori

    def _log_series(self, t: float, sol: torch.Tensor):
        """Write CSV rows for planned targets, solver output, and kinematic state.

        This is intentionally lightweight; we log a minimal subset matching the
        original run_wbc.py debug outputs.
        """

        # planned targets
        tp = self.target_pos
        to = self.target_ori

        # COM target (pos + euler from R)
        Rcom = to['com']
        # simple XYZ euler extraction without scipy
        # (matches typical roll-pitch-yaw for rotation matrix)
        sy = torch.sqrt(Rcom[0, 0] * Rcom[0, 0] + Rcom[1, 0] * Rcom[1, 0])
        singular = sy < 1e-9
        if not bool(singular.item()):
            roll = torch.atan2(Rcom[2, 1], Rcom[2, 2])
            pitch = torch.atan2(-Rcom[2, 0], sy)
            yaw = torch.atan2(Rcom[1, 0], Rcom[0, 0])
        else:
            roll = torch.atan2(-Rcom[1, 2], Rcom[1, 1])
            pitch = torch.atan2(-Rcom[2, 0], sy)
            yaw = torch.zeros((), device=self.device, dtype=self.dtype)

        self.logger.write_row(
            name='com_target',
            filename='com_target_data.csv',
            header=['t', 'com_x', 'com_y', 'com_z', 'roll', 'pitch', 'yaw'],
            row=[
                float(t),
                float(tp['com'][0].detach().cpu()),
                float(tp['com'][1].detach().cpu()),
                float(tp['com'][2].detach().cpu()),
                float(roll.detach().cpu()),
                float(pitch.detach().cpu()),
                float(yaw.detach().cpu()),
            ],
        )

        # RF target
        self.logger.write_row(
            name='rf_target',
            filename='RF_target_data.csv',
            header=['t', 'x', 'y', 'z'],
            row=[
                float(t),
                float(tp['RF'][0].detach().cpu()),
                float(tp['RF'][1].detach().cpu()),
                float(tp['RF'][2].detach().cpu()),
            ],
        )

        # solver "optimal" outputs (kept consistent with old script slices)
        if sol is not None and sol.numel() >= 3:
            self.logger.write_row(
                name='com_optimal',
                filename='com_opimal_data.csv',
                header=['t', 'dx', 'dy', 'dz'],
                row=[
                    float(t),
                    float(sol[0].detach().cpu()),
                    float(sol[1].detach().cpu()),
                    float(sol[2].detach().cpu()),
                ],
            )
        if sol is not None and sol.numel() >= 10:
            self.logger.write_row(
                name='rf_optimal',
                filename='RF_opimal_data.csv',
                header=['t', 'dx', 'dy', 'dz'],
                row=[
                    float(t),
                    float(sol[7].detach().cpu()),
                    float(sol[8].detach().cpu()),
                    float(sol[9].detach().cpu()),
                ],
            )

        # state (from info, if available)
        try:
            st = self.info.getstate()
            it = self.info.getinput()
            # state assumed to start with COM position
            # and contain base orientation quaternion at indices [3:7] as (x,y,z,w)
            # (per user's CentroidalModelInfoSimple usage)
            qx, qy, qz, qw = st[3], st[4], st[5], st[6]
            # quaternion (x,y,z,w) -> roll/pitch/yaw (XYZ, intrinsic RPY)
            sinr_cosp = 2.0 * (qw * qx + qy * qz)
            cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
            roll_s = torch.atan2(sinr_cosp, cosr_cosp)

            sinp = 2.0 * (qw * qy - qz * qx)
            # clamp for numerical stability
            sinp = torch.clamp(sinp, -1.0, 1.0)
            pitch_s = torch.asin(sinp)

            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw_s = torch.atan2(siny_cosp, cosy_cosp)

            self.logger.write_row(
                name='com_state',
                filename='com_state_data.csv',
                header=['t', 'com_x', 'com_y', 'com_z', 'roll', 'pitch', 'yaw'],
                row=[
                    float(t),
                    float(st[0].detach().cpu()),
                    float(st[1].detach().cpu()),
                    float(st[2].detach().cpu()),
                    float(roll_s.detach().cpu()),
                    float(pitch_s.detach().cpu()),
                    float(yaw_s.detach().cpu()),
                ],
            )
            
            rf_state = self.robot.getPosition( st, it, "RF_FOOT")

            self.logger.write_row(
                name='rf_state',
                filename='RF_state_data.csv',
                header=['t', 'x', 'y', 'z'],
                row=[
                    float(t),
                    float(rf_state[0].detach().cpu()),
                    float(rf_state[1].detach().cpu()),
                    float(rf_state[2].detach().cpu()),
                ],
            )
        except Exception:
            pass
        
        #  error logs
        try:
            com_ori_error = self.com_OrientationTask.error
            self.logger.write_row(
                name='com_ori_error',
                filename='com_ori_error_data.csv',
                header=['t', 'err_x', 'err_y', 'err_z'],
                row=[
                    float(t),
                    float(com_ori_error[0].detach().cpu()),
                    float(com_ori_error[1].detach().cpu()),
                    float(com_ori_error[2].detach().cpu()),
                ],
            )

            rf_ori_error = self.RF_PositionTask.error
            self.logger.write_row(
                name='rf_error',
                filename='RF_error_data.csv',
                header=['t', 'err_x', 'err_y', 'err_z'],
                row=[
                    float(t),
                    float(rf_ori_error[0].detach().cpu()),
                    float(rf_ori_error[1].detach().cpu()),
                    float(rf_ori_error[2].detach().cpu()),
                ],
            )
        except Exception:
            pass

    def step_with_cmd(
        self,
        measured_rbd_state: torch.Tensor,
        input_desired: torch.Tensor,
        dt: float,
        cmd_vxyz: torch.Tensor,
        cmd_yaw_rate: torch.Tensor,
        mode: int = 0,
    ) -> torch.Tensor:
        """High level step: gait planning -> update targets -> solve -> log.

        Returns:
            state_desired (same shape as measured_rbd_state)
        """

        self._t += float(dt)

        # plan new targets
        plan = self.gait.update(
            t=self._t,
            target_pos=self.target_pos,
            target_ori=self.target_ori,
            cmd_vxyz=cmd_vxyz,
            cmd_yaw_rate=cmd_yaw_rate,
            dt=dt,
        )
        self._last_plan = plan
        self.update_targets(plan.target_pos, plan.target_ori)

        # solve
        sol = self.update(measured_rbd_state, input_desired, mode=mode)
        self._last_sol = sol

        # apply solution to construct desired next state (keep old behavior)
        state_desired = measured_rbd_state.clone()
        if sol is not None and sol.numel() >= 18:
            # ---- per-step norms diagnostics (sol/dp/dq)
            try:
                dp = sol[0:3]
                dq = sol[6:18]
                self.logger.write_row(
                    name='step_norms',
                    filename='step_norms_data.csv',
                    header=['t', 'sol_norm', 'dp_norm', 'dq_norm'],
                    row=[
                        float(self._t),
                        float(torch.linalg.norm(sol).detach().cpu()),
                        float(torch.linalg.norm(dp).detach().cpu()),
                        float(torch.linalg.norm(dq).detach().cpu()),
                    ],
                )
            except Exception:
                pass

            state_desired[0:3] = measured_rbd_state[0:3] + sol[0:3]
            # base orientation update: treat sol[3:6] as small-angle rotation vector (phi)
            # and integrate it into quaternion state_desired[3:7] (x,y,z,w).
            try:
                if measured_rbd_state.numel() >= 7 and sol.numel() >= 6:
                    # NOTE:
                    # In this codebase, `sol` is produced by HoQp from kinematic tasks.
                    # It behaves like an increment per-step, but its absolute scale can
                    # drift when the optimization becomes ill-conditioned.
                    # We therefore:
                    # 1) scale by dt (treat as angular velocity-like increment)
                    # 2) clamp magnitude to avoid a single bad solve exploding the state
                    phi = sol[3:6]
                    # clamp to at most ~10 deg per step (tunable)
                    phi_norm = torch.linalg.norm(phi)
                    phi_max = torch.tensor(0.17453292519943295, device=phi.device, dtype=phi.dtype)  # 10 deg
                    if phi_norm > phi_max:
                        phi = phi * (phi_max / (phi_norm + 1e-12))

                    angle = torch.linalg.norm(phi)
                    # small-angle stable conversion
                    half = 0.5 * angle
                    axis = phi / (angle + 1e-12)
                    sin_half = torch.sin(half)
                    dq = torch.stack([axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, torch.cos(half)])

                    q = measured_rbd_state[3:7]
                    # Hamilton product q_new = dq ⊗ q
                    qx, qy, qz, qw = dq[0], dq[1], dq[2], dq[3]
                    rx, ry, rz, rw = q[0], q[1], q[2], q[3]
                    vx = qw * rx + rw * qx + (qy * rz - qz * ry)
                    vy = qw * ry + rw * qy + (qz * rx - qx * rz)
                    vz = qw * rz + rw * qz + (qx * ry - qy * rx)
                    vw = qw * rw - (qx * rx + qy * ry + qz * rz)
                    q_new = torch.stack([vx, vy, vz, vw])
                    q_new = q_new / (torch.linalg.norm(q_new) + 1e-12)
                    state_desired[3:7] = q_new

                    # lightweight diagnostics for divergence hunting
                    try:
                        self.logger.write_row(
                            name='sol_norm',
                            filename='sol_norm_data.csv',
                            header=['t', 'dp_norm', 'dphi_norm_raw', 'dphi_norm_used'],
                            row=[
                                float(self._t),
                                float(torch.linalg.norm(sol[0:3]).detach().cpu()),
                                float(torch.linalg.norm(sol[3:6]).detach().cpu()),
                                float(torch.linalg.norm(phi).detach().cpu()),
                            ],
                        )
                    except Exception:
                        pass
            except Exception:
                # never let quaternion integration break the loop
                pass
            # joints slice consistent with run_wbc.py: state[7:19] += sol[6:18]
            state_desired[7:19] = measured_rbd_state[7:19] + sol[6:18]

        # keep info state updated so downstream kinematics can read it
        try:
            self.info.update_state_input(state_desired, input_desired)
        except Exception:
            pass

        # log
        try:
            self._log_series(self._t, sol)
        except Exception:
            # never let logging break control loop
            if self.verbose:
                print('[WBC] logging failed (ignored).')

        return state_desired

    def update(self, measured_rbd_state: Sequence[float], input_desired: Sequence[float], mode: int):
        """Main update call that returns the stacked HoQp solution vector.
        
        The implementation here builds a few tasks (some are placeholders)
        and runs the Python HoQp stack to produce a solution vector compatible
        with the decision variable layout described in the header.
        """
        
        device = self.device
        dtype = self.dtype

        self.info.update_state_input(
            measured_rbd_state,
            input_desired
        )

        com_weight = 1.0     # Base weight for COM position
        lf_weight = 1.0      # Base weight for foot position
        
        task_com_pos = self.com_PositionTask.as_task(
            target_world=self.target_pos["com"],
            axises="xyz",
            frame="task",
            weight=com_weight
        )
        task_com_ori = self.com_OrientationTask.as_task(
            target_attitude=self.target_ori["com"],
            axises="xyz",
            frame="task",
            weight=com_weight
        )
    # Combine COM position and orientation into the high-priority task so the
    # optimizer actively tracks both position and attitude of the base.
        task_com_frame =  task_com_pos + task_com_ori
        # print('task_com_frame.a_:', task_com_frame.a_)
        ho_high = HoQp(task_com_frame, higher_problem=None, device=device, dtype=dtype, task_weight=100.0)

        task_LF_pos = self.LF_PositionTask.as_task(
            target_world=self.target_pos["LF"],
            axises="xyz",
            frame="task",
            weight=lf_weight
        )
        task_LH_pos = self.LH_PositionTask.as_task(
            target_world=self.target_pos["LH"],
            axises="xyz",
            frame="task",
            weight=lf_weight
        )
        task_RF_pos = self.RF_PositionTask.as_task(
            target_world=self.target_pos["RF"],
            axises="xyz",
            frame="task",
            weight=lf_weight
        )
        task_RH_pos = self.RH_PositionTask.as_task(
            target_world=self.target_pos["RH"],
            axises="xyz",
            frame="task",
            weight=lf_weight
        )
        constraint = self.base_constraint.as_task(weight=.01)
        high_priority_weight = 1.0    # Reasonable weight for high priority tasks  
        low_priority_weight = .1     # Reasonable weight for low priority tasks
        ho_RF = HoQp(task_LF_pos, higher_problem=None, device=device, dtype=dtype, task_weight=high_priority_weight)

        combined_foot_task = task_LF_pos + task_RF_pos + task_LH_pos + task_RH_pos + constraint
        const_task = HoQp(constraint, higher_problem=None, device=device, dtype=dtype, task_weight=low_priority_weight)
        # print('combined_foot_task.a_:', combined_foot_task.a_)
        combine_foot_ho = HoQp(combined_foot_task, higher_problem=None, device=device, dtype=dtype, task_weight=low_priority_weight)
        combined_ho = HoQp(task_com_frame, higher_problem=combine_foot_ho, device=device, dtype=dtype, task_weight=low_priority_weight)
    # Return the full stacked solution so both high-priority (COM pos+ori)
    # and lower-priority (foot positions) objectives are considered.
        # Optionally add base constraint as the lowest-priority task on top of
        # the existing stack.

        sol = combined_ho.getSolutions()

        # ---- HoQp debug chain (Hessian conditioning, residual norms, nullspace dims)
        try:
            chain = combined_ho.get_debug_chain() if hasattr(combined_ho, 'get_debug_chain') else []
            # Each entry corresponds to one HoQp level from high->low.
            # We log a compact per-level line for post-mortem analysis.
            for lvl, dbg in enumerate(chain):
                if not isinstance(dbg, dict) or len(dbg) == 0:
                    continue

                # stable columns; missing keys become NaN
                def _get(key, default=float('nan')):
                    v = dbg.get(key, default)
                    try:
                        return float(v)
                    except Exception:
                        return default

                self.logger.write_row(
                    name=f'hoqp_debug_lvl{lvl}',
                    filename='hoqp_debug_data.csv',
                    header=[
                        't', 'level',
                        'qp_dim', 'H_cond', 'H_min_eig',
                        'res_task_norm', 'res_task_rms', 'res_task_dim',
                        'res_stacked_norm', 'res_stacked_rms', 'res_stacked_dim',
                        'null_prev_dim', 'null_dim', 'A_proj_rank',
                        'debug_error',
                    ],
                    row=[
                        float(self._t), float(lvl),
                        _get('qp_dim'), _get('H_cond'), _get('H_min_eig'),
                        _get('res_task_norm'), _get('res_task_rms'), _get('res_task_dim'),
                        _get('res_stacked_norm'), _get('res_stacked_rms'), _get('res_stacked_dim'),
                        _get('null_prev_dim'), _get('null_dim'), _get('A_proj_rank'),
                        _get('debug_error', 0.0),
                    ],
                )
        except Exception:
            pass

        return sol


if __name__ == "__main__":
    # Quick smoke test to ensure integration with ho_qp works
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", dev)
    dtype = torch.float64
    info = CentroidalModelInfoSimple(generalizedCoordinatesNum=18, actuatedDofNum=12, numThreeDofContacts=4, robotMass=30.0)
    pino = FakePinocchioInterface(info, device=dev, dtype=dtype)
    w = Wbc(task_file="", pino_interface=pino, info=info, device=dev, dtype=dtype)
    target_pos = {
            "com": torch.tensor([0., 0., 0.0], device=dev, dtype=dtype),
            "LH": torch.tensor([-0.1, 0.5, -0.1], device=dev, dtype=dtype),
            "LF": torch.tensor([0.1, 0.5, -0.1], device=dev, dtype=dtype),
            "RF": torch.tensor([0.1, -0.5, -0.2], device=dev, dtype=dtype),
            "RH": torch.tensor([-0.1, -0.5, -0.1], device=dev, dtype=dtype),
        }

    target_ori = {
        "com": torch.eye(3, device=dev, dtype=dtype),
        "LH": torch.eye(3, device=dev, dtype=dtype),
        "LF": torch.eye(3, device=dev, dtype=dtype),
        "RF": torch.eye(3, device=dev, dtype=dtype),
        "RH": torch.eye(3, device=dev, dtype=dtype),
    }
    # w.update_targets(target_pos, target_ori)

    # build trivial inputs
    measured = torch.tensor([0., 0., 0., 0., 0., 0., 0.,
                                            0., 0., 0.,
                                            0., 0., 0.0,
                                            0., 0., 0.0,
                                            0., 0., 0.0], dtype=dtype)
    input_desired = torch.tensor([0., 0., 0.0, 0., 0., 0.,
                                            0., 0., 0.0,
                                            0., 0., 0.0,
                                            0., 0., 0.0,
                                            0., 0., 0.0], dtype=dtype)

    sol = w.update( measured, input_desired, mode=0)
    print("Solution vector shape:", sol.shape)
    print("Solution vector:", sol)
