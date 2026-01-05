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

        return combined_ho.getSolutions()


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
