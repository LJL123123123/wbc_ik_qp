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
    # If running as package, try relative import
    from .ho_qp import Task, HoQp

from Centroidal import CentroidalModelInfoSimple
from task.position_task import PositionTask
from task.orientation_task import OrientationTask

@dataclass
class WbcTask:
    position: PositionTask
    orientation: OrientationTask

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

        # Construct a PositionTask for a representative frame. The project's
        # PositionTask implementation expects (info, frame_name, target_world,
        # device, dtype) and will fall back to a minimal robot if the full
        # kinematic backend is not available. We choose a foot frame name by
        # convention; users can override after Wbc construction if desired.
        try:
            self.com_PositionTask = PositionTask(
                info,
                frame_name='com',
                device=self.device,
                dtype=self.dtype,
            )
        except Exception:
            print("Failed to create com_PositionTask, setting to None.")
            self.com_PositionTask = None
        try:
            self.com_OrientationTask = OrientationTask(
                info,
                frame_name='com',
                device=self.device,
                dtype=self.dtype,
            )
        except Exception:
            print("Failed to create com_PositionTask, setting to None.")
            self.com_PositionTask = None
        try:
            self.LF_PositionTask = PositionTask(
                info,
                frame_name='LF_FOOT',
                device=self.device,
                dtype=self.dtype,
            )
        except Exception:
            print("Failed to create LF_PositionTask, setting to None.")
            self.LF_PositionTask = None
        try:
            self.LH_PositionTask = PositionTask(
                info,
                frame_name='LH_FOOT',
                device=self.device,
                dtype=self.dtype,
            )
        except Exception:
            print("Failed to create LH_PositionTask, setting to None.")
            self.LH_PositionTask = None
        try:
            self.RF_PositionTask = PositionTask(
                info,
                frame_name='RF_FOOT',
                device=self.device,
                dtype=self.dtype,
            )
        except Exception:
            print("Failed to create RF_PositionTask, setting to None.")
            self.RF_PositionTask = None
        try:
            self.RH_PositionTask = PositionTask(
                info,
                frame_name='RH_FOOT',
                device=self.device,
                dtype=self.dtype,
            )
        except Exception:
            print("Failed to create RH_PositionTask, setting to None.")
            self.RH_PositionTask = None

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
    def update(self, state_desired: Sequence[float], input_desired: Sequence[float],
               measured_rbd_state: Sequence[float], mode: int):
        """Main update call that returns the stacked HoQp solution vector.

        The implementation here builds a few tasks (some are placeholders)
        and runs the Python HoQp stack to produce a solution vector compatible
        with the decision variable layout described in the header.
        """
        # Convert inputs to torch tensors for potential use
        device = self.device
        dtype = self.dtype

        measured_rbd_state = torch.tensor([0., 0., 0., 0., 0., 0., 0.,
                                            0., 0., 0.0,
                                            0., 0., 0.0,
                                            0., 0., 0.0,
                                            0., 0., 0.0], dtype=dtype)
        input_desired = torch.tensor([0., 0., 0.0, 0., 0., 0.,
                                            0., 0., 0.0,
                                            0., 0., 0.0,
                                            0., 0., 0.0,
                                            0., 0., 0.0], dtype=dtype)
        self.info.update_state_input(
            measured_rbd_state,
            input_desired
        )
        # Build tasks roughly following the C++ order
        # Define task weights - higher priority tasks get higher weights
        com_weight = 100.0   # High priority for COM position
        lf_weight = 1000.0   # Increased weight for foot position to see more effect
        
        com_target_world = torch.tensor([0.0, 0., 0.0], device=device, dtype=dtype)
        task_com_pos = self.com_PositionTask.as_task(
            target_world=com_target_world,
            axises="xyz",
            frame="task",
            weight=com_weight
        )
        # OrientationTask expects a 3x3 rotation matrix, not quaternion
        com_target_attitude = torch.eye(3, device=device, dtype=dtype)  # Identity rotation matrix
        task_com_ori = self.com_OrientationTask.as_task(
            target_attitude=com_target_attitude,
            axises="xyz",
            frame="task",
            weight=com_weight
        )
        LF_target_world = torch.tensor([0.1,0.5,-0.1], device=device, dtype=dtype)
        task_LF_pos = self.LF_PositionTask.as_task(
            target_world=LF_target_world,
            axises="xyz",
            frame="task",
            weight=lf_weight
        )
        LH_target_world = torch.tensor([-0.1,0.5,-0.1], device=device, dtype=dtype)
        task_LH_pos = self.LH_PositionTask.as_task(
            target_world=LH_target_world,
            axises="xyz",
            frame="task",
            weight=lf_weight
        )
        RF_target_world = torch.tensor([0.1,-0.5,-0.2], device=device, dtype=dtype)
        task_RF_pos = self.RF_PositionTask.as_task(
            target_world=RF_target_world,
            axises="xyz",
            frame="task",
            weight=lf_weight
        )
        RH_target_world = torch.tensor([-0.1,-0.5,-0.1], device=device, dtype=dtype)
        task_RH_pos = self.RH_PositionTask.as_task(
            target_world=RH_target_world,
            axises="xyz",
            frame="task",
            weight=lf_weight
        )
        # Create nested hierarchical problems: task_com_pos is the higher (first) priority
        # HoQp expects a Task and an optional higher_problem (HoQp). Construct the
        # higher-priority problem first, then pass it as higher_problem to the
        # lower-priority HoQp instance.
        dev = device if device is not None else torch.device('cpu')

        # Debug: show task shapes and some stats
        print('task_com_pos.a_.shape =', task_com_pos.a_.shape, ' nonzero:', (task_com_pos.a_ != 0).sum().item())
        print('task_LF_pos.a_.shape =', task_LF_pos.a_.shape, ' nonzero:', (task_LF_pos.a_ != 0).sum().item())
        print('COM target error (b) =', task_com_pos.b_)
        print('LF target error (b) =', task_LF_pos.b_)

        # Define hierarchical weights for different priorities
        high_priority_weight = 100.0   # High weight for high priority tasks
        low_priority_weight = 100.0    # Increased weight for low priority tasks
        
        # Combine COM position and orientation tasks into a single high-priority task
        combined_com_task = task_com_pos + task_com_ori
        ho_high = HoQp(combined_com_task, higher_problem=None, device=dev, dtype=dtype, task_weight=high_priority_weight)
        # Diagnostic prints for ho_high
        print('\n--- ho_high diagnostic ---')
        print('ho_high.num_decision_vars_ =', ho_high.num_decision_vars_)
        print('ho_high.num_slack_vars_ =', ho_high.num_slack_vars_)
        print('ho_high.has_eq_constraints_ =', ho_high.has_eq_constraints_)
        print('ho_high.task_weight_ =', ho_high.task_weight_)
        print('ho_high.task_.weight_ =', ho_high.task_.weight_)
        print('ho_high.stacked_z_.shape =', ho_high.getStackedZMatrix().shape)
        print('ho_high.h_.shape =', ho_high.h_.shape)
        print('ho_high.c_.shape =', ho_high.c_.shape)
        print('ho_high.getSolutions() =', ho_high.getSolutions())

        ho0 = HoQp(task_LH_pos, None, device=dev, dtype=dtype, task_weight=low_priority_weight)
        print('\n--- ho0 (LH standalone) diagnostic ---')
        print('ho0.num_decision_vars_ =', ho0.num_decision_vars_)
        print('ho0.task_weight_ =', ho0.task_weight_)
        print('ho0.task_.weight_ =', ho0.task_.weight_)
        print('ho0.stacked_z_.shape =', ho0.getStackedZMatrix().shape)
        print('ho0.h_.shape =', ho0.h_.shape)
        print('ho0.c_.shape =', ho0.c_.shape)
        print('ho0.getSolutions() =', ho0.getSolutions())

        ho1 = HoQp(task_LF_pos, None, device=dev, dtype=dtype, task_weight=low_priority_weight)
        print('\n--- ho1 (LF standalone) diagnostic ---')
        print('ho1.num_decision_vars_ =', ho1.num_decision_vars_)
        print('ho1.task_weight_ =', ho1.task_weight_)
        print('ho1.task_.weight_ =', ho1.task_.weight_)
        print('ho1.stacked_z_.shape =', ho1.getStackedZMatrix().shape)
        print('ho1.h_.shape =', ho1.h_.shape)
        print('ho1.c_.shape =', ho1.c_.shape)
        print('ho1.getSolutions() =', ho1.getSolutions())

        ho2 = HoQp(task_RH_pos, None, device=dev, dtype=dtype, task_weight=low_priority_weight)
        print('\n--- ho2 (RH standalone) diagnostic ---')
        print('ho2.num_decision_vars_ =', ho2.num_decision_vars_)
        print('ho2.task_weight_ =', ho2.task_weight_)
        print('ho2.task_.weight_ =', ho2.task_.weight_)
        print('ho2.stacked_z_.shape =', ho2.getStackedZMatrix().shape)
        print('ho2.h_.shape =', ho2.h_.shape)
        print('ho2.c_.shape =', ho2.c_.shape)
        print('ho2.getSolutions() =', ho2.getSolutions())

        ho3 = HoQp(task_RF_pos, None, device=dev, dtype=dtype, task_weight=low_priority_weight)
        print('\n--- ho3 (RF standalone) diagnostic ---')
        print('ho3.num_decision_vars_ =', ho3.num_decision_vars_)
        print('ho3.task_weight_ =', ho3.task_weight_)
        print('ho3.task_.weight_ =', ho3.task_.weight_)
        print('ho3.stacked_z_.shape =', ho3.getStackedZMatrix().shape)
        print('ho3.h_.shape =', ho3.h_.shape)
        print('ho3.c_.shape =', ho3.c_.shape)
        print('ho3.getSolutions() =', ho3.getSolutions())

        combined_ho = HoQp(task_RF_pos, higher_problem=ho_high, device=dev, dtype=dtype, task_weight=low_priority_weight)
        print('\n--- combined_ho diagnostic ---')
        print('combined.num_decision_vars_ =', combined_ho.num_decision_vars_)
        print('combined.num_slack_vars_ =', combined_ho.num_slack_vars_)
        print('combined.has_eq_constraints_ =', combined_ho.has_eq_constraints_)
        print('combined.task_weight_ =', combined_ho.task_weight_)
        print('combined.task_.weight_ =', combined_ho.task_.weight_)
        print('combined.stacked_z_prev_.shape =', combined_ho.stacked_z_prev_.shape)
        print('combined.getStackedZMatrix().shape =', combined_ho.getStackedZMatrix().shape)
        print('combined.x_prev_.shape =', combined_ho.x_prev_.shape)
        print('combined.h_.shape =', combined_ho.h_.shape)
        print('combined.c_.shape =', combined_ho.c_.shape)
        print('combined.d_.shape =', combined_ho.d_.shape)
        print('combined.f_.shape =', combined_ho.f_.shape)
        print('combined.getSolutions() =', combined_ho.getSolutions())
        return combined_ho.getSolutions()


if __name__ == "__main__":
    # Quick smoke test to ensure integration with ho_qp works
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", dev)
    dtype = torch.float64
    info = CentroidalModelInfoSimple(generalizedCoordinatesNum=18, actuatedDofNum=12, numThreeDofContacts=4, robotMass=30.0)
    pino = FakePinocchioInterface(info, device=dev, dtype=dtype)
    w = Wbc(task_file="", pino_interface=pino, info=info, device=dev, dtype=dtype)

    # build trivial inputs
    state_desired = [0.0] * (info.generalizedCoordinatesNum * 2)
    input_desired = [0.0] * info.actuatedDofNum
    measured = [0.0] * (info.generalizedCoordinatesNum + info.generalizedCoordinatesNum)

    sol = w.update(state_desired, input_desired, measured, mode=0)
    print("Solution vector shape:", sol.shape)
    print("Solution vector:", sol)
