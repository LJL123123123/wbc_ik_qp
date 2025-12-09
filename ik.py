import torch
from src import *
from casadi import *
from cusadi import *

@dataclass
class CentroidalModelInfoSimple:
    """Minimal subset of CentroidalModelInfo used by the Wbc python port."""
    generalizedCoordinatesNum: int
    actuatedDofNum: int
    numThreeDofContacts: int
    robotMass: float = 1.0

class Model_cusadi:
    """A tiny fake Pinocchio interface exposing the minimal model/data used
    by the simplified Wbc implementation."""

    def __init__(self, info: CentroidalModelInfoSimple, device=None, dtype=torch.float64):
        self.info = info
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        self._model = FakePinocchioModel(info.generalizedCoordinatesNum)
        self._data.nv = info.generalizedCoordinatesNum
        self._data.nu = info.actuatedDofNum
        self._data.state = torch.zeros(self._data.nv, device=self.device, dtype=self.dtype)
        self._data.input = torch.zeros(self._data.nu, device=self.device, dtype=self.dtype)

        #CoM_cusadi
        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_LF_FOOT_position.casadi"))

        # Foot_cusadi
        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_LF_FOOT_position.casadi"))
        self.LF_FOOT_position_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE)
        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_LH_FOOT_position.casadi"))
        self.LH_FOOT_position_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE)
        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_RH_FOOT_position.casadi"))
        self.RH_FOOT_position_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE)
        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_RF_FOOT_position.casadi"))
        self.RF_FOOT_position_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE)

        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_LF_FOOT_attitude.casadi"))
        self.LF_FOOT_attitude_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE)
        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_LH_FOOT_attitude.casadi"))
        self.LH_FOOT_attitude_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE)
        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_RH_FOOT_attitude.casadi"))
        self.RH_FOOT_attitude_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE)
        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_RF_FOOT_attitude.casadi"))
        self.RF_FOOT_attitude_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE)

        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_LF_FOOT_jacobian.casadi"))
        self.LF_FOOT_jacobian_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE)
        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_LH_FOOT_jacobian.casadi"))
        self.LH_FOOT_jacobian_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE)
        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_RH_FOOT_jacobian.casadi"))
        self.RH_FOOT_jacobian_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE)
        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_RF_FOOT_jacobian.casadi"))
        self.RF_FOOT_jacobian_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE)

        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_LF_FOOT_velocity.casadi"))
        self.LF_FOOT_velocity_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE)
        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_LH_FOOT_velocity.casadi"))
        self.LH_FOOT_velocity_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE)
        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_RH_FOOT_velocity.casadi"))
        self.RH_FOOT_velocity_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE)
        kinematic_casadi = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "anymal_example_RF_FOOT_velocity.casadi"))
        self.RF_FOOT_velocity_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE)

    def getPosition(self, x0, u0, __name__):
        if __name__ == "LF_FOOT":
            self.LF_FOOT_position_cusadi.evaluate((x0, u0))
            position = self.LF_FOOT_position_cusadi.getDenseOutput(0)
        elif __name__ == "LH_FOOT":
            self.LH_FOOT_position_cusadi.evaluate((x0, u0))
            position = self.LH_FOOT_position_cusadi.getDenseOutput(0)
        elif __name__ == "RH_FOOT":
            self.RH_FOOT_position_cusadi.evaluate((x0, u0))
            position = self.RH_FOOT_position_cusadi.getDenseOutput(0)
        elif __name__ == "RF_FOOT":
            self.RF_FOOT_position_cusadi.evaluate((x0, u0))
            position = self.RF_FOOT_position_cusadi.getDenseOutput(0)
        return position

    def getAttitude(self, x0, u0, __name__):
        if __name__ == "LF_FOOT":
            self.LF_FOOT_attitude_cusadi.evaluate((x0, u0))
            attitude = self.LF_FOOT_attitude_cusadi.getDenseOutput(0)
        elif __name__ == "LH_FOOT":
            self.LH_FOOT_attitude_cusadi.evaluate((x0, u0))
            attitude = self.LH_FOOT_attitude_cusadi.getDenseOutput(0)
        elif __name__ == "RH_FOOT":
            self.RH_FOOT_attitude_cusadi.evaluate((x0, u0))
            attitude = self.RH_FOOT_attitude_cusadi.getDenseOutput(0)
        elif __name__ == "RF_FOOT":
            self.RF_FOOT_attitude_cusadi.evaluate((x0, u0))
            attitude = self.RF_FOOT_attitude_cusadi.getDenseOutput(0)
        return attitude

    def getJacobian(self, x0, u0, __name__):
        if __name__ == "LF_FOOT":
            self.LF_FOOT_jacobian_cusadi.evaluate((x0, u0))
            jacobian = self.LF_FOOT_jacobian_cusadi.getDenseOutput(0)
        elif __name__ == "LH_FOOT":
            self.LH_FOOT_jacobian_cusadi.evaluate((x0, u0))
            jacobian = self.LH_FOOT_jacobian_cusadi.getDenseOutput(0)
        elif __name__ == "RH_FOOT":
            self.RH_FOOT_jacobian_cusadi.evaluate((x0, u0))
            jacobian = self.RH_FOOT_jacobian_cusadi.getDenseOutput(0)
        elif __name__ == "RF_FOOT":
            self.RF_FOOT_jacobian_cusadi.evaluate((x0, u0))
            jacobian = self.RF_FOOT_jacobian_cusadi.getDenseOutput(0)
        return jacobian

    def getVelocity(self, x0, u0, __name__):
        if __name__ == "LF_FOOT":
            self.LF_FOOT_velocity_cusadi.evaluate((x0, u0))
            velocity = self.LF_FOOT_velocity_cusadi.getDenseOutput(0)
        elif __name__ == "LH_FOOT":
            self.LH_FOOT_velocity_cusadi.evaluate((x0, u0))
            velocity = self.LH_FOOT_velocity_cusadi.getDenseOutput(0)
        elif __name__ == "RH_FOOT":
            self.RH_FOOT_velocity_cusadi.evaluate((x0, u0))
            velocity = self.RH_FOOT_velocity_cusadi.getDenseOutput(0)
        elif __name__ == "RF_FOOT":
            self.RF_FOOT_velocity_cusadi.evaluate((x0, u0))
            velocity = self.RF_FOOT_velocity_cusadi.getDenseOutput(0)
        return velocity

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


if __name__ == "__main__":
    print(f"✓ Created Model_cusadi")
    info = CentroidalModelInfoSimple(generalizedCoordinatesNum=19,
                                   actuatedDofNum=18,
                                   numThreeDofContacts=4,
                                   robotMass=30.0)
    model = Model_cusadi(info, device='cuda', dtype=torch.double)
    print(f"  - generalizedCoordinatesNum: {model.info.generalizedCoordinatesNum}")
    print(f"  - actuatedDofNum: {model.info.actuatedDofNum}")
    print(f"  - numThreeDofContacts: {model.info.numThreeDofContacts}")
    print(f"  - robotMass: {model.info.robotMass}") 