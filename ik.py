import os
import numpy as np
import torch
from src import *
from casadi import *
from cusadi import *
from dataclasses import dataclass
from typing import Optional, Sequence
from wbc_ik_qp.Centroidal import CentroidalModelInfoSimple


class Model_Cusadi:
    """A tiny fake Pinocchio interface exposing the minimal model/data used
    by the simplified Wbc implementation."""

    def __init__(self, info: CentroidalModelInfoSimple, device=None, dtype=torch.float64,BATCH_SIZE=1):
        self.info = info
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        
        # COM offset compensation to match placo behavior
        # This offset makes COM position zero at zero configuration
        # COM offset to match placo's COM position at zero configuration
        # placo COM at zero config: [-0.00108229, -0.00078014, -0.03434109]
        # Raw cusadi COM at zero config: [-0.0145, 0.0045, 0.0166] (need to verify)
        # Offset = raw - target
        self.COM_OFFSET = torch.tensor([-0.01341771, 0.00528014, 0.05094109], device=self.device, dtype=self.dtype)
        
        casadi_dir = "/home/ReLUQP-py/wbc_ik_qp/src/casadi_functions"
        robot_name = "go1"
        #CoM_cusadi
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_com_position.casadi"))
        self.CoM_position_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_com_attitude.casadi"))
        self.CoM_attitude_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_com_jacobian.casadi"))
        self.CoM_jacobian_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_com_velocity.casadi"))
        self.CoM_velocity_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)

        # Foot_cusadi
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_FL_foot_position.casadi"))
        self.LF_FOOT_position_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_RL_foot_position.casadi"))
        self.LH_FOOT_position_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_RR_foot_position.casadi"))
        self.RH_FOOT_position_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_FR_foot_position.casadi"))
        self.RF_FOOT_position_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)

        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_FL_foot_attitude.casadi"))
        self.LF_FOOT_attitude_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_RL_foot_attitude.casadi"))
        self.LH_FOOT_attitude_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_RR_foot_attitude.casadi"))
        self.RH_FOOT_attitude_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_FR_foot_attitude.casadi"))
        self.RF_FOOT_attitude_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)

        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_FL_foot_jacobian.casadi"))
        self.LF_FOOT_jacobian_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_RL_foot_jacobian.casadi"))
        self.LH_FOOT_jacobian_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_RR_foot_jacobian.casadi"))
        self.RH_FOOT_jacobian_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_FR_foot_jacobian.casadi"))
        self.RF_FOOT_jacobian_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)

        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_FL_foot_velocity.casadi"))
        self.LF_FOOT_velocity_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_RL_foot_velocity.casadi"))
        self.LH_FOOT_velocity_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_RR_foot_velocity.casadi"))
        self.RH_FOOT_velocity_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)
        kinematic_casadi = casadi.Function.load(os.path.join(casadi_dir, f"{robot_name}_FR_foot_velocity.casadi"))
        self.RF_FOOT_velocity_cusadi = CusadiFunction(kinematic_casadi, BATCH_SIZE, robot_name)

    def getPosition(self, x0, u0, __name__):
        # support center-of-mass query
        if __name__.lower() in ("com", "centroid", "center_of_mass"):
            if getattr(self, 'CoM_position_cusadi', None) is None:
                raise RuntimeError("CoM position cusadi function not available")
            self.CoM_position_cusadi.evaluate((x0, u0))
            position = self.CoM_position_cusadi.getDenseOutput(0)
        elif __name__ == "LF_FOOT":
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
        else:
            raise ValueError(f"Unknown frame name for getPosition: {__name__}")

        # convert to torch tensor on the model device/dtype
        try:
            arr = np.asarray(position)
        except Exception:
            arr = position
        
        pos_tensor = torch.as_tensor(arr, device=self.device, dtype=self.dtype).reshape(3,)
        
        # Apply COM offset compensation for COM frame to match placo behavior
        if __name__.lower() in ("com", "centroid", "center_of_mass"):
            pos_tensor = pos_tensor - self.COM_OFFSET
        
        return pos_tensor

    def getAttitude(self, x0, u0, __name__):
        if __name__.lower() in ("com", "centroid", "center_of_mass"):
            if getattr(self, 'CoM_attitude_cusadi', None) is None:
                raise RuntimeError("CoM attitude cusadi function not available")
            self.CoM_attitude_cusadi.evaluate((x0, u0))
            attitude = self.CoM_attitude_cusadi.getDenseOutput(0)
        elif __name__ == "LF_FOOT":
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
        else:
            raise ValueError(f"Unknown frame name for getAttitude: {__name__}")

        try:
            arr = np.asarray(attitude)
        except Exception:
            arr = attitude
        return torch.as_tensor(arr, device=self.device, dtype=self.dtype).reshape(3,3)

    def getJacobian(self, x0, u0, __name__):
        if __name__.lower() in ("com", "centroid", "center_of_mass"):
            if getattr(self, 'CoM_jacobian_cusadi', None) is None:
                raise RuntimeError("CoM jacobian cusadi function not available")
            self.CoM_jacobian_cusadi.evaluate((x0, u0))
            jacobian = self.CoM_jacobian_cusadi.getDenseOutput(0)
        elif __name__ == "LF_FOOT":
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
        else:
            raise ValueError(f"Unknown frame name for getJacobian: {__name__}")

        return jacobian

    def getVelocity(self, x0, u0, __name__):
        if __name__.lower() in ("com", "centroid", "center_of_mass"):
            if getattr(self, 'CoM_velocity_cusadi', None) is None:
                raise RuntimeError("CoM velocity cusadi function not available")
            self.CoM_velocity_cusadi.evaluate((x0, u0))
            velocity = self.CoM_velocity_cusadi.getDenseOutput(0)
        elif __name__ == "LF_FOOT":
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
        else:
            raise ValueError(f"Unknown frame name for getVelocity: {__name__}")

        try:
            arr = np.asarray(velocity)
        except Exception:
            arr = velocity
        return torch.as_tensor(arr, device=self.device, dtype=self.dtype)

    def getModel(self):
        return None

    def getData(self):
        return None

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
    model = Model_Cusadi(info, device='cuda', dtype=torch.double)
    print(f"  - generalizedCoordinatesNum: {model.info.generalizedCoordinatesNum}")
    print(f"  - actuatedDofNum: {model.info.actuatedDofNum}")
    print(f"  - numThreeDofContacts: {model.info.numThreeDofContacts}")
    print(f"  - robotMass: {model.info.robotMass}") 
    x0 = torch.zeros((19,), dtype=torch.double, device='cuda')
    u0 = torch.zeros((18,), dtype=torch.double, device='cuda')
    # pos = model.getPosition(x0, u0, 'com')
    # print("CoM position:", pos)
    # att = model.getAttitude(x0, u0, 'com')
    # print("CoM attitude:", att)
    # jac = model.getJacobian(x0, u0, 'com')
    # print("CoM jacobian:", jac)
    # vel = model.getVelocity(x0, u0, 'com')
    # print("CoM velocity:", vel)
    
    pos = model.getPosition(x0, u0, 'LF_FOOT')
    print("LF_FOOT position:", pos)
    # att = model.getAttitude(x0, u0, 'LF_FOOT')
    # print("LF_FOOT attitude:", att)
    jac = model.getJacobian(x0, u0, 'LF_FOOT')
    print("LF_FOOT jacobian shape:", jac.shape)
    print("LF_FOOT jacobian:", jac)
    # vel = model.getVelocity(x0, u0, 'LF_FOOT')
    # print("LF_FOOT velocity:", vel)