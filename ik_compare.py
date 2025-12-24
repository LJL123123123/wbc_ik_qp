from wbc_ik_qp.ik import Model_Cusadi
from wbc_ik_qp.Centroidal import CentroidalModelInfoSimple
import torch
from typing import Optional, Sequence
import numpy as np
import random
import pandas as pd
import os
import csv

from src import *
from casadi import *

kinematic_casadi = casadi.Function.load("/home/casadi/wbc_ik_qp/src/casadi_functions/anymal_example_LF_FOOT_jacobian.casadi")

x0 = np.array([0.1, 0.2, 0.3, 0.3390053, 0.620543, 0.6205462, -0.3390048,
                                    1.5708, 1.5708, 0.71,
                                    0, 0, 0,
                                    0, 0, 0,
                                    0, 0, 0])
u0 = np.array([1.0, 0., 0., 0., 0., 0.,
                                    0.1, 0, 0,
                                    0, 0, 0,
                                    0, 0, 0,
                                    0, 0, 0])

J_casadi = kinematic_casadi(x0)
print("Casadi Jacobian:\n", J_casadi)

dtype = torch.float64
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

info = CentroidalModelInfoSimple(generalizedCoordinatesNum=18, actuatedDofNum=12, numThreeDofContacts=4, robotMass=30.0)
robot_q = np.array([0.9, 0.6, -0.9, 0., 0., 0., 1.,
            0.9, 0.7, -0.1,
            0.9, 0.5, -0.9,
            0.1, 0.6, 0.2,
            0., 0., 0.0])
robot_dq = np.array([1., 0.1, 0., 0., 0., 0.,
            0., 0., 0.,
            0., 0., 0.,
            0., 0., 0.0,
            0., 0., 0.0])

measured = torch.tensor(robot_q,device=dev, dtype=dtype)
input_desired = torch.tensor(robot_dq,device=dev, dtype=dtype)
info.update_state_input(
            measured,
            input_desired
            )
cusadi_robot = Model_Cusadi(info)  # This line will now raise an error due to


cusadi_pos = cusadi_robot.getPosition(info.getstate(), info.getinput(), "LF_FOOT")
cusadi_att = cusadi_robot.getAttitude(info.getstate(), info.getinput(), "LF_FOOT")
cusadi_J = cusadi_robot.getJacobian(info.getstate(), info.getinput(), "LF_FOOT")

# print("Position:", cusadi_pos)
# print("Attitude:", cusadi_att)
# print("Jacobian:", cusadi_J)

import placo

placo_robot = placo.RobotWrapper("/home/casadi/anymal_b_simple_description/urdf", placo.Flags.ignore_collisions)
placo_robot.state.q = robot_q
placo_robot.state.qd = robot_dq
placo_robot.update_kinematics()

com_error_csv_path = '/home/casadi/wbc_ik_qp/debug/com_error_data.csv'
FL_error_csv_path = '/home/casadi/wbc_ik_qp/debug/FL_error_data.csv'
RL_error_csv_path = '/home/casadi/wbc_ik_qp/debug/RL_error_data.csv'
FH_error_csv_path = '/home/casadi/wbc_ik_qp/debug/FH_error_data.csv'
RH_error_csv_path = '/home/casadi/wbc_ik_qp/debug/RH_error_data.csv'

com_error_csv_initialized = False
FL_error_csv_initialized = False
RL_error_csv_initialized = False
FH_error_csv_initialized = False
RH_error_csv_initialized = False

def compare_cusadi_placo(__name__: str,i:int=0):
    global robot_q, robot_dq, cusadi_robot, placo_robot, info,com_error_csv_initialized  , FL_error_csv_initialized, RL_error_csv_initialized, FH_error_csv_initialized, RH_error_csv_initialized
    robot_q = np.random.rand(19)
    robot_dq = np.random.rand(18)

    # set quaternion (indices 3:7) to a random unit quaternion
    q = np.random.randn(4)
    q /= np.linalg.norm(q)
    robot_q[3:7] = q

    measured = torch.tensor(robot_q,device=dev, dtype=dtype)
    input_desired = torch.tensor(robot_dq,device=dev, dtype=dtype)
    info.update_state_input(
            measured,
            input_desired
            )
    cusadi_pos = cusadi_robot.getPosition(info.getstate(), info.getinput(), __name__)
    cusadi_att = cusadi_robot.getAttitude(info.getstate(), info.getinput(), __name__)
    cusadi_J = cusadi_robot.getJacobian(info.getstate(), info.getinput(), __name__)   

    # print(f"cusadi {__name__} Jacobian:\n{cusadi_J}\n")
    # print(f"cusadi_pos.cpu().numpy():\n{cusadi_pos.cpu().numpy()}\n")

    placo_robot.state.q = robot_q
    placo_robot.state.qd = robot_dq
    placo_robot.update_kinematics()
    if __name__ == "COM":
        __name__ = "base"
    foot_pos = placo_robot.get_T_world_frame(__name__)[:3,3]
    foot_att = placo_robot.get_T_world_frame(__name__)[:3,:3]
    foot_J = placo_robot.frame_jacobian(__name__,"local")
    # print(f"placo {__name__} Jacobian:\n{foot_J}\n")
    # print(f"foot_pos:\n{foot_pos}\n")
    # print(f"cusadi_{__name__}_J.cpu().numpy():\n{cusadi_J.cpu().numpy()[0]}\n")

    pos_diff = np.linalg.norm(cusadi_pos.cpu().numpy() - foot_pos)
    att_diff = np.linalg.norm(cusadi_att.cpu().numpy() - foot_att)
    J_diff = np.linalg.norm(cusadi_J.cpu().numpy()[0] - foot_J)

    # J_casadi = kinematic_casadi(robot_q)
    # J_ca_cu_diff = np.linalg.norm(cusadi_J.cpu().numpy()[0] - J_casadi)
    # J_ca_p_diff = np.linalg.norm(foot_J - J_casadi)

    # print(f"Comparing {__name__}:")
    # print(f"Position difference: {pos_diff:.6e}")
    # print(f"Attitude difference: {att_diff:.6e}")
    # print(f"Jacobian difference: {J_diff:.6e}")
    
    # 初始化（仅第一次循环写入表头）
    if __name__ == "base":
        error_csv_path = com_error_csv_path
        if not com_error_csv_initialized:
            os.makedirs(os.path.dirname(error_csv_path), exist_ok=True)
            with open(error_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['i', 'Position difference', 'Attitude difference', 'Jacobian difference'])
            com_error_csv_initialized = True
    elif __name__ == "LF_FOOT":
        error_csv_path = FL_error_csv_path
        if not FL_error_csv_initialized:
            os.makedirs(os.path.dirname(error_csv_path), exist_ok=True)
            with open(error_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['i', 'Position difference', 'Attitude difference', 'Jacobian difference'])
            FL_error_csv_initialized = True
    elif __name__ == "RF_FOOT":
        error_csv_path = RL_error_csv_path
        if not RL_error_csv_initialized:
            os.makedirs(os.path.dirname(error_csv_path), exist_ok=True)
            with open(error_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['i', 'Position difference', 'Attitude difference', 'Jacobian difference'])
            RL_error_csv_initialized = True
    elif __name__ == "LH_FOOT":
        error_csv_path = FH_error_csv_path
        if not FH_error_csv_initialized:
            os.makedirs(os.path.dirname(error_csv_path), exist_ok=True)
            with open(error_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['i', 'Position difference', 'Attitude difference', 'Jacobian difference'])
            FH_error_csv_initialized = True
    elif __name__ == "RH_FOOT":
        error_csv_path = RH_error_csv_path
        if not RH_error_csv_initialized:
            os.makedirs(os.path.dirname(error_csv_path), exist_ok=True)
            with open(error_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['i', 'Position difference', 'Attitude difference', 'Jacobian difference'])
            RH_error_csv_initialized = True
    row = [int(i), float(pos_diff), float(att_diff), float(J_diff)]
    # 追加到 CSV
    with open(error_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


for i in range(100):
    compare_cusadi_placo("COM",i)
    compare_cusadi_placo("LF_FOOT",i)
    compare_cusadi_placo("RF_FOOT",i)
    compare_cusadi_placo("LH_FOOT",i)
    compare_cusadi_placo("RH_FOOT",i)

