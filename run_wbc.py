from ischedule import schedule, run_loop
from wbc_ik_qp.Centroidal import CentroidalModelInfoSimple
from wbc_ik_qp.wbc import Wbc, FakePinocchioInterface
from wbc_ik_qp.ik_visualization import URDFModel,URDFMeshcatViewer
from wbc_ik_qp.ik import Model_Cusadi

import argparse
import time
import math
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, List, Optional

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

import numpy as np
import torch
import sys

import pandas as pd
import os
import csv

com_error_csv_path = '/home/ReLUQP-py/wbc_ik_qp/debug/com_error_data.csv'
FL_error_csv_path = '/home/ReLUQP-py/wbc_ik_qp/debug/FL_error_data.csv'
RL_error_csv_path = '/home/ReLUQP-py/wbc_ik_qp/debug/RL_error_data.csv'
FH_error_csv_path = '/home/ReLUQP-py/wbc_ik_qp/debug/FH_error_data.csv'
RH_error_csv_path = '/home/ReLUQP-py/wbc_ik_qp/debug/RH_error_data.csv'

com_error_csv_initialized = False
FL_error_csv_initialized = False
RL_error_csv_initialized = False
FH_error_csv_initialized = False
RH_error_csv_initialized = False
sys.path.append('/home/ReLUQP-py/wbc_ik_qp')
sys.path.append('/home/ReLUQP-py')

parser = argparse.ArgumentParser(description='Simple URDF MeshCat viewer (no pinocchio/placo)')
parser.add_argument('path', help='Path to URDF file')
parser.add_argument('--frames', nargs='+', help='Frame names to display (unused currently)')
parser.add_argument('--animate', action='store_true', help='Animate joints (sinusoidal)')
parser.add_argument('--no-browser', action='store_true', help='Do not try to open browser automatically')
args = parser.parse_args()
model = URDFModel(args.path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64
info = CentroidalModelInfoSimple(19, 12, 4)
robot = Model_Cusadi(info, device=device, dtype=dtype)
pino_interface = FakePinocchioInterface(info, device=device, dtype=dtype)
wbc = Wbc("", pino_interface, info, verbose=True, device=device, dtype=dtype)

target_pos = {
            "com": torch.tensor([0., 0., 0.], device=device, dtype=dtype),
            "LH": torch.tensor([-0.44, 0.27, -0.55], device=device, dtype=dtype),
            "LF": torch.tensor([0.44, 0.27, -0.55], device=device, dtype=dtype),
            "RF": torch.tensor([0.44, -0.27, -0.55], device=device, dtype=dtype),
            "RH": torch.tensor([-0.44, -0.27, -0.55], device=device, dtype=dtype),
        }
target_ori = {
            "com": torch.eye(3, device=device, dtype=dtype),
            "LH": torch.eye(3, device=device, dtype=dtype),
            "LF": torch.eye(3, device=device, dtype=dtype),
            "RF": torch.eye(3, device=device, dtype=dtype),
            "RH": torch.eye(3, device=device, dtype=dtype),
        }

motor_map = {
    "FL_hip_joint": 7,
    "FL_thigh_joint": 8,
    "FL_calf_joint": 9,
    "RL_hip_joint": 10,
    "RL_thigh_joint": 11,
    "RL_calf_joint": 12,
    "FR_hip_joint": 13,
    "FR_thigh_joint": 14,
    "FR_calf_joint": 15,
    "RR_hip_joint": 16,
    "RR_thigh_joint": 17,
    "RR_calf_joint": 18
}
viewer = URDFMeshcatViewer(model, open_browser=not args.no_browser, motor_map_=motor_map)

wbc.update_targets(target_pos, target_ori)

measured = torch.tensor([0., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0.,
                                        0., 0., 0.0,
                                        0., 0., 0.0,
                                        0., 0., 0.0],device=device, dtype=dtype)
input_desired = torch.tensor([0., 0., 0.0, 0., 0., 0.,
                                        0., 0., 0.0,
                                        0., 0., 0.0,
                                        0., 0., 0.0,
                                        0., 0., 0.0],device=device, dtype=dtype)

sol = wbc.update( measured, input_desired, mode=0)
state_desired = measured
state_desired[0:3] = measured[0:3] + sol[0:3]
state_desired[7:19] = measured[7:19] + sol[6:18]
dt = 0.01
t = 0.0

@schedule(interval=dt)
def loop():
    global t,state_desired,target_pos,measured,FL_error_csv_initialized
    t += 1.*dt
    # target_pos["com"][0] = 0.0 + 0.1*t
    h = 0.1 / -0.25
    l = 0.2
    if t%2.0 < 1.0:
        target_pos["LF"][0] = 0.44 + -l * (t % 2.0) 
        target_pos["LF"][1] = 0.27
        target_pos["LF"][2] = -0.55 

        target_pos["RH"][0] = -0.44 + -l * (t % 2.0) + 0.1
        target_pos["RH"][1] = -0.27
        target_pos["RH"][2] = -0.55

        target_pos["LH"][0] = -0.44 + l * (t % 2.0) 
        target_pos["LH"][2] = -0.55 + h*(t % 2.0)*(t % 2.0-1.0)

        target_pos["RF"][0] = 0.44 + l * (t % 2.0) - l
        target_pos["RF"][2] = -0.55 + h*(t % 2.0)*(t % 2.0-1.0)

    else :
        target_pos["LH"][0] = -0.44 + -l * (t % 2.0 - 1.0) +l
        target_pos["LH"][2] = -0.55 

        target_pos["RF"][0] = 0.44 + -l * (t % 2.0 - 1.0)
        target_pos["RF"][2] = -0.55 

        target_pos["LF"][0] = 0.44 + l * ((t % 2.0) - 1.0) - l 
        target_pos["LF"][1] = 0.27
        target_pos["LF"][2] = -0.55 + h*((t % 2.0) - 1.0)*((t % 2.0)-1.0 - 1.0)

        target_pos["RH"][0] = -0.44 + l * ((t % 2.0) - 1.0) - l + 0.1
        target_pos["RH"][1] = -0.27
        target_pos["RH"][2] = -0.55 + h*((t % 2.0) - 1.0)*((t % 2.0) - 1.0 - 1.0)
        
    # target_pos["LF"][0] = 0.44 + 0.5*math.sin(2.0*math.pi*0.5*t)
    # target_pos["LF"][1] = 0.27 + -0.5*math.sin(1.0*math.pi*0.5*t)
    # target_pos["LF"][2] = -0.55 + -0.5*math.sin(2.0*math.pi*0.5*t)

    # target_pos["RF"][0] = 0.44 + 0.2*math.sin(2.0*math.pi*0.5*t)
    # target_pos["RF"][1] = -0.27 + -0.2*math.sin(1.0*math.pi*0.5*t)
    # target_pos["RF"][2] = -0.55 + -0.2*math.sin(2.0*math.pi*0.5*t)

    # target_pos["LH"][0] = -0.44 + 0.2*math.sin(2.0*math.pi*0.5*t)
    # target_pos["LH"][1] = 0.27 + -0.2*math.sin(1.0*math.pi*0.5*t)
    # target_pos["LH"][2] = -0.55 + -0.2*math.sin(2.0*math.pi*0.5*t)

    # target_pos["RH"][0] = -0.44 + -0.2*math.sin(2.0*math.pi*0.5*t)
    # target_pos["RH"][1] = -0.27 + 0.2*math.sin(1.0*math.pi*0.5*t)
    # target_pos["RH"][2] = -0.55 + 0.5 * t

    # print(f"Time: {t:.2f} sec, LF z target: {target_pos['LF'][2]:.3f}")
    wbc.update_targets(target_pos, target_ori)
    sol = wbc.update( measured, input_desired, mode=0)
    state_desired = measured
    state_desired[0:3] = measured[0:3] + sol[0:3]
    state_desired[7:19] = measured[7:19] + sol[6:18]
    measured = state_desired
    LF_pos = robot.getPosition(info.getstate(), info.getinput(), "LF_FOOT")
    pos_error = (LF_pos.cpu().numpy() - target_pos["LF"].cpu().numpy())
    error_csv_path = FL_error_csv_path
    if not FL_error_csv_initialized:
        os.makedirs(os.path.dirname(error_csv_path), exist_ok=True)
        with open(error_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['i', 'pos_error[0]', 'pos_error[1]', 'pos_error[2]'])
        FL_error_csv_initialized = True

    row = [float(t), float(pos_error[0]), float(pos_error[1]), float(pos_error[2])]
    # 追加到 CSV
    with open(error_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    viewer.animate_state(state_desired=state_desired, rate=60.0)
run_loop()
