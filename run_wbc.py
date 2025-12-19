from ischedule import schedule, run_loop
from wbc_ik_qp.Centroidal import CentroidalModelInfoSimple
from wbc_ik_qp.wbc import Wbc, FakePinocchioInterface
from wbc_ik_qp.ik_visualization import URDFModel,URDFMeshcatViewer

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
import os
sys.path.append('/home/ReLUQP-py/wbc_ik_qp')
sys.path.append('/home/ReLUQP-py')

parser = argparse.ArgumentParser(description='Simple URDF MeshCat viewer (no pinocchio/placo)')
parser.add_argument('path', help='Path to URDF file')
parser.add_argument('--frames', nargs='+', help='Frame names to display (unused currently)')
parser.add_argument('--animate', action='store_true', help='Animate joints (sinusoidal)')
parser.add_argument('--no-browser', action='store_true', help='Do not try to open browser automatically')
args = parser.parse_args()
model = URDFModel("/home/ReLUQP-py/anymal_b_simple_description/urdf/robot.urdf")
viewer = URDFMeshcatViewer(model, open_browser=not args.no_browser)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64
info = CentroidalModelInfoSimple(19, 12, 4)
pino_interface = FakePinocchioInterface(info, device, dtype)
wbc = Wbc("", pino_interface, info, verbose=True, device=device, dtype=dtype)

target_pos = {
            "com": torch.tensor([0.3, 0., 0.0], device=device, dtype=dtype),
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
    global t,state_desired,target_pos,measured
    t += dt
    target_pos["LF"][0] = 0.44 + 0.5*math.sin(2.0*math.pi*0.5*t)
    print(f"Time: {t:.2f} sec, LF z target: {target_pos['LF'][2]:.3f}")
    wbc.update_targets(target_pos, target_ori)
    sol = wbc.update( measured, input_desired, mode=0)
    state_desired = measured
    state_desired[0:3] = measured[0:3] + sol[0:3]
    state_desired[7:19] = measured[7:19] + sol[6:18]
    measured = state_desired
    viewer.animate_state(state_desired=state_desired, rate=60.0)
run_loop()
