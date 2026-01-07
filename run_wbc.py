from ischedule import schedule, run_loop
from Centroidal import CentroidalModelInfoSimple
from wbc import Wbc, FakePinocchioInterface
from ik_visualization import URDFModel,URDFMeshcatViewer
from ik import Model_Cusadi

import argparse
import time
import math
# removed unused imports: xml.etree.ElementTree, typing.*

# meshcat and numpy not used directly in this script
import torch
import sys
import select
import termios
import tty
import atexit

# pandas not used
import os

sys.path.append('.')
sys.path.append('/home/wbc_ik_qp')

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
height = 0.26
target_pos = {
            "com": torch.tensor([0., 0., height], device=device, dtype=dtype),
            "LH": torch.tensor([-0.25, 0.15, 0.], device=device, dtype=dtype),
            "LF": torch.tensor([0.14, 0.15, 0.], device=device, dtype=dtype),
            "RF": torch.tensor([0.14, -0.15, 0.], device=device, dtype=dtype),
            "RH": torch.tensor([-0.25, -0.15, -0.], device=device, dtype=dtype),
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


measured = torch.tensor([0., 0., 0.26, 0., 0., 0., 1.,
                                        0., 1.08, -1.80,
                                        0., 1.08, -1.80,
                                        0., 1.08, -1.80,
                                        0., 1.08, -1.80,],device=device, dtype=dtype)
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

# smoothing state for solver outputs to avoid large sudden command jumps
# prev_sol holds the previous solution vector (torch tensor on device)
try:
    prev_sol
except NameError:
    prev_sol = sol.clone()
    # smoothing factor alpha in [0,1], larger -> smoother/slower
    sol_smooth_alpha = 0.85
    # max allowed angle step (rad) for orientation increment per control step
    # this prevents large instantaneous rotations that can destabilize the loop
    sol_max_angle_step = 0.05  # ~2.8 degrees per dt
    # dedicated smoothing for orientation increment (phi)
    sol_phi_smooth_alpha = 0.9
    try:
        prev_phi
    except NameError:
        prev_phi = torch.zeros((3,), device=device, dtype=dtype)

# ------------------ keyboard control setup ------------------
# we'll read single-key presses non-blocking from stdin. Holding a key
# typically generates repeated key events; we treat a key as "pressed"
# when we've seen an event within `press_timeout` seconds.
try:
    orig_termios = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
except Exception:
    orig_termios = None

def restore_terminal():
    if orig_termios is not None:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_termios)
        except Exception:
            pass

atexit.register(restore_terminal)

last_pressed = {}
press_timeout = 0.18  # seconds: how long since last key event we consider key held

def poll_keyboard():
    # read all available characters from stdin and update last_pressed times
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        try:
            ch = sys.stdin.read(1)
        except Exception:
            return
        if ch:
            ch = ch.lower()
            # register timestamp for this key
            # For movement keys, treat a new key press as exclusive: clear other movement keys
            movement_keys = set(['w', 'a', 's', 'd', 'q', 'e', 'r', 'f'])
            now = time.time()
            last_pressed[ch] = now
            if ch in movement_keys:
                # remove any other movement key entries so opposite or new direction takes effect immediately
                for k in list(last_pressed.keys()):
                    if k != ch and k in movement_keys:
                        del last_pressed[k]

def key_held(key: str) -> bool:
    now = time.time()
    return (key in last_pressed) and (now - last_pressed[key] < press_timeout)

# control parameters (you can tune speeds)
speed_forward = 0.1  # m/s
speed_lateral = 0.1  # m/s
yaw_speed = 0.1       # rad/s

# height control: r up / f down. We send it as cmd_vxyz.z (m/s).
height_max = 0.26
height_min = 0.0
height_speed = 0.05   # m/s


from scipy.spatial.transform import Rotation as R
def quaternion2rot(quaternion):
    r = R.from_quat(quaternion)
    rot = r.as_matrix()
    return rot

def quat_mul(q, r):
    """
    Hamilton product q ⊗ r for quaternions in (x,y,z,w) order.
    q, r: tensors with shape (..., 4)
    returns: tensor shape (...,4)
    """
    # ensure shapes: (...,4)
    qx, qy, qz, qw = q.unbind(-1)
    rx, ry, rz, rw = r.unbind(-1)

    # vector part
    vx = qw * rx + rw * qx + (qy * rz - qz * ry)
    vy = qw * ry + rw * qy + (qz * rx - qx * rz)
    vz = qw * rz + rw * qz + (qx * ry - qy * rx)
    # scalar part
    vw = qw * rw - (qx * rx + qy * ry + qz * rz)

    return torch.stack([vx, vy, vz, vw], dim=-1)

def quat_normalize(q, eps=1e-12):
    norm = torch.linalg.norm(q, dim=-1, keepdim=True)
    return q / (norm + eps)

quaternion = [0, 0, 0, 1.]
prev_phi = torch.zeros((3,), device=device, dtype=dtype)
print(quaternion2rot(quaternion))
@schedule(interval=dt)
def loop():
    global t,state_desired,measured
    # advance internal time (kept as original scaling)
    delta_t = 1.*dt
    t += delta_t

    # poll keyboard for recent key presses
    poll_keyboard()

    # compute commanded velocities from keys (w/a/s/d/q/e)
    vx = 0.0
    if key_held('w'):
        vx += speed_forward
    if key_held('s'):
        vx -= speed_forward

    vy = 0.0
    if key_held('a'):
        vy += speed_lateral
    if key_held('d'):
        vy -= speed_lateral

    yaw_rate_cmd = 0.0
    # q: counter-clockwise (positive); e: clockwise (negative)
    if key_held('q'):
        yaw_rate_cmd += yaw_speed
    if key_held('e'):
        yaw_rate_cmd -= yaw_speed

    vz = 0.0
    if key_held('r'):
        vz += height_speed
    if key_held('f'):
        vz -= height_speed

    # Hard clamp: if current measured height already at limit, don't command further.
    try:
        z_now = float(measured[2].item())
    except Exception:
        z_now = float(measured[2])
    if (z_now >= height_max and vz > 0.0) or (z_now <= height_min and vz < 0.0):
        vz = 0.0

    # delegate planning + solve + logging to WBC
    cmd_vxyz = torch.tensor([vx, vy, vz], device=device, dtype=dtype)
    cmd_yaw = torch.tensor(yaw_rate_cmd, device=device, dtype=dtype)
    state_desired = wbc.step_with_cmd(
        measured_rbd_state=measured,
        input_desired=input_desired,
        dt=dt,
        cmd_vxyz=cmd_vxyz,
        cmd_yaw_rate=cmd_yaw,
        mode=0,
    )

    measured = state_desired
    viewer.animate_state(state_desired=state_desired.cpu().detach().numpy(), rate=60.0)
run_loop()
