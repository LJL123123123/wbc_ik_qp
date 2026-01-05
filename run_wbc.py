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
import csv

com_target_csv_path = './debug/com_target_data.csv'
com_target_csv_initialized = False
FL_target_csv_path = './debug/FL_target_data.csv'
FL_target_csv_initialized = False
RF_target_csv_path = './debug/RF_target_data.csv'
RF_target_csv_initialized = False 
LH_target_csv_path = './debug/LH_target_data.csv'
LH_target_csv_initialized = False
RH_target_csv_path = './debug/RH_target_data.csv'
RH_target_csv_initialized = False

com_opimal_csv_path = './debug/com_opimal_data.csv'
com_opimal_csv_initialized = False
FL_opimal_csv_path = './debug/FL_opimal_data.csv'
FL_opimal_csv_initialized = False
RF_opimal_csv_path = './debug/RF_opimal_data.csv'
RF_opimal_csv_initialized = False
LH_opimal_csv_path = './debug/LH_opimal_data.csv'
LH_opimal_csv_initialized = False
RH_opimal_csv_path = './debug/RH_opimal_data.csv'
RH_opimal_csv_initialized = False

com_state_csv_path = './debug/com_state_data.csv'
com_state_csv_initialized = False
FL_state_csv_path = './debug/FL_state_data.csv'
FL_state_csv_initialized = False
RF_state_csv_path = './debug/RF_state_data.csv'
RF_state_csv_initialized = False
LH_state_csv_path = './debug/LH_state_data.csv'
LH_state_csv_initialized = False
RH_state_csv_path = './debug/RH_state_data.csv'
RH_state_csv_initialized = False

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

# store nominal foot positions (world) to return to when idle
nominal_feet = {k: v.clone() for k, v in target_pos.items() if k != 'com'}

# compute nominal offsets of feet in the body frame so feet can follow the body
# offsets are vectors from COM to foot expressed in body frame
nominal_body_offsets = {}
for k, v in nominal_feet.items():
    # offset in world: foot - com
    offset_world = v - target_pos['com']
    # express in body frame: R_body^T * offset_world
    nominal_body_offsets[k] = target_ori['com'].T @ offset_world

# swing planner parameters
cycle_period = 1.0  # full gait cycle (s)
swing_duration = cycle_period / 2.0  # each half cycle one pair swings
swing_height = 0.06  # max foot lift (m)
lookahead = cycle_period  # time to lookahead when placing foothold

# grouping for trot: group1 swings during first half, group2 during second half
group1 = ['LF', 'RH']
group2 = ['RF', 'LH']

# current world positions of feet (updated when swing completes)
# initialize from COM+R*offset so they follow body pose
current_feet_world = {}
for k, off in nominal_body_offsets.items():
    current_feet_world[k] = target_pos['com'] + target_ori['com'] @ off

# per-leg swing state to keep p0/pf consistent during a swing
swing_state = {k: False for k in nominal_body_offsets.keys()}
swing_start_pos = {k: current_feet_world[k].clone() for k in nominal_body_offsets.keys()}
swing_pf = {k: current_feet_world[k].clone() for k in nominal_body_offsets.keys()}

# smoothing transition state: when command on/off toggles we interpolate foot targets
prev_any_cmd = False
transition_active = False
trans_start_t = 0.0
trans_duration = 0.2  # seconds for the auto-differential transition
trans_start_positions = {k: v.clone() for k, v in target_pos.items() if k != 'com'}
trans_end_positions = {}

# small threshold
eps_cmd = 1e-4


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

# CSV logging for raw vs clamped orientation increment (phi)
sol_phi_csv_path = './debug/sol_phi_data.csv'
sol_phi_csv_initialized = False

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
            movement_keys = set(['w', 'a', 's', 'd', 'q', 'e'])
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
    global t,state_desired,measured,prev_phi,sol_phi_csv_initialized
    global com_target_csv_initialized, FL_target_csv_initialized, RF_target_csv_initialized, LH_target_csv_initialized, RH_target_csv_initialized
    global com_opimal_csv_initialized, FL_opimal_csv_initialized, RF_opimal_csv_initialized, LH_opimal_csv_initialized, RH_opimal_csv_initialized
    global com_state_csv_initialized, FL_state_csv_initialized, RF_state_csv_initialized, LH_state_csv_initialized, RH_state_csv_initialized
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

    # Interpret vx,vy as body-frame velocities; convert to world-frame using target orientation
    v_body = torch.tensor([vx, vy, 0.0], device=device, dtype=dtype)
    v_world = target_ori["com"] @ v_body
    # integrate commanded world velocity into target COM
    target_pos["com"] = target_pos["com"] + v_world * delta_t

    # integrate yaw into target orientation (rotate around z)
    if abs(yaw_rate_cmd) > 1e-9:
        angle = yaw_rate_cmd * delta_t
        c = math.cos(angle)
        s = math.sin(angle)
        R_inc = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        target_ori["com"] = R_inc @ target_ori["com"]
    # Determine whether any movement command is active
    any_cmd = (abs(vx) > eps_cmd) or (abs(vy) > eps_cmd) or (abs(yaw_rate_cmd) > eps_cmd)

    # helper cubic interpolator (zero velocity endpoints)
    def cubic_interpolate(p0, pf, s):
        s = float(s)
        s2 = s * s
        s3 = s2 * s
        alpha = 3.0 * s2 - 2.0 * s3
        return p0 + (pf - p0) * alpha

    # if command changed, start a smooth transition between current foot targets
    global prev_any_cmd, transition_active, trans_start_t, trans_duration
    global trans_start_positions, trans_end_positions
    if any_cmd != prev_any_cmd and not transition_active:
        transition_active = True
        trans_start_t = t
        # snapshot start positions from current target_pos
        trans_start_positions = {leg: target_pos[leg].clone() for leg in nominal_body_offsets.keys()}
        trans_end_positions = {}
        if any_cmd:
            # starting movement: compute footholds as the planned end targets
            body_trans = v_world * lookahead
            yaw_angle = yaw_rate_cmd * lookahead
            ca = math.cos(yaw_angle)
            sa = math.sin(yaw_angle)
            R_yaw = torch.tensor([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
            com_pos = target_pos['com']
            for leg in nominal_body_offsets.keys():
                rel_world = target_ori['com'] @ nominal_body_offsets[leg]
                pf = com_pos + (R_yaw @ rel_world) + body_trans
                trans_end_positions[leg] = pf
            print(f"[TRANSITION] start->move at t={t:.2f}, trans_end RF: {trans_end_positions.get('RF')} ")
        else:
            # stopping movement: end target is body-relative nominal foot positions
            for leg in nominal_body_offsets.keys():
                trans_end_positions[leg] = target_pos['com'] + target_ori['com'] @ nominal_body_offsets[leg]
            print(f"[TRANSITION] start->stop at t={t:.2f}, trans_end RF: {trans_end_positions.get('RF')} ")

    # If a transition is active, interpolate targets and skip normal gait logic until finished
    if transition_active:
        s = (t - trans_start_t) / trans_duration
        s_clamped = max(0.0, min(1.0, s))
        for leg in nominal_body_offsets.keys():
            p0 = trans_start_positions.get(leg, target_pos[leg])
            pf = trans_end_positions.get(leg, target_pos[leg])
            pos = cubic_interpolate(p0, pf, s_clamped)
            target_pos[leg] = pos.clone()
        if s_clamped >= 1.0:
            # transition finished
            transition_active = False
            prev_any_cmd = any_cmd
            # sync contact positions to the end of transition
            for leg in nominal_body_offsets.keys():
                current_feet_world[leg] = trans_end_positions.get(leg, target_pos[leg]).clone()
            # debug print: show sync event
            print(f"[TRANSITION] finished at t={t:.2f}, synced current_feet_world RF: {current_feet_world.get('RF')} ")
    else:
        # no transition in progress: use existing idle / gait planner logic
        if not any_cmd:
            # idle: keep feet at nominal positions (body-relative)
            # NOTE: do NOT overwrite current_feet_world here every frame.
            # Overwriting contact positions on every idle loop causes problems when
            # key-hold detection briefly flickers (key repeat or missed events):
            # a transient any_cmd==False would reset the stored contact positions
            # and produce large instantaneous jumps in subsequent gait planning.
            # current_feet_world should be updated explicitly when transitions finish
            # or when a swing completes; here we only set the visual/target pose.
            for leg in nominal_body_offsets.keys():
                world_nom = target_pos['com'] + target_ori['com'] @ nominal_body_offsets[leg]
                target_pos[leg] = world_nom.clone()
        else:
            # compute foothold targets based on commanded velocities and yaw
            # body translation in world coordinates (use v_world)
            body_trans = v_world * lookahead
            yaw_angle = yaw_rate_cmd * lookahead
            ca = math.cos(yaw_angle)
            sa = math.sin(yaw_angle)
            R_yaw = torch.tensor([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)

            # compute desired foothold positions (pf) for each leg
            footholds = {}
            com_pos = target_pos['com']
            for leg in nominal_body_offsets.keys():
                # rel_world is the vector from COM to nominal foot in world frame
                rel_world = target_ori['com'] @ nominal_body_offsets[leg]
                pf = com_pos + (R_yaw @ rel_world) + body_trans
                footholds[leg] = pf
            # print(f"Time: {t:.2f} sec, RF target pos: {target_pos['RF']}")

            # gait phase: which group swings now
            phase = (t % cycle_period)
            if phase < swing_duration:
                swinging_group = group1
                s_phase = phase / swing_duration
            else:
                swinging_group = group2
                s_phase = (phase - swing_duration) / swing_duration

            for leg in nominal_feet.keys():
                # ensure consistent start/target for the swing
                # if this leg just entered swing, cache its start and pf
                if leg in swinging_group:
                    if not swing_state.get(leg, False):
                        swing_state[leg] = True
                        swing_start_pos[leg] = current_feet_world[leg].clone()
                        swing_pf[leg] = footholds[leg].clone()
                        print(f"[SWING] enter {leg} at t={t:.2f}, p0={swing_start_pos[leg]}, pf={swing_pf[leg]}")
                    p0 = swing_start_pos[leg]
                    pf = swing_pf[leg]
                    s = max(0.0, min(1.0, s_phase))
                    pos = cubic_interpolate(p0, pf, s)
                    # add vertical bump
                    z_bump = swing_height * 4.0 * s * (1.0 - s)
                    pos = pos.clone()
                    pos[2] = pos[2] + z_bump
                    target_pos[leg] = pos
                    # when swing completed, update world contact position and clear swing state
                    if s >= 0.999:
                        current_feet_world[leg] = pf.clone()
                        swing_state[leg] = False
                        print(f"[SWING] complete {leg} at t={t:.2f}, new contact={current_feet_world[leg]}")
                else:
                    # stance foot: slowly follow planned foothold (footholds computed each loop)
                    p0 = current_feet_world[leg]
                    pf = footholds[leg]
                    frac = float(min(1.0, max(0.0, delta_t / max(1e-6, lookahead))))
                    pos = cubic_interpolate(p0, pf, frac)
                    target_pos[leg] = pos.clone()

    # print(f"Time: {t:.2f} sec, LF z target: {target_pos['LF'][2]:.3f}")
    wbc.update_targets(target_pos, target_ori)
    sol = wbc.update( measured, input_desired, mode=0)

    # Exponential smoothing on solver output to reduce high-frequency/large spikes
    # (applies to full solution vector). prev_sol and sol_smooth_alpha are defined
    # outside the loop when the script starts.
    try:
        prev_sol
    except NameError:
        prev_sol = sol.clone()
    # blend: smoothed = alpha * prev + (1-alpha) * current
    sol = sol_smooth_alpha * prev_sol + (1.0 - sol_smooth_alpha) * sol
    prev_sol = sol.clone()

    state_desired = measured
    state_desired[0:3] = measured[0:3] + sol[0:3]
    state_desired[7:19] = measured[7:19] + sol[6:18]
    # convert small Euler-angle increment sol[3:6] (rx,ry,rz) to a delta quaternion
    # sol is a torch tensor; ensure we use the correct device/dtype
    phi_raw = sol[3:6].to(device=device, dtype=dtype)
    # apply dedicated exponential smoothing to phi (helps remove high-freq chatter)
    try:
        sol_phi_smooth_alpha
    except NameError:
        sol_phi_smooth_alpha = 0.9
    phi_smoothed = sol_phi_smooth_alpha * prev_phi + (1.0 - sol_phi_smooth_alpha) * phi_raw

    # Per-step clamping: limit the magnitude of the rotation vector phi to
    # avoid large instantaneous rotations that destabilize the controller.
    phi_to_apply = phi_smoothed.clone()
    phi_norm = torch.norm(phi_to_apply)
    max_step = sol_max_angle_step if 'sol_max_angle_step' in globals() else 0.05
    phi_applied = phi_to_apply
    if phi_norm.item() > max_step:
        phi_applied = phi_to_apply * (max_step / phi_norm)
        # small debug print to indicate clamping occurred
        print(f"[WBC] Clamped phi from {phi_raw.cpu().numpy()} (raw_norm={torch.norm(phi_raw).item():.4f}) to {phi_applied.cpu().numpy()} (max={max_step})")
    # update prev_phi for next iteration
    prev_phi = phi_applied.clone()
    # final phi used to form delta quaternion
    phi = phi_applied

    angle = torch.norm(phi)
    # numerical threshold for small angle
    tol = 1e-8
    vec = torch.zeros(3, device=device, dtype=dtype)
    w = torch.tensor(1.0, device=device, dtype=dtype)
    if angle.item() < tol:
        # small-angle approximation: q ~= [0.5*phi, 1 - |phi|^2/8]
        vec = 0.5 * phi
        w = 1.0 - (angle * angle) / 8.0
    else:
        axis = phi / angle
        s = torch.sin(angle / 2.0)
        vec = axis * s
        w = torch.cos(angle / 2.0)
    delta_quaternion = torch.empty(4, device=device, dtype=dtype)
    delta_quaternion[0:3] = vec
    delta_quaternion[3] = w
    # Note: the code originally performed an elementwise addition of quaternion components.
    # Keeping that behavior here (downstream you may prefer quaternion multiplication instead).
    delta_q = delta_quaternion   # shape (4,), on device
    q_meas = measured[3:7]       # shape (4,), on device

    q_new = quat_mul(delta_q.unsqueeze(0), q_meas.unsqueeze(0))[0]
    q_new = quat_normalize(q_new)
    # 如果要写回 state_desired (保持(x,y,z,w)顺序)
    state_desired[3:7] = q_new
    # print(f"state_desired[3:7]: {state_desired[3:7]}")
    measured = state_desired
    # --- log raw vs applied orientation increment (phi) ---
    if not sol_phi_csv_initialized:
        os.makedirs(os.path.dirname(sol_phi_csv_path), exist_ok=True)
        with open(sol_phi_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['t', 'phi_raw_x', 'phi_raw_y', 'phi_raw_z', 'phi_raw_norm', 'phi_smoothed_x', 'phi_smoothed_y', 'phi_smoothed_z', 'phi_smoothed_norm', 'phi_applied_x', 'phi_applied_y', 'phi_applied_z', 'phi_applied_norm'])
        sol_phi_csv_initialized = True
    with open(sol_phi_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([float(t), float(phi_raw[0].cpu().numpy()), float(phi_raw[1].cpu().numpy()), float(phi_raw[2].cpu().numpy()), float(torch.norm(phi_raw).cpu().numpy()), float(phi_smoothed[0].cpu().numpy()), float(phi_smoothed[1].cpu().numpy()), float(phi_smoothed[2].cpu().numpy()), float(torch.norm(phi_smoothed).cpu().numpy()), float(phi[0].cpu().numpy()), float(phi[1].cpu().numpy()), float(phi[2].cpu().numpy()), float(torch.norm(phi).cpu().numpy())])
    # Update the shared model info state/input so robot kinematics reflect the
    # new measured state. Without this, robot.getPosition/getAttitude will read
    # stale None or previous values from `info` and the logged `com_ori_state`
    # will not follow `state_desired`.
    try:
        info.update_state_input(measured, input_desired)
    except Exception:
        # be robust in case info is not configured for this call
        pass

    viewer.animate_state(state_desired=state_desired.cpu().detach().numpy(), rate=60.0)
    
    # log COM position error
    # convert 3x3 rotation matrix to Euler angles (XYZ convention, radians)
    rot_mat = target_ori["com"].cpu().numpy()
    target_ori_euler = R.from_matrix(rot_mat).as_euler('xyz', degrees=False)
    com_pos_target = target_pos["com"].cpu().numpy()
    com_ori_target = target_ori_euler
    if not com_target_csv_initialized:
        os.makedirs(os.path.dirname(com_target_csv_path), exist_ok=True)
        with open(com_target_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['i', 'com_pos_target[0]', 'com_pos_target[1]', 'com_pos_target[2]', 'com_ori_target[0]', 'com_ori_target[1]', 'com_ori_target[2]'])
        com_target_csv_initialized = True
    with open(com_target_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([float(t), float(com_pos_target[0]), float(com_pos_target[1]), float(com_pos_target[2]), float(com_ori_target[0]), float(com_ori_target[1]), float(com_ori_target[2])])

    com_opimal = sol[0:3].cpu().numpy()
    if not com_opimal_csv_initialized:
        os.makedirs(os.path.dirname(com_opimal_csv_path), exist_ok=True)
        with open(com_opimal_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['i', 'com_opimal[0]', 'com_opimal[1]', 'com_opimal[2]'])
        com_opimal_csv_initialized = True
    with open(com_opimal_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([float(t), float(com_opimal[0]), float(com_opimal[1]), float(com_opimal[2])])

    com_pos_state = robot.getPosition(info.getstate(), info.getinput(), "COM").cpu().numpy()
    rot_mat = robot.getAttitude(info.getstate(), info.getinput(), "COM").cpu().numpy()
    # print(f"info.getstate():{info.getstate()}\nrot_mat: {rot_mat}")
    state_ori_euler = R.from_matrix(rot_mat).as_euler('xyz', degrees=False)
    com_ori_state = state_ori_euler
    if not com_state_csv_initialized:
        os.makedirs(os.path.dirname(com_state_csv_path), exist_ok=True)
        with open(com_state_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['i', 'com_state[0]', 'com_state[1]', 'com_state[2]', 'com_ori_state[0]', 'com_ori_state[1]', 'com_ori_state[2]'])
        com_state_csv_initialized = True
    with open(com_state_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([float(t), float(com_pos_state[0]), float(com_pos_state[1]), float(com_pos_state[2]), float(com_ori_state[0]), float(com_ori_state[1]), float(com_ori_state[2])])
    
    RF_target = target_pos["RF"].cpu().numpy()
    if not RF_target_csv_initialized:
        os.makedirs(os.path.dirname(RF_target_csv_path), exist_ok=True)
        with open(RF_target_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['i', 'RF_target[0]', 'RF_target[1]', 'RF_target[2]'])
        RF_target_csv_initialized = True
    with open(RF_target_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([float(t), float(RF_target[0]), float(RF_target[1]), float(RF_target[2])])
    
    RF_opimal = sol[7:10].cpu().numpy()
    if not RF_opimal_csv_initialized:
        os.makedirs(os.path.dirname(RF_opimal_csv_path), exist_ok=True)
        with open(RF_opimal_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['i', 'RF_opimal[0]', 'RF_opimal[1]', 'RF_opimal[2]'])
        RF_opimal_csv_initialized = True
    with open(RF_opimal_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([float(t), float(RF_opimal[0]), float(RF_opimal[1]), float(RF_opimal[2])])

    RF_state = robot.getPosition(info.getstate(), info.getinput(), "RF_FOOT").cpu().numpy()
    if not RF_state_csv_initialized:
        os.makedirs(os.path.dirname(RF_state_csv_path), exist_ok=True)
        with open(RF_state_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['i', 'RF_state[0]', 'RF_state[1]', 'RF_state[2]'])
        RF_state_csv_initialized = True
    with open(RF_state_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([float(t), float(RF_state[0]), float(RF_state[1]), float(RF_state[2])]) 
run_loop()
