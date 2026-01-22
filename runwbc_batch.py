"""
Modified Batch run_wbc.py
- Supports 2000 robot instances parallel simulation on GPU.
- Visualizes only the 1st robot (Batch 0) to maintain performance.
"""
from ischedule import schedule, run_loop
from Centroidal import CentroidalModelInfoSimple
from wbc_batch import Wbc  # 使用 batch 版本的 WBC
from ik_visualization import URDFModel, URDFMeshcatViewer
from ik import Model_Cusadi

import argparse
import time
import torch
import sys
import select
import termios
import tty
import atexit
import os

sys.path.append('.')
sys.path.append('/home/wbc_ik_qp')

parser = argparse.ArgumentParser(description='Batch URDF Simulation')
parser.add_argument('--path', default="/home/wbc_ik_qp/unitree_model/robots/go1_description/urdf/go1.urdf", help='Path to URDF file')
parser.add_argument('--batch', type=int, default=2000, help='Batch size')
parser.add_argument('--no-browser', action='store_true', help='Do not try to open browser automatically')
args = parser.parse_args()

# --- 1. 基础参数与设备设置 ---
B_SIZE = args.batch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64
print(f"Initializing Batch WBC with size: {B_SIZE} on {device}")

# --- 2. 初始化批量环境 ---
model = URDFModel(args.path)
info = CentroidalModelInfoSimple(19, 12, 4)
robot = Model_Cusadi(info, device=device, dtype=dtype)
wbc = Wbc("", info, batch_size=B_SIZE, device=device, dtype=dtype)

# --- 3. 构造批量初始状态 ---
height = 0.26
# 初始姿态张量化 [B, 19]
initial_q = torch.tensor([0., 0., 0.26, 0., 0., 0.149, 0.989,
                          0., 1.08, -1.80,
                          0., 1.08, -1.80,
                          0., 1.08, -1.80,
                          0., 1.08, -1.80], device=device, dtype=dtype)
measured = initial_q.unsqueeze(0).expand(B_SIZE, -1).clone()

# 初始期望输入 [B, 18]
input_desired = torch.zeros((B_SIZE, 18), device=device, dtype=dtype)

# 批量目标位置 [B, 3]
target_pos = {
    "com": torch.tensor([0., 0., height], device=device, dtype=dtype).unsqueeze(0).expand(B_SIZE, 3),
    "LH": torch.tensor([-0.25, 0.15, 0.], device=device, dtype=dtype).unsqueeze(0).expand(B_SIZE, 3),
    "LF": torch.tensor([0.14, 0.15, 0.], device=device, dtype=dtype).unsqueeze(0).expand(B_SIZE, 3),
    "RF": torch.tensor([0.14, -0.15, 0.], device=device, dtype=dtype).unsqueeze(0).expand(B_SIZE, 3),
    "RH": torch.tensor([-0.25, -0.15, 0.], device=device, dtype=dtype).unsqueeze(0).expand(B_SIZE, 3),
}
# 批量目标姿态 [B, 3, 3]
target_ori = {
    k: torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B_SIZE, 3, 3).contiguous()
    for k in ["com", "LH", "LF", "RF", "RH"]
}

wbc.update_targets(target_pos, target_ori)

# --- 4. 可视化设置 (仅显示一个实例) ---
motor_map = {
    "FL_hip_joint": 7, "FL_thigh_joint": 8, "FL_calf_joint": 9,
    "RL_hip_joint": 10, "RL_thigh_joint": 11, "RL_calf_joint": 12,
    "FR_hip_joint": 13, "FR_thigh_joint": 14, "FR_calf_joint": 15,
    "RR_hip_joint": 16, "RR_thigh_joint": 17, "RR_calf_joint": 18
}
viewer = URDFMeshcatViewer(model, open_browser=not args.no_browser, motor_map_=motor_map)
viewer1 = URDFMeshcatViewer(model, open_browser=not args.no_browser, motor_map_=motor_map)

# --- 5. 键盘控制逻辑 (保持全局指令) ---
try:
    orig_termios = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
except Exception:
    orig_termios = None

def restore_terminal():
    if orig_termios is not None:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_termios)

atexit.register(restore_terminal)

last_pressed = {}
press_timeout = 0.18

def poll_keyboard():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        ch = sys.stdin.read(1).lower()
        if ch:
            movement_keys = {'w', 'a', 's', 'd', 'q', 'e', 'r', 'f'}
            now = time.time()
            last_pressed[ch] = now
            if ch in movement_keys:
                for k in list(last_pressed.keys()):
                    if k != ch and k in movement_keys:
                        del last_pressed[k]

def key_held(key: str) -> bool:
    return (key in last_pressed) and (time.time() - last_pressed[key] < press_timeout)

speed_forward, speed_lateral, yaw_speed, height_speed = 0.8, 0.5, 0.5, 0.05
height_max, height_min = 0.26, 0.15

# --- 6. 主循环 ---
dt = 0.01
t = 0.0

@schedule(interval=dt)
def loop():
    global t, measured
    poll_keyboard()

    # 计算全局指令分量
    vx = speed_forward if key_held('w') else (-speed_forward if key_held('s') else 0.0)
    vy = speed_lateral if key_held('a') else (-speed_lateral if key_held('d') else 0.0)
    yaw_rate = yaw_speed if key_held('q') else (-yaw_speed if key_held('e') else 0.0)
    vz = height_speed if key_held('r') else (-height_speed if key_held('f') else 0.0)

    # 简单的批量高度限制检查 (基于 Batch 0)
    z_now = measured[0, 2].item()
    if (z_now >= height_max and vz > 0.0) or (z_now <= height_min and vz < 0.0):
        vz = 0.0

    # 构造批量指令张量 [B, 3] 和 [B, 1]
    base_cmd = torch.tensor([vx, vy, vz], device=device, dtype=dtype)
    cmd_vxyz = base_cmd.repeat(B_SIZE, 1)   # 真复制，每一行独立
    cmd_vxyz[0] = torch.tensor([0.5, vy, vz], device=device, dtype=dtype)

    # cmd_vxyz[1, :] = 0.0  # 仅 Batch 1 静止（示例）
    # print(cmd_vxyz[0, :], cmd_vxyz[1, :])

    base_yaw = torch.tensor([yaw_rate], device=device, dtype=dtype)
    cmd_yaw = base_yaw.repeat(B_SIZE, 1)  # 真复制，每一行独立
    cmd_yaw[0]= torch.tensor([0.5], device=device, dtype=dtype)
    cmd_yaw[1]= torch.tensor([-0.6], device=device, dtype=dtype)
    # 调用批量 WBC 步进
    # 返回 [B, 19]
    state_desired = wbc.step_with_cmd(
        measured_rbd_state=measured,
        input_desired=input_desired,
        dt=dt,
        cmd_vxyz_batch=cmd_vxyz,
        cmd_yaw_rate_batch=cmd_yaw,
        mode=0,
    )

    measured = state_desired
    
    # 仅可视化 Batch 0 的机器人状态
    # Visualize Batch 0 (keep consistent with LOG_BATCH_IDX default)
    viewer.animate_state(state_desired=state_desired[0].cpu().detach().numpy(), rate=60.0)
    viewer1.animate_state(state_desired=state_desired[1].cpu().detach().numpy(), rate=60.0)
    
    t += dt

print("Batch Simulation Running... Press W/A/S/D to move.")
run_loop()