"""
Modified Batch WBC Implementation - FIXED Batch Size Passing
- Supports parallel execution of multiple QP problems on GPU.
- Integrated with qpth for batched solver.
"""
import os
import sys
import torch
import math
import numpy as np
import casadi as ca
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Sequence, Dict

# 导入自定义模块
from wbc_logger import WbcCsvLogger
from gait_manager_batch import GaitCycleManagerCuda, GaitParamsCuda
from tools.integrrate_batch import integrate_freeflyer_quat_xyzw, quat_to_rotmat_xyzw
from Centroidal import CentroidalModelInfoSimple
from ik import Model_Cusadi

# 动态导入 cusadi
try:
    from cusadi import CusadiFunction
except ModuleNotFoundError:
    cusadi_repo = "/home/cusadi-main"
    if cusadi_repo not in sys.path:
        sys.path.insert(0, cusadi_repo)
    from src import CusadiFunction

# 导入并行求解器
from qpth.qp import QPFunction

@dataclass
class Wbc:
    def __init__(self, task_file: str, 
                 info: CentroidalModelInfoSimple, batch_size: int = 2000, 
                 log_batch_idx: int = 0,
                 device=None, dtype=torch.float64):
        self.device = device if device is not None else torch.device('cuda')
        self.dtype = dtype
        self.info = info
        self.BATCH_SIZE = batch_size
        self.LOG_BATCH_IDX = log_batch_idx # 记录索引
        self.plan = None
        self.verbose = True # 开启日志开关
        # one-time initialization: sync gait manager & WBC targets to measured state
        self._gait_inited = False
        self.robot_name = "go2"

        # --- 1. CasADi 并行函数加载 ---
        self.wbik_qp_casadi = ca.Function.load("./dockerbuild/cusadi_build/go2/go2_wbik_qp.casadi")
        self.wbik_qp = CusadiFunction(self.wbik_qp_casadi, self.BATCH_SIZE, self.robot_name)

        # --- 2. 求解器初始化 ---
        self.qp_solver = QPFunction(verbose=-1) 

        # --- 3. 步态管理器 (核心修复点：明确传递 batch_size) ---
        self.cmd_vxyz_batch = torch.zeros((batch_size, 3), device=self.device, dtype=self.dtype)
        self.cmd_yaw_rate_batch = torch.zeros((batch_size, 1), device=self.device, dtype=self.dtype)
        self.gait = GaitCycleManagerCuda(
            batch_size=self.BATCH_SIZE,
            device=self.device,
            dtype=self.dtype,
            params=GaitParamsCuda()
        )

        # 4. 批量化内部状态
        self.t_batch = torch.zeros((batch_size,), device=self.device, dtype=self.dtype)
        self.measured_q = torch.zeros((batch_size, 19), device=self.device, dtype=self.dtype)
        self.measured_v = torch.zeros((batch_size, 18), device=self.device, dtype=self.dtype)

        target_pos = {
            "com": torch.tensor([0., 0., 0.26], device=device, dtype=dtype),
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
        # self.gait.set_stand_targets(target_pos,target_ori)
        # NOTE: we always need FK for reliable gait initialization (stance reset)
        self.robot = Model_Cusadi(info, device=self.device, dtype=self.dtype)
        if self.verbose:
            self.logger = WbcCsvLogger(base_dir=Path('./debug'))

        # --- 4. 批处理目标状态初始化 ---
        self.target_pos = {
            k: torch.zeros((self.BATCH_SIZE, 3), device=self.device, dtype=self.dtype)
            for k in ["com", "LH", "LF", "RF", "RH"]
        }
        self.target_ori = {
            k: torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0).expand(self.BATCH_SIZE, 3, 3).contiguous()
            for k in ["com", "LH", "LF", "RF", "RH"]
        }

        # --- 5. 限制与参数 (Batched) ---
        self.torque_limits_ = torch.ones((self.BATCH_SIZE, info.actuatedDofNum), device=self.device, dtype=self.dtype) * 50.0
        self._t = 0.0

    def reset_batch(self, batch_idx: int):
        """
        重启指定的 batch 接口
        1. 重置 WBC 内部时间及测量状态
        2. 顺带重启该 batch 的 gait_manager
        """
        if batch_idx >= self.BATCH_SIZE:
            return

        self.t_batch[batch_idx] = 0.0
        self.measured_q[batch_idx].zero_()
        self.measured_v[batch_idx].zero_()
        
        # 同步重启步态管理器中的对应 batch
        self.gait.reset_batch(batch_idx)
        # 允许下一步重新用测量值初始化 gait
        self._gait_inited = False
        if self.verbose:
            print(f"[WbcBatch] Batch {batch_idx} and its GaitManager have been reset.")

    @torch.no_grad()
    def _init_gait_from_measured(self, measured_rbd_state: torch.Tensor, input_desired: torch.Tensor):
        """Initialize gait internal (world_yaw/world_com_pos/stance) from measured state.

        This removes the large discontinuity at t=0 (targets starting from yaw=0 while state yaw!=0),
        and ensures swing trajectories start from the actual stance foot positions.
        """
        B = measured_rbd_state.shape[0]
        device, dtype = self.device, self.dtype

        p0 = measured_rbd_state[:, 0:3].to(device=device, dtype=dtype)
        quat0 = measured_rbd_state[:, 3:7].to(device=device, dtype=dtype)
        R0 = quat_to_rotmat_xyzw(quat0)  # [B,3,3]

        # Feet stance positions from FK (world)
        # Note: getPosition signature is (q, dq, frame_name)
        feet0 = {
            "LH": self.robot.getPosition(measured_rbd_state, input_desired, "LH_FOOT"),
            "LF": self.robot.getPosition(measured_rbd_state, input_desired, "LF_FOOT"),
            "RF": self.robot.getPosition(measured_rbd_state, input_desired, "RF_FOOT"),
            "RH": self.robot.getPosition(measured_rbd_state, input_desired, "RH_FOOT"),
        }
        for k in feet0:
            feet0[k] = feet0[k].to(device=device, dtype=dtype)
            # For gait planning we typically keep stance feet on ground (z=0)
            feet0[k][2] = 0.0

        init_target_pos = {
            "com": p0,
            "LH": feet0["LH"],
            "LF": feet0["LF"],
            "RF": feet0["RF"],
            "RH": feet0["RH"],
        }
        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, 3, 3)
        init_target_ori = {"com": R0, "LH": eye, "LF": eye, "RF": eye, "RH": eye}

        # Sync both WBC targets and gait internal memory
        self.update_targets(init_target_pos, init_target_ori)
        self.gait.set_stand_targets(init_target_pos, init_target_ori)
        self._gait_inited = True

    def _log_series(self, t: float, sol_batch: torch.Tensor, measured_batch: torch.Tensor, input_batch: torch.Tensor):
        """仅提取并记录 LOG_BATCH_IDX 指定的机器人数据"""
        idx = self.LOG_BATCH_IDX
        
        # 1. 提取指定 Batch 的目标数据
        tp = {k: v[idx] for k, v in self.target_pos.items()}
        to = {k: v[idx] for k, v in self.target_ori.items()}
        cmd_RF_batch = self.plan.target_pos['RF'][idx]
        cmd_vxyz_batch = self.cmd_vxyz_batch[idx]
        cmd_yaw_rate_batch = self.cmd_yaw_rate_batch[idx]
        st = measured_batch[idx]  # 当前状态
        it = input_batch[idx]     # 当前输入
        sol = sol_batch[idx]
        st = measured_batch[idx]

        # 2. COM 姿态 RPY 转换 (保持原 wbc.py 逻辑)
        Rcom = to['com']
        sy = torch.sqrt(Rcom[0, 0]**2 + Rcom[1, 0]**2)
        if sy > 1e-9:
            roll, pitch, yaw = torch.atan2(Rcom[2, 1], Rcom[2, 2]), torch.atan2(-Rcom[2, 0], sy), torch.atan2(Rcom[1, 0], Rcom[0, 0])
        else:
            roll, pitch, yaw = torch.atan2(-Rcom[1, 2], Rcom[1, 1]), torch.atan2(-Rcom[2, 0], sy), torch.zeros((), device=self.device)

        # 3. 写入 COM 目标日志
        self.logger.write_row(
            name='com_target', filename='com_target_data.csv',
            header=['t', 'com_x', 'com_y', 'com_z', 'roll', 'pitch', 'yaw'],
            row=[float(t), *tp['com'].tolist(), roll.item(), pitch.item(), yaw.item()]
        )

        # 记录 batch0 / batch1 的指令（用于确认每个 batch 的命令是否不同）
        # NOTE: 之前 com_cmd0/com_cmd1 都写了同一个 idx 的数据，所以看起来永远一样。
        for i, name, fname in [
            (0, 'com_cmd0', 'com_cmd0_data.csv'),
            (1, 'com_cmd1', 'com_cmd1_data.csv'),
        ]:
            if i >= self.BATCH_SIZE:
                continue

            v_i = self.cmd_vxyz_batch[i]
            yaw_i = self.cmd_yaw_rate_batch[i].view(-1)[0]
            self.logger.write_row(
                name=name, filename=fname,
                header=['t', 'cmd_vx', 'cmd_vy', 'cmd_vz', 'cmd_yaw_rate'],
                row=[float(t), v_i[0].item(), v_i[1].item(), v_i[2].item(), yaw_i.item()]
            )

        # --- 新增：RF 足端目标日志 ---
        self.logger.write_row(
            name='rf_target', filename='RF_target_data.csv',
            header=['t', 'x', 'y', 'z'],
            row=[float(t), *tp['RF'].tolist()]
        )

        self.logger.write_row(
            name='rf_cmd', filename='RF_cmd_data.csv',
            header=['t', 'cmd_x', 'cmd_y', 'cmd_z'],
            row=[float(t), cmd_RF_batch[0].item(), cmd_RF_batch[1].item(), cmd_RF_batch[2].item()]
        )

        # 4. 写入解的增量日志
        self.logger.write_row(
            name='com_optimal', filename='com_opimal_data.csv',
            header=['t', 'dx', 'dy', 'dz'],
            row=[float(t), *sol[0:3].tolist()]
        )

        # 5. 写入当前状态日志 (包含四元数转 RPY)
        qx, qy, qz, qw = st[3], st[4], st[5], st[6]
        roll_s = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch_s = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
        yaw_s = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        self.logger.write_row(
            name='com_state', filename='com_state_data.csv',
            header=['t', 'com_x', 'com_y', 'com_z', 'roll', 'pitch', 'yaw'],
            row=[float(t), *st[0:3].tolist(), roll_s.item(), pitch_s.item(), yaw_s.item()]
        )

        try:
            # 使用 robot 模型获取 RF 足端在世界坐标系下的位置
            rf_pos = self.robot.getPosition(st, it, "RF_FOOT")
            
            self.logger.write_row(
                name='rf_state', 
                filename='RF_state_data.csv',
                header=['t', 'x', 'y', 'z'],
                row=[float(t), *rf_pos.tolist()]
            )
        except Exception as e:
            # 记录失败时不中断主循环
            pass

    def update_targets(self, target_pos: Dict[str, torch.Tensor], target_ori: Dict[str, torch.Tensor]):
        for k in self.target_pos.keys():
            self.target_pos[k].copy_(target_pos[k])
        self.target_ori["com"].copy_(target_ori["com"])

    def update(self, measured_rbd_state: torch.Tensor, input_desired: torch.Tensor, mode: int):
        B = self.BATCH_SIZE
        device = self.device
        dtype = self.dtype

        # 1. 构造输入 [Batch, Dim]
        p_feet_des = torch.cat([
            self.target_pos["LH"], self.target_pos["LF"],
            self.target_pos["RF"], self.target_pos["RH"]
        ], dim=-1)

        # Keep weights consistent with the non-batch baseline (COM first, feet second)
        w_trunk_pos = torch.full((B, 1), 1e3, device=device, dtype=dtype)
        w_trunk_ori = torch.full((B, 1), 1e3, device=device, dtype=dtype)
        w_feet = torch.full((B, 1), 1e2, device=device, dtype=dtype)
        lam = torch.full((B, 1), 1e-6, device=device, dtype=dtype)
        # Use real control dt (important if QP uses dt for discretization)
        dt_tensor = torch.full((B, 1), float(self._last_dt), device=device, dtype=dtype)

        # 2. 调用 Cusadi 评估
        inputs = (
            measured_rbd_state, 
            self.target_pos["com"], 
            # IMPORTANT: Cusadi/CasADi typically expects column-major vec(R).
            # torch.view(B,9) is row-major -> effectively passes R^T -> yaw sign flips.
            self.target_ori["com"].transpose(1, 2).reshape(B, 9), 
            p_feet_des,
            w_trunk_pos, w_trunk_ori, w_feet,
            lam, dt_tensor,
            torch.ones((B, 1), device=device, dtype=dtype),
            torch.ones((B, 1), device=device, dtype=dtype)
        )
        self.wbik_qp.evaluate(inputs)

        # 3. 提取矩阵并转换为 qpth 格式 (Gx <= h)
        H = self.wbik_qp.outputs_sparse[0].view(B, 18, 18)
        g = self.wbik_qp.outputs_sparse[1].view(B, 18)
        A_constr = self.wbik_qp.outputs_sparse[2].view(B, 18, 18)
        l_constr = self.wbik_qp.outputs_sparse[3].view(B, 18)
        u_constr = self.wbik_qp.outputs_sparse[4].view(B, 18)

        G = torch.cat([A_constr, -A_constr], dim=1) 
        h = torch.cat([u_constr, -l_constr], dim=1)

        # 4. 数值正则化
        # qpth requires Q to be symmetric PSD
        Hsym = 0.5 * (H + H.transpose(-1, -2))
        reg = 1e-4 * torch.eye(18, device=device, dtype=dtype).unsqueeze(0).expand(B, 18, 18)
        H_reg = Hsym + reg

        # 5. 并行求解 QP
        e = torch.Tensor().to(device=device, dtype=dtype)
        sol = self.qp_solver(H_reg, g, G, h, e, e)

        return sol

    def step_with_cmd(
        self,
        measured_rbd_state: torch.Tensor,
        input_desired: torch.Tensor,
        dt: float,
        cmd_vxyz_batch: torch.Tensor,
        cmd_yaw_rate_batch: torch.Tensor,
        mode: int = 0,
    ) -> torch.Tensor:
        self._t += float(dt)
        self._last_dt = float(dt)
        self.t_batch += dt
        # One-time gait initialization from measured state (fix t=0 discontinuity)
        if not self._gait_inited:
            self._init_gait_from_measured(measured_rbd_state, input_desired)
        self.cmd_vxyz_batch = cmd_vxyz_batch
        self.cmd_yaw_rate_batch = cmd_yaw_rate_batch

        self.plan = self.gait.update(
            t=self.t_batch,
            cmd_vxyz=self.cmd_vxyz_batch,
            cmd_yaw_rate=self.cmd_yaw_rate_batch,
            dt=dt
        )
        self.update_targets(self.plan.target_pos, self.plan.target_ori)

        sol = self.update(measured_rbd_state, input_desired, mode=mode)
        if self.verbose:
            try:
                self._log_series(self._t, sol, measured_rbd_state, input_desired)
            except Exception as e:
                pass # 防止日志报错中断控制循环

        state_desired = measured_rbd_state.clone()
        state_desired[:, 0:7] = integrate_freeflyer_quat_xyzw(
            measured_rbd_state[:, 0:7],
            sol[:, 0:3],
            sol[:, 3:6],
            right_multiply=True
        )
        # joints are 12 DoF -> indices 7..18 inclusive (slice 7:19)
        state_desired[:, 7:19] += sol[:, 6:18]

        return state_desired

if __name__ == "__main__":
    B_SIZE = 2000
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    
    info = CentroidalModelInfoSimple(generalizedCoordinatesNum=18, actuatedDofNum=12, numThreeDofContacts=4, robotMass=30.0)
    
    wbc = Wbc(task_file="",  info=info, batch_size=B_SIZE, device=dev, dtype=dtype)
    
    # 构造模拟输入 [2000, 19] 和 [2000, 18]
    m_state = torch.zeros((B_SIZE, 19), device=dev, dtype=dtype)
    m_state[:, 6] = 1.0 
    i_des = torch.zeros((B_SIZE, 18), device=dev, dtype=dtype)
    
    # 指令也必须是批处理维度的 [2000, 3] 和 [2000, 1]
    cmd_v = torch.zeros((B_SIZE, 3), device=dev, dtype=dtype)
    cmd_y = torch.zeros((B_SIZE, 1), device=dev, dtype=dtype)

    new_state = wbc.step_with_cmd(m_state, i_des, 0.01, cmd_v, cmd_y)
    print(f"Batch solving complete. Output shape: {new_state.shape}")