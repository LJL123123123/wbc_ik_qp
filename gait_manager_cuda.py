import torch
import math
from dataclasses import dataclass
from typing import Dict

class GaitPlan:
    def __init__(self, target_pos: Dict[str, torch.Tensor], target_ori: Dict[str, torch.Tensor]):
        self.target_pos = target_pos
        self.target_ori = target_ori

@dataclass
class GaitParamsCuda:
    default_gait_period: float = 0.5
    min_gait_period: float = 0.25
    duty_factor: float = 0.5
    max_stride_x: float = 0.2
    max_stride_y: float = 0.15
    max_stride_yaw: float = 0.5
    step_height: float = 0.08
    nominal_height: float = 0.30
    mode: int = 0 
    wait_time: float = 0.5        
    transition_time: float = 0.5  

class GaitCycleManagerCuda:
    def __init__(self, device, dtype, params: GaitParamsCuda):
        self.device = device
        self.dtype = dtype
        self.params = params
        
        self.leg_names = ["LH", "LF", "RF", "RH"]
        self.world_com_pos = torch.zeros(3, device=device, dtype=dtype)
        self.world_yaw = 0.0
        
        self.zero_vel_timer = 0.0
        self.stand_fraction = 1.0  
        self.last_valid_v = torch.zeros(3, device=device, dtype=dtype)
        self.last_valid_yaw = 0.0
        self.smooth_v = torch.zeros(3, device=device, dtype=dtype)
        self.smooth_yaw = 0.0

        self.last_stance_pos = {
            "LH": torch.tensor([-0.25,  0.15, 0.0], device=device, dtype=dtype),
            "LF": torch.tensor([ 0.15,  0.15, 0.0], device=device, dtype=dtype),
            "RF": torch.tensor([ 0.15, -0.15, 0.0], device=device, dtype=dtype),
            "RH": torch.tensor([-0.25, -0.15, 0.0], device=device, dtype=dtype),
        }
        self.nominal_offsets = {
            "LH": torch.tensor([-0.25,  0.15, 0.0], device=device, dtype=dtype),
            "LF": torch.tensor([ 0.15,  0.15, 0.0], device=device, dtype=dtype),
            "RF": torch.tensor([ 0.15, -0.15, 0.0], device=device, dtype=dtype),
            "RH": torch.tensor([-0.25, -0.15, 0.0], device=device, dtype=dtype),
        }
        self.current_gait_period = params.default_gait_period

    def update(self, t, target_pos, target_ori, cmd_vxyz, cmd_yaw_rate, dt):
        # 1. 速度输入监控
        cmd_norm = torch.norm(cmd_vxyz) + abs(cmd_yaw_rate)
        if cmd_norm < 1e-4:
            self.zero_vel_timer += dt
        else:
            self.zero_vel_timer = 0.0
            self.last_valid_v = cmd_vxyz.clone()
            self.last_valid_yaw = cmd_yaw_rate

        # 2. 状态机切换逻辑
        if self.zero_vel_timer < self.params.wait_time:
            target_stand_fraction = 0.0
            target_v = self.last_valid_v if self.zero_vel_timer > 0 else cmd_vxyz
            target_yaw = self.last_valid_yaw if self.zero_vel_timer > 0 else cmd_yaw_rate
        else:
            transition_progress = (self.zero_vel_timer - self.params.wait_time) / self.params.transition_time
            target_stand_fraction = min(1.0, transition_progress)
            target_v = torch.zeros_like(cmd_vxyz)
            target_yaw = 0.0

        # 平滑过渡因子和速度
        alpha = 0.1
        self.stand_fraction = (1 - alpha) * self.stand_fraction + alpha * target_stand_fraction
        self.smooth_v = (1 - alpha) * self.smooth_v + alpha * target_v
        self.smooth_yaw = (1 - alpha) * self.smooth_yaw + alpha * target_yaw

        # 3. 核心修复：世界坐标系速度映射
        self.world_yaw += self.smooth_yaw * dt
        rot_mat = self._yaw_to_rot_mat(self.world_yaw)
        
        # 修正后的旋转变换：V_world = R * V_local
        # vx_world = cos(yaw)*vx - sin(yaw)*vy
        # vy_world = sin(yaw)*vx + cos(yaw)*vy
        vx_world = rot_mat[0, 0] * self.smooth_v[0] + rot_mat[0, 1] * self.smooth_v[1]
        vy_world = rot_mat[1, 0] * self.smooth_v[0] + rot_mat[1, 1] * self.smooth_v[1]
        
        self.world_com_pos[0] += vx_world * dt
        self.world_com_pos[1] += vy_world * dt
        self.world_com_pos[2] = self.params.nominal_height

        new_target_pos = {"com": self.world_com_pos.clone()}
        new_target_ori = {"com": rot_mat}
        offsets = self._get_offsets(self.params.mode)

        # 4. 足端轨迹生成（保持平滑回归逻辑）
        for i, name in enumerate(self.leg_names):
            nominal_world_pos = rot_mat @ self.nominal_offsets[name] + self.world_com_pos
            nominal_world_pos[2] = 0.0
            leg_phase = ((t / self.current_gait_period) + offsets[i]) % 1.0
            
            if leg_phase < self.params.duty_factor:
                # 支撑相混合
                gait_pos = self.last_stance_pos[name].clone()
                pos = (1 - self.stand_fraction) * gait_pos + self.stand_fraction * nominal_world_pos
            else:
                # 摆动相混合
                swing_phase = (leg_phase - self.params.duty_factor) / (1.0 - self.params.duty_factor)
                p_start = self.last_stance_pos[name].clone()
                # 落脚点预测同样需要使用修正后的世界速度
                p_end = nominal_world_pos + (self.current_gait_period * self.params.duty_factor / 2.0) * \
                        torch.tensor([vx_world, vy_world, 0.0], device=self.device, dtype=self.dtype)
                
                gait_swing_pos = p_start + (p_end - p_start) * swing_phase
                current_step_height = self.params.step_height * (1.0 - self.stand_fraction)
                gait_swing_pos[2] = self._bezier_height(swing_phase) * current_step_height
                
                pos = (1 - self.stand_fraction) * gait_swing_pos + self.stand_fraction * nominal_world_pos
                
                if swing_phase > 0.98:
                    self.last_stance_pos[name] = p_end.clone()

            new_target_pos[name] = pos
            new_target_ori[name] = torch.eye(3, device=self.device, dtype=self.dtype)

        return GaitPlan(new_target_pos, new_target_ori)

    def _bezier_height(self, phase):
        return (1-phase)**3 * 0.0 + 3*(1-phase)**2 * phase * 1.5 + 3*(1-phase) * phase**2 * 1.5 + phase**3 * 0.0

    def _get_offsets(self, mode):
        if mode == 0: return torch.tensor([0.5, 0.0, 0.5, 0.0], device=self.device)
        return torch.zeros(4, device=self.device)

    def _yaw_to_rot_mat(self, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        # 标准右手系旋转矩阵：[[cos, -sin], [sin, cos]]
        return torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], device=self.device, dtype=self.dtype)