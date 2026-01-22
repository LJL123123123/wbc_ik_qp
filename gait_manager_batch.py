import torch
import math
from dataclasses import dataclass
from typing import Dict, Optional


class GaitPlan:
    """Batched gait plan.

    target_pos: Dict[str, Tensor]
        - "com", "LH", "LF", "RF", "RH" -> [B, 3]
    target_ori: Dict[str, Tensor]
        - "com" -> [B, 3, 3]
        - legs  -> [B, 3, 3] (identity)
    """

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
    # NOTE: match runwbc_batch.py default initial height (0.26)
    nominal_height: float = 0.26
    mode: int = 0
    wait_time: float = 0.5
    transition_time: float = 0.5


class GaitCycleManagerCuda:
    """GPU batched gait manager.

    This is a batched version of gait_manager_cuda.GaitCycleManagerCuda.
    It is designed to be called by wbc_batch.py:

        plan = gait.update(t=[B], cmd_vxyz=[B,3], cmd_yaw_rate=[B,1], dt=float)
    """

    def __init__(self, batch_size: int, device, dtype, params: Optional[GaitParamsCuda] = None):
        self.batch_size = int(batch_size)
        self.device = device
        self.dtype = dtype
        self.params = params if params is not None else GaitParamsCuda()

        self.leg_names = ["LH", "LF", "RF", "RH"]

        # --- batched internal states ---
        B = self.batch_size
        self.world_com_pos = torch.zeros((B, 3), device=device, dtype=dtype)
        self.world_com_pos[:, 2] = float(self.params.nominal_height)

        self.world_yaw = torch.zeros((B,), device=device, dtype=dtype)

        self.zero_vel_timer = torch.zeros((B,), device=device, dtype=dtype)
        self.stand_fraction = torch.ones((B,), device=device, dtype=dtype)

        self.last_valid_v = torch.zeros((B, 3), device=device, dtype=dtype)
        self.last_valid_yaw = torch.zeros((B,), device=device, dtype=dtype)
        self.smooth_v = torch.zeros((B, 3), device=device, dtype=dtype)
        self.smooth_yaw = torch.zeros((B,), device=device, dtype=dtype)

        # stance memory in world frame
        self.last_stance_pos: Dict[str, torch.Tensor] = {
            "LH": torch.tensor([-0.25, 0.15, 0.0], device=device, dtype=dtype).view(1, 3).repeat(B, 1),
            "LF": torch.tensor([0.15, 0.15, 0.0], device=device, dtype=dtype).view(1, 3).repeat(B, 1),
            "RF": torch.tensor([0.15, -0.15, 0.0], device=device, dtype=dtype).view(1, 3).repeat(B, 1),
            "RH": torch.tensor([-0.25, -0.15, 0.0], device=device, dtype=dtype).view(1, 3).repeat(B, 1),
        }

        # nominal offsets in trunk frame (constant)
        self.nominal_offsets: Dict[str, torch.Tensor] = {
            "LH": torch.tensor([-0.25, 0.15, 0.0], device=device, dtype=dtype),
            "LF": torch.tensor([0.15, 0.15, 0.0], device=device, dtype=dtype),
            "RF": torch.tensor([0.15, -0.15, 0.0], device=device, dtype=dtype),
            "RH": torch.tensor([-0.25, -0.15, 0.0], device=device, dtype=dtype),
        }

        self.current_gait_period = float(self.params.default_gait_period)

    # ----------------- public APIs -----------------
    @torch.no_grad()
    def reset_batch(self, batch_idx: int):
        """Reset internal states for one robot in the batch."""
        i = int(batch_idx)
        if i < 0 or i >= self.batch_size:
            return

        self.world_com_pos[i].zero_()
        self.world_com_pos[i, 2] = float(self.params.nominal_height)
        self.world_yaw[i] = 0.0

        self.zero_vel_timer[i] = 0.0
        self.stand_fraction[i] = 1.0
        self.last_valid_v[i].zero_()
        self.last_valid_yaw[i] = 0.0
        self.smooth_v[i].zero_()
        self.smooth_yaw[i] = 0.0

        # Reset stance positions around current com
        for name in self.leg_names:
            p = self.nominal_offsets[name].clone()
            p[0:2] += self.world_com_pos[i, 0:2]
            p[2] = 0.0
            self.last_stance_pos[name][i] = p

    @torch.no_grad()
    def set_stand_targets(self, target_pos: Dict[str, torch.Tensor], target_ori: Dict[str, torch.Tensor]):
        """(Optional) Initialize internal states from provided targets.

        Accepts either per-batch tensors ([B,*]) or single tensors ([*])
        and broadcasts them to batch.
        """

        com = target_pos["com"]
        if com.dim() == 1:
            com = com.view(1, 3).repeat(self.batch_size, 1)
        self.world_com_pos.copy_(com)

        R = target_ori["com"]
        if R.dim() == 2:
            R = R.view(1, 3, 3).repeat(self.batch_size, 1, 1)
        # yaw from rotation matrix
        self.world_yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])

        # reset stance positions consistent with current com + yaw
        rot = self._yaw_to_rot_mat(self.world_yaw)
        for name in self.leg_names:
            off = self.nominal_offsets[name].view(1, 3, 1).repeat(self.batch_size, 1, 1)
            p = torch.matmul(rot, off).squeeze(-1) + self.world_com_pos
            p[:, 2] = 0.0
            self.last_stance_pos[name].copy_(p)

    @torch.no_grad()
    def update(self, t: torch.Tensor, cmd_vxyz: torch.Tensor, cmd_yaw_rate: torch.Tensor, dt: float) -> GaitPlan:
        """Update gait plan.

        Args:
            t: [B] time for each instance
            cmd_vxyz: [B,3] command velocity in trunk/body frame (vx, vy, vz)
            cmd_yaw_rate: [B,1] yaw rate command (rad/s)
            dt: scalar
        """

        B = self.batch_size
        assert t.shape[0] == B, f"t must be [B], got {t.shape}"
        assert cmd_vxyz.shape[0] == B and cmd_vxyz.shape[1] == 3, f"cmd_vxyz must be [B,3], got {cmd_vxyz.shape}"
        assert cmd_yaw_rate.shape[0] == B, f"cmd_yaw_rate must be [B,1] or [B], got {cmd_yaw_rate.shape}"

        yaw_rate = cmd_yaw_rate.view(B).to(device=self.device, dtype=self.dtype)
        v_cmd = cmd_vxyz.to(device=self.device, dtype=self.dtype)
        t = t.to(device=self.device, dtype=self.dtype)

        # 1) velocity monitoring (per-batch)
        cmd_norm = torch.linalg.norm(v_cmd, dim=1) + yaw_rate.abs()
        zero_mask = cmd_norm < 1e-4

        # update timers
        self.zero_vel_timer = torch.where(
            zero_mask,
            self.zero_vel_timer + float(dt),
            torch.zeros_like(self.zero_vel_timer),
        )

        # update last valid command when not zero
        nz = ~zero_mask
        if nz.any():
            self.last_valid_v[nz] = v_cmd[nz]
            self.last_valid_yaw[nz] = yaw_rate[nz]

        # 2) state-machine transition (per-batch)
        moving_mask = self.zero_vel_timer < float(self.params.wait_time)
        use_last_mask = (self.zero_vel_timer > 0.0) & moving_mask

        target_stand_fraction = torch.zeros((B,), device=self.device, dtype=self.dtype)
        target_v = torch.zeros((B, 3), device=self.device, dtype=self.dtype)
        target_yaw = torch.zeros((B,), device=self.device, dtype=self.dtype)

        if moving_mask.any():
            m = moving_mask
            target_stand_fraction[m] = 0.0
            target_v[m] = torch.where(use_last_mask[m].unsqueeze(-1), self.last_valid_v[m], v_cmd[m])
            target_yaw[m] = torch.where(use_last_mask[m], self.last_valid_yaw[m], yaw_rate[m])

        standing_mask = ~moving_mask
        if standing_mask.any():
            prog = (self.zero_vel_timer[standing_mask] - float(self.params.wait_time)) / float(self.params.transition_time)
            target_stand_fraction[standing_mask] = torch.clamp(prog, 0.0, 1.0)
            # target_v/target_yaw remain zero

        # 3) smooth transition
        alpha = 0.1
        self.stand_fraction = (1.0 - alpha) * self.stand_fraction + alpha * target_stand_fraction
        self.smooth_v = (1.0 - alpha) * self.smooth_v + alpha * target_v
        self.smooth_yaw = (1.0 - alpha) * self.smooth_yaw + alpha * target_yaw

        # 4) integrate COM pose in world
        self.world_yaw = self.world_yaw + self.smooth_yaw * float(dt)
        rot_mat = self._yaw_to_rot_mat(self.world_yaw)  # [B,3,3]

        c = torch.cos(self.world_yaw)
        s = torch.sin(self.world_yaw)
        vx_world = c * self.smooth_v[:, 0] - s * self.smooth_v[:, 1]
        vy_world = s * self.smooth_v[:, 0] + c * self.smooth_v[:, 1]
        vz_world = self.smooth_v[:, 2]

        self.world_com_pos[:, 0] += vx_world * float(dt)
        self.world_com_pos[:, 1] += vy_world * float(dt)
        self.world_com_pos[:, 2] += vz_world * float(dt)

        new_target_pos: Dict[str, torch.Tensor] = {"com": self.world_com_pos.clone()}
        new_target_ori: Dict[str, torch.Tensor] = {"com": rot_mat.clone()}
        offsets = self._get_offsets(self.params.mode)  # [4]

        stand_f = self.stand_fraction.view(B, 1)

        # 5) foot trajectories
        for i_leg, name in enumerate(self.leg_names):
            off = self.nominal_offsets[name].view(1, 3, 1).repeat(B, 1, 1)
            nominal_world_pos = torch.matmul(rot_mat, off).squeeze(-1) + self.world_com_pos
            nominal_world_pos[:, 2] = 0.0

            leg_phase = torch.remainder((t / self.current_gait_period) + offsets[i_leg], 1.0)
            stance_mask = leg_phase < float(self.params.duty_factor)

            # stance
            gait_pos = self.last_stance_pos[name]
            pos_stance = (1.0 - stand_f) * gait_pos + stand_f * nominal_world_pos

            # swing
            swing_phase = (leg_phase - float(self.params.duty_factor)) / (1.0 - float(self.params.duty_factor))
            swing_phase = torch.clamp(swing_phase, 0.0, 1.0)
            p_start = gait_pos

            mid = float(self.current_gait_period * self.params.duty_factor / 2.0)
            delta = torch.stack([vx_world, vy_world, torch.zeros_like(vx_world)], dim=-1) * mid
            p_end = nominal_world_pos + delta

            gait_swing_pos = p_start + (p_end - p_start) * swing_phase.unsqueeze(-1)
            current_step_height = float(self.params.step_height) * (1.0 - self.stand_fraction)
            gait_swing_pos[:, 2] = self._bezier_height(swing_phase) * current_step_height

            pos_swing = (1.0 - stand_f) * gait_swing_pos + stand_f * nominal_world_pos

            # select stance/swing
            pos = torch.where(stance_mask.unsqueeze(-1), pos_stance, pos_swing)

            # update last stance positions near swing end
            # NOTE:
            # swing_phase advances by dt / (T*(1-duty)). With typical dt=0.01, T=0.5, duty=0.5,
            # swing_phase increments ~0.04, so a hard threshold like 0.98 can be skipped (0.96 -> wrap to 0.0),
            # causing last_stance_pos to miss the current landing and produce a large discontinuity at phase wrap.
            swing_inc = float(dt) / (self.current_gait_period * (1.0 - float(self.params.duty_factor)) + 1e-12)
            land_threshold = max(0.0, 1.0 - swing_inc - 1e-6)
            land_mask = (~stance_mask) & (swing_phase >= land_threshold)

            # If we're at the last swing sample, snap to the landing point to remove any pop.
            if land_mask.any():
                pos = torch.where(land_mask.unsqueeze(-1), p_end, pos)
                pos[:, 2] = torch.where(land_mask, torch.zeros_like(pos[:, 2]), pos[:, 2])
                self.last_stance_pos[name][land_mask] = p_end[land_mask]

            new_target_pos[name] = pos
            new_target_ori[name] = torch.eye(3, device=self.device, dtype=self.dtype).view(1, 3, 3).repeat(B, 1, 1)

        return GaitPlan(new_target_pos, new_target_ori)

    # ----------------- helpers -----------------
    def _bezier_height(self, phase: torch.Tensor) -> torch.Tensor:
        # phase: [B]
        return (
            (1 - phase) ** 3 * 0.0
            + 3 * (1 - phase) ** 2 * phase * 1.5
            + 3 * (1 - phase) * phase ** 2 * 1.5
            + phase ** 3 * 0.0
        )

    def _get_offsets(self, mode: int) -> torch.Tensor:
        if int(mode) == 0:
            return torch.tensor([0.5, 0.0, 0.5, 0.0], device=self.device, dtype=self.dtype)
        return torch.zeros((4,), device=self.device, dtype=self.dtype)

    def _yaw_to_rot_mat(self, yaw: torch.Tensor) -> torch.Tensor:
        """yaw: [B] -> rot_mat: [B,3,3]"""
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        B = yaw.shape[0]
        rot = torch.zeros((B, 3, 3), device=self.device, dtype=self.dtype)
        rot[:, 0, 0] = c
        rot[:, 0, 1] = -s
        rot[:, 1, 0] = s
        rot[:, 1, 1] = c
        rot[:, 2, 2] = 1.0
        return rot