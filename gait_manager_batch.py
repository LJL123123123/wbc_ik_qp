import torch
from dataclasses import dataclass
from typing import Dict, Optional


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
    nominal_height: float = 0.26
    mode: int = 0
    wait_time: float = 0.5
    transition_time: float = 0.5


class GaitCycleManagerCuda:
    """GPU batched gait manager.

    plan = gait.update(t=[B], cmd_vxyz=[B,3], cmd_yaw_rate=[B,1], dt=float)
    """

    def __init__(self, batch_size: int, device, dtype, params: Optional[GaitParamsCuda] = None):
        self.batch_size = int(batch_size)
        self.device = device
        self.dtype = dtype
        self.params = params if params is not None else GaitParamsCuda()

        self.leg_names = ["LH", "LF", "RF", "RH"]
        self._n_legs = 4
        B = self.batch_size

        # internal states
        self.world_com_pos = torch.zeros((B, 3), device=device, dtype=dtype)
        self.world_com_pos[:, 2] = float(self.params.nominal_height)
        self.world_yaw = torch.zeros((B,), device=device, dtype=dtype)

        self.zero_vel_timer = torch.zeros((B,), device=device, dtype=dtype)
        self.stand_fraction = torch.ones((B,), device=device, dtype=dtype)

        self.last_valid_v = torch.zeros((B, 3), device=device, dtype=dtype)
        self.last_valid_yaw = torch.zeros((B,), device=device, dtype=dtype)
        self.smooth_v = torch.zeros((B, 3), device=device, dtype=dtype)
        self.smooth_yaw = torch.zeros((B,), device=device, dtype=dtype)

        # nominal offsets in trunk frame (constant)
        nominal = {
            "LH": [-0.25, 0.15, 0.0],
            "LF": [0.15, 0.15, 0.0],
            "RF": [0.15, -0.15, 0.0],
            "RH": [-0.25, -0.15, 0.0],
        }
        self.nominal_offsets = {k: torch.tensor(v, device=device, dtype=dtype) for k, v in nominal.items()}
        self._nominal_offsets_stack = torch.stack([self.nominal_offsets[n] for n in self.leg_names], dim=0)  # [4,3]

        # stance memory in world frame
        self.last_stance_pos: Dict[str, torch.Tensor] = {}
        for name in self.leg_names:
            base = self.nominal_offsets[name].view(1, 3).expand(B, 3).clone()
            base[:, 2] = 0.0
            self.last_stance_pos[name] = base

        # gait phase offsets (trot)
        self._offsets_mode0 = torch.tensor([0.5, 0.0, 0.5, 0.0], device=device, dtype=dtype)
        self._offsets_zero = torch.zeros((4,), device=device, dtype=dtype)

        # reusable buffers
        self._eye3 = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3)       # [1,3,3]
        self._rot_buf = torch.empty((B, 3, 3), device=device, dtype=dtype)        # [B,3,3]

        self.current_gait_period = float(self.params.default_gait_period)

    @torch.no_grad()
    def reset_batch(self, batch_idx: int):
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

        # reset stance around current com
        for name in self.leg_names:
            p = self.nominal_offsets[name].clone()
            p[0:2] += self.world_com_pos[i, 0:2]
            p[2] = 0.0
            self.last_stance_pos[name][i].copy_(p)

    @torch.no_grad()
    def set_stand_targets(self, target_pos: Dict[str, torch.Tensor], target_ori: Dict[str, torch.Tensor]):
        """Initialize internal states from provided targets (broadcastable)."""
        B = self.batch_size

        com = target_pos["com"]
        if com.dim() == 1:
            com = com.view(1, 3).expand(B, 3)
        self.world_com_pos.copy_(com.to(self.device, self.dtype))

        R = target_ori["com"]
        if R.dim() == 2:
            R = R.view(1, 3, 3).expand(B, 3, 3)
        R = R.to(self.device, self.dtype)
        self.world_yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])

        rot = self._yaw_to_rot_mat_inplace(self.world_yaw)

        for i_leg, name in enumerate(self.leg_names):
            if name in target_pos:
                p = target_pos[name]
                if p.dim() == 1:
                    p = p.view(1, 3).expand(B, 3)
                p = p.to(self.device, self.dtype).clone()
                p[:, 2] = 0.0
                self.last_stance_pos[name].copy_(p)
            else:
                off = self._nominal_offsets_stack[i_leg].view(1, 3, 1).expand(B, 3, 1)
                p = torch.bmm(rot, off).squeeze(-1) + self.world_com_pos
                p = p.clone()
                p[:, 2] = 0.0
                self.last_stance_pos[name].copy_(p)

    @torch.no_grad()
    def update(self, t: torch.Tensor, cmd_vxyz: torch.Tensor, cmd_yaw_rate: torch.Tensor, dt: float) -> GaitPlan:
        B = self.batch_size
        dt_f = float(dt)
        duty = float(self.params.duty_factor)

        t = t.to(self.device, self.dtype).view(B)
        v_cmd = cmd_vxyz.to(self.device, self.dtype).view(B, 3)
        yaw_rate = cmd_yaw_rate.to(self.device, self.dtype).view(B)

        # ---- (1) zero-vel detection (no Python if / no .any sync) ----
        cmd_norm = torch.linalg.norm(v_cmd, dim=1) + yaw_rate.abs()
        zero_mask = cmd_norm < 1e-4
        nz = ~zero_mask

        # timer: timer = (timer + dt) if zero else 0   (in-place, no alloc)
        self.zero_vel_timer.add_(dt_f)
        self.zero_vel_timer.mul_(zero_mask)

        # last valid cmd (mask update; empty mask is OK, no .any needed)
        self.last_valid_v[nz] = v_cmd[nz]
        self.last_valid_yaw[nz] = yaw_rate[nz]

        # ---- (2) state machine (fully tensorized) ----
        wait_t = float(self.params.wait_time)
        trans_t = float(self.params.transition_time)

        moving_mask = self.zero_vel_timer < wait_t
        use_last_mask = (self.zero_vel_timer > 0.0) & moving_mask

        # stand fraction ramps when standing
        prog = (self.zero_vel_timer - wait_t) / (trans_t + 1e-12)
        target_stand_fraction = torch.clamp(prog, 0.0, 1.0)
        target_stand_fraction.mul_(~moving_mask)  # moving => 0

        # target v/yaw when moving
        target_v = torch.where(use_last_mask.unsqueeze(-1), self.last_valid_v, v_cmd)
        target_v.mul_(moving_mask.unsqueeze(-1))
        target_yaw = torch.where(use_last_mask, self.last_valid_yaw, yaw_rate)
        target_yaw.mul_(moving_mask)

        # ---- (3) smoothing ----
        alpha = 0.1
        self.stand_fraction.mul_(1.0 - alpha).add_(alpha * target_stand_fraction)
        self.smooth_v.mul_(1.0 - alpha).add_(alpha * target_v)
        self.smooth_yaw.mul_(1.0 - alpha).add_(alpha * target_yaw)

        # ---- (4) integrate COM pose (world) ----
        self.world_yaw.add_(self.smooth_yaw * dt_f)
        rot_mat = self._yaw_to_rot_mat_inplace(self.world_yaw)

        c = torch.cos(self.world_yaw)
        s = torch.sin(self.world_yaw)
        vx_world = c * self.smooth_v[:, 0] - s * self.smooth_v[:, 1]
        vy_world = s * self.smooth_v[:, 0] + c * self.smooth_v[:, 1]
        vz_world = self.smooth_v[:, 2]

        self.world_com_pos[:, 0].add_(vx_world * dt_f)
        self.world_com_pos[:, 1].add_(vy_world * dt_f)
        self.world_com_pos[:, 2].add_(vz_world * dt_f)

        new_target_pos: Dict[str, torch.Tensor] = {"com": self.world_com_pos}
        new_target_ori: Dict[str, torch.Tensor] = {"com": rot_mat}

        offsets = self._get_offsets(self.params.mode)  # [4]
        stand_f = self.stand_fraction.view(B, 1)

        # dt-adaptive landing threshold (precompute once)
        swing_inc = dt_f / (self.current_gait_period * (1.0 - duty) + 1e-12)
        land_threshold = max(0.0, 1.0 - swing_inc - 1e-6)

        mid = float(self.current_gait_period * duty / 2.0)
        delta = torch.stack([vx_world, vy_world, torch.zeros_like(vx_world)], dim=-1) * mid  # [B,3]

        # ---- (5) foot trajectories ----
        for i_leg, name in enumerate(self.leg_names):
            off = self._nominal_offsets_stack[i_leg].view(1, 3, 1).expand(B, 3, 1)  # no alloc
            nominal_world_pos = torch.bmm(rot_mat, off).squeeze(-1) + self.world_com_pos
            nominal_world_pos = nominal_world_pos.clone()
            nominal_world_pos[:, 2] = 0.0

            leg_phase = torch.remainder((t / self.current_gait_period) + offsets[i_leg], 1.0)
            stance_mask = leg_phase < duty

            gait_pos = self.last_stance_pos[name]

            # stance interpolation
            pos_stance = (1.0 - stand_f) * gait_pos + stand_f * nominal_world_pos

            # swing interpolation
            swing_phase = (leg_phase - duty) / (1.0 - duty + 1e-12)
            swing_phase = torch.clamp(swing_phase, 0.0, 1.0)

            p_start = gait_pos
            p_end = nominal_world_pos + delta

            gait_swing_pos = p_start + (p_end - p_start) * swing_phase.unsqueeze(-1)
            current_step_height = float(self.params.step_height) * (1.0 - self.stand_fraction)
            gait_swing_pos[:, 2] = self._bezier_height(swing_phase) * current_step_height

            pos_swing = (1.0 - stand_f) * gait_swing_pos + stand_f * nominal_world_pos

            pos = torch.where(stance_mask.unsqueeze(-1), pos_stance, pos_swing)

            # landing update WITHOUT .any() sync
            land_mask = (~stance_mask) & (swing_phase >= land_threshold)
            pos = torch.where(land_mask.unsqueeze(-1), p_end, pos)
            pos[:, 2] = torch.where(land_mask, torch.zeros_like(pos[:, 2]), pos[:, 2])

            # update stance memory (no boolean indexing assignment)
            updated = torch.where(land_mask.unsqueeze(-1), p_end, self.last_stance_pos[name])
            self.last_stance_pos[name].copy_(updated)

            new_target_pos[name] = pos
            new_target_ori[name] = self._eye3.expand(B, 3, 3)

        return GaitPlan(new_target_pos, new_target_ori)

    # ---------------- helpers ----------------
    def _bezier_height(self, phase: torch.Tensor) -> torch.Tensor:
        return (
            (1 - phase) ** 3 * 0.0
            + 3 * (1 - phase) ** 2 * phase * 1.5
            + 3 * (1 - phase) * phase**2 * 1.5
            + phase**3 * 0.0
        )

    def _get_offsets(self, mode: int) -> torch.Tensor:
        return self._offsets_mode0 if int(mode) == 0 else self._offsets_zero

    def _yaw_to_rot_mat_inplace(self, yaw: torch.Tensor) -> torch.Tensor:
        """yaw: [B] -> rot_mat: [B,3,3] (writes into self._rot_buf)"""
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        rot = self._rot_buf
        rot.zero_()
        rot[:, 0, 0] = c
        rot[:, 0, 1] = -s
        rot[:, 1, 0] = s
        rot[:, 1, 1] = c
        rot[:, 2, 2] = 1.0
        return rot
