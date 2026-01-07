"""gait_manager_cuda.py

CUDA-friendly gait cycle manager for this repo.

Design goals (matches user request):
- Accept velocity commands (vx, vy, vz, yaw_rate) and generate target COM + feet.
- When no command:
  - Keep current targets (robot holds steady).
  - If no command for >2s: exit walking cycle and smoothly transition to stand.
- When command present:
  - If not in cycle: initialize and enter a cycle.
  - If in cycle: manage trot gait and generate swing trajectories.

Notes:
- This is a kinematic target generator. It does not require Pinocchio.
- All math is torch-based and runs on self.device (CUDA when available).
- Internally uses a small amount of Python control flow (per-leg state machine),
  but all vector math stays on GPU tensors.

Expected target dict keys: 'com', 'LF', 'RF', 'LH', 'RH'
Expected target_ori['com']: (3,3) rotation matrix
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import torch


PAIR1 = ['LF', 'RH']
PAIR2 = ['RF', 'LH']
ALL_LEGS = ['LF', 'RF', 'LH', 'RH']


@dataclass
class GaitParamsCuda:
    cycle_period: float = 1.0
    swing_height: float = 0.06
    lookahead: float = 1.0
    cmd_epsilon: float = 1e-4
    cmd_timeout: float = 2.0
    stand_transition_duration: float = 0.4


@dataclass
class GaitPlan:
    in_cycle: bool
    stance_legs: List[str]
    swing_legs: List[str]
    target_pos: Dict[str, torch.Tensor]
    target_ori: Dict[str, torch.Tensor]


class GaitCycleManagerCuda:
    def __init__(self, device=None, dtype=torch.float64, params: Optional[GaitParamsCuda] = None):
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        self.params = params or GaitParamsCuda()

        self.in_cycle = False
        self._t0 = 0.0
        self._last_cmd_wall: Optional[float] = None

        # Stand targets set on first call (or explicit call)
        self._stand_pos: Optional[Dict[str, torch.Tensor]] = None
        self._stand_ori: Optional[Dict[str, torch.Tensor]] = None

        # Persistent foot state in world
        self._feet_world: Dict[str, torch.Tensor] = {}
        self._swing_active = {k: False for k in ALL_LEGS}
        self._swing_start: Dict[str, torch.Tensor] = {}
        self._swing_goal: Dict[str, torch.Tensor] = {}

        # Stand transition
        self._stand_trans_active = False
        self._stand_trans_t0 = 0.0
        self._stand_trans_p0: Optional[Dict[str, torch.Tensor]] = None
        # Two transition modes:
        # - "normal": feet start from world contact points and immediately align
        #             to COM-relative stand offsets (old behavior).
        # - "timeout": when exiting a gait due to cmd timeout, start from the
        #              CURRENT foot targets (may be mid-swing) and smoothly
        #              interpolate feet XY to COM-relative stand offsets.
        self._stand_trans_mode: str = "normal"

        # When transitioning to stand after command timeout, we don't want to
        # pull COM x/y back to the original stand pose, and we don't want to
        # pull yaw back either. We freeze the reference x/y/yaw when the
        # transition starts.
        self._freeze_com_xy: Optional[torch.Tensor] = None  # shape (2,)
        self._freeze_yaw: Optional[torch.Tensor] = None     # scalar

        # Track previous half-cycle to make phase boundary handling explicit.
        # This is critical for world-fixed stance + continuous touchdown.
        self._prev_first_half = None  # type: Optional[bool]

    def set_stand_targets(self, target_pos: Dict[str, torch.Tensor], target_ori: Dict[str, torch.Tensor]):
        self._stand_pos = {k: v.to(self.device, self.dtype).clone() for k, v in target_pos.items()}
        self._stand_ori = {k: v.to(self.device, self.dtype).clone() for k, v in target_ori.items()}

        # Cache nominal feet COM-frame offsets at stand (x/y only).
        # These offsets are used when transitioning back to stand so feet return
        # to a COM-relative configuration rather than the world origin.
        if 'com' in self._stand_pos:
            com_xy = self._stand_pos['com'][0:2]
            self._stand_foot_offset_xy = {
                leg: (self._stand_pos[leg][0:2] - com_xy).clone()
                for leg in ALL_LEGS
            }

    def _now(self) -> float:
        return time.time()

    def _cmd_active(self, vxyz: torch.Tensor, yaw_rate: torch.Tensor) -> bool:
        # vxyz: (3,), yaw_rate: scalar
        eps = self.params.cmd_epsilon
        return (torch.linalg.norm(vxyz).item() > eps) or (abs(float(yaw_rate)) > eps)

    def _cmd_walk_active(self, vxyz: torch.Tensor, yaw_rate: torch.Tensor) -> bool:
        """Whether the command should trigger/maintain a walking gait.

        Height-only commands (non-zero z with ~zero xy and yaw) should NOT enter
        a walking cycle.
        """
        eps = self.params.cmd_epsilon
        vxy = torch.linalg.norm(vxyz[0:2]).item()
        return (vxy > eps) or (abs(float(yaw_rate)) > eps)

    def _cubic(self, p0: torch.Tensor, pf: torch.Tensor, s: float) -> torch.Tensor:
        s = float(max(0.0, min(1.0, s)))
        a = 3.0 * s * s - 2.0 * s * s * s
        return p0 + (pf - p0) * a

    def _swing_traj(self, p0: torch.Tensor, pf: torch.Tensor, s: float) -> torch.Tensor:
        p = self._cubic(p0, pf, s)
        z_bump = 4.0 * self.params.swing_height * float(s) * (1.0 - float(s))
        p2 = p.clone()
        p2[2] = p2[2] + z_bump
        return p2

    def update(
        self,
        t: float,
        target_pos: Dict[str, torch.Tensor],
        target_ori: Dict[str, torch.Tensor],
        cmd_vxyz: torch.Tensor,
        cmd_yaw_rate: torch.Tensor,
        dt: float,
    ) -> GaitPlan:
        """Update and return planned targets.

        Inputs:
        - t: controller time (seconds)
        - target_pos/target_ori: current targets (will not be mutated)
        - cmd_vxyz: desired body-frame velocity (vx,vy,vz)
        - cmd_yaw_rate: desired yaw rate (omega_z)
        - dt: controller step
        """

        # move cmd onto device
        cmd_vxyz = cmd_vxyz.to(device=self.device, dtype=self.dtype).reshape(3)
        cmd_yaw_rate = cmd_yaw_rate.to(device=self.device, dtype=self.dtype).reshape(())

        # initialize stand targets once
        if self._stand_pos is None:
            self.set_stand_targets(target_pos, target_ori)

        active = self._cmd_active(cmd_vxyz, cmd_yaw_rate)
        walk_active = self._cmd_walk_active(cmd_vxyz, cmd_yaw_rate)
        # yaw-only spin in place (no commanded vxy). Used to keep stance feet
        # fixed in world while swing feet stay approximately fixed in body frame.
        eps = self.params.cmd_epsilon
        spin_in_place = (torch.linalg.norm(cmd_vxyz[0:2]).item() <= eps) and (abs(float(cmd_yaw_rate)) > eps)
        now = self._now()
        if active:
            self._last_cmd_wall = now

        # timeout -> exit cycle
        # IMPORTANT: do not abruptly flip to stand targets, otherwise feet/com
        # targets can jump (stance feet are world-fixed in _feet_world).
        # Instead, trigger a stand transition starting from the LAST planned
        # world contact points.
        if (not active) and (self._last_cmd_wall is not None):
            if now - self._last_cmd_wall > self.params.cmd_timeout:
                if self.in_cycle:
                    # Start stand transition from a continuous pose.
                    if not self._stand_trans_active:
                        self._stand_trans_active = True
                        self._stand_trans_t0 = t
                        # timeout exit: start from CURRENT targets (may be mid-swing)
                        self._stand_trans_mode = "timeout"
                        p0 = {k: target_pos[k].to(self.device, self.dtype).clone() for k in ['com'] + ALL_LEGS}
                        self._stand_trans_p0 = p0

                        com0 = p0['com']
                        self._freeze_com_xy = com0[0:2].clone()
                        R0 = target_ori['com'].to(self.device, self.dtype)
                        yaw0 = torch.atan2(R0[1, 0], R0[0, 0])
                        self._freeze_yaw = yaw0.clone()

                    # Exit cycle; subsequent not-active branch will execute stand transition.
                    self.in_cycle = False
                else:
                    self.in_cycle = False

        # enter cycle if needed (walking-triggered only)
        if walk_active and (not self.in_cycle):
            self.in_cycle = True
            # Start phase from 0 at the moment we enter the cycle.
            # Using (t - dt) makes the first update land at a small positive phase
            # instead of potentially skipping the very beginning due to caller timing.
            self._t0 = float(t) - float(dt)
            # capture current COM-frame foot offsets (x/y) so stance feet can be
            # planned as COM-relative points that rotate with yaw.
            com_xy0 = target_pos['com'].to(self.device, self.dtype)[0:2].clone()
            self._cycle_foot_offset_xy = {
                leg: (target_pos[leg].to(self.device, self.dtype)[0:2].clone() - com_xy0)
                for leg in ALL_LEGS
            }
            # init feet world from current targets
            for leg in ALL_LEGS:
                self._feet_world[leg] = target_pos[leg].to(self.device, self.dtype).clone()
                self._swing_active[leg] = False
                self._swing_start[leg] = self._feet_world[leg].clone()
                self._swing_goal[leg] = self._feet_world[leg].clone()
            # cancel any stand transition
            self._stand_trans_active = False
            self._prev_first_half = None

        # Height-only command: no walking. Keep all feet as stance and only move COM z.
        if active and (not walk_active):
            new_pos = {k: v.to(self.device, self.dtype).clone() for k, v in target_pos.items()}
            new_ori = {k: v.to(self.device, self.dtype).clone() for k, v in target_ori.items()}

            com_next = new_pos['com'].clone()
            com_next[2] = com_next[2] + cmd_vxyz[2] * float(dt)
            z_hi = float(self._stand_pos['com'][2].item()) if self._stand_pos is not None else 0.26
            z_lo = 0.0
            com_next[2] = torch.clamp(com_next[2], min=z_lo, max=z_hi)
            new_pos['com'] = com_next

            # ensure no gait cycle
            self.in_cycle = False
            self._stand_trans_active = False
            self._freeze_com_xy = None
            self._freeze_yaw = None

            return GaitPlan(
                in_cycle=self.in_cycle,
                stance_legs=ALL_LEGS,
                swing_legs=[],
                target_pos=new_pos,
                target_ori=new_ori,
            )

        # no command: hold pose (do not inject drift)
        if not active:
            if self.in_cycle:
                # in-cycle but cmd absent: keep current targets steady (hold)
                return GaitPlan(
                    in_cycle=self.in_cycle,
                    stance_legs=ALL_LEGS,
                    swing_legs=[],
                    target_pos={k: v.to(self.device, self.dtype).clone() for k, v in target_pos.items()},
                    target_ori={k: v.to(self.device, self.dtype).clone() for k, v in target_ori.items()},
                )

            # not in cycle: smoothly go back to stand
            if not self._stand_trans_active:
                self._stand_trans_active = True
                self._stand_trans_t0 = t
                self._stand_trans_p0 = {k: target_pos[k].to(self.device, self.dtype).clone() for k in ['com'] + ALL_LEGS}
                self._stand_trans_mode = "normal"
                # Ensure feet start from world-fixed contact points if available.
                for leg in ALL_LEGS:
                    if leg in self._feet_world:
                        self._stand_trans_p0[leg] = self._feet_world[leg].clone()

                # freeze COM x/y and current yaw at the moment we start the stand transition
                com0 = self._stand_trans_p0['com']
                self._freeze_com_xy = com0[0:2].clone()
                # compute yaw from current com rotation matrix
                R0 = target_ori['com'].to(self.device, self.dtype)
                yaw0 = torch.atan2(R0[1, 0], R0[0, 0])
                self._freeze_yaw = yaw0.clone()

            s = (t - self._stand_trans_t0) / max(1e-6, self.params.stand_transition_duration)
            s = max(0.0, min(1.0, s))
            new_pos = dict(target_pos)
            for k in ['com'] + ALL_LEGS:
                new_pos[k] = self._cubic(self._stand_trans_p0[k], self._stand_pos[k], s)

            # Override COM behavior during stand transition:
            # - keep x/y frozen
            # - keep z as-is (maintain height-control result)
            com = new_pos['com'].clone()
            if self._freeze_com_xy is not None:
                com[0:2] = self._freeze_com_xy
            # keep current z (do not pull back to stand height)
            com[2] = self._stand_trans_p0['com'][2]
            new_pos['com'] = com

            # Override feet behavior during stand transition.
            # We always want to end at the COM-relative stand offsets with frozen yaw,
            # but:
            # - normal mode: immediately enforce COM-relative XY (old behavior)
            # - timeout mode: interpolate XY from current feet to those offsets
            if self._freeze_com_xy is not None:
                yaw = self._freeze_yaw
                if yaw is None:
                    yaw = torch.zeros((), device=self.device, dtype=self.dtype)
                cy = torch.cos(yaw)
                sy = torch.sin(yaw)
                Rz2 = torch.stack([
                    torch.stack([cy, -sy]),
                    torch.stack([sy,  cy]),
                ])  # (2,2)
                for leg in ALL_LEGS:
                    # fall back to stand absolute pose if offsets missing
                    if not hasattr(self, '_stand_foot_offset_xy'):
                        continue
                    if leg not in self._stand_foot_offset_xy:
                        continue
                    off_xy = self._stand_foot_offset_xy[leg]
                    leg_xy = self._freeze_com_xy + (Rz2 @ off_xy)
                    if getattr(self, '_stand_trans_mode', 'normal') == 'timeout':
                        # smooth XY: interpolate from p0.xy to desired COM-relative xy
                        leg_p = new_pos[leg].clone()
                        p0_xy = self._stand_trans_p0[leg][0:2]
                        leg_p[0:2] = self._cubic(p0_xy, leg_xy, s)
                        new_pos[leg] = leg_p
                    else:
                        # normal: snap XY to desired COM-relative xy
                        leg_p = new_pos[leg].clone()
                        leg_p[0:2] = leg_xy
                        new_pos[leg] = leg_p

            new_ori = dict(target_ori)

            # Orientation behavior during stand transition:
            # - roll/pitch -> 0
            # - yaw kept frozen
            # Build R = Rz(yaw) * Ry(0) * Rx(0) = Rz(yaw)
            yaw = self._freeze_yaw
            if yaw is None:
                yaw = torch.zeros((), device=self.device, dtype=self.dtype)
            cy = torch.cos(yaw)
            sy = torch.sin(yaw)
            new_ori['com'] = torch.stack(
                [
                    torch.stack([cy, -sy, torch.zeros((), device=self.device, dtype=self.dtype)]),
                    torch.stack([sy,  cy, torch.zeros((), device=self.device, dtype=self.dtype)]),
                    torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=self.dtype),
                ]
            )

            if s >= 1.0:
                self._stand_trans_active = False
                self._freeze_com_xy = None
                self._freeze_yaw = None
                self._stand_trans_mode = "normal"

            return GaitPlan(
                in_cycle=self.in_cycle,
                stance_legs=ALL_LEGS,
                swing_legs=[],
                target_pos=new_pos,
                target_ori=new_ori,
            )

        # active command & in cycle: integrate COM, yaw, and trot feet
        # integrate COM in world: v_world = R_com @ v_body
        Rcom = target_ori['com'].to(self.device, self.dtype)
        v_world = Rcom @ cmd_vxyz

        new_pos = {k: v.to(self.device, self.dtype).clone() for k, v in target_pos.items()}
        new_ori = {k: v.to(self.device, self.dtype).clone() for k, v in target_ori.items()}

        # COM integration: use x/y from rotated command, and z from cmd_vxyz[2]
        # so height control is decoupled from body yaw (z is world-up).
        com_next = new_pos['com'] + v_world * float(dt)
        com_next[2] = new_pos['com'][2] + cmd_vxyz[2] * float(dt)

        # Clamp height to stand limits (use stand com z as upper bound).
        z_hi = float(self._stand_pos['com'][2].item()) if self._stand_pos is not None else 0.26
        z_lo = 0.0
        com_next[2] = torch.clamp(com_next[2], min=z_lo, max=z_hi)
        new_pos['com'] = com_next

        # yaw integration around z axis
        omega = float(cmd_yaw_rate)
        if abs(omega) > 1e-12:
            ang = omega * float(dt)
            c = torch.cos(torch.tensor(ang, device=self.device, dtype=self.dtype))
            s = torch.sin(torch.tensor(ang, device=self.device, dtype=self.dtype))
            Rz = torch.stack([
                torch.stack([c, -s, torch.zeros((), device=self.device, dtype=self.dtype)]),
                torch.stack([s,  c, torch.zeros((), device=self.device, dtype=self.dtype)]),
                torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=self.dtype),
            ])
            new_ori['com'] = Rz @ new_ori['com']

        # Extract current yaw from planned COM orientation (after yaw integration)
        yaw = torch.atan2(new_ori['com'][1, 0], new_ori['com'][0, 0])
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)
        Rz2 = torch.stack([
            torch.stack([cy, -sy]),
            torch.stack([sy,  cy]),
        ])  # (2,2)

        # helper: get desired COM-relative XY in world from rotated offset
        def com_relative_xy_for_leg(leg: str) -> torch.Tensor:
            if hasattr(self, '_cycle_foot_offset_xy') and (leg in self._cycle_foot_offset_xy):
                off = self._cycle_foot_offset_xy[leg]
            elif hasattr(self, '_stand_foot_offset_xy') and (leg in self._stand_foot_offset_xy):
                off = self._stand_foot_offset_xy[leg]
            else:
                return self._stand_pos[leg][0:2].clone()
            return com_next[0:2] + (Rz2 @ off)

        # gait phase
        phase = ((t - self._t0) % self.params.cycle_period) / self.params.cycle_period
        first_half = phase < 0.5
        swing_legs = PAIR1 if first_half else PAIR2
        stance_legs = [l for l in ALL_LEGS if l not in swing_legs]
        s_phase = (phase / 0.5) if first_half else ((phase - 0.5) / 0.5)
        s_phase = max(0.0, min(1.0, float(s_phase)))

        # Detect half-cycle switch. At the exact boundary, the previous swing legs
        # must be forced to touchdown (world contact point becomes their swing_goal).
        # Otherwise, stance logic will immediately snap the foot back to the previous
        # _feet_world and create a jump.
        half_changed = (self._prev_first_half is not None) and (first_half != self._prev_first_half)
        if half_changed:
            prev_swing = PAIR1 if self._prev_first_half else PAIR2
            for leg in prev_swing:
                pf = self._swing_goal[leg].clone() if leg in self._swing_goal else self._feet_world[leg].clone()
                self._feet_world[leg] = pf.clone()
                new_pos[leg] = pf.clone()
                self._swing_active[leg] = False
                self._swing_start[leg] = pf.clone()
                self._swing_goal[leg] = pf.clone()
                if hasattr(self, '_spin_swing_off_b') and (leg in getattr(self, '_spin_swing_off_b')):
                    # Clear cached offset so next lift-off re-freezes correctly.
                    del self._spin_swing_off_b[leg]

            # Also ensure the other pair (new stance legs) uses world-fixed feet.
            prev_stance = [l for l in ALL_LEGS if l not in prev_swing]
            for leg in prev_stance:
                new_pos[leg] = self._feet_world[leg].clone()

        self._prev_first_half = first_half

        # update swing start & goal when entering swing
        lookahead = float(self.params.lookahead)
        vxy_world = torch.tensor([v_world[0], v_world[1], 0.0], device=self.device, dtype=self.dtype)
        for leg in swing_legs:
            if not self._swing_active[leg]:
                self._swing_active[leg] = True
                # Start swing from the last touchdown world point.
                # This guarantees continuity at phase boundaries even if
                # the caller's target_pos/new_pos has drifted.
                p0 = self._feet_world[leg].clone()
                self._swing_start[leg] = p0

                # For spin-in-place, freeze the foot's body-frame XY offset at lift-off.
                # This guarantees the commanded target point is body-fixed even while
                # yaw continues to integrate during the swing.
                if spin_in_place:
                    # off_world = p0_xy - com_xy
                    off_w = (p0[0:2] - com_next[0:2])
                    # off_body = Rz(-yaw) * off_world
                    off_b = torch.stack([
                        cy * off_w[0] + sy * off_w[1],
                        -sy * off_w[0] + cy * off_w[1],
                    ])
                    if not hasattr(self, '_spin_swing_off_b'):
                        self._spin_swing_off_b = {}
                    self._spin_swing_off_b[leg] = off_b

                pf = p0.clone()
                if spin_in_place:
                    # spin-in-place: keep swing foot fixed in body frame
                    off_b = None
                    if hasattr(self, '_spin_swing_off_b') and (leg in self._spin_swing_off_b):
                        off_b = self._spin_swing_off_b[leg]
                    if off_b is None:
                        # fallback to nominal cycle/stand offset
                        pf[0:2] = com_relative_xy_for_leg(leg)
                    else:
                        # pf_xy = com_xy + Rz(yaw) * off_body
                        pf[0:2] = com_next[0:2] + (Rz2 @ off_b)
                else:
                    # walking: freeze landing target for the whole swing.
                    # If we keep updating pf during the swing (because COM keeps moving),
                    # the swing trajectory will "chase" a moving target and can create
                    # a jump at swing->stance.
                    pf[0:2] = com_relative_xy_for_leg(leg) + vxy_world[0:2] * lookahead
                pf[2] = self._stand_pos[leg][2]
                self._swing_goal[leg] = pf

        # stance legs
        for leg in stance_legs:
            self._swing_active[leg] = False
            # stance feet keep world position fixed.
            # (Their relationship to COM will change as COM moves forward, which is expected.
            #  They will be re-positioned only when they become swing legs.)
            leg_p = self._feet_world[leg].clone()
            new_pos[leg] = leg_p

        # generate swing trajectories
        for leg in swing_legs:
            p0 = self._swing_start[leg]
            pf = self._swing_goal[leg]
            if spin_in_place and hasattr(self, '_spin_swing_off_b') and (leg in self._spin_swing_off_b):
                # During spin-in-place, the desired world landing point should rotate with yaw
                # while remaining fixed in the body frame. So we re-compute pf every step.
                off_b = self._spin_swing_off_b[leg]
                pf = pf.clone()
                pf[0:2] = com_next[0:2] + (Rz2 @ off_b)
                pf[2] = self._stand_pos[leg][2]
                self._swing_goal[leg] = pf

            new_pos[leg] = self._swing_traj(p0, pf, s_phase)
            if s_phase >= 1.0 - 1e-6:
                self._feet_world[leg] = pf.clone()
                self._swing_active[leg] = False
                if spin_in_place and hasattr(self, '_spin_swing_off_b') and (leg in self._spin_swing_off_b):
                    # Clear cached offset once the swing is finished.
                    # It will be re-frozen at the next lift-off.
                    del self._spin_swing_off_b[leg]

        # NOTE: feet_world represents the world contact point (last touchdown).
        # We update it only when a swing finishes (touchdown), not continuously
        # during stance.

        return GaitPlan(
            in_cycle=self.in_cycle,
            stance_legs=stance_legs,
            swing_legs=swing_legs,
            target_pos=new_pos,
            target_ori=new_ori,
        )
