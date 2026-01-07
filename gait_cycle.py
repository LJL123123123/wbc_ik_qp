"""gait_cycle.py

A lightweight gait cycle manager for this repo's simplified WBC.

Responsibilities:
- Track whether we're in a walking cycle.
- Detect command timeout.
- When command is absent: keep robot steady (hold current targets).
- If command absent for >2s: exit cycle and smoothly go to stand targets.
- When command becomes present: if not in cycle, initialize and enter cycle.
- When in cycle: manage stance/swing sets and generate swing trajectories.

This is NOT a full dynamics gait planner; it's a kinematic target generator
compatible with `wbc.update_targets(...)` and `wbc.set_contact_state(...)`.

Assumptions:
- Target dictionaries include keys: 'com','LF','RF','LH','RH'.
- Orientation for 'com' is 3x3 rotation matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import time
import torch


LEG_ORDER = ['LF', 'RH', 'RF', 'LH']  # trot pairs: (LF,RH) and (RF,LH)
PAIR1 = ['LF', 'RH']
PAIR2 = ['RF', 'LH']


@dataclass
class GaitParams:
    cycle_period: float = 1.0
    swing_height: float = 0.06
    lookahead: float = 1.0
    transition_duration: float = 0.4  # stand transition
    cmd_timeout: float = 2.0


@dataclass
class GaitOutputs:
    stance_legs: List[str]
    swing_legs: List[str]
    target_pos: Dict[str, torch.Tensor]
    target_ori: Dict[str, torch.Tensor]


class GaitCycleManager:
    def __init__(self, device=None, dtype=torch.float64, params: GaitParams | None = None):
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        self.params = params or GaitParams()

        self.in_cycle = False
        self.t0 = 0.0
        self.last_cmd_time = None  # wall time

        self._stand_target_pos = None
        self._stand_target_ori = None

        # persistent foot state
        self._feet_world = {}
        self._swing_start = {}
        self._swing_goal = {}
        self._swing_active = {k: False for k in ['LF', 'RF', 'LH', 'RH']}

        # stand transition
        self._stand_transition_active = False
        self._stand_trans_start = 0.0
        self._stand_trans_p0 = None
        self._stand_trans_pf = None

    def _now(self):
        return time.time()

    def set_stand_targets(self, target_pos: Dict[str, torch.Tensor], target_ori: Dict[str, torch.Tensor]):
        # expected to be tensors already
        self._stand_target_pos = {k: v.clone() for k, v in target_pos.items()}
        self._stand_target_ori = {k: v.clone() for k, v in target_ori.items()}

    def _cmd_active(self, cmd_vxy: torch.Tensor, cmd_yaw_rate: torch.Tensor, eps: float = 1e-4) -> bool:
        return (torch.norm(cmd_vxy).item() > eps) or (abs(float(cmd_yaw_rate)) > eps)

    def _cubic(self, p0: torch.Tensor, pf: torch.Tensor, s: float) -> torch.Tensor:
        s = max(0.0, min(1.0, float(s)))
        a = 3 * s * s - 2 * s * s * s
        return p0 + (pf - p0) * a

    def _swing_traj(self, p0: torch.Tensor, pf: torch.Tensor, s: float) -> torch.Tensor:
        # horizontal cubic + vertical parabola bump
        p = self._cubic(p0, pf, s)
        z_bump = 4.0 * self.params.swing_height * s * (1.0 - s)
        p2 = p.clone()
        p2[2] = p2[2] + z_bump
        return p2

    def update(
        self,
        t: float,
        target_pos: Dict[str, torch.Tensor],
        target_ori: Dict[str, torch.Tensor],
        cmd_vxy: torch.Tensor,
        cmd_yaw_rate: torch.Tensor,
    ) -> GaitOutputs:
        """Update gait state and return new targets & stance/swing sets."""

        # ensure tensors on right device
        cmd_vxy = cmd_vxy.to(device=self.device, dtype=self.dtype).reshape(2)
        cmd_yaw_rate = cmd_yaw_rate.to(device=self.device, dtype=self.dtype).reshape(())

        active = self._cmd_active(cmd_vxy, cmd_yaw_rate)
        now = self._now()
        if active:
            self.last_cmd_time = now

        # init stand targets if not set
        if self._stand_target_pos is None:
            self.set_stand_targets(target_pos, target_ori)

        # timeout logic
        if (not active) and (self.last_cmd_time is not None):
            if now - self.last_cmd_time > self.params.cmd_timeout:
                # exit cycle and go to stand
                self.in_cycle = False

        # entering cycle
        if active and (not self.in_cycle):
            self.in_cycle = True
            self.t0 = t
            # initialize foot world positions from current targets
            for leg in ['LF', 'RF', 'LH', 'RH']:
                self._feet_world[leg] = target_pos[leg].clone()
                self._swing_active[leg] = False
                self._swing_start[leg] = target_pos[leg].clone()
                self._swing_goal[leg] = target_pos[leg].clone()

        # if not in cycle: either hold or smoothly return to stand
        if not self.in_cycle:
            # if we are not at stand pose, start/continue a smooth transition
            if not self._stand_transition_active:
                self._stand_transition_active = True
                self._stand_trans_start = t
                self._stand_trans_p0 = {k: target_pos[k].clone() for k in ['LF', 'RF', 'LH', 'RH', 'com']}
                self._stand_trans_pf = {k: self._stand_target_pos[k].clone() for k in ['LF', 'RF', 'LH', 'RH', 'com']}

            s = (t - self._stand_trans_start) / max(1e-6, self.params.transition_duration)
            s = max(0.0, min(1.0, s))
            new_pos = dict(target_pos)
            for k in ['LF', 'RF', 'LH', 'RH', 'com']:
                new_pos[k] = self._cubic(self._stand_trans_p0[k], self._stand_trans_pf[k], s)
            new_ori = dict(target_ori)
            new_ori['com'] = self._stand_target_ori['com'].clone()

            if s >= 1.0:
                self._stand_transition_active = False

            return GaitOutputs(
                stance_legs=['LF', 'RF', 'LH', 'RH'],
                swing_legs=[],
                target_pos=new_pos,
                target_ori=new_ori,
            )

        # in cycle: manage trot timing
        phase = ((t - self.t0) % self.params.cycle_period) / self.params.cycle_period
        first_half = phase < 0.5
        swing_legs = PAIR1 if first_half else PAIR2
        stance_legs = [l for l in ['LF', 'RF', 'LH', 'RH'] if l not in swing_legs]

        # compute swing progress in [0,1]
        s = (phase / 0.5) if first_half else ((phase - 0.5) / 0.5)
        s = max(0.0, min(1.0, s))

        # update foot goals at swing start
        for leg in swing_legs:
            if not self._swing_active[leg]:
                self._swing_active[leg] = True
                self._swing_start[leg] = self._feet_world[leg].clone()

                # simple foothold: current foot + v_world * lookahead
                v_world = torch.tensor([cmd_vxy[0], cmd_vxy[1], 0.0], device=self.device, dtype=self.dtype)
                pf = self._feet_world[leg] + v_world * float(self.params.lookahead)
                pf[2] = self._stand_target_pos[leg][2]  # land at nominal ground height
                self._swing_goal[leg] = pf

        # mark stance legs as not swinging, keep their world position fixed
        for leg in stance_legs:
            self._swing_active[leg] = False
            self._feet_world[leg] = target_pos[leg].clone()

        # produce new targets
        new_pos = dict(target_pos)
        for leg in swing_legs:
            p0 = self._swing_start[leg]
            pf = self._swing_goal[leg]
            new_pos[leg] = self._swing_traj(p0, pf, s)
            if s >= 1.0 - 1e-6:
                self._feet_world[leg] = pf.clone()
                self._swing_active[leg] = False

        # com and orientation are expected to be managed by higher-level logic;
        # here we simply keep the existing targets.
        new_ori = dict(target_ori)

        return GaitOutputs(
            stance_legs=stance_legs,
            swing_legs=swing_legs,
            target_pos=new_pos,
            target_ori=new_ori,
        )
