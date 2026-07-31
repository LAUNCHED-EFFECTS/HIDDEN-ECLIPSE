"""Turn a trained policy into a mission plan for a specific scenario."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from defenses import DefenseSite
from env import STEP_SECONDS, MissionEnv, Scenario
from geo import Position, great_circle_km, initial_bearing_deg
from ppo import ActorCritic, RunningNorm


@dataclass
class Waypoint:
    """A turn point on the route, with the state the aircraft holds through it."""

    index: int
    position: Position
    heading: float
    elapsed_min: float


@dataclass
class MissionPlan:
    outcome: str
    track: list[Position]
    waypoints: list[Waypoint]
    route_km: float
    duration_min: float
    closest_margin_km: float
    threatening_site: str | None

    @property
    def succeeded(self) -> bool:
        return self.outcome == "target_destroyed"

    def summary(self) -> str:
        verdict = {
            "target_destroyed": "target destroyed",
            "shot_down": "BLUE lost to air defence",
            "out_of_fuel": "aborted — out of fuel",
        }.get(self.outcome, self.outcome)
        return (
            f"{verdict} · {self.route_km:,.0f} km routed over "
            f"{self.duration_min:,.0f} min · {len(self.waypoints)} waypoints"
        )


def _closest_margin(track: list[Position], sites: list[DefenseSite]) -> tuple[float, str | None]:
    """Smallest gap between the route and any live engagement envelope.

    Negative means the route entered one. Sites the aircraft was flying above
    are skipped for the leg where that held, since they could not reach it.
    """
    best, culprit = float("inf"), None
    for pos in track:
        for site in sites:
            if pos.alt_m > site.kind.ceiling_m:
                continue
            margin = great_circle_km(pos, site.position) - site.kind.engagement_km
            if margin < best:
                best, culprit = margin, site.designator
    return (best, culprit) if culprit else (float("nan"), None)


def _waypoints(track: list[Position], turn_threshold_deg: float = 8.0) -> list[Waypoint]:
    """Compress the per-minute track into turn points a briefing could use.

    The policy emits a heading correction every step; most are tiny. Only
    changes past the threshold become waypoints, plus the start and the target.
    """
    if len(track) < 2:
        return []

    points = [Waypoint(0, track[0], initial_bearing_deg(track[0], track[1]), 0.0)]
    heading = points[0].heading

    for i in range(1, len(track) - 1):
        leg = initial_bearing_deg(track[i], track[i + 1])
        turn = abs((leg - heading + 180.0) % 360.0 - 180.0)
        if turn >= turn_threshold_deg:
            points.append(Waypoint(i, track[i], leg, i * STEP_SECONDS / 60.0))
            heading = leg

    last = len(track) - 1
    points.append(
        Waypoint(last, track[last], heading, last * STEP_SECONDS / 60.0)
    )
    return points


def plan_mission(
    scenario: Scenario,
    model: ActorCritic,
    norm: RunningNorm,
    seed: int = 0,
) -> MissionPlan:
    """Fly the deterministic policy through `scenario` and package the result."""
    env = MissionEnv(random.Random(seed))
    obs = env.reset(scenario)

    while True:
        with torch.no_grad():
            action = model.actor(torch.as_tensor(norm(obs[None]))).numpy()[0]
        obs, _, done, info = env.step(action)
        if done:
            break

    track = env.track
    route_km = sum(great_circle_km(a, b) for a, b in zip(track, track[1:]))
    margin, culprit = _closest_margin(track, scenario.defenses)

    return MissionPlan(
        outcome=info["outcome"],
        track=track,
        waypoints=_waypoints(track),
        route_km=route_km,
        duration_min=(len(track) - 1) * STEP_SECONDS / 60.0,
        closest_margin_km=margin,
        threatening_site=culprit,
    )


def load_and_plan(policy_path: Path, scenario: Scenario, seed: int = 0) -> MissionPlan:
    from ppo import load_policy

    model, norm = load_policy(policy_path)
    return plan_mission(scenario, model, norm, seed)