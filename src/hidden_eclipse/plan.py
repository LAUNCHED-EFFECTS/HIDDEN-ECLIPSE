"""Turn a trained policy into a mission plan for a specific scenario."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .defences import DefenceSite
from .env import STEP_SECONDS, MissionEnv, Scenario
from .geo import Position, great_circle_km, initial_bearing_deg
from .ppo import ActorCritic, RunningNorm


@dataclass
class Waypoint:
    """A turn point on the route, with the state the aircraft holds through it."""

    index: int
    position: Position
    heading: float
    elapsed_min: float


@dataclass
class AssetPlan:
    """One aircraft's leg of the package."""

    label: str
    fate: str                     # on_target | lost | returned
    track: list[Position]
    waypoints: list[Waypoint]
    route_km: float
    arrived_min: float | None
    closest_margin_km: float
    threatening_site: str | None

    @property
    def succeeded(self) -> bool:
        return self.fate == "on_target"

    def summary(self) -> str:
        if self.fate == "on_target":
            return f"{self.label}: on target at T+{self.arrived_min:,.0f} min"
        if self.fate == "lost":
            return f"{self.label}: lost to {self.threatening_site or 'air defence'}"
        return f"{self.label}: did not reach the target"


@dataclass
class PackagePlan:
    """The whole strike package, and how the mission came out."""

    outcome: str
    assets: list[AssetPlan] = field(default_factory=list)
    coordination: float = 0.0
    duration_min: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.outcome == "target_destroyed"

    @property
    def on_target(self) -> list[AssetPlan]:
        return [a for a in self.assets if a.succeeded]

    @property
    def tot_spread_min(self) -> float:
        """Minutes between the first and last arrival.

        Derived from the assets rather than the environment's window bookkeeping,
        which reports a sentinel when no qualifying strike ever formed.
        """
        times = sorted(a.arrived_min for a in self.on_target)
        return times[-1] - times[0] if len(times) > 1 else 0.0

    def summary(self) -> str:
        verdict = {
            "target_destroyed": "target destroyed",
            "team_attrited": "package lost to air defence",
            "out_of_fuel": "aborted — no aircraft reached the target",
        }.get(self.outcome, self.outcome)

        parts = [verdict, f"{len(self.on_target)}/{len(self.assets)} on target"]
        if len(self.on_target) > 1:
            parts.append(f"{self.tot_spread_min:,.0f} min apart")
        return " · ".join(parts)


def _closest_margin(
    track: list[Position], sites: list[DefenceSite]
) -> tuple[float, str | None]:
    """Smallest gap between a route and any live engagement envelope.

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
    changes past the threshold become waypoints, plus the start and the end.
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
    points.append(Waypoint(last, track[last], heading, last * STEP_SECONDS / 60.0))
    return points


def plan_mission(
    scenario: Scenario,
    model: ActorCritic,
    norm: RunningNorm,
    seed: int = 0,
    labels: list[str] | None = None,
) -> PackagePlan:
    """Fly the deterministic policy through `scenario` and package the result."""
    env = MissionEnv(random.Random(seed))
    obs = env.reset(scenario)

    while True:
        with torch.no_grad():
            actions = model.actor(torch.as_tensor(norm(obs))).numpy()
        obs, _, _, info = env.step(actions)
        if info:
            break

    assets = []
    for i, asset in enumerate(env.assets):
        track = env.tracks[i]
        margin, culprit = _closest_margin(track, scenario.defences)
        if asset.arrived_at is not None:
            fate = "on_target"
        elif not asset.alive:
            fate = "lost"
        else:
            fate = "returned"

        assets.append(
            AssetPlan(
                label=(labels[i] if labels and i < len(labels) else f"BLUE {i + 1}"),
                fate=fate,
                track=track,
                waypoints=_waypoints(track),
                route_km=sum(
                    great_circle_km(a, b) for a, b in zip(track, track[1:])
                ),
                arrived_min=(
                    asset.arrived_at * STEP_SECONDS / 60.0
                    if asset.arrived_at is not None
                    else None
                ),
                closest_margin_km=margin,
                threatening_site=culprit,
            )
        )

    return PackagePlan(
        outcome=info["outcome"],
        assets=assets,
        coordination=info.get("coordination", 0.0),
        duration_min=env.steps * STEP_SECONDS / 60.0,
    )


def load_and_plan(
    policy_path: Path,
    scenario: Scenario,
    seed: int = 0,
    labels: list[str] | None = None,
) -> PackagePlan:
    from .ppo import load_policy

    model, norm = load_policy(policy_path)
    return plan_mission(scenario, model, norm, seed, labels)