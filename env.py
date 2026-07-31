"""Strike-mission environment: fly BLUE to RED without entering a SAM envelope.

The agent flies on the actual sphere — each step walks a fixed distance along
the current heading with `geo.destination` — so a trained route can be dropped
straight onto the globe without a projection step.

Two controls, both continuous in [-1, 1]: turn rate and climb rate. Altitude is
a real decision rather than decoration, because every defence site has a ceiling
as well as a radius: climbing over a short-range site is a valid alternative to
going around it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from defenses import DefenseSite, random_defenses
from geo import Position, destination, great_circle_km, initial_bearing_deg

# --- airframe and episode limits -------------------------------------------
SPEED_KMH = 900.0
STEP_SECONDS = 60.0
STEP_KM = SPEED_KMH * STEP_SECONDS / 3600.0  # 15 km per step
MAX_TURN_DEG = 15.0                          # per step
MAX_CLIMB_M = 600.0                          # per step
MAX_ALT_M = 15000.0
MAX_STEPS = 220                              # ~3,300 km of range
STRIKE_RADIUS_KM = 20.0                      # close enough to release on RED

# --- observation shape ------------------------------------------------------
TRACKED_THREATS = 4      # nearest N sites are visible to the policy
THREAT_FEATURES = 7
OBS_DIM = 4 + TRACKED_THREATS * THREAT_FEATURES + 1
ACT_DIM = 2

# --- reward weights ---------------------------------------------------------
R_PROGRESS = 1.0     # per unit of "fraction of a step's distance closed"
R_STEP = -0.01       # mild time cost
R_PRESSURE = -0.30   # per unit of proximity pressure, capped below
R_KILLED = -30.0
R_DESTROYED = 50.0
PRESSURE_CAP = 3.0
PRESSURE_MARGIN_KM = 50.0  # how far outside a ring the agent starts feeling it

# --- scenario randomisation -------------------------------------------------
START_RANGE_KM = (600.0, 1200.0)
DEFENSE_COUNT = (3, 8)
DEFENSE_SPREAD_KM = (150.0, 400.0)
# Poles are excluded: bearing maths is fine there, but the scenarios are
# degenerate and it keeps the training distribution closer to the useful case.
TARGET_LAT_LIMIT = 60.0


@dataclass
class Scenario:
    """A single mission setup — fixed inputs the episode plays out from."""

    target: Position
    start: Position
    start_heading: float
    defenses: list[DefenseSite]


def sealing_sites(target: Position, sites: list[DefenseSite]) -> list[DefenseSite]:
    """Sites that make the target unreachable, not merely well defended.

    A site whose ceiling is above the airframe's cannot be overflown, so if its
    envelope also swallows the release point there is no way in — the episode is
    a guaranteed loss whatever the policy does. Training on those teaches
    nothing and drags the measured success rate down by a fixed offset.
    """
    return [
        s
        for s in sites
        if s.kind.ceiling_m >= MAX_ALT_M
        and great_circle_km(target, s.position) <= s.kind.engagement_km - STRIKE_RADIUS_KM
    ]


def random_scenario(rng: random.Random, max_attempts: int = 12) -> Scenario:
    """Draw a random target, defence laydown, and BLUE start position.

    Laydowns that seal the target are redrawn; if the draw keeps failing the
    blocking sites are removed, so this always terminates with a viable mission.
    """
    target = Position(
        lat=math.degrees(math.asin(rng.uniform(
            math.sin(math.radians(-TARGET_LAT_LIMIT)),
            math.sin(math.radians(TARGET_LAT_LIMIT)),
        ))),
        lon=rng.uniform(-180.0, 180.0),
        alt_m=0.0,
    )
    for _ in range(max_attempts):
        sites = random_defenses(
            target,
            rng,
            count=rng.randint(*DEFENSE_COUNT),
            spread_km=rng.uniform(*DEFENSE_SPREAD_KM),
        )
        if not sealing_sites(target, sites):
            break
    else:
        blocked = set(sealing_sites(target, sites))
        sites = [s for s in sites if s not in blocked]
    approach = rng.uniform(0.0, 360.0)
    start = destination(target, approach, rng.uniform(*START_RANGE_KM))
    start = Position(start.lat, start.lon, rng.uniform(6000.0, 12000.0))

    # Start pointed at the target, so the policy learns evasion rather than
    # how to turn around.
    return Scenario(target, start, initial_bearing_deg(start, target), sites)


class MissionEnv:
    """Single-agent episode. Reset/step, no gym dependency."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.scenario: Scenario | None = None
        self.position: Position = Position(0.0, 0.0)
        self.heading: float = 0.0
        self.steps: int = 0
        self.track: list[Position] = []

    # -- episode lifecycle ---------------------------------------------------

    def reset(self, scenario: Scenario | None = None) -> np.ndarray:
        self.scenario = scenario or random_scenario(self.rng)
        self.position = self.scenario.start
        self.heading = self.scenario.start_heading
        self.steps = 0
        self.track = [self.position]
        self._prev_dist = great_circle_km(self.position, self.scenario.target)
        return self._observe()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        turn = float(np.clip(action[0], -1.0, 1.0))
        climb = float(np.clip(action[1], -1.0, 1.0))

        self.heading = (self.heading + turn * MAX_TURN_DEG) % 360.0
        alt = float(np.clip(self.position.alt_m + climb * MAX_CLIMB_M, 0.0, MAX_ALT_M))
        moved = destination(self.position, self.heading, STEP_KM)
        self.position = Position(moved.lat, moved.lon, alt)
        self.track.append(self.position)
        self.steps += 1

        dist = great_circle_km(self.position, self.scenario.target)
        reward = R_PROGRESS * (self._prev_dist - dist) / STEP_KM + R_STEP
        self._prev_dist = dist

        # Being near a live ring costs something before it becomes fatal —
        # a purely terminal penalty gives no gradient until it is too late.
        reward += R_PRESSURE * min(self._threat_pressure(), PRESSURE_CAP)

        engaged = [s for s in self.scenario.defenses if s.engages(self.position)]
        if engaged:
            return self._observe(), reward + R_KILLED, True, {
                "outcome": "shot_down",
                "by": engaged[0].designator,
                "distance_km": dist,
            }

        if dist <= STRIKE_RADIUS_KM:
            return self._observe(), reward + R_DESTROYED, True, {
                "outcome": "target_destroyed",
                "steps": self.steps,
                "distance_km": dist,
            }

        if self.steps >= MAX_STEPS:
            return self._observe(), reward, True, {
                "outcome": "out_of_fuel",
                "distance_km": dist,
            }

        return self._observe(), reward, False, {}

    # -- internals -----------------------------------------------------------

    def _threat_pressure(self) -> float:
        """Soft cost that ramps up as the aircraft closes on a live envelope.

        Sites whose ceiling is below the current altitude contribute nothing,
        which is what makes climbing a genuine alternative to turning.
        """
        total = 0.0
        for site in self.scenario.defenses:
            if self.position.alt_m > site.kind.ceiling_m:
                continue
            reach = site.kind.engagement_km + PRESSURE_MARGIN_KM
            d = great_circle_km(self.position, site.position)
            if d < reach:
                total += 1.0 - d / reach
        return total

    def _threat_features(self) -> list[float]:
        """Nearest sites, ordered by how little margin is left to their edge."""
        rows = []
        for site in self.scenario.defenses:
            d = great_circle_km(self.position, site.position)
            rows.append((d - site.kind.engagement_km, d, site))
        rows.sort(key=lambda r: r[0])

        feats: list[float] = []
        for margin, d, site in rows[:TRACKED_THREATS]:
            rel = math.radians(initial_bearing_deg(self.position, site.position) - self.heading)
            live = self.position.alt_m <= site.kind.ceiling_m
            feats += [
                d / 500.0,
                math.sin(rel),
                math.cos(rel),
                site.kind.engagement_km / 250.0,
                site.kind.ceiling_m / MAX_ALT_M,
                margin / 500.0,
                1.0 if live else 0.0,
            ]

        # Pad with a sentinel that reads as "nothing out there": far away, no
        # reach, not live. Zero-padding would look like a site at range 0.
        while len(feats) < TRACKED_THREATS * THREAT_FEATURES:
            feats += [4.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0]
        return feats

    def _observe(self) -> np.ndarray:
        target = self.scenario.target
        dist = great_circle_km(self.position, target)
        rel = math.radians(initial_bearing_deg(self.position, target) - self.heading)

        obs = [
            dist / 1000.0,
            math.sin(rel),
            math.cos(rel),
            self.position.alt_m / MAX_ALT_M,
        ]
        obs += self._threat_features()
        obs.append(1.0 - self.steps / MAX_STEPS)
        return np.asarray(obs, dtype=np.float32)


class VecEnv:
    """A fixed number of independent envs stepped together, auto-resetting."""

    def __init__(self, n: int, seed: int = 0):
        self.envs = [MissionEnv(random.Random(seed + i)) for i in range(n)]
        self.n = n

    def reset(self) -> np.ndarray:
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions: np.ndarray):
        obs, rews, dones, infos = [], [], [], []
        for env, act in zip(self.envs, actions):
            o, r, d, info = env.step(act)
            if d:
                infos.append(info)
                o = env.reset()  # the terminal obs is unused; value is bootstrapped off
            obs.append(o)
            rews.append(r)
            dones.append(d)
        return (
            np.stack(obs),
            np.asarray(rews, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            infos,
        )