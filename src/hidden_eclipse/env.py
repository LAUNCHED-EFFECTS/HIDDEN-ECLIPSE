"""Strike-mission environment: fly a BLUE package to RED without losing it to SAMs.

The agents fly on the actual sphere — each step walks a fixed distance along the
current heading with `geo.destination` — so a trained route can be dropped
straight onto the globe without a projection step.

Two controls per asset, both continuous in [-1, 1]: turn rate and climb rate.
Altitude is a real decision rather than decoration, because every defence site
has a ceiling as well as a radius: climbing over a short-range site is a valid
alternative to going around it.

One aircraft reaching RED unengaged destroys the target — the strike does not
depend on the rest of the package arriving. Teaming is therefore an incentive
rather than a gate:

  1. The first arrival earns the destroy reward: the mission is won there.
  2. A follow-up arrival inside TOT_WINDOW_STEPS earns the coordination bonus,
     so a package that presses the attack together scores higher than one that
     trickles in — but never at the cost of calling a successful strike a
     failure.
  3. A mutual-support penalty for two assets sitting inside one battery's
     envelope at the same time, where a single engagement can take both.

The episode runs until every aircraft has released or been lost, rather than
stopping at the first arrival, because ending there would leave no opportunity
for a coordinated follow-up to exist at all.

Arrival rewards are paid at the moment of arrival, not at the end: an aircraft
that released at minute 40 has already finished its own trajectory and cannot
be credited afterwards.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .defences import DefenceSite, random_defences
from .geo import Position, destination, great_circle_km, initial_bearing_deg

# --- airframe and episode limits -------------------------------------------
SPEED_KMH = 900.0
STEP_SECONDS = 60.0
STEP_KM = SPEED_KMH * STEP_SECONDS / 3600.0  # 15 km per step
MAX_TURN_DEG = 15.0                          # per step
MAX_CLIMB_M = 600.0                          # per step
MAX_ALT_M = 15000.0
MAX_STEPS = 220                              # ~3,300 km of range
STRIKE_RADIUS_KM = 20.0                      # close enough to release on RED

# --- team -------------------------------------------------------------------
# Slot count the policy is *trained* against — it fixes the rollout buffer
# shape, so training scenarios never exceed it. It is deliberately not a limit
# on how many aircraft an episode can fly: the policy is per-aircraft and sees
# only its nearest TRACKED_TEAMMATES, so it applies to a package of any size.
# MissionEnv sizes its arrays to the actual package for exactly that reason.
MAX_TEAM = 4
TEAM_SIZE_RANGE = (2, 4)     # assets per training scenario
TRACKED_TEAMMATES = 3        # nearest N teammates visible to each asset
# Arrivals needed to destroy the target. One aircraft reaching RED unengaged is
# a successful strike; further coordinated arrivals earn a bonus but are never
# required.
REQUIRED_STRIKES = 1
TOT_WINDOW_STEPS = 12        # arrivals inside this window count as coordinated
MUTUAL_SUPPORT_KM = 60.0     # closer than this inside one envelope is stacking

# --- observation shape ------------------------------------------------------
TRACKED_THREATS = 4
THREAT_FEATURES = 7
TEAMMATE_FEATURES = 5
OBS_DIM = 4 + TRACKED_THREATS * THREAT_FEATURES + TRACKED_TEAMMATES * TEAMMATE_FEATURES + 4
ACT_DIM = 2

# --- reward weights ---------------------------------------------------------
R_PROGRESS = 1.0     # per unit of "fraction of a step's distance closed"
R_STEP = -0.01       # mild time cost
R_PRESSURE = -0.30   # per unit of proximity pressure, capped below
R_KILLED = -30.0     # to the asset that is lost
R_TEAM_LOSS = -8.0   # to every surviving asset, so losses are the team's problem
R_ARRIVAL = 15.0     # per asset that reaches the target
R_DESTROYED = 50.0   # shared, once enough arrivals land
R_TOT_BONUS = 20.0   # shared, scaled by how tightly the arrivals cluster
R_STACKING = -0.25   # per stacked pair per step
PRESSURE_CAP = 3.0
PRESSURE_MARGIN_KM = 50.0

# --- scenario randomisation -------------------------------------------------
START_RANGE_KM = (600.0, 1200.0)
DEFENCE_COUNT = (3, 8)
DEFENCE_SPREAD_KM = (150.0, 400.0)
PACKAGE_SPREAD_DEG = 25.0    # how far apart the assets start, around one bearing
TARGET_LAT_LIMIT = 60.0


@dataclass
class Scenario:
    """A single mission setup — fixed inputs the episode plays out from."""

    target: Position
    starts: list[Position]
    start_headings: list[float]
    defences: list[DefenceSite]

    @property
    def team_size(self) -> int:
        return len(self.starts)

    @property
    def required_strikes(self) -> int:
        return min(REQUIRED_STRIKES, self.team_size)


def sealing_sites(target: Position, sites: list[DefenceSite]) -> list[DefenceSite]:
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


def random_scenario(
    rng: random.Random,
    team_size: int | None = None,
    max_attempts: int = 12,
) -> Scenario:
    """Draw a random target, defence laydown, and BLUE package.

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
        sites = random_defences(
            target,
            rng,
            count=rng.randint(*DEFENCE_COUNT),
            spread_km=rng.uniform(*DEFENCE_SPREAD_KM),
        )
        if not sealing_sites(target, sites):
            break
    else:
        blocked = set(sealing_sites(target, sites))
        sites = [s for s in sites if s not in blocked]

    size = team_size or rng.randint(*TEAM_SIZE_RANGE)
    # The package launches from a common direction, fanned out around it — so
    # splitting up or staying together is the policy's decision, not the
    # scenario's.
    approach = rng.uniform(0.0, 360.0)
    starts, headings = [], []
    for i in range(size):
        offset = 0.0 if size == 1 else PACKAGE_SPREAD_DEG * (i / (size - 1) - 0.5)
        at = destination(target, approach + offset, rng.uniform(*START_RANGE_KM))
        start = Position(at.lat, at.lon, rng.uniform(6000.0, 12000.0))
        starts.append(start)
        headings.append(initial_bearing_deg(start, target))

    return Scenario(target, starts, headings, sites)


@dataclass
class _Asset:
    """Per-asset mutable state inside an episode."""

    position: Position
    heading: float
    alive: bool = True
    arrived_at: int | None = None
    prev_dist: float = 0.0

    @property
    def active(self) -> bool:
        """Still flying — neither lost nor already on target."""
        return self.alive and self.arrived_at is None


class MissionEnv:
    """One episode for a whole BLUE package. Reset/step, no gym dependency.

    Observations, rewards and dones are per slot, padded to at least MAX_TEAM
    so training buffers keep a fixed shape — but a larger package simply gets
    more slots rather than overflowing. The `active` mask says which slots carry
    a real, still-flying asset; the trainer uses it to keep padding and
    casualties out of the gradient.
    """

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.scenario: Scenario | None = None
        self.assets: list[_Asset] = []
        self.steps = 0
        self.slots = MAX_TEAM
        self.tracks: list[list[Position]] = []

    # -- episode lifecycle ---------------------------------------------------

    def reset(self, scenario: Scenario | None = None) -> np.ndarray:
        self.scenario = scenario or random_scenario(self.rng)
        self.assets = [
            _Asset(
                position=start,
                heading=heading,
                prev_dist=great_circle_km(start, self.scenario.target),
            )
            for start, heading in zip(self.scenario.starts, self.scenario.start_headings)
        ]
        # At least MAX_TEAM so the training buffers keep their shape, but never
        # fewer than the package actually has.
        self.slots = max(MAX_TEAM, len(self.assets))
        self.steps = 0
        self.arrivals: list[int] = []
        self.tracks = [[a.position] for a in self.assets]
        self.outcome: str | None = None
        return self.observe()

    def step(self, actions: np.ndarray):
        """Advance every active asset by one minute.

        Returns (obs, rewards, dones, info) with per-slot arrays of length
        `self.slots`. `dones` marks a slot's trajectory as finished — the asset was
        lost, reached the target, or the episode ended under it.
        """
        rewards = np.zeros(self.slots, dtype=np.float32)
        dones = np.zeros(self.slots, dtype=np.float32)
        was_active = [a.active for a in self.assets]

        self.steps += 1
        lost_this_step = 0

        for i, asset in enumerate(self.assets):
            if not asset.active:
                continue

            turn = float(np.clip(actions[i][0], -1.0, 1.0))
            climb = float(np.clip(actions[i][1], -1.0, 1.0))

            asset.heading = (asset.heading + turn * MAX_TURN_DEG) % 360.0
            alt = float(np.clip(asset.position.alt_m + climb * MAX_CLIMB_M, 0.0, MAX_ALT_M))
            moved = destination(asset.position, asset.heading, STEP_KM)
            asset.position = Position(moved.lat, moved.lon, alt)
            self.tracks[i].append(asset.position)

            dist = great_circle_km(asset.position, self.scenario.target)
            rewards[i] += R_PROGRESS * (asset.prev_dist - dist) / STEP_KM + R_STEP
            asset.prev_dist = dist

            # Being near a live ring costs something before it becomes fatal —
            # a purely terminal penalty gives no gradient until it is too late.
            rewards[i] += R_PRESSURE * min(self._threat_pressure(asset), PRESSURE_CAP)

            engaged = [s for s in self.scenario.defences if s.engages(asset.position)]
            if engaged:
                asset.alive = False
                rewards[i] += R_KILLED
                dones[i] = 1.0
                lost_this_step += 1
                continue

            if dist <= STRIKE_RADIUS_KM:
                # Paid at the moment of arrival, because this aircraft's
                # trajectory ends here and nothing later in the episode could be
                # credited to it.
                #
                #   first arrival  -> it destroyed the target: full reward
                #   later, in the coordination window -> the follow-up bonus
                #   later, outside it -> the arrival alone
                #
                # So a lone aircraft reaching the target is a success on its
                # own, and a coordinated second run is worth more on top.
                rewards[i] += R_ARRIVAL
                if not self.arrivals:
                    rewards[i] += R_DESTROYED
                elif self.arrivals_in_window():
                    rewards[i] += R_TOT_BONUS

                asset.arrived_at = self.steps
                self.arrivals.append(self.steps)
                dones[i] = 1.0

        # Mutual support: two assets inside one battery's envelope can be taken
        # by a single engagement, so stacking is charged to both.
        stacked = self._stacked_pairs()
        if stacked:
            for i in range(len(self.assets)):
                if was_active[i]:
                    rewards[i] += R_STACKING * stacked

        # A loss is the team's problem, not just the casualty's.
        if lost_this_step:
            for i, asset in enumerate(self.assets):
                if was_active[i] and asset.alive:
                    rewards[i] += R_TEAM_LOSS * lost_this_step

        env_done, team_reward, outcome = self._resolve()
        if env_done:
            self.outcome = outcome
            for i in range(len(self.assets)):
                if was_active[i]:
                    rewards[i] += team_reward
                    dones[i] = 1.0

        info = {"outcome": outcome} if env_done else {}
        if env_done:
            info.update(self.team_stats())
        return self.observe(), rewards, dones, info

    # -- outcome -------------------------------------------------------------

    def arrivals_in_window(self) -> list[int]:
        """Arrivals still counting toward the strike.

        An arrival older than TOT_WINDOW_STEPS has expired: that aircraft
        released alone and is gone, and the target is still standing.
        """
        return [a for a in self.arrivals if self.steps - a < TOT_WINDOW_STEPS]

    def window_remaining(self) -> int:
        """Steps left before the current partial strike expires."""
        live = self.arrivals_in_window()
        if not live:
            return TOT_WINDOW_STEPS
        return max(0, TOT_WINDOW_STEPS - (self.steps - min(live)))

    def _resolve(self) -> tuple[bool, float, str | None]:
        """Decide whether the mission has ended, and what the team earns.

        One aircraft reaching the target unengaged is a successful strike. The
        episode still runs on until the rest of the package is resolved, so a
        second aircraft can follow up and earn the coordination bonus — ending
        the moment the first one released would make teaming impossible to
        reward at all.

        Rewards for arrivals are paid as they happen (see `step`), not here: an
        aircraft that released at minute 40 has already ended its own
        trajectory and cannot be credited at minute 90.
        """
        if any(a.active for a in self.assets):
            if self.steps < MAX_STEPS:
                return False, 0.0, None
            # Out of fuel, but the target may already be down.
            outcome = "target_destroyed" if self.arrivals else "out_of_fuel"
            return True, 0.0, outcome

        # Nobody left flying: everyone either released or was lost.
        if self.arrivals:
            return True, 0.0, "target_destroyed"
        return True, 0.0, "team_attrited"

    def coordination(self) -> float:
        """How tightly the arrivals landed, in [0, 1].

        1.0 means simultaneous, 0.0 a full window or more apart. With fewer
        than two arrivals there is nothing to coordinate, so it reports 1.0 —
        callers filter on `arrivals` before reading it.
        """
        if len(self.arrivals) < 2:
            return 1.0
        spread = self.arrivals[-1] - self.arrivals[0]
        return max(0.0, 1.0 - spread / TOT_WINDOW_STEPS)

    def _stacked_pairs(self) -> int:
        """Pairs of live assets sharing one envelope and sitting close together."""
        live = [a for a in self.assets if a.active]
        pairs = 0
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                if great_circle_km(live[i].position, live[j].position) > MUTUAL_SUPPORT_KM:
                    continue
                for site in self.scenario.defences:
                    reach = site.kind.engagement_km + PRESSURE_MARGIN_KM
                    covers_both = all(
                        a.position.alt_m <= site.kind.ceiling_m
                        and great_circle_km(a.position, site.position) < reach
                        for a in (live[i], live[j])
                    )
                    if covers_both:
                        pairs += 1
                        break
        return pairs

    def team_stats(self) -> dict:
        return {
            "team_size": self.scenario.team_size,
            "required": self.scenario.required_strikes,
            "arrivals": len(self.arrivals),
            "survivors": sum(1 for a in self.assets if a.alive),
            "coordination": self.coordination(),
            "tot_spread": (
                self.arrivals[-1] - self.arrivals[0] if len(self.arrivals) > 1 else 0
            ),
        }

    # -- observations --------------------------------------------------------

    def _threat_pressure(self, asset: _Asset) -> float:
        """Soft cost that ramps up as an asset closes on a live envelope.

        Sites whose ceiling is below the asset's altitude contribute nothing,
        which is what makes climbing a genuine alternative to turning.
        """
        total = 0.0
        for site in self.scenario.defences:
            if asset.position.alt_m > site.kind.ceiling_m:
                continue
            reach = site.kind.engagement_km + PRESSURE_MARGIN_KM
            d = great_circle_km(asset.position, site.position)
            if d < reach:
                total += 1.0 - d / reach
        return total

    def _threat_features(self, asset: _Asset) -> list[float]:
        """Nearest sites, ordered by how little margin is left to their edge."""
        rows = []
        for site in self.scenario.defences:
            d = great_circle_km(asset.position, site.position)
            rows.append((d - site.kind.engagement_km, d, site))
        rows.sort(key=lambda r: r[0])

        feats: list[float] = []
        for margin, d, site in rows[:TRACKED_THREATS]:
            rel = math.radians(
                initial_bearing_deg(asset.position, site.position) - asset.heading
            )
            feats += [
                d / 500.0,
                math.sin(rel),
                math.cos(rel),
                site.kind.engagement_km / 250.0,
                site.kind.ceiling_m / MAX_ALT_M,
                margin / 500.0,
                1.0 if asset.position.alt_m <= site.kind.ceiling_m else 0.0,
            ]

        # Sentinel that reads as "nothing out there": far away, no reach, not
        # live. Zero-padding would look like a site at range 0.
        while len(feats) < TRACKED_THREATS * THREAT_FEATURES:
            feats += [4.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0]
        return feats

    def _teammate_features(self, index: int) -> list[float]:
        """Nearest live teammates, so the policy can position relative to them."""
        me = self.assets[index]
        rows = []
        for j, other in enumerate(self.assets):
            if j == index or not other.active:
                continue
            rows.append((great_circle_km(me.position, other.position), other))
        rows.sort(key=lambda r: r[0])

        feats: list[float] = []
        for d, other in rows[:TRACKED_TEAMMATES]:
            rel = math.radians(
                initial_bearing_deg(me.position, other.position) - me.heading
            )
            feats += [
                d / 500.0,
                math.sin(rel),
                math.cos(rel),
                (other.position.alt_m - me.position.alt_m) / MAX_ALT_M,
                great_circle_km(other.position, self.scenario.target) / 1000.0,
            ]

        # Sentinel for "no teammate there": far away, level, and far from the
        # target, so an empty slot never reads as company.
        while len(feats) < TRACKED_TEAMMATES * TEAMMATE_FEATURES:
            feats += [4.0, 0.0, 1.0, 0.0, 4.0]
        return feats

    def _observe_asset(self, index: int) -> list[float]:
        asset = self.assets[index]
        target = self.scenario.target
        dist = great_circle_km(asset.position, target)
        rel = math.radians(initial_bearing_deg(asset.position, target) - asset.heading)

        obs = [
            dist / 1000.0,
            math.sin(rel),
            math.cos(rel),
            asset.position.alt_m / MAX_ALT_M,
        ]
        obs += self._threat_features(asset)
        obs += self._teammate_features(index)

        # Team status. The window terms are what make the strike requirement
        # learnable: without them the deadline is invisible and an asset has no
        # way to know whether to press in or hold off.
        required = self.scenario.required_strikes
        obs += [
            min(1.0, len(self.arrivals) / max(len(self.assets), 1)),
            self.window_remaining() / TOT_WINDOW_STEPS,
            # Clamped: training only ever saw up to MAX_TEAM aircraft, so a
            # larger package must not push this feature outside the range the
            # policy was fitted on.
            min(1.0, sum(1 for a in self.assets if a.active) / MAX_TEAM),
            1.0 - self.steps / MAX_STEPS,
        ]
        return obs

    def observe(self) -> np.ndarray:
        obs = np.zeros((self.slots, OBS_DIM), dtype=np.float32)
        for i, asset in enumerate(self.assets):
            if asset.active:
                obs[i] = np.asarray(self._observe_asset(i), dtype=np.float32)
        return obs

    def active_mask(self) -> np.ndarray:
        mask = np.zeros(self.slots, dtype=np.float32)
        for i, asset in enumerate(self.assets):
            mask[i] = 1.0 if asset.active else 0.0
        return mask


class VecEnv:
    """A fixed number of independent episodes stepped together, auto-resetting.

    Arrays are (num_envs, MAX_TEAM, ...) — one trajectory stream per slot.
    """

    def __init__(self, n: int, seed: int = 0):
        self.envs = [MissionEnv(random.Random(seed + i)) for i in range(n)]
        self.n = n

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        obs = np.stack([e.reset() for e in self.envs])
        mask = np.stack([e.active_mask() for e in self.envs])
        return obs, mask

    def step(self, actions: np.ndarray):
        """Returns (obs, rewards, dones, masks, env_dones, infos).

        `masks` describes the state *after* the step, ready for the next
        action; `env_dones` flags episodes that ended and were auto-reset,
        which the caller needs since the returned obs already belongs to the
        replacement episode.
        """
        obs, rews, dones, masks, env_dones, infos = [], [], [], [], [], []
        for env, act in zip(self.envs, actions):
            o, r, d, info = env.step(act)
            finished = bool(info)
            if finished:
                infos.append(info)
                o = env.reset()
            obs.append(o)
            rews.append(r)
            dones.append(d)
            masks.append(env.active_mask())
            env_dones.append(finished)
        return (
            np.stack(obs),
            np.stack(rews).astype(np.float32),
            np.stack(dones).astype(np.float32),
            np.stack(masks).astype(np.float32),
            np.asarray(env_dones, dtype=bool),
            infos,
        )