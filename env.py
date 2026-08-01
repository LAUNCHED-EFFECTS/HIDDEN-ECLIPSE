"""Strike-mission environment: fly a BLUE package to RED without losing it to SAMs.

The agents fly on the actual sphere — each step walks a fixed distance along the
current heading with `geo.destination` — so a trained route can be dropped
straight onto the globe without a projection step.

Two controls per asset, both continuous in [-1, 1]: turn rate and climb rate.
Altitude is a real decision rather than decoration, because every defence site
has a ceiling as well as a radius: climbing over a short-range site is a valid
alternative to going around it.

Teaming is not assumed, it is priced. Three terms make a coordinated package
beat the same aircraft flying independently:

  1. The target needs REQUIRED_STRIKES arrivals *within TOT_WINDOW_STEPS of
     each other*. Arrivals outside that window expire, and an asset that has
     released is out of the fight — so an early, uncoordinated run does not
     merely score less, it spends an aircraft for nothing.
  2. A time-on-target bonus on top, largest when the arrivals land together.
  3. A mutual-support penalty for two assets sitting inside one battery's
     envelope at the same time, where a single engagement can take both.

An earlier version made the window a bonus rather than a requirement. Measured
against a straight-in baseline the trained policy's arrival tightness was
identical (0.36 vs 0.35) — the incentive was real but too weak to change
behaviour, so it flew well individually and never teamed.

The reward is otherwise shared: every living asset receives the team's terminal
outcome, with individual shaping on top so each still learns to fly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from defences import DefenceSite, random_defences
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

# --- team -------------------------------------------------------------------
# Slot count the policy is *trained* against — it fixes the rollout buffer
# shape, so training scenarios never exceed it. It is deliberately not a limit
# on how many aircraft an episode can fly: the policy is per-aircraft and sees
# only its nearest TRACKED_TEAMMATES, so it applies to a package of any size.
# MissionEnv sizes its arrays to the actual package for exactly that reason.
MAX_TEAM = 4
TEAM_SIZE_RANGE = (2, 4)     # assets per training scenario
TRACKED_TEAMMATES = 3        # nearest N teammates visible to each asset
# Arrivals needed to destroy the target, capped by team size so a one-asset
# package still behaves exactly as it did before teaming existed.
REQUIRED_STRIKES = 2
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
        self._final_window: int | None = None
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
                # Paying for the arrival itself rewarded racing in alone, which
                # is exactly the behaviour teaming is supposed to replace. The
                # payment is scaled by how ready the rest of the package is, so
                # the signal arrives at the moment the decision is made rather
                # than at the end of the episode where it cannot be credited.
                rewards[i] += R_ARRIVAL * self._arrival_quality(i)
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

    def _arrival_quality(self, index: int) -> float:
        """How ready the package is at the moment asset `index` releases, in [0, 1].

        Counts the other assets that could still make the strike window: ones
        that have already released inside it, plus ones close enough to reach
        the target before it expires. 1.0 means releasing now completes a
        coordinated strike; 0.0 means going in alone.
        """
        need = self.scenario.required_strikes - 1
        if need <= 0:
            return 1.0

        reach = TOT_WINDOW_STEPS * STEP_KM  # how far out a teammate can still be
        in_position = len(self.arrivals_in_window())
        for j, other in enumerate(self.assets):
            if j == index or not other.active:
                continue
            if great_circle_km(other.position, self.scenario.target) <= reach:
                in_position += 1

        return min(1.0, in_position / need)

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
        """Decide whether the mission has ended, and what the team earns."""
        required = self.scenario.required_strikes
        together = self.arrivals_in_window()

        if len(together) >= required:
            self._final_window = together[-1] - together[0]
            bonus = R_DESTROYED + R_TOT_BONUS * self.coordination()
            return True, bonus, "target_destroyed"

        still_flying = sum(1 for a in self.assets if a.active)
        if len(together) + still_flying < required:
            # Not enough aircraft left inside the window to finish. If some
            # already released, the package was spent piecemeal rather than
            # simply shot down — worth telling apart in the metrics.
            outcome = "uncoordinated" if self.arrivals else "team_attrited"
            return True, 0.0, outcome

        if self.steps >= MAX_STEPS:
            return True, 0.0, "out_of_fuel"

        return False, 0.0, None

    def coordination(self) -> float:
        """How tightly the qualifying arrivals landed, in [0, 1].

        1.0 means simultaneous; 0.0 means a full window apart. A single-asset
        package scores 1.0 — there is nothing to coordinate.
        """
        required = self.scenario.required_strikes
        if required < 2:
            return 1.0
        window = getattr(self, "_final_window", None)
        if window is None:
            return 0.0
        return max(0.0, 1.0 - window / TOT_WINDOW_STEPS)

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
            "tot_spread": self._final_window if self._final_window is not None else -1,
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
            len(self.arrivals_in_window()) / max(required, 1),
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