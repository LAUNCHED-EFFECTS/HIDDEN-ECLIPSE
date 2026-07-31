"""Scenario construction shared by the CLI and the server.

Both entry points need the same thing — a random RED, a random BLUE, and a
defence laydown — so it lives here rather than in either one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from defenses import DefenseSite, random_defenses
from env import Scenario
from geo import Position, initial_bearing_deg, random_position


@dataclass
class World:
    """One generated tactical picture."""

    hostile: Position
    friendly: Position
    defenses: list[DefenseSite]

    def to_scenario(self) -> Scenario:
        """The same picture in the form the RL environment expects."""
        return Scenario(
            target=self.hostile,
            start=self.friendly,
            start_heading=initial_bearing_deg(self.friendly, self.hostile),
            defenses=self.defenses,
        )


def generate_world(
    rng: random.Random,
    lat_range: tuple[float, float] = (-90.0, 90.0),
    lon_range: tuple[float, float] = (-180.0, 180.0),
    alt_range: tuple[float, float] = (0.0, 15000.0),
    defense_count: int = 5,
    defense_spread_km: float = 400.0,
) -> World:
    hostile = random_position(rng, lat_range, lon_range, alt_range)
    friendly = random_position(rng, lat_range, lon_range, alt_range)
    sites = random_defenses(hostile, rng, defense_count, defense_spread_km)
    return World(hostile, friendly, sites)