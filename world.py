"""Scenario construction shared by the CLI and the server.

Both entry points need the same thing — a random RED, a random BLUE, and a
defence laydown — so it lives here rather than in either one.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from defences import DefenceSite, random_defences
from env import Scenario
from geo import Position, initial_bearing_deg, random_position


CALLSIGN_MAX = 24
_CALLSIGN_ALLOWED = re.compile(r"[^A-Za-z0-9 '\-]")


def clean_callsign(callsign: str) -> str:
    """Normalise a user-supplied callsign.

    The label ends up in an HTML attribute, a plotly trace name and a hover
    template, so it is restricted at the input boundary rather than relying on
    every one of those to escape correctly.
    """
    text = _CALLSIGN_ALLOWED.sub("", str(callsign or ""))
    return " ".join(text.split())[:CALLSIGN_MAX]


def blue_label(number: int, callsign: str = "") -> str:
    """The friendly asset's callsign: "BLUE 2", or "BLUE 2 (Viper)" with one."""
    label = f"BLUE {int(number)}"
    callsign = clean_callsign(callsign)
    return f"{label} ({callsign})" if callsign else label


@dataclass
class World:
    """One generated tactical picture."""

    hostile: Position
    friendly: Position
    defences: list[DefenceSite]
    friendly_number: int = 1
    friendly_callsign: str = ""

    @property
    def friendly_label(self) -> str:
        return blue_label(self.friendly_number, self.friendly_callsign)

    def to_scenario(self) -> Scenario:
        """The same picture in the form the RL environment expects."""
        return Scenario(
            target=self.hostile,
            start=self.friendly,
            start_heading=initial_bearing_deg(self.friendly, self.hostile),
            defences=self.defences,
        )


def generate_world(
    rng: random.Random,
    lat_range: tuple[float, float] = (-90.0, 90.0),
    lon_range: tuple[float, float] = (-180.0, 180.0),
    alt_range: tuple[float, float] = (0.0, 15000.0),
    defence_count: int = 5,
    defence_spread_km: float = 400.0,
    friendly_number: int = 1,
    friendly_callsign: str = "",
) -> World:
    hostile = random_position(rng, lat_range, lon_range, alt_range)
    friendly = random_position(rng, lat_range, lon_range, alt_range)
    sites = random_defences(hostile, rng, defence_count, defence_spread_km)
    return World(hostile, friendly, sites, friendly_number, friendly_callsign)