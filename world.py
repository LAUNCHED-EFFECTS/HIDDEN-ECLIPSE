"""Scenario construction shared by the CLI and the server.

Both entry points need the same thing — a random RED, a random BLUE package,
and a defence laydown — so it lives here rather than in either one.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from defences import DefenceSite, random_defences
from env import Scenario
from geo import Position, destination, initial_bearing_deg, random_position

CALLSIGN_MAX = 24
_CALLSIGN_ALLOWED = re.compile(r"[^A-Za-z0-9 '\-]")

# Where a newly added asset appears, relative to the one it joins.
NEW_ASSET_OFFSET_KM = 120.0


def clean_callsign(callsign: str) -> str:
    """Normalise a user-supplied callsign.

    The label ends up in an HTML attribute, a plotly trace name and a hover
    template, so it is restricted at the input boundary rather than relying on
    every one of those to escape correctly.
    """
    text = _CALLSIGN_ALLOWED.sub("", str(callsign or ""))
    return " ".join(text.split())[:CALLSIGN_MAX]


def blue_label(number: int, callsign: str = "") -> str:
    """One asset's call sign: "BLUE 2", or "BLUE 2 (Viper)" with one."""
    label = f"BLUE {int(number)}"
    callsign = clean_callsign(callsign)
    return f"{label} ({callsign})" if callsign else label


@dataclass
class Asset:
    """A friendly aircraft: where it is, and what it is called.

    Identity is per aircraft rather than per package, so BLUE 2 can be renamed
    without touching BLUE 1.
    """

    position: Position
    number: int
    callsign: str = ""

    @property
    def label(self) -> str:
        return blue_label(self.number, self.callsign)


@dataclass
class World:
    """One generated tactical picture."""

    hostile: Position
    assets: list[Asset]
    defences: list[DefenceSite]

    # -- views the renderer and the environment want ------------------------

    @property
    def friendlies(self) -> list[Position]:
        return [a.position for a in self.assets]

    @property
    def friendly_labels(self) -> list[str]:
        return [a.label for a in self.assets]

    @property
    def lead(self) -> Position:
        """The first asset — what the title's ranges and bearings refer to."""
        return self.assets[0].position

    def to_scenario(self) -> Scenario:
        """The same picture in the form the RL environment expects."""
        return Scenario(
            target=self.hostile,
            starts=[a.position for a in self.assets],
            start_headings=[
                initial_bearing_deg(a.position, self.hostile) for a in self.assets
            ],
            defences=self.defences,
        )

    # -- editing ------------------------------------------------------------

    def add_asset(self, rng: random.Random | None = None) -> Asset:
        """Append an aircraft, offset from the last one so it is not hidden."""
        rng = rng or random
        anchor = self.assets[-1] if self.assets else None
        if anchor is None:
            position = Position(0.0, 0.0, 9000.0)
        else:
            at = destination(
                anchor.position, rng.uniform(0.0, 360.0), NEW_ASSET_OFFSET_KM
            )
            position = Position(at.lat, at.lon, anchor.position.alt_m)

        number = max((a.number for a in self.assets), default=0) + 1
        asset = Asset(position, number, anchor.callsign if anchor else "")
        self.assets.append(asset)
        return asset

    def remove_asset(self, index: int) -> bool:
        """Delete one aircraft. The package always keeps at least one."""
        if len(self.assets) <= 1 or not 0 <= index < len(self.assets):
            return False
        self.assets.pop(index)
        return True


def generate_world(
    rng: random.Random,
    lat_range: tuple[float, float] = (-90.0, 90.0),
    lon_range: tuple[float, float] = (-180.0, 180.0),
    alt_range: tuple[float, float] = (0.0, 15000.0),
    defence_count: int = 5,
    defence_spread_km: float = 400.0,
    friendly_number: int = 1,
    friendly_callsign: str = "",
    blues: int = 2,
) -> World:
    hostile = random_position(rng, lat_range, lon_range, alt_range)
    assets = [
        Asset(
            position=random_position(rng, lat_range, lon_range, alt_range),
            number=friendly_number + i,
            callsign=friendly_callsign,
        )
        for i in range(max(1, blues))
    ]
    sites = random_defences(hostile, rng, defence_count, defence_spread_km)
    return World(hostile, assets, sites)
