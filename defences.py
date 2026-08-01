"""Enemy air-defence sites: random laydown around a target, and threat checks.

The site types are generic tiers rather than real-world systems — enough to give
each site a distinct envelope without pretending to model anything specific.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from geo import Position, destination, slant_range_km


@dataclass(frozen=True)
class DefenceType:
    """A class of air-defence site."""

    name: str
    code: str
    engagement_km: float
    ceiling_m: float
    weight: float  # relative frequency in a random laydown


# Long-range sites are rarer than point defence, so the laydown skews toward the
# short end rather than picking uniformly.
DEFENCE_TYPES = (
    DefenceType("Long-range SAM", "LR", engagement_km=250.0, ceiling_m=25000.0, weight=1.0),
    DefenceType("Medium-range SAM", "MR", engagement_km=90.0, ceiling_m=18000.0, weight=2.0),
    DefenceType("Short-range SAM", "SR", engagement_km=35.0, ceiling_m=10000.0, weight=3.0),
    DefenceType("Point defence", "PD", engagement_km=12.0, ceiling_m=4500.0, weight=3.0),
)


@dataclass(frozen=True)
class DefenceSite:
    """A placed air-defence site."""

    designator: str
    kind: DefenceType
    position: Position

    def engages(self, target: Position) -> bool:
        """True if `target` is inside this site's envelope.

        Both conditions have to hold: within slant range *and* below the
        ceiling. A target directly overhead but above the ceiling is untouched.
        """
        if target.alt_m > self.kind.ceiling_m:
            return False
        return slant_range_km(self.position, target) <= self.kind.engagement_km

    def __str__(self) -> str:
        return (
            f"{self.designator:<6} {self.kind.name:<17} {self.position.coords}"
            f"  range {self.kind.engagement_km:>5,.0f} km"
            f"  ceiling {self.kind.ceiling_m:>6,.0f} m"
        )


def random_defences(
    target: Position,
    rng: random.Random | None = None,
    count: int = 5,
    spread_km: float = 400.0,
) -> list[DefenceSite]:
    """Scatter `count` sites around `target`, within `spread_km` of it.

    Sites are placed on the ground (altitude 0) and distributed uniformly by
    *area* over the disc — drawing the radius uniformly would crowd them toward
    the centre, since a ring's area grows with its radius.
    """
    rng = rng or random
    kinds = list(DEFENCE_TYPES)
    weights = [k.weight for k in kinds]
    counters: dict[str, int] = {}
    sites = []

    for _ in range(count):
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        bearing = rng.uniform(0.0, 360.0)
        radius = spread_km * math.sqrt(rng.random())

        at = destination(target, bearing, radius)
        counters[kind.code] = counters.get(kind.code, 0) + 1
        sites.append(
            DefenceSite(
                designator=f"{kind.code}-{counters[kind.code]:02d}",
                kind=kind,
                position=Position(at.lat, at.lon, 0.0),
            )
        )
    return sites


def engaging(sites: list[DefenceSite], target: Position) -> list[DefenceSite]:
    """The subset of `sites` whose envelope currently covers `target`."""
    return [s for s in sites if s.engages(target)]
