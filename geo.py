"""Geospatial primitives: uniform position sampling and great-circle math."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# IUGG mean Earth radius.
EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class Position:
    """A point on the Earth's surface, in decimal degrees."""

    lat: float
    lon: float

    def __str__(self) -> str:
        ns = "N" if self.lat >= 0 else "S"
        ew = "E" if self.lon >= 0 else "W"
        return f"{abs(self.lat):.4f}°{ns}, {abs(self.lon):.4f}°{ew}"

    def to_unit_vector(self) -> tuple[float, float, float]:
        """Cartesian unit vector on the sphere."""
        lat, lon = math.radians(self.lat), math.radians(self.lon)
        return (
            math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
        )


def _from_unit_vector(x: float, y: float, z: float) -> Position:
    norm = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm
    return Position(math.degrees(math.asin(z)), math.degrees(math.atan2(y, x)))


def random_position(
    rng: random.Random | None = None,
    lat_range: tuple[float, float] = (-90.0, 90.0),
    lon_range: tuple[float, float] = (-180.0, 180.0),
) -> Position:
    """Sample a position uniformly over the sphere's *surface*.

    Drawing latitude uniformly from [-90, 90] would bunch points toward the
    poles, where circles of latitude are short. Sampling sin(lat) uniformly
    instead gives every unit of area an equal chance, which is what "random
    point on the globe" should mean.
    """
    rng = rng or random
    lat_lo, lat_hi = sorted(lat_range)
    lon_lo, lon_hi = sorted(lon_range)

    sin_lo, sin_hi = math.sin(math.radians(lat_lo)), math.sin(math.radians(lat_hi))
    lat = math.degrees(math.asin(rng.uniform(sin_lo, sin_hi)))
    lon = rng.uniform(lon_lo, lon_hi)
    return Position(lat, lon)


def great_circle_km(a: Position, b: Position) -> float:
    """Shortest surface distance between two positions, in kilometres."""
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat = lat2 - lat1
    dlon = math.radians(b.lon - a.lon)

    # Haversine — numerically well behaved at small separations, where the
    # spherical law of cosines loses precision.
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def initial_bearing_deg(a: Position, b: Position) -> float:
    """Compass bearing at `a` when departing along the great circle to `b`."""
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlon = math.radians(b.lon - a.lon)

    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0


def great_circle_path(a: Position, b: Position, segments: int = 128) -> list[Position]:
    """Interpolate the great-circle arc from `a` to `b` as a list of points.

    Plotly draws map lines straight through projected space, so the arc has to
    be sampled here for it to sit on the sphere correctly.
    """
    v1, v2 = a.to_unit_vector(), b.to_unit_vector()
    dot = max(-1.0, min(1.0, sum(p * q for p, q in zip(v1, v2))))
    omega = math.acos(dot)

    if math.isclose(omega, 0.0, abs_tol=1e-12):
        return [a, b]

    path = []
    for i in range(segments + 1):
        t = i / segments
        # Spherical linear interpolation.
        c1 = math.sin((1 - t) * omega) / math.sin(omega)
        c2 = math.sin(t * omega) / math.sin(omega)
        path.append(
            _from_unit_vector(*(c1 * p + c2 * q for p, q in zip(v1, v2)))
        )
    return path


def midpoint(a: Position, b: Position) -> Position:
    """Great-circle midpoint, used to aim the globe's camera between two points."""
    v1, v2 = a.to_unit_vector(), b.to_unit_vector()
    return _from_unit_vector(*(p + q for p, q in zip(v1, v2)))