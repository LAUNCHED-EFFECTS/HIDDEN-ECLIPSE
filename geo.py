"""Geospatial primitives: uniform position sampling and great-circle math.

Positions carry an altitude, so ranges come in two flavours: `great_circle_km`
measures across the ground, `slant_range_km` measures line-of-sight.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# IUGG mean Earth radius.
EARTH_RADIUS_KM = 6371.0088


METRES_PER_FOOT = 0.3048


@dataclass(frozen=True)
class Position:
    """A point in decimal degrees, with altitude in metres above mean sea level."""

    lat: float
    lon: float
    alt_m: float = 0.0

    @property
    def alt_ft(self) -> float:
        return self.alt_m / METRES_PER_FOOT

    @property
    def coords(self) -> str:
        """Latitude/longitude alone — for ground features, where altitude is noise."""
        ns = "N" if self.lat >= 0 else "S"
        ew = "E" if self.lon >= 0 else "W"
        return f"{abs(self.lat):.4f}°{ns}, {abs(self.lon):.4f}°{ew}"

    def __str__(self) -> str:
        return f"{self.coords}  @ {self.alt_m:,.0f} m ({self.alt_ft:,.0f} ft)"

    def to_unit_vector(self) -> tuple[float, float, float]:
        """Cartesian unit vector on the sphere — direction only, altitude ignored."""
        lat, lon = math.radians(self.lat), math.radians(self.lon)
        return (
            math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
        )

    def to_ecef_km(self) -> tuple[float, float, float]:
        """Earth-centred cartesian position, scaled by radius *plus* altitude."""
        r = EARTH_RADIUS_KM + self.alt_m / 1000.0
        return tuple(r * c for c in self.to_unit_vector())


def _from_unit_vector(x: float, y: float, z: float) -> Position:
    norm = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm
    return Position(math.degrees(math.asin(z)), math.degrees(math.atan2(y, x)))


def random_position(
    rng: random.Random | None = None,
    lat_range: tuple[float, float] = (-90.0, 90.0),
    lon_range: tuple[float, float] = (-180.0, 180.0),
    alt_range: tuple[float, float] = (0.0, 15000.0),
) -> Position:
    """Sample a position uniformly over the sphere's *surface*, at a random altitude.

    Drawing latitude uniformly from [-90, 90] would bunch points toward the
    poles, where circles of latitude are short. Sampling sin(lat) uniformly
    instead gives every unit of area an equal chance, which is what "random
    point on the globe" should mean.

    Altitude is metres above mean sea level, drawn uniformly — there is no
    terrain model here, so a point can sit below the ground it is over.
    """
    rng = rng or random
    lat_lo, lat_hi = sorted(lat_range)
    lon_lo, lon_hi = sorted(lon_range)
    alt_lo, alt_hi = sorted(alt_range)

    sin_lo, sin_hi = math.sin(math.radians(lat_lo)), math.sin(math.radians(lat_hi))
    lat = math.degrees(math.asin(rng.uniform(sin_lo, sin_hi)))
    lon = rng.uniform(lon_lo, lon_hi)
    alt = rng.uniform(alt_lo, alt_hi)
    return Position(lat, lon, alt)


def great_circle_km(a: Position, b: Position) -> float:
    """Ground range: surface distance below the two positions, in kilometres.

    Altitude is deliberately ignored here — use `slant_range_km` for the
    line-of-sight distance between the points themselves.
    """
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat = lat2 - lat1
    dlon = math.radians(b.lon - a.lon)

    # Haversine — numerically well behaved at small separations, where the
    # spherical law of cosines loses precision.
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def destination(origin: Position, bearing_deg: float, distance_km: float) -> Position:
    """Walk `distance_km` from `origin` along `bearing_deg`, staying on the sphere.

    The direct geodesic problem. Altitude is carried over from the origin.
    """
    ang = distance_km / EARTH_RADIUS_KM
    lat1, lon1 = math.radians(origin.lat), math.radians(origin.lon)
    theta = math.radians(bearing_deg)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(theta)
    )
    lon2 = lon1 + math.atan2(
        math.sin(theta) * math.sin(ang) * math.cos(lat1),
        math.cos(ang) - math.sin(lat1) * math.sin(lat2),
    )
    # Normalise into [-180, 180] so the renderer does not draw a seam.
    lon2 = (math.degrees(lon2) + 540.0) % 360.0 - 180.0
    return Position(math.degrees(lat2), lon2, origin.alt_m)


def circle_path(center: Position, radius_km: float, segments: int = 96) -> list[Position]:
    """Points forming a circle of constant surface radius around `center`.

    A true circle on the sphere, not a circle in projected space — so it
    stretches correctly toward the poles.
    """
    return [
        destination(center, 360.0 * i / segments, radius_km) for i in range(segments + 1)
    ]


def slant_range_km(a: Position, b: Position) -> float:
    """Straight-line distance through space between two positions."""
    va, vb = a.to_ecef_km(), b.to_ecef_km()
    return math.sqrt(sum((q - p) ** 2 for p, q in zip(va, vb)))


def elevation_angle_deg(a: Position, b: Position) -> float:
    """Look angle from `a` up to `b`, in degrees above `a`'s local horizontal.

    Negative means looking down. Earth curvature is in this: a distant target
    at the same altitude sits *below* the horizon, not level with it.
    """
    va, vb = a.to_ecef_km(), b.to_ecef_km()
    los = [q - p for p, q in zip(va, vb)]
    dist = math.sqrt(sum(c * c for c in los))
    if math.isclose(dist, 0.0, abs_tol=1e-12):
        return 0.0

    up = a.to_unit_vector()  # local vertical at the observer
    sin_el = sum(u * c for u, c in zip(up, los)) / dist
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))


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