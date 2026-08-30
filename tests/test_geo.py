"""Great-circle math: the numbers everything else is measured in."""

from __future__ import annotations

import math
import random

import pytest

from hidden_eclipse.geo import (
    EARTH_RADIUS_KM,
    Position,
    destination,
    elevation_angle_deg,
    great_circle_km,
    initial_bearing_deg,
    midpoint,
    random_position,
    slant_range_km,
)


def test_great_circle_quarter_of_the_planet() -> None:
    """Equator to pole is a quarter circumference, whatever the longitude."""
    equator = Position(lat=0.0, lon=17.0, alt_m=0.0)
    pole = Position(lat=90.0, lon=-140.0, alt_m=0.0)
    assert great_circle_km(equator, pole) == pytest.approx(0.5 * math.pi * EARTH_RADIUS_KM)


def test_destination_round_trips_through_bearing_and_range() -> None:
    origin = Position(lat=12.5, lon=-3.25, alt_m=8000.0)
    for bearing in (0.0, 47.0, 180.0, 271.5):
        far = destination(origin, bearing, 350.0)
        assert great_circle_km(origin, far) == pytest.approx(350.0)
        assert initial_bearing_deg(origin, far) == pytest.approx(bearing, abs=1e-6)


def test_midpoint_is_equidistant() -> None:
    a = Position(lat=-20.0, lon=100.0, alt_m=0.0)
    b = Position(lat=35.0, lon=-70.0, alt_m=0.0)
    mid = midpoint(a, b)
    assert great_circle_km(a, mid) == pytest.approx(great_circle_km(mid, b))


def test_slant_range_is_the_chord_not_the_arc() -> None:
    """Line of sight cuts through the planet, so it never exceeds the ground track."""
    rng = random.Random(4)
    for _ in range(200):
        a = random_position(rng, alt_range=(0.0, 0.0))
        b = random_position(rng, alt_range=(0.0, 0.0))
        assert slant_range_km(a, b) <= great_circle_km(a, b) + 1e-9


def test_slant_range_closes_on_the_arc_at_short_ranges() -> None:
    origin = Position(lat=5.0, lon=5.0, alt_m=0.0)
    near = destination(origin, 75.0, 1.0)
    assert slant_range_km(origin, near) == pytest.approx(great_circle_km(origin, near), rel=1e-6)


def test_slant_range_measures_altitude_when_directly_overhead() -> None:
    ground = Position(lat=-8.0, lon=140.0, alt_m=0.0)
    above = Position(lat=-8.0, lon=140.0, alt_m=12000.0)
    assert slant_range_km(ground, above) == pytest.approx(12.0)
    assert slant_range_km(above, ground) == pytest.approx(12.0)


def test_elevation_angle_signs_with_altitude() -> None:
    low = Position(lat=0.0, lon=0.0, alt_m=0.0)
    high = Position(lat=0.0, lon=0.5, alt_m=12000.0)
    assert elevation_angle_deg(low, high) > 0.0
    assert elevation_angle_deg(high, low) < 0.0


def test_random_position_respects_its_ranges() -> None:
    rng = random.Random(11)
    for _ in range(500):
        p = random_position(
            rng, lat_range=(10.0, 20.0), lon_range=(-5.0, 5.0), alt_range=(0.0, 100.0)
        )
        assert 10.0 <= p.lat <= 20.0
        assert -5.0 <= p.lon <= 5.0
        assert 0.0 <= p.alt_m <= 100.0


def test_random_position_is_seed_reproducible() -> None:
    assert random_position(random.Random(7)) == random_position(random.Random(7))