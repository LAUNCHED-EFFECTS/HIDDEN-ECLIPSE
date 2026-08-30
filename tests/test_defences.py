"""Air-defence laydown and the threat check the globe colours sites by."""

from __future__ import annotations

import random

from hidden_eclipse.defences import DEFENCE_TYPES, engaging, random_defences
from hidden_eclipse.geo import Position, destination, slant_range_km


TARGET = Position(lat=30.0, lon=45.0, alt_m=0.0)


def test_laydown_size_and_spread() -> None:
    sites = random_defences(TARGET, random.Random(3), 6, 400.0)
    assert len(sites) == 6
    for site in sites:
        assert site.kind in DEFENCE_TYPES


def test_laydown_is_seed_reproducible() -> None:
    a = random_defences(TARGET, random.Random(19), 5, 300.0)
    b = random_defences(TARGET, random.Random(19), 5, 300.0)
    assert [(s.position, s.kind.name) for s in a] == [(s.position, s.kind.name) for s in b]


def test_engaging_picks_out_the_sites_that_can_reach() -> None:
    """A point sat on top of a site is engaged; one far outside every envelope is not."""
    sites = random_defences(TARGET, random.Random(8), 5, 400.0)
    site = sites[0]

    overhead = Position(lat=site.position.lat, lon=site.position.lon, alt_m=1000.0)
    assert site in engaging(sites, overhead)

    far = destination(TARGET, 90.0, 5000.0)
    assert engaging(sites, Position(lat=far.lat, lon=far.lon, alt_m=1000.0)) == []


def test_engaging_agrees_with_the_envelope_it_is_derived_from() -> None:
    sites = random_defences(TARGET, random.Random(21), 6, 400.0)
    probe = destination(TARGET, 30.0, 120.0)
    probe = Position(lat=probe.lat, lon=probe.lon, alt_m=6000.0)

    threats = engaging(sites, probe)
    for site in sites:
        in_range = (
            slant_range_km(site.position, probe) <= site.kind.engagement_km
            and probe.alt_m <= site.kind.ceiling_m
        )
        assert (site in threats) == in_range