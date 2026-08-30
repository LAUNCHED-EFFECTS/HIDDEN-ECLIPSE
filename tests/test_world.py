"""Scenario construction and the callsign sanitiser at the input boundary."""

from __future__ import annotations

import random

from hidden_eclipse.world import CALLSIGN_MAX, blue_label, clean_callsign, generate_world


def test_clean_callsign_strips_markup_and_collapses_whitespace() -> None:
    assert clean_callsign("  Viper   Lead ") == "Viper Lead"
    assert clean_callsign("<script>alert(1)</script>") == "scriptalert1script"
    assert clean_callsign('Bad" onmouseover="x') == "Bad onmouseoverx"


def test_clean_callsign_keeps_the_characters_a_callsign_needs() -> None:
    assert clean_callsign("O'Hara-2") == "O'Hara-2"


def test_clean_callsign_is_bounded() -> None:
    assert len(clean_callsign("A" * 200)) == CALLSIGN_MAX


def test_clean_callsign_tolerates_nothing() -> None:
    assert clean_callsign("") == ""
    assert clean_callsign(None) == ""


def test_blue_label_shows_the_callsign_only_when_there_is_one() -> None:
    assert blue_label(2) == "BLUE 2"
    assert blue_label(2, "Viper") == "BLUE 2 (Viper)"
    assert blue_label(2, "<>") == "BLUE 2"


def test_generate_world_numbers_the_package_consecutively() -> None:
    world = generate_world(random.Random(5), defence_count=4, blues=3, friendly_number=1)
    assert [a.number for a in world.assets] == [1, 2, 3]
    assert len(world.defences) == 4


def test_generate_world_always_makes_at_least_one_asset() -> None:
    world = generate_world(random.Random(5), blues=0)
    assert len(world.assets) == 1


def test_generate_world_is_seed_reproducible() -> None:
    a = generate_world(random.Random(31), blues=2)
    b = generate_world(random.Random(31), blues=2)
    assert a.hostile == b.hostile
    assert [x.position for x in a.assets] == [x.position for x in b.assets]


def test_add_asset_numbers_beyond_the_highest_in_the_package() -> None:
    world = generate_world(random.Random(1), blues=2)
    added = world.add_asset(random.Random(1))
    assert added.number == 3
    assert world.assets[-1] is added


def test_remove_asset_keeps_the_package_non_empty() -> None:
    world = generate_world(random.Random(1), blues=2)
    assert world.remove_asset(0) is True
    assert world.remove_asset(0) is False
    assert len(world.assets) == 1


def test_remove_asset_rejects_an_index_that_is_not_there() -> None:
    world = generate_world(random.Random(1), blues=3)
    assert world.remove_asset(9) is False
    assert world.remove_asset(-1) is False
    assert len(world.assets) == 3


def test_to_scenario_carries_the_package_across() -> None:
    world = generate_world(random.Random(14), blues=3, defence_count=4)
    scenario = world.to_scenario()
    assert scenario.target == world.hostile
    assert scenario.starts == world.friendlies
    assert len(scenario.start_headings) == 3
    assert scenario.defences == world.defences


def test_generate_world_respects_the_sampling_box() -> None:
    world = generate_world(
        random.Random(2),
        lat_range=(0.0, 10.0),
        lon_range=(20.0, 30.0),
        alt_range=(1000.0, 2000.0),
        blues=3,
    )
    for position in [world.hostile] + [a.position for a in world.assets]:
        assert 0.0 <= position.lat <= 10.0
        assert 20.0 <= position.lon <= 30.0
        assert 1000.0 <= position.alt_m <= 2000.0
