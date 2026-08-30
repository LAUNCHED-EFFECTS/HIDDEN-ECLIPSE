"""The trained policy, end to end: checkpoint on disk to a mission plan."""

from __future__ import annotations

import random

import pytest

from hidden_eclipse.env import random_scenario
from hidden_eclipse.paths import DEFAULT_POLICY
from hidden_eclipse.plan import load_and_plan, plan_mission
from hidden_eclipse.ppo import load_policy


pytestmark = pytest.mark.skipif(
    not DEFAULT_POLICY.exists(), reason=f"no checkpoint at {DEFAULT_POLICY}"
)


@pytest.fixture(scope="module")
def policy():
    return load_policy(DEFAULT_POLICY)


def test_the_shipped_checkpoint_loads_from_its_default_path() -> None:
    model, norm = load_policy(DEFAULT_POLICY)
    assert not model.training


def test_plan_covers_every_asset_in_the_package(policy) -> None:
    model, norm = policy
    scenario = random_scenario(random.Random(12), team_size=3)
    plan = plan_mission(scenario, model, norm, seed=12)

    assert plan.outcome
    assert len(plan.assets) == scenario.team_size
    assert plan.duration_min > 0.0
    for asset in plan.assets:
        assert asset.fate in {"on_target", "lost", "returned"}
        assert len(asset.track) >= 1
        assert asset.route_km >= 0.0
        assert (asset.arrived_min is not None) == asset.succeeded


def test_labels_carry_through_to_the_plan(policy) -> None:
    model, norm = policy
    scenario = random_scenario(random.Random(6), team_size=2)
    plan = plan_mission(scenario, model, norm, seed=6, labels=["BLUE 1 (Viper)", "BLUE 2"])
    assert [a.label for a in plan.assets] == ["BLUE 1 (Viper)", "BLUE 2"]


def test_missing_labels_fall_back_to_slot_numbers(policy) -> None:
    model, norm = policy
    scenario = random_scenario(random.Random(6), team_size=2)
    plan = plan_mission(scenario, model, norm, seed=6, labels=["BLUE 1 (Viper)"])
    assert [a.label for a in plan.assets] == ["BLUE 1 (Viper)", "BLUE 2"]


def test_planning_is_deterministic_for_a_fixed_seed(policy) -> None:
    """The globe and the server plan the same scenario twice; they must agree."""
    model, norm = policy
    scenario = random_scenario(random.Random(9), team_size=2)
    first = plan_mission(scenario, model, norm, seed=9)
    second = plan_mission(scenario, model, norm, seed=9)

    assert first.outcome == second.outcome
    assert [a.track for a in first.assets] == [a.track for a in second.assets]


def test_summary_reads_as_a_sentence(policy) -> None:
    model, norm = policy
    plan = plan_mission(random_scenario(random.Random(4), team_size=2), model, norm, seed=4)
    assert " · " in plan.summary()
    for asset in plan.assets:
        assert asset.label in asset.summary()


def test_load_and_plan_matches_the_two_step_route() -> None:
    model, norm = load_policy(DEFAULT_POLICY)
    scenario = random_scenario(random.Random(15), team_size=2)
    assert (
        load_and_plan(DEFAULT_POLICY, scenario, seed=15).outcome
        == plan_mission(scenario, model, norm, seed=15).outcome
    )
