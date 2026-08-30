"""The episode the policy is trained against: shapes, padding, and termination."""

from __future__ import annotations

import random

import numpy as np

from hidden_eclipse.env import (
    ACT_DIM,
    MAX_TEAM,
    OBS_DIM,
    MissionEnv,
    Scenario,
    random_scenario,
    sealing_sites,
)


def test_random_scenario_is_internally_consistent() -> None:
    scenario = random_scenario(random.Random(0))
    assert scenario.team_size == len(scenario.starts) == len(scenario.start_headings)
    assert 1 <= scenario.team_size <= MAX_TEAM
    assert scenario.required_strikes <= scenario.team_size


def test_random_scenario_honours_a_fixed_team_size() -> None:
    for size in range(1, MAX_TEAM + 1):
        assert random_scenario(random.Random(size), team_size=size).team_size == size


def test_random_scenario_is_never_sealed() -> None:
    """A sealed laydown is an automatic loss, so the generator must not emit one."""
    for seed in range(40):
        scenario = random_scenario(random.Random(seed))
        assert sealing_sites(scenario.target, scenario.defences) == []


def test_reset_pads_observations_to_the_training_shape() -> None:
    env = MissionEnv(random.Random(1))
    obs = env.reset(random_scenario(random.Random(1), team_size=2))
    assert obs.shape == (MAX_TEAM, OBS_DIM)
    assert np.isfinite(obs).all()


def test_slots_grow_with_an_oversized_package() -> None:
    """More assets than MAX_TEAM gets more slots rather than dropping anyone."""
    base = random_scenario(random.Random(3), team_size=MAX_TEAM)
    big = Scenario(
        target=base.target,
        starts=base.starts * 2,
        start_headings=base.start_headings * 2,
        defences=base.defences,
    )
    env = MissionEnv(random.Random(3))
    obs = env.reset(big)
    assert env.slots == 2 * MAX_TEAM
    assert obs.shape == (2 * MAX_TEAM, OBS_DIM)


def test_episode_terminates_and_reports_an_outcome() -> None:
    env = MissionEnv(random.Random(2))
    obs = env.reset(random_scenario(random.Random(2)))
    rng = np.random.default_rng(2)

    for _ in range(10_000):
        obs, rewards, dones, info = env.step(rng.uniform(-1.0, 1.0, size=(env.slots, ACT_DIM)))
        assert obs.shape == (env.slots, OBS_DIM)
        assert rewards.shape == (env.slots,)
        assert dones.shape == (env.slots,)
        if info:
            break
    else:
        raise AssertionError("episode never terminated")

    assert info["outcome"]
    assert len(env.tracks) == env.scenario.team_size
    assert all(len(track) >= 1 for track in env.tracks)
