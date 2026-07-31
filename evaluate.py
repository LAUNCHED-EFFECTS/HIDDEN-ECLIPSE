"""Score a trained policy against baselines on held-out scenarios.

    python3 evaluate.py --episodes 500

Scenario seeds start well above the ones the trainer uses, so nothing here was
seen during training.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from env import MissionEnv, Scenario, random_scenario
from ppo import load_policy

HELD_OUT_SEED_BASE = 1_000_000


def policy_action(model, norm, obs: np.ndarray) -> np.ndarray:
    """Deterministic action — the distribution mean, not a sample."""
    with torch.no_grad():
        return model.actor(torch.as_tensor(norm(obs[None]))).numpy()[0]


def run_episode(scenario: Scenario, act_fn, seed: int) -> tuple[str, list]:
    env = MissionEnv(random.Random(seed))
    obs = env.reset(scenario)
    while True:
        obs, _, done, info = env.step(act_fn(obs))
        if done:
            return info["outcome"], env.track


def evaluate(model, norm, episodes: int) -> dict[str, dict[str, float]]:
    strategies = {
        "PPO policy": lambda obs: policy_action(model, norm, obs),
        "straight-in": lambda obs: np.array([0.0, 0.0]),
        "random": lambda obs: np.random.uniform(-1.0, 1.0, size=2),
    }
    tallies = {name: {} for name in strategies}

    for i in range(episodes):
        seed = HELD_OUT_SEED_BASE + i
        scenario = random_scenario(random.Random(seed))
        for name, fn in strategies.items():
            outcome, _ = run_episode(scenario, fn, seed)
            tallies[name][outcome] = tallies[name].get(outcome, 0) + 1

    return tallies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", type=Path, default=Path("policy.pt"))
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    np.random.seed(0)
    model, norm = load_policy(args.policy)
    tallies = evaluate(model, norm, args.episodes)

    print(f"\n  {args.episodes} held-out scenarios, identical for every strategy\n")
    print(f"  {'strategy':<14} {'destroyed':>10} {'shot down':>11} {'out of fuel':>12}")
    for name, counts in tallies.items():
        n = max(sum(counts.values()), 1)
        print(
            f"  {name:<14} {counts.get('target_destroyed', 0)/n:>9.1%} "
            f"{counts.get('shot_down', 0)/n:>10.1%} "
            f"{counts.get('out_of_fuel', 0)/n:>11.1%}"
        )


if __name__ == "__main__":
    main()