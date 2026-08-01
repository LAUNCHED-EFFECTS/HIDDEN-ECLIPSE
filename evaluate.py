"""Score a trained policy against baselines on held-out scenarios.

    python3 evaluate.py --episodes 400

Reports mission outcome and, separately, whether the package actually behaves
as a team — arrival tightness, survivors, and time spent stacked inside one
battery's envelope. A policy can win by luck; those three say whether it is
coordinating or just flying well.

Scenario seeds start well above the ones the trainer uses, so nothing here was
seen during training.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from env import MAX_TEAM, MissionEnv, Scenario, random_scenario
from ppo import load_policy

HELD_OUT_SEED_BASE = 1_000_000


def policy_actions(model, norm, obs: np.ndarray) -> np.ndarray:
    """Deterministic actions for every slot — the distribution mean."""
    with torch.no_grad():
        return model.actor(torch.as_tensor(norm(obs))).numpy()


def run_episode(scenario: Scenario, act_fn, seed: int) -> dict:
    env = MissionEnv(random.Random(seed))
    obs = env.reset(scenario)
    stacked_steps = 0

    while True:
        obs, _, _, info = env.step(act_fn(obs))
        stacked_steps += 1 if env._stacked_pairs() else 0
        if info:
            info = dict(info)
            info["stacked_steps"] = stacked_steps
            info["steps"] = env.steps
            return info


def evaluate(model, norm, episodes: int, team_size: int | None) -> dict:
    strategies = {
        "PPO policy": lambda obs: policy_actions(model, norm, obs),
        "straight-in": lambda obs: np.zeros((MAX_TEAM, 2)),
        "random": lambda obs: np.random.uniform(-1.0, 1.0, size=(MAX_TEAM, 2)),
    }
    results = {name: [] for name in strategies}

    for i in range(episodes):
        seed = HELD_OUT_SEED_BASE + i
        scenario = random_scenario(random.Random(seed), team_size=team_size)
        for name, fn in strategies.items():
            results[name].append(run_episode(scenario, fn, seed))

    return results


def summarise(name: str, runs: list[dict]) -> str:
    n = len(runs)
    wins = [r for r in runs if r["outcome"] == "target_destroyed"]
    destroyed = len(wins) / n
    attrited = sum(r["outcome"] == "team_attrited" for r in runs) / n
    uncoord = sum(r["outcome"] == "uncoordinated" for r in runs) / n
    fuel = sum(r["outcome"] == "out_of_fuel" for r in runs) / n

    # Teaming metrics are only meaningful on missions that succeeded — arrival
    # spread is undefined when nothing arrived.
    coord = np.mean([w["coordination"] for w in wins]) if wins else float("nan")
    spread = np.mean([w["tot_spread"] for w in wins]) if wins else float("nan")
    survivors = np.mean([w["survivors"] for w in wins]) if wins else float("nan")
    stacked = np.mean([r["stacked_steps"] / max(r["steps"], 1) for r in runs])

    return (
        f"  {name:<13} {destroyed:>8.1%} {attrited:>9.1%} {uncoord:>13.1%} {fuel:>9.1%}   "
        f"{coord:>6.2f} {spread:>8.1f} {survivors:>10.2f} {stacked:>9.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", type=Path, default=Path("policy.pt"))
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument(
        "--team-size", type=int, default=None, help="fix the package size (default: random)"
    )
    args = parser.parse_args()

    np.random.seed(0)
    model, norm = load_policy(args.policy)
    results = evaluate(model, norm, args.episodes, args.team_size)

    size = args.team_size or "random 2-4"
    print(f"\n  {args.episodes} held-out scenarios, team size {size}, "
          f"identical for every strategy\n")
    print(f"  {'strategy':<13} {'destroyed':>8} {'attrited':>9} {'uncoordinated':>13} {'no fuel':>9}   "
          f"{'coord':>6} {'spread':>8} {'survivors':>10} {'stacked':>9}")
    for name, runs in results.items():
        print(summarise(name, runs))
    print("\n  coord 1.0 = simultaneous arrivals, 0.0 = a full window apart")
    print("  spread    = minutes between first and last arrival")
    print("  stacked   = share of steps with two assets inside one envelope")


if __name__ == "__main__":
    main()