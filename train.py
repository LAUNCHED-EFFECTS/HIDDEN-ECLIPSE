"""Train the PPO mission planner.

    python3 train.py                      # default 400k steps
    python3 train.py --steps 100000       # shorter run
    python3 train.py --output policy.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ppo import PPOConfig, PPOTrainer, TrainStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=400_000, help="total environment steps")
    parser.add_argument("--envs", type=int, default=16, help="parallel environments")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--output", type=Path, default=Path("policy.pt"), help="checkpoint path"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PPOConfig(
        total_steps=args.steps, num_envs=args.envs, seed=args.seed, learning_rate=args.lr
    )
    trainer = PPOTrainer(cfg)

    print(f"  training PPO for {cfg.total_steps:,} steps "
          f"({cfg.num_envs} envs x {cfg.rollout_steps} steps per batch)")
    print(f"  {'iter':>6}  {'steps':>9}  {'return':>8}  {'success':>8}  "
          f"{'shot down':>10}  {'no fuel':>8}  {'entropy':>8}")

    started = time.time()

    def report(it: int, total: int, s: TrainStats) -> None:
        out = s.outcomes
        n = max(sum(out.values()), 1)
        print(
            f"  {it:>4}/{total:<3}  {s.steps:>9,}  {s.mean_return:>8.1f}  "
            f"{s.success_rate:>7.1%}  {out.get('shot_down', 0)/n:>9.1%}  "
            f"{out.get('out_of_fuel', 0)/n:>7.1%}  {s.entropy:>8.3f}"
        )

    history = trainer.train(on_log=report)
    trainer.save(args.output)

    elapsed = time.time() - started
    final = history[-1]
    print(f"\n  done in {elapsed:,.0f}s — final success rate {final.success_rate:.1%}")
    print(f"  checkpoint written to {args.output.resolve()}")


if __name__ == "__main__":
    main()