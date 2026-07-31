"""Proximal Policy Optimization for the strike-mission environment.

A standard clipped-surrogate PPO: Gaussian policy with a state-independent
log-std, GAE(lambda) advantages, several epochs of minibatch updates per batch
of rollouts, and an entropy bonus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from env import ACT_DIM, OBS_DIM, VecEnv


@dataclass
class PPOConfig:
    total_steps: int = 400_000
    num_envs: int = 16
    rollout_steps: int = 256
    epochs: int = 10
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    hidden: int = 64
    seed: int = 0
    device: str = "cpu"

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps


class RunningNorm:
    """Welford running mean/variance, used to standardise observations.

    Feature scales here differ by orders of magnitude, and an unnormalised
    input makes the value function much harder to fit.
    """

    def __init__(self, dim: int):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean, batch_var, batch_count = x.mean(0), x.var(0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count

        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0).astype(
            np.float32
        )

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean, self.var, self.count = state["mean"], state["var"], state["count"]


def _layer(in_dim: int, out_dim: int, std: float = np.sqrt(2)) -> nn.Linear:
    """Orthogonal init — the usual PPO default; plain init trains noticeably worse."""
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """Separate policy and value trunks with a shared observation format."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
            # Small final-layer gain keeps the initial policy near-deterministic
            # and stops the first updates from thrashing.
            _layer(hidden, act_dim, std=0.01),
        )
        self.critic = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
            _layer(hidden, 1, std=1.0),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, self.log_std.exp())

    def act(self, obs: torch.Tensor):
        dist = self.distribution(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), self.value(obs)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        dist = self.distribution(obs)
        return (
            dist.log_prob(action).sum(-1),
            dist.entropy().sum(-1),
            self.value(obs),
        )


@dataclass
class TrainStats:
    """Per-iteration diagnostics, kept so training progress can be asserted on."""

    steps: int = 0
    outcomes: dict = field(default_factory=dict)
    mean_return: float = 0.0
    success_rate: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    lam: float,
):
    """Generalised advantage estimation over a (T, N) rollout.

    Every termination here is a real end state — destroyed, shot down, or out
    of fuel — so no value is bootstrapped across a `done`.
    """
    T = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros_like(last_value)

    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae

    return advantages, advantages + values


class PPOTrainer:
    def __init__(self, config: PPOConfig):
        self.cfg = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.device = torch.device(config.device)
        self.envs = VecEnv(config.num_envs, seed=config.seed)
        self.model = ActorCritic(hidden=config.hidden).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, eps=1e-5
        )
        self.obs_norm = RunningNorm(OBS_DIM)
        self.history: list[TrainStats] = []

    def train(self, log_every: int = 5, on_log=None) -> list[TrainStats]:
        cfg = self.cfg
        raw_obs = self.envs.reset()
        self.obs_norm.update(raw_obs)
        obs = self.obs_norm(raw_obs)

        iterations = cfg.total_steps // cfg.batch_size
        episode_returns = np.zeros(cfg.num_envs, dtype=np.float32)
        recent_returns: list[float] = []
        recent_outcomes: list[str] = []
        step_count = 0

        for it in range(1, iterations + 1):
            # Linear learning-rate decay, standard for PPO stability late on.
            frac = 1.0 - (it - 1) / iterations
            for group in self.optimizer.param_groups:
                group["lr"] = frac * cfg.learning_rate

            obs_buf = np.zeros((cfg.rollout_steps, cfg.num_envs, OBS_DIM), dtype=np.float32)
            act_buf = np.zeros((cfg.rollout_steps, cfg.num_envs, ACT_DIM), dtype=np.float32)
            logp_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
            rew_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
            done_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
            val_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)

            for t in range(cfg.rollout_steps):
                obs_buf[t] = obs
                with torch.no_grad():
                    tensor_obs = torch.as_tensor(obs, device=self.device)
                    action, logp, value = self.model.act(tensor_obs)

                act_buf[t] = action.cpu().numpy()
                logp_buf[t] = logp.cpu().numpy()
                val_buf[t] = value.cpu().numpy()

                raw_obs, reward, done, infos = self.envs.step(act_buf[t])
                self.obs_norm.update(raw_obs)
                obs = self.obs_norm(raw_obs)

                rew_buf[t] = reward
                done_buf[t] = done
                episode_returns += reward
                step_count += cfg.num_envs

                for i, d in enumerate(done):
                    if d:
                        recent_returns.append(float(episode_returns[i]))
                        episode_returns[i] = 0.0
                for info in infos:
                    recent_outcomes.append(info["outcome"])

            with torch.no_grad():
                last_value = (
                    self.model.value(torch.as_tensor(obs, device=self.device))
                    .cpu()
                    .numpy()
                )

            advantages, returns = compute_gae(
                rew_buf, val_buf, done_buf, last_value, cfg.gamma, cfg.gae_lambda
            )
            stats = self._update(obs_buf, act_buf, logp_buf, advantages, returns)

            stats.steps = step_count
            window_out = recent_outcomes[-400:]
            stats.outcomes = {o: window_out.count(o) for o in set(window_out)}
            stats.success_rate = (
                window_out.count("target_destroyed") / len(window_out) if window_out else 0.0
            )
            stats.mean_return = float(np.mean(recent_returns[-200:])) if recent_returns else 0.0
            self.history.append(stats)

            if on_log and (it % log_every == 0 or it == iterations):
                on_log(it, iterations, stats)

        return self.history

    def _update(self, obs_buf, act_buf, logp_buf, advantages, returns) -> TrainStats:
        cfg = self.cfg
        b_obs = torch.as_tensor(obs_buf.reshape(-1, OBS_DIM), device=self.device)
        b_act = torch.as_tensor(act_buf.reshape(-1, ACT_DIM), device=self.device)
        b_logp = torch.as_tensor(logp_buf.reshape(-1), device=self.device)
        b_adv = torch.as_tensor(advantages.reshape(-1), device=self.device)
        b_ret = torch.as_tensor(returns.reshape(-1), device=self.device)

        indices = np.arange(cfg.batch_size)
        minibatch_size = cfg.batch_size // cfg.minibatches
        losses = {"policy": [], "value": [], "entropy": []}

        for _ in range(cfg.epochs):
            np.random.shuffle(indices)
            for start in range(0, cfg.batch_size, minibatch_size):
                idx = indices[start : start + minibatch_size]
                new_logp, entropy, value = self.model.evaluate(b_obs[idx], b_act[idx])

                ratio = (new_logp - b_logp[idx]).exp()
                # Normalise per minibatch, which is what the reference
                # implementation does and is more stable than whole-batch.
                adv = b_adv[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                unclipped = -adv * ratio
                clipped = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                policy_loss = torch.max(unclipped, clipped).mean()
                value_loss = 0.5 * ((value - b_ret[idx]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                losses["policy"].append(policy_loss.item())
                losses["value"].append(value_loss.item())
                losses["entropy"].append(entropy_loss.item())

        return TrainStats(
            policy_loss=float(np.mean(losses["policy"])),
            value_loss=float(np.mean(losses["value"])),
            entropy=float(np.mean(losses["entropy"])),
        )

    def save(self, path: Path) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "obs_norm": self.obs_norm.state_dict(),
                "hidden": self.cfg.hidden,
            },
            path,
        )


def load_policy(path: Path, device: str = "cpu") -> tuple[ActorCritic, RunningNorm]:
    """Restore a trained policy and the observation statistics it expects."""
    ckpt = torch.load(path, map_location=device)
    model = ActorCritic(hidden=ckpt.get("hidden", 64)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    norm = RunningNorm(OBS_DIM)
    norm.load_state_dict(ckpt["obs_norm"])
    return model, norm