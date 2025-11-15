import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from utils import pick_device

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class MAPPO:
    """
    Pretty minimal MAPPO collector for setup.
    """

    def __init__(
        self, env, gamma=0.99, lr=3e-4, device=None, policy=None, use_wandb=True
    ):
        self.env = env
        self.gamma = gamma
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if use_wandb and not WANDB_AVAILABLE:
            logging.warning(
                "wandb requested but not available. Install with: pip install wandb"
            )
        self.device = pick_device()

        self.policy = (
            policy
            if policy is not None
            else MLPPolicyNet(env.obs_dim, env.action_dim).to(self.device)
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.trajectory = []

        self.logger = logging.getLogger("MAPPO")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
        self.logger.info(f"MAPPO initialized on device: {self.device}")

    def collect_trajectories(self, steps=10):
        """
        I basically collect episodic experience to inspect policy network outputs here
        """
        obs, _ = self.env.reset()
        obs = {
            k: torch.tensor(v, dtype=torch.float32).to(self.device)
            for k, v in obs.items()
        }

        self.logger.info(f"Starting trajectory collection for {steps} steps...")

        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0.0
        current_episode_length = 0

        for step in range(steps):
            actions, log_probs, values = {}, {}, {}

            for agent in self.env.agents:
                a, logp = self.policy.get_action(obs[agent])
                _, _, v = self.policy.forward(obs[agent])

                actions[agent] = a.cpu().detach().numpy()
                log_probs[agent] = logp.cpu().detach().item()
                values[agent] = v.cpu().detach().item()

            next_obs, rewards, terminations, truncations, _ = self.env.step(actions)

            # Track episode statistics
            step_reward = sum(rewards.values()) / len(
                rewards
            )  # Average reward across agents
            current_episode_reward += step_reward
            current_episode_length += 1

            for agent in self.env.agents:
                self.trajectory.append(
                    {
                        "agent": agent,
                        "obs": obs[agent].cpu().numpy(),
                        "action": actions[agent],
                        "log_prob": log_probs[agent],
                        "value": values[agent],
                        "reward": rewards[agent],
                        "done": terminations[agent] or truncations[agent],
                    }
                )

            done = all(terminations.values()) or all(truncations.values())
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0.0
                current_episode_length = 0
                next_obs, _ = self.env.reset()

            obs = {
                k: torch.tensor(v, dtype=torch.float32).to(self.device)
                for k, v in next_obs.items()
            }

        # Store episode stats for logging
        self.episode_rewards = episode_rewards
        self.episode_lengths = episode_lengths

        self.logger.info("Trajectory collection complete. Sample of collected steps:")

        for step_data in self.trajectory[: min(5, len(self.trajectory))]:
            self.logger.info(step_data)

        return self.trajectory

    def _compute_returns_and_advantages(self):
        """
        Discounted return, advantages = returns - values.
        """
        rewards = [x["reward"] for x in self.trajectory]
        dones = [x["done"] for x in self.trajectory]
        values = [x["value"] for x in self.trajectory]

        R = 0.0
        returns = [0.0] * len(rewards)
        for t in reversed(range(len(rewards))):
            if dones[t]:
                R = 0.0
            R = rewards[t] + self.gamma * R
            returns[t] = R

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values, dtype=torch.float32, device=self.device)
        adv_t = returns_t - values_t
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        return returns_t, adv_t

    def _tensorize_batch(self):
        obs = torch.tensor(
            np.stack([x["obs"] for x in self.trajectory], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            np.stack([x["action"] for x in self.trajectory], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        logp = torch.tensor(
            [x["log_prob"] for x in self.trajectory],
            dtype=torch.float32,
            device=self.device,
        )
        return obs, actions, logp

    def train(
        self,
        epochs=100,
        steps_per_epoch=512,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    ):
        """Train the shared policy/value network using a minimal REINFORCE-with-baseline update.

        Args:
            epochs (int, optional): Number of training epochs. Defaults to 100.
            steps_per_epoch (int, optional): Environment steps to collect per epoch. Defaults to 512.
            value_coef (float, optional): Weight for the value-function regression loss. Defaults to 0.5.
            entropy_coef (float, optional): Weight for the entropy bonus to encourage exploration. Defaults to 0.01.
            max_grad_norm (float, optional): Gradient clipping norm to stabilize updates. Defaults to 0.5.
        Returns:
            None
        """

        # Starting with a minimal REINFORCE-with-baseline update,
        # might need to update to a PPO based approach later?

        for ep in range(1, epochs + 1):
            self.trajectory = []
            self.collect_trajectories(steps=steps_per_epoch)

            returns_t, adv_t = self._compute_returns_and_advantages()
            obs_t, actions_t, logp_old_t = self._tensorize_batch()

            mean, std, values_now = self.policy.forward(obs_t)
            normal = torch.distributions.Normal(mean, std)
            entropy = normal.entropy().sum(-1).mean()

            policy_loss = -(logp_old_t * adv_t).mean()
            value_loss = 0.5 * (returns_t - values_now.squeeze(-1)).pow(2).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), max_grad_norm
            )
            self.optimizer.step()

            # Compute metrics
            avg_reward = float(np.mean([x["reward"] for x in self.trajectory]))
            avg_len = float(len(self.trajectory) / max(1, len(self.env.agents)))

            # Episode statistics
            avg_episode_reward = (
                float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
            )
            avg_episode_length = (
                float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0
            )
            max_episode_reward = (
                float(np.max(self.episode_rewards)) if self.episode_rewards else 0.0
            )
            min_episode_reward = (
                float(np.min(self.episode_rewards)) if self.episode_rewards else 0.0
            )

            # Advantage statistics
            adv_mean = float(adv_t.mean().item())
            adv_std = float(adv_t.std().item())

            # Value statistics
            value_mean = float(values_now.mean().item())
            value_std = float(values_now.std().item())

            # Log to console
            self.logger.info(
                f"[epoch {ep}] loss={loss.item():.3f} "
                f"pi={policy_loss.item():.3f} vf={value_loss.item():.3f} "
                f"H={entropy.item():.3f} R/step={avg_reward:.3f} steps~{avg_len:.0f}"
            )

            # Log to wandb
            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": ep,
                        "loss/total": loss.item(),
                        "loss/policy": policy_loss.item(),
                        "loss/value": value_loss.item(),
                        "loss/entropy": entropy.item(),
                        "metrics/avg_reward_per_step": avg_reward,
                        "metrics/avg_episode_reward": avg_episode_reward,
                        "metrics/max_episode_reward": max_episode_reward,
                        "metrics/min_episode_reward": min_episode_reward,
                        "metrics/avg_episode_length": avg_episode_length,
                        "metrics/avg_trajectory_length": avg_len,
                        "metrics/num_episodes": len(self.episode_rewards),
                        "advantage/mean": adv_mean,
                        "advantage/std": adv_std,
                        "value/mean": value_mean,
                        "value/std": value_std,
                        "training/grad_norm": grad_norm.item(),
                    }
                )

            self.trajectory = []
