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
        self.device = (
            torch.device(device) if device is not None else torch.device(pick_device())
        )

        if policy is None:
            raise ValueError(
                "MAPPO requires a `policy` instance. "
                "Pass your MLP/KAN policy (obs_dim, action_dim compatible)."
            )
        self.policy = policy.to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # rollout buffers for efficiency while training
        self._obs_buf: List[torch.Tensor] = []
        self._act_buf: List[torch.Tensor] = []
        self._logp_buf: List[torch.Tensor] = []
        self._val_buf: List[torch.Tensor] = []
        self._rew_buf: List[float] = []
        self._done_buf: List[bool] = []

        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

        self.logger = logging.getLogger("MAPPO")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
        self.logger.info(f"MAPPO initialized on device: {self.device}")

    @staticmethod
    def _tanh_log_prob(
        u: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6
    ):
        normal = torch.distributions.Normal(mean, std)
        logp_u = normal.log_prob(u).sum(-1)
        log_det = torch.log1p(-torch.tanh(u).pow(2) + eps).sum(-1)
        return logp_u - log_det

    def _sample_actions(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Sample actions with tanh-squash and map to (0,1):
        Returns (actions, log_prob)
        """
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()
        z = torch.tanh(u)
        a = 0.5 * (z + 1.0)

        logp = self._tanh_log_prob(u, mean, std)
        k_log2 = mean.shape[-1] * torch.log(
            torch.tensor(2.0, device=mean.device, dtype=mean.dtype)
        )
        logp = logp + k_log2
        return a, logp

    def _reset_buffers(self) -> None:
        self._obs_buf.clear()
        self._act_buf.clear()
        self._logp_buf.clear()
        self._val_buf.clear()
        self._rew_buf.clear()
        self._done_buf.clear()

    def collect_trajectories(self, steps: int = 10) -> None:
        """
        Vectorized per-step rollout across agents.
        Stores buffers on self for training.
        """
        self._reset_buffers()
        self.episode_rewards.clear()
        self.episode_lengths.clear()

        obs_dict, _ = self.env.reset()
        # keep observations as device tensors
        obs = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in obs_dict.items()
        }
        self.logger.info(f"Starting trajectory collection for {steps} steps...")

        ep_ret = 0.0
        ep_len = 0
        
        # batch process all agents
        for _ in range(steps):
            agent_ids = list(self.env.agents)
            obs_batch = torch.stack([obs[a] for a in agent_ids], dim=0)  # [N, obs_dim]

            with torch.no_grad():
                mean, std, values = self.policy(obs_batch)
                actions, logp = self._sample_actions(mean, std)

            # scatter back per agent
            action_dict: Dict[str, np.ndarray] = {
                agent_ids[i]: actions[i].detach().cpu().numpy()
                for i in range(len(agent_ids))
            }

            # Step environment
            next_obs, rewards, terminations, truncations, _ = self.env.step(action_dict)

            # Average multi-agent reward (you can change this aggregation if desired)
            step_rew = float(sum(rewards.values()) / max(1, len(rewards)))
            ep_ret += step_rew
            ep_len += 1

            # Store per-agent transitions to buffers
            for i, aid in enumerate(agent_ids):
                self._obs_buf.append(obs[aid].detach())
                self._act_buf.append(actions[i].detach())
                self._logp_buf.append(logp[i].detach())
                self._val_buf.append(values[i].detach().squeeze(-1))
                self._rew_buf.append(float(rewards[aid]))
                self._done_buf.append(bool(terminations[aid] or truncations[aid]))

            # Episode end?
            done_all = all(terminations.values()) or all(truncations.values())
            if done_all:
                self.episode_rewards.append(ep_ret)
                self.episode_lengths.append(ep_len)
                ep_ret, ep_len = 0.0, 0
                next_obs, _ = self.env.reset()

            obs = {
                k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
                for k, v in next_obs.items()
            }

        self.logger.info(
            f"Rollout done: steps={steps}, "
            f"episodes={len(self.episode_rewards)}, "
            f"avg_ep_ret={np.mean(self.episode_rewards) if self.episode_rewards else 0:.3f}, "
            f"avg_ep_len={np.mean(self.episode_lengths) if self.episode_lengths else 0:.1f}"
        )

    def _compute_returns_and_advantages(self):
        """
        compute discounted returns; advantages = returns - values.
        """
        rewards = np.asarray(self._rew_buf, dtype=np.float32)
        dones = np.asarray(self._done_buf, dtype=np.bool_)
        values = torch.stack(self._val_buf, dim=0).to(self.device) 

        R = 0.0
        rets = np.zeros_like(rewards, dtype=np.float32)
        for t in range(len(rewards) - 1, -1, -1):
            if dones[t]:
                R = 0.0
            R = rewards[t] + self.gamma * R
            rets[t] = R

        returns_t = torch.as_tensor(rets, dtype=torch.float32, device=self.device)
        adv_t = returns_t - values
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        return returns_t, adv_t

    def train(
        self,
        epochs: int = 100,
        steps_per_epoch: int = 512,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ) -> None:
        """
        Minimal REINFORCE-with-baseline:
            loss = -(logp * adv) + value_coef * MSE(V, ret) - entropy_coef * entropy
        Uses stored rollout log-probabilities (no importance ratios).
        """
        for ep in range(1, epochs + 1):
            # Collect rollout 
            self.collect_trajectories(steps=steps_per_epoch)

            #  Prepare batch 
            obs_t = torch.stack(self._obs_buf, dim=0)
            act_t = torch.stack(self._act_buf, dim=0) 
            logp_old_t = torch.stack(self._logp_buf, dim=0)  # [T]
            returns_t, adv_t = self._compute_returns_and_advantages()

            #  Forward current policy for entropy + value loss 
            mean, std, values_now = self.policy(obs_t) 
            normal = torch.distributions.Normal(mean, std)
            entropy = normal.entropy().sum(-1).mean()

            # REINFORCE
            policy_loss = -(logp_old_t * adv_t).mean()
            value_loss = 0.5 * (returns_t - values_now.squeeze(-1)).pow(2).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), max_grad_norm
            )
            self.optimizer.step()

            avg_reward_per_step = (
                float(np.mean(self._rew_buf)) if self._rew_buf else 0.0
            )
            avg_traj_len = float(len(self._rew_buf) / max(1, len(self.env.agents)))
            avg_ep_ret = (
                float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
            )
            max_ep_ret = (
                float(np.max(self.episode_rewards)) if self.episode_rewards else 0.0
            )
            min_ep_ret = (
                float(np.min(self.episode_rewards)) if self.episode_rewards else 0.0
            )
            avg_ep_len = (
                float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0
            )

            self.logger.info(
                f"[epoch {ep}] "
                f"loss={loss.item():.3f} pi={policy_loss.item():.3f} vf={value_loss.item():.3f} "
                f"H={entropy.item():.3f} grad={grad_norm.item():.3f} | "
                f"R/step={avg_reward_per_step:.3f} traj_len~{avg_traj_len:.0f} "
                f"ep_R(avg/max/min)=({avg_ep_ret:.2f}/{max_ep_ret:.2f}/{min_ep_ret:.2f}) ep_len={avg_ep_len:.1f}"
            )

            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": ep,
                        "loss/total": loss.item(),
                        "loss/policy": policy_loss.item(),
                        "loss/value": value_loss.item(),
                        "entropy": entropy.item(),
                        "grad_norm": grad_norm.item(),
                        "metrics/avg_reward_per_step": avg_reward_per_step,
                        "metrics/avg_episode_reward": avg_ep_ret,
                        "metrics/max_episode_reward": max_ep_ret,
                        "metrics/min_episode_reward": min_ep_ret,
                        "metrics/avg_episode_length": avg_ep_len,
                        "metrics/avg_trajectory_length": avg_traj_len,
                        "metrics/num_episodes": len(self.episode_rewards),
                    }
                )
            self._reset_buffers()
