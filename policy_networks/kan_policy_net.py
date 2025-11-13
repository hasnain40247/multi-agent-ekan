import torch
import torch.nn as nn
from kan import *

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class KANPolicyNet(nn.Module):
    """
    Baseline version using normal KANs, written as a drop in replacement for the MLP,
    but using KANs
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=64, *, grid=16, k=3, kw=None):
        """
        Args:
            obs_dim (int): observation dim
            action_dim (int): action dim
            hidden_dim (int): width of the backbone output fed into heads
            grid (int): pykan grid size (typical values: 10-32)
            k (int): spline order/degree-like hyperparameter
            kw (dict|None): extra kwargs passed to pykan.KAN constructor
        """
        super().__init__()
        kw = kw or {}

        self.backbone = KAN(
            width=[obs_dim, hidden_dim],
            grid=grid,
            k=k,
            **kw,
        )

        # Heads are identical the MLP
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input observation tensor of shape [batch_size, obs_dim].

        Returns:
            mean (torch.Tensor): Mean of the action distribution for each action dimension.
            std (torch.Tensor): Standard deviation of the action distribution.
            value (torch.Tensor): Predicted state value (V(s)).
        """
        # dimension check
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        h = self.backbone(x)
        mean = self.actor_mean(h)

        logstd = torch.clamp(self.actor_logstd, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(logstd).expand_as(mean)
        value = self.critic(h)

        if single:
            mean = mean.squeeze(0)
            std = std.squeeze(0)
            value = value.squeeze(0)
        return mean, std, value

    @staticmethod
    def _tanh_log_prob(raw, mean, std, eps=1e-6):
        """
        for increased stability while training
        """
        normal = torch.distributions.Normal(mean, std)
        logp_raw = normal.log_prob(raw).sum(-1)
        log_det = torch.log(1 - torch.tanh(raw).pow(2) + eps).sum(-1)
        return logp_raw - log_det

    def get_action(self, obs):
        mean, std, _ = self.forward(obs)

        # check dimension
        single = obs.dim() == 1
        if single:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)

        dist = torch.distributions.Normal(mean, std)
        r_action = dist.rsample()
        action = torch.tanh(r_action)
        action = 0.5 * (action + 1.0)

        log_prob = self._tanh_log_prob(r_action, mean, std)
        k_log2 = mean.shape[-1] * torch.log(torch.tensor(2.0, device=mean.device))
        log_prob = log_prob + k_log2

        if single:
            return action.squeeze(0), log_prob.squeeze(0)
        return action, log_prob
