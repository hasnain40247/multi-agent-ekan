import torch
import torch.nn as nn
from ekan import EKAN
from utils import pick_device

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class EKANPolicyNet(nn.Module):
    """
    Policy/value network using EKAN as the backbone.

    I styled this like the KANPolicyNet, but instead of specifying plain dimensions
    for the backbone, you pass in representation objects and the group for EKAN.
    """

    def __init__(
        self,
        rep_in,
        rep_out,
        group,
        action_dim,
        *,
        ekan_kwargs=None,
    ):
        """
        Args:
            rep_in:   input representation (must satisfy rep_in.size() == obs_dim)
            rep_out:  output/latent representation for the backbone
            group:    group object used by EKAN (same one you pass to EKAN)
            action_dim (int): dimension of the (continuous) action space
            ekan_kwargs (dict|None): extra kwargs forwarded to EKAN.__init__,
                e.g. width, grid, grid_range, device, seed, classify, ...
        """
        super().__init__()
        ekan_kwargs = ekan_kwargs or {}
        ekan_kwargs.setdefault("device", pick_device())

        # EKAN backbone
        self.backbone = EKAN(rep_in, rep_out, group, **ekan_kwargs)

        hidden_dim = rep_out.size()

        # Heads are identical to the KANPolicyNet version
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

        # For sanity checks later
        self._obs_dim = rep_in.size()
        self._hidden_dim = hidden_dim
        self._action_dim = action_dim

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the EKAN backbone + actor/critic heads.

        Args:
            x (torch.Tensor): observation tensor of shape
                              [batch_size, obs_dim] or [obs_dim].

        Returns:
            mean (torch.Tensor): Mean of the action distribution for each action dimension.
            std (torch.Tensor): Standard deviation of the action distribution.
            value (torch.Tensor): Predicted state value (V(s)).
        """
        # Dimension check
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        assert (
            x.size(-1) == self._obs_dim
        ), f"Expected obs_dim={self._obs_dim}, got {x.size(-1)}"

        # EKAN backbone: returns features of size rep_out.size() == hidden_dim
        h = self.backbone(x)

        # Actor mean
        mean = self.actor_mean(h)

        # Log std is a free parameter per action dim (as in KANPolicyNet)
        logstd = torch.clamp(self.actor_logstd, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(logstd).expand_as(mean)

        # Critic value
        value = self.critic(h)

        if single:
            mean = mean.squeeze(0)
            std = std.squeeze(0)
            value = value.squeeze(0)

        return mean, std, value.squeeze(-1)

    @staticmethod
    def _tanh_log_prob(raw, mean, std, eps=1e-6):
        """
        for increased stability while training
        """
        normal = torch.distributions.Normal(mean, std)
        logp_raw = normal.log_prob(raw).sum(-1)
        log_det = torch.log(1 - torch.tanh(raw).pow(2) + eps).sum(-1)
        return logp_raw - log_det

    def get_action(self, obs: torch.Tensor):
        mean, std, _ = self.forward(obs)

        single = obs.dim() == 1
        if single:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)

        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        action = 0.5 * (action + 1.0)

        log_prob = self._tanh_log_prob(raw_action, mean, std)
        k_log2 = mean.shape[-1] * torch.log(torch.tensor(2.0, device=mean.device))
        log_prob = log_prob + k_log2

        if single:
            return action.squeeze(0), log_prob.squeeze(0)

        return action, log_prob
