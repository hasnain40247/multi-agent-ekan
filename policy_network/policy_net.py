import torch
import torch.nn as nn
import torch.optim as optim

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class PolicyNet(nn.Module):
    """
    Kinda simple Actor-Critic network. I was just trying to have a shared backbone
    that feeds into both the policy and value function.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        """

        Args:
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int, optional): Number of hidden units in the shared layer. Defaults to 64.
        """
        super().__init__()

        self.fc = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
        # pretty basic stuff we may have to train this better or maybe even
        # use a CNN but maybe stick to an NN if we're going towards KANs?
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

        x = self.fc(x)
        mean = self.actor_mean(x)

        logstd = torch.clamp(self.actor_logstd, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(logstd)
        value = self.critic(x)

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
