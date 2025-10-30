import torch
import torch.nn as nn
import torch.optim as optim

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

        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )
        # pretty basic stuff we may have to train this better or maybe even use a CNN but maybe stick to an NN if we're going towards KANs?
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
        x = self.fc(x)
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        value = self.critic(x)
        return mean, std, value

    def get_action(self, obs):
        mean, std, _ = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.tanh(action)      
        action = 0.5 * (action + 1.0)     
        log_prob = dist.log_prob(action).sum(-1) 
        return action, log_prob

