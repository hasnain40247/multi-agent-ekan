"""
Neural Network architectures for DDPG
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_weights(module, output_layer=None, init_w=3e-3):
    """Initialize network weights using standard PyTorch initialization
    
    Args:
        module: PyTorch module to initialize
        output_layer: The output layer that should use uniform initialization
        init_w: Weight initialization range for output layer
    """
    if isinstance(module, nn.Linear):
        if module == output_layer:  # Output layer
            # Use uniform initialization for the final layer
            nn.init.uniform_(module.weight, -init_w, init_w)
            nn.init.uniform_(module.bias, -init_w, init_w)
        else:  # Hidden layers
            # Use Kaiming initialization for ReLU layers
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(module.bias)

def _init_weights_approx(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)  # Stable gain
        nn.init.constant_(module.bias, 0)
    if hasattr(module, 'fc3') and module is module.fc3:  # Final layer
        nn.init.uniform_(module.weight, -3e-3, 3e-3)
        nn.init.constant_(module.bias, 0)



import torch
import torch.nn as nn
import torch.nn.functional as F




class Actor(nn.Module):
    """Actor (Policy) Model"""
    
    def __init__(self, state_size, action_size, hidden_sizes=(64, 64), init_w=3e-3, 
                 action_low=-1.0, action_high=1.0):
        """
        Initialize parameters and build model.

        [   self_vel, [(vx, vy)]
            self_pos, (x, y)
            landmark_rel_positions, [(x1, y1),(x2, y2),(x3, y3)]
            other_agent_rel_positions, [(x1, y1),(x2, y2)]
            communication [a,b,c,d]
        ] 

        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_sizes (tuple): Sizes of hidden layers
            init_w (float): Final layer weight initialization
            action_low (float or array): Lower bound of the action space (default: -1.0)
            action_high (float or array): Upper bound of the action space (default: 1.0)
        """
        super(Actor, self).__init__()
        
        self.action_low = action_low
        self.action_high = action_high
        self.scale = (action_high - action_low) / 2.0
        self.bias = (action_high + action_low) / 2.0
        print("action size")
        print(action_size)
        # input()
        
        # Build the network
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 2)
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, self.fc3, init_w))
     
        
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions"""
        # print("state:")
        # print(state)
        # print()
        # print("Decoding The State")
        # self_vel = state[0:2]                  # shape (2,)
        # self_pos = state[2:4]                  # shape (2,)

        # landmarks = state[4:10].reshape(3, 2)  # 3 landmarks × 2 dims
        # other_agents = state[10:14].reshape(2, 2)

        # print("Agent Self Velocities: ")
        # print(self_vel)
        # print()
        # print("Agent Self Position: ")
        # print(self_pos)
        # print()
        # print("Relative Landmark Positions: ")
     
        # print(landmarks)
        # print()
        # print("Relative Agent Positions: ")
        # print(other_agents)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output is in range [-1, 1]
        
    
        return self._scale_action(x)
    
    def _scale_action(self, action):
        """Scale action from [-1, 1] to [action_low, action_high]"""
        return self.scale * action  + self.bias

class Critic(nn.Module):
    """Critic (Value) Model"""
    
    def __init__(self, state_size, action_size, hidden_sizes=(64, 64), init_w=3e-3):
        """
        Initialize parameters and build model.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_sizes (tuple): Sizes of hidden layers
            init_w (float): Final layer weight initialization
        """
        super(Critic, self).__init__()

        # print("state size")
        # print(state_size)
        # print("action size")
        # print(action_size)
        # input()
        
        # Build the network - concatenate state and action at the first layer
        self.fc1 = nn.Linear(state_size + action_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, self.fc3, init_w))
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values"""
        # Concatenate state and action at the first layer
        # print("state that the critic takes in ")
        # print(state)
        # print(state.shape)
        # print("actions that the critic takes in")
        # print(action)
        # print(action.shape)

       
        # input()
        x = torch.cat((state, action), dim=1)

   
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
   
        return self.fc3(x)  # Output is Q-value 

import emlp.nn.pytorch as emlp
from emlp.reps import T
from emlp.groups import SO


# class Critic(nn.Module):
#     """Centralized Critic (Value) Model - SO(2) Invariant"""
    
#     def __init__(self, state_size, action_size, hidden_sizes=(64, 64), init_w=3e-3):
#         """
#         Initialize parameters and build model.
        
#         Args:
#             state_size (int): Dimension of full concatenated state (54 for 3 agents)
#             action_size (int): Dimension of full concatenated actions (15 for 3 agents)
#         """
#         super(Critic, self).__init__()
        
#         self.group = SO(2)
        
#         # States: 3 agents × 18D each = 54D total
#         # Per agent breakdown (18D):
#         #   - velocity: 2D → T(1)
#         #   - position: 2D → T(1)  
#         #   - 3 landmark relative positions: 6D → 3*T(1)
#         #   - 2 other agent relative positions: 4D → 2*T(1)
#         #   - communication: 4D → 2*T(1) (assuming 2 other agents × 2D comm)
#         # Total per agent: 9*T(1) = 18D
#         # Total for 3 agents: 27*T(1) = 54D
        
#         # Actions: 3 agents × 5D each = 15D total
#         # Per agent: [unused, left, right, down, up]
#         #   - unused: 1D → T(0) (scalar)
#         #   - left/right: 2D → T(1) (vector)
#         #   - down/up: 2D → T(1) (vector)
#         # Total per agent: 1*T(0) + 2*T(1) = 5D
#         # Total for 3 agents: 3*T(0) + 6*T(1) = 15D
        
#         # Combined input representation:
#         # States: 27*T(1)
#         # Actions: 6*T(1) + 3*T(0)
#         # Total: 33*T(1) + 3*T(0)
#         self.repin = 33*T(1) + 3*T(0)
        
#         # Output: Single scalar Q-value (invariant to rotations)
#         self.repout = T(0)
        
#         self.emlp = emlp.EMLP(self.repin, self.repout, group=self.group, 
#                               num_layers=1)
        
#     def forward(self, state, action):
#         """
#         Build a critic network that maps (state, action) pairs -> Q-values
        
#         Args:
#             state: Concatenated states of all agents [batch_size, 54]
#             action: Concatenated actions of all agents [batch_size, 15]
        
#         Returns:
#             Q-value [batch_size, 1] - invariant to rotations
#         """
#         # Concatenate state and action
#         x = torch.cat((state, action), dim=1)  # [batch_size, 69]
        
      
            
#         # Pass through invariant EMLP
#         q_value = self.emlp(x)
        
#         # # Convert back to tensor
#         # q_value = torch.tensor(q_value_np, dtype=torch.float32, 
#         #                       device=state.device if isinstance(state, torch.Tensor) else 'cpu')
        
#         # Ensure output shape is [batch_size, 1]
#         if q_value.dim() == 1:
#             q_value = q_value.unsqueeze(-1)

      
        
#         return q_value

# class Actor(nn.Module):
#     """Individual agent's actor - SO(2) equivariant"""
    
#     def __init__(self, state_size, action_size, hidden_sizes=(128, 128), 
#                  action_low=-1, action_high=1.0):
#         super(Actor, self).__init__()
        
#         self.action_low = action_low
#         self.action_high = action_high
#         self.scale = (action_high - action_low) / 2.0
#         self.bias = (action_high + action_low) / 2.0
        
#         self.group = SO(2)
        
#         # Input: 7 scalars + 4 2D vectors = 7*T(0) + 4*T(1)
#         self.repin = 7*T(1) + 4*T(0)
        
#         # Output: 4 action values that form 2 vectors (left/right for x, down/up for y)
#         # These pair up as: [left, right] and [down, up]
#         # Each pair represents a force direction, so 2 vectors
#         self.repout = 2*T(1)  # 2 vectors in SO(2), gives 4 output values
        
#         self.emlp = emlp.EMLP(self.repin, self.repout, group=self.group, 
#                               num_layers=1)
        
#     def forward(self, state):
#         # emlp outputs 4 values as 2 vectors
#         action_np = self.emlp(state)  # Shape: (..., 4)
        
#         # Convert to torch tensor
#         action = torch.tensor(action_np, dtype=torch.float32, 
#                             device=state.device if isinstance(state, torch.Tensor) else 'cpu')
        
#         # Apply tanh to get values in [-1, 1]
#         action = torch.tanh(action)
        
#         # Scale from [-1, 1] to [action_low, action_high] using your method
#         action_scaled = self._scale_action(action)  
        
      
#         action_5d = torch.cat([
#             torch.full((*action_scaled.shape[:-1], 1), 0.5, device=action.device),
#             action_scaled
#         ], dim=-1)

#         # print("action")
#         # print(action_5d)
        
#         return action_5d
    
#     def _scale_action(self, action):
#         """Scale action from [-1, 1] to [action_low, action_high]"""
#         return self.scale * action + self.bias

# Problem: TanhTransform in TransformedDistribution computes log_prob by inverting 
# the transform: atanh((action - bias) / scale). For your setup (bias=0.5, scale=0.5), 
# actions of 0 or 1 map to atanh(-1) or atanh(1), which are undefined (infinite in 
# theory, NaN in practice due to numerical limits).
# 
# Solution: Clamp the action to [-0.999999, 0.999999] before inverting the transform.
# This avoids the undefined values and the NaN in practice (edge cases).
# 
# Other Solution is adjust the action space definition 
class SafeTanhTransform(torch.distributions.transforms.TanhTransform):
    """Safe Tanh Transform"""

    def _inverse(self, y):
        """Inverse of the TanhTransform"""
        # Clamp to avoid exact -1 or 1
        y = torch.clamp(y, -0.999999, 0.999999)
        return torch.atanh(y)

class ApproxActor(nn.Module):
    """Approximate Actor Network"""

    def __init__(self, state_size, action_size, hidden_sizes=(64, 64), init_w=3e-3, 
                 action_low=-1.0, action_high=1.0):
        super(ApproxActor, self).__init__()
        
        # Build the network
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_size * 2)

        self.action_low = action_low
        self.action_high = action_high
        self.scale = (action_high - action_low) / 2.0
        self.bias = (action_high + action_low) / 2.0

        self.apply(_init_weights_approx)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu, log_std = self.fc3(x).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std

    def _get_dist(self, mu, log_std):
        """Get the distribution of the action"""
        base_distribution = torch.distributions.Normal(mu, torch.exp(log_std))
        # tanh_transform = torch.distributions.transforms.TanhTransform(cache_size=1)
        tanh_transform = SafeTanhTransform(cache_size=1)
        scale_transform = torch.distributions.transforms.AffineTransform(self.bias, self.scale)
        squashed_and_scaled_dist = torch.distributions.TransformedDistribution(base_distribution, [tanh_transform, scale_transform])
        return squashed_and_scaled_dist, base_distribution

    def sample(self, state, deterministic=False):
        """Sample an action from the actor network"""
        mu, log_std = self.forward(state)
    
        if deterministic:
            action = torch.tanh(mu) * self.scale + self.bias
            return action, None, None
        
        dist, base_dist = self._get_dist(mu, log_std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = base_dist.entropy().sum(-1, keepdim=True) 

        return action, log_prob, entropy
    
    def evaluate_actions(self, state, action):
        """Evaluate the log probability of actions"""
        mu, log_std = self.forward(state)
        
        dist, base_dist = self._get_dist(mu, log_std)

        # Old way of clamping the action
        # action = torch.clamp(action, self.action_low + 1e-6, self.action_high - 1e-6)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = base_dist.entropy().sum(-1, keepdim=True) # 

        return log_prob, entropy