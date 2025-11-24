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
        self.fc3 = nn.Linear(hidden_sizes[1], action_size)
        
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


class SafeTanhTransform(torch.distributions.transforms.TanhTransform):
    """Safe Tanh Transform"""

    def _inverse(self, y):
        """Inverse of the TanhTransform"""
        # Clamp to avoid exact -1 or 1
        y = torch.clamp(y, -0.999999, 0.999999)
        return torch.atanh(y)