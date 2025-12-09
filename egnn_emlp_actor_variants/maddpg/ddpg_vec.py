
import os

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from maddpg.models import model_factory


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def adjust_lr(optimizer, init_lr, episode_i, num_episode, start_episode):
    if episode_i < start_episode:
        return init_lr
    lr = init_lr * (1 - (episode_i - start_episode) / (num_episode - start_episode))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


nn.LayerNorm = LayerNorm


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs, action_range):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

        self.action_range = action_range

    def forward(self, inputs):

        x = inputs
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        mu = self.mu(x)
        if self.action_range is not None:
           mu = torch.tanh(mu) * self.action_range

    
        return mu
    


import emlp.nn.pytorch as em
from emlp.groups import SO,C
from emlp.reps import T

# class EMLPActor(nn.Module):
#     def __init__(self, hidden_size, num_inputs, num_outputs, action_range):
#         super(EMLPActor, self).__init__()
        
#         # Build equivariant representations
#         # Input: [v_self(2), p_self(2), l1(2), l2(2), l3(2), o1(2), o2(2)] = 7 vectors
#         vec = T(1, 0)
#         rep_in = 7 * vec
#         rep_out = vec  # 2D output (fx, fy)
#         group = SO(2)
        
#         self.emlp = em.EMLP(
#             rep_in=rep_in,
#             rep_out=rep_out,
#             group=group,
#             ch=hidden_size,
#             num_layers=2
#         )
        
    
        
#         self.action_range = action_range

#     def forward(self, inputs):
     
#         x = inputs
#         x = self.emlp(x)  # outputs (B, 2)
#         mu = x
#         if self.action_range is not None:
#             mu = torch.tanh(mu) * self.action_range
      
#         return mu

vec = T(1, 0)
rep_in = 7 * vec
rep_out = vec  # output force vector fx, fy
group = SO(2)
class EMLPActor(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs, action_range):
        super(EMLPActor, self).__init__()
        
        # 7 vectors, each 2D
     

        self.emlp = em.EMLP(
            rep_in=rep_in,
            rep_out=rep_out,
            group=group,
            ch=hidden_size,
            num_layers=2
        )
        
        self.action_range = action_range

    def forward(self, inputs):
        # inputs shape = (B,14)
        x = inputs.clone()

   
        x[:, 2:4] = 0.0  

        out = self.emlp(x)

        # print("action range")
        # print(self.action_range)

        # if self.action_range is not None:
        #     out = torch.tanh(out) * self.action_range

        return out



class ActorG(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs, num_agents, critic_type='mlp', group=None):
        super(ActorG, self).__init__()
        assert num_agents == sum(group)
        self.num_agents = num_agents
        self.critic_type = critic_type
        sa_dim = int(num_inputs / num_agents)
        self.net_fn = model_factory.get_model_fn(critic_type)
        if group is None:
            self.net = self.net_fn(sa_dim, num_agents, hidden_size)
        else:
            self.net = self.net_fn(sa_dim, num_agents, hidden_size, group)
        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, inputs):
        bz = inputs.size()[0]
        x = inputs.view(bz, self.num_agents, -1)
        x = self.net(x)
        mu = self.mu(x)
        return mu


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs, num_agents, critic_type='mlp', agent_id=0, group=None):
        super(Critic, self).__init__()

        self.num_agents = num_agents
        self.critic_type = critic_type
        sa_dim = int((num_inputs + num_outputs) / num_agents)
        self.agent_id = agent_id
        self.net_fn = model_factory.get_model_fn(critic_type)
        if group is None:
            self.net = self.net_fn(sa_dim, num_agents, hidden_size)
        else:
            self.net = self.net_fn(sa_dim, num_agents, hidden_size, group)

    def forward(self, inputs, actions):
        bz = inputs.size()[0]
        s_n = inputs.view(bz, self.num_agents, -1)
        a_n = actions.view(bz, self.num_agents, -1)
        x = torch.cat((s_n, a_n), dim=2)
        V = self.net(x)
        return V

