from maddpg.EGNN import EGNN_Vel
import numpy as np
import torch
from maddpg.ddpg_vec import soft_update, hard_update, adjust_lr, Actor,EMLPActor, Critic
from maddpg.ounoise import OUNoise


class DDPG(object):
    def __init__(self,
        gamma, continuous, obs_dim, n_action, n_agent, obs_dims, action_range, 
        actor_type, critic_type, actor_hidden_size, critic_hidden_size, 
        actor_lr, critic_lr, fixed_lr, num_episodes, 
        train_noise,
        target_update_mode, tau,
        actor_clip_grad_norm=0.5,
        device='cpu',
    ):
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.tau = tau
        self.continuous = continuous
        self.action_range = action_range
        if not self.continuous:
            raise NotImplementedError
        self.n_agent = n_agent
        self.n_action = n_action
        self.train_noise = train_noise
        self.target_update_mode = target_update_mode
        self.device = device

        self.actor_type = actor_type
        self.critic_type = critic_type
        
        # actor, actor_optim
        if actor_type in ['mlp']:
            self.actor = Actor(actor_hidden_size, obs_dim, n_action, action_range).to(self.device)
            self.actor_target = Actor(actor_hidden_size, obs_dim, n_action, action_range).to(self.device)
        elif actor_type in ["emlp"]:
            self.actor = EMLPActor(actor_hidden_size, obs_dim, n_action, action_range).to(self.device)
            self.actor_target = EMLPActor(actor_hidden_size, obs_dim, n_action, action_range).to(self.device)
            
        elif actor_type in ["egnn"]:
            n_nodes = n_agent + 3  # 3 agents + 3 landmarks
            in_node_nf = 1         # node type only (agent=1, landmark=0)
            hidden_nf = actor_hidden_size
            out_node_nf = hidden_nf  # output features (not used for action, we use v)
            in_edge_nf = 1         # edge attributes
            n_layers = 4
            
            self.actor = EGNN_Vel(
                in_node_nf=in_node_nf,
                hidden_nf=hidden_nf,
                out_node_nf=out_node_nf,
                in_edge_nf=in_edge_nf,
                device=self.device,
                n_layers=n_layers,
                residual=True,
                attention=False,
                normalize=False,
                tanh=True
            ).to(self.device)
            
            self.actor_target = EGNN_Vel(
                in_node_nf=in_node_nf,
                hidden_nf=hidden_nf,
                out_node_nf=out_node_nf,
                in_edge_nf=in_edge_nf,
                device=self.device,
                n_layers=n_layers,
                residual=True,
                attention=False,
                normalize=False,
                tanh=True
            ).to(self.device)
            
            # Store for later use
            self.n_nodes = n_nodes
            self.n_landmarks = 3
        

        else:
            raise NotImplementedError
        
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # centralized critic, critic_optim
        if critic_type in ['mlp', 'gcn_max']:
            self.critic = Critic(critic_hidden_size, np.sum(obs_dims),
                                n_action * n_agent, n_agent, critic_type).to(self.device)
            self.critic_target = Critic(critic_hidden_size, np.sum(obs_dims), 
                                n_action * n_agent, n_agent, critic_type).to(self.device)
        else:
            raise NotImplementedError
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Initialize target 
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Exploration noise
        if self.continuous:
            self.exploration_noise_n = [OUNoise(n_action) for _ in range(n_agent)]
        
        # Adjust lr
        self.fixed_lr = fixed_lr
        self.init_act_lr = actor_lr
        self.init_critic_lr = critic_lr
        self.actor_clip_grad_norm = actor_clip_grad_norm
        self.num_episodes = num_episodes
        self.start_episode = 0

    def reset_noise(self):
        if self.continuous:
            for exploration_noise in self.exploration_noise_n:
                exploration_noise.reset() 

    def scale_noise(self, scale):
        if self.continuous:
            for exploration_noise in self.exploration_noise_n:
                exploration_noise.scale = scale
    
    def adjust_lr(self, i_episode):
        adjust_lr(self.actor_optim, self.init_act_lr, i_episode, self.num_episodes, self.start_episode)
        adjust_lr(self.critic_optim, self.init_critic_lr, i_episode, self.num_episodes, self.start_episode)
    
    def select_action(self, batch_obs_n, action_noise=False, param_noise=False, grad=False):
        # batch_obs_n: (batch_size, n_agent, obs_dim) or (n_agent, obs_dim)
        if self.actor_type in ['mlp',"emlp"]:
            actor_in = torch.tensor(batch_obs_n, dtype=torch.get_default_dtype(), device=self.device)
            actor_in = actor_in.reshape((-1, self.obs_dim))
            self.actor.eval()
            mu = self.actor(actor_in)  # (batch*n_agent, action_dim)
            action_dim = mu.shape[-1]
            mu = torch.reshape(mu, (-1, self.n_agent, action_dim))  # (batch_size, n_agent, action_dim)
        elif self.actor_type == 'egnn':
            batch_obs_n = np.array(batch_obs_n)
            if batch_obs_n.ndim == 2:
                batch_obs_n = batch_obs_n[np.newaxis, ...]
            
            batch_size = batch_obs_n.shape[0]
            n_nodes = self.n_agent + self.n_landmarks  # 6
            
            # Build graph (flattened format)
            h, x, v, edges, edge_attr = self.build_graph_from_obs_batch(batch_obs_n)
            
            self.actor.eval()
            h_out, x_out, v_out = self.actor(h, x, v, edges, edge_attr)
            
            # Reshape x_out: (batch*6, 2) -> (batch, 6, 2)
            x_out = x_out.reshape(batch_size, n_nodes, 2)
            
            # Actions = coordinate output of agent nodes (first 3 nodes)
            mu = x_out[:, :self.n_agent, :]  # (batch, 3, 2)
            # print(mu.shape)
            # batch_obs_n = np.array(batch_obs_n)
            # print("batch:")
            # print(batch_obs_n.shape)
            # print(batch_obs_n.ndim)
            # if batch_obs_n.ndim == 2:
            #     batch_obs_n = batch_obs_n[np.newaxis, ...]  # (1, n_agent, obs_dim)
            
            # # Build graph
            # print("build graph for ")
            # print(batch_obs_n.shape)
           
            # h, x, edge_index = self.build_graph_from_obs_batch(batch_obs_n)
            
            # # self.actor.eval()
            # # # Your E2GN2Policy takes (h, x, edge_index) and returns actions for agents only
            # # mu = self.actor(h, x, edge_index) 
            # actor_in = torch.tensor(batch_obs_n, dtype=torch.get_default_dtype(), device=self.device)
            # actor_in = actor_in.reshape((-1, self.obs_dim))
            # self.actor.eval()
            # mu = self.actor(actor_in)  # (batch*n_agent, action_dim)
            # action_dim = mu.shape[-1]
            # mu = torch.reshape(mu, (-1, self.n_agent, action_dim))  # (batch_size, n_agent, action_dim)
            
        else:
            raise NotImplementedError

        
        self.actor.train()
        
        if not grad:
            mu = mu.data

        if not self.continuous:
            raise NotImplementedError
        else:
            action = mu
            if action_noise:
                noise = [exploration_noise.noise() for exploration_noise in self.exploration_noise_n]
                noise = torch.tensor(np.array(noise), dtype=torch.get_default_dtype(), device=self.device)
                noise = torch.unsqueeze(noise, dim=0)  # (n_agent, action_dim) -> (1, n_agent, action_dim)
                action = action + noise  # (batch_size, n_agent, action_dim)
            action = action.clamp(-self.action_range, self.action_range)  # (-1, 1) for MPE

        if not grad:
            return action
        else:
            return action, mu
    
    def update_critic_parameters(self, batch, agent_id=0, eval=False):
        
        batch_next_obs_n = torch.tensor(batch.next_obs_n, dtype=torch.get_default_dtype(), device=self.device)
        critic_obs_in = batch_next_obs_n.reshape((-1, self.obs_dim * self.n_agent))
 


        batch_next_action_n = self.select_action(batch.next_obs_n, action_noise=self.train_noise)
        critic_action_in = batch_next_action_n.view(-1, self.n_action * self.n_agent)
        next_state_action_values = self.critic_target(critic_obs_in, critic_action_in)

        reward_n_batch = torch.tensor(batch.reward_n, dtype=torch.get_default_dtype(), device=self.device)
        mask_n_batch = torch.tensor(batch.mask_n, dtype=torch.get_default_dtype(), device=self.device)
        reward_batch = reward_n_batch[:, agent_id].unsqueeze(1)
        mask_batch = mask_n_batch[:, agent_id].unsqueeze(1)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)
        
        batch_obs_n = torch.tensor(batch.obs_n, dtype=torch.get_default_dtype(), device=self.device)
        critic_obs_in = batch_obs_n.reshape((-1, self.obs_dim * self.n_agent))
        batch_action_n = torch.tensor(batch.action_n, dtype=torch.get_default_dtype(), device=self.device)
        critic_action_in = batch_action_n.view(-1, self.n_action * self.n_agent)
        state_action_batch = self.critic(critic_obs_in, critic_action_in)

        value_loss = ((state_action_batch - expected_state_action_batch) ** 2).mean()
        if eval:
            return value_loss.item()
        
        self.critic_optim.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()
        if self.target_update_mode == 'soft':
            soft_update(self.critic_target, self.critic, self.tau)
        elif self.target_update_mode == 'hard':
            hard_update(self.critic_target, self.critic)
        return value_loss.item()
    
    def update_actor_parameters(self, batch):
        self.actor_optim.zero_grad()
        if not self.continuous:
            raise NotImplementedError
        else:
     
            batch_action_n, mu = self.select_action(batch.obs_n, action_noise=self.train_noise, grad=True)

        critic_action_in = batch_action_n.reshape(-1, self.n_action * self.n_agent)
        batch_obs_n = torch.tensor(batch.obs_n, dtype=torch.get_default_dtype(), device=self.device)
        critic_obs_in = batch_obs_n.reshape((-1, self.obs_dim * self.n_agent))
        policy_loss = -self.critic(critic_obs_in, critic_action_in).mean()

        if not self.continuous:
            raise NotImplementedError
        else:
            policy_loss = policy_loss + 1e-3 * (mu ** 2).mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_clip_grad_norm)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return policy_loss.item()
    
    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            param = params[name]
            param = param + torch.randn(param.shape) * param_noise.current_stddev

    # # def build_global_graph_from_obs_batch(batch_obs_n, n_agents, n_landmarks=3):
    # def build_graph_from_obs_batch(self, batch_obs_n, n_agents=3, n_landmarks=3):
    #     """
    #     Args:
    #         batch_obs_n: (batch_size, n_agents, obs_dim=14) numpy array or tensor
            
    #     Returns:
    #         h: (batch_size * n_nodes, feature_dim) - node features (FLATTENED)
    #         x: (batch_size * n_nodes, 2) - node coordinates (FLATTENED)
    #         edges: [row, col] - batched edge indices
    #         edge_attr: (n_edges, 1) - edge attributes
    #     """
    #     if isinstance(batch_obs_n, np.ndarray):
    #         batch_obs_n = torch.tensor(batch_obs_n, dtype=torch.get_default_dtype(), device=self.device)
        
    #     batch_size = batch_obs_n.shape[0]
    #     n_nodes = n_agents + n_landmarks  # 6
        
    #     # === Parse observations ===
    #     # obs per agent: [vel(2), pos(2), landmark_rel(6), other_agent_rel(4)]
        
    #     # Agent positions (absolute) - indices 2:4
    #     agent_pos = batch_obs_n[:, :, 2:4]  # (batch, 3, 2)
        
    #     # Landmark positions (reconstruct from agent 0's relative observations)
    #     agent0_pos = agent_pos[:, 0:1, :]  # (batch, 1, 2)
    #     landmark_rel_to_agent0 = batch_obs_n[:, 0, 4:10].reshape(batch_size, 3, 2)
    #     landmark_pos = agent0_pos + landmark_rel_to_agent0  # (batch, 3, 2)
        
    #     # === Build coordinate tensor x ===
    #     x = torch.cat([agent_pos, landmark_pos], dim=1)  # (batch, 6, 2)
    #     x = x.reshape(batch_size * n_nodes, 2)  # (batch*6, 2) FLATTENED
        
    #     # === Build feature tensor h ===
    #     # Node type: 1 for agents, 0 for landmarks (as per E2GN2 paper)
    #     agent_type = torch.ones(batch_size, n_agents, 1, device=self.device)
    #     landmark_type = torch.zeros(batch_size, n_landmarks, 1, device=self.device)
    #     h = torch.cat([agent_type, landmark_type], dim=1)  # (batch, 6, 1)
    #     h = h.reshape(batch_size * n_nodes, 1)  # (batch*6, 1) FLATTENED
        
    #     # === Build batched edges ===
    #     edges, edge_attr = self._get_edges_batch(n_nodes, batch_size)
        
    #     return h, x, edges, edge_attr

    def _get_edges_batch(self, n_nodes, batch_size):
        """
        Returns batched edge indices for multiple graphs.
        
        For batch_size=2, n_nodes=6:
            Graph 0: nodes 0,1,2,3,4,5
            Graph 1: nodes 6,7,8,9,10,11
        """
        # Base edges for single graph
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        
        edge_attr = torch.ones(len(rows) * batch_size, 1, device=self.device)
        
        if batch_size == 1:
            edges = [
                torch.LongTensor(rows).to(self.device), 
                torch.LongTensor(cols).to(self.device)
            ]
            return edges, edge_attr
        
        # Offset edges for each graph in batch
        all_rows, all_cols = [], []
        for i in range(batch_size):
            offset = n_nodes * i
            all_rows.extend([r + offset for r in rows])
            all_cols.extend([c + offset for c in cols])
        
        edges = [
            torch.LongTensor(all_rows).to(self.device),
            torch.LongTensor(all_cols).to(self.device)
        ]
        return edges, edge_attr
        
    def build_graph_from_obs_batch(self, batch_obs_n, n_agents=3, n_landmarks=3):
        """
        Args:
            batch_obs_n: (batch_size, n_agents, obs_dim=14) numpy array or tensor
            
        Returns:
            h: (batch_size * n_nodes, feature_dim) - node features (FLATTENED) - INVARIANT
            x: (batch_size * n_nodes, 2) - node coordinates (FLATTENED) - EQUIVARIANT
            v: (batch_size * n_nodes, 2) - node velocities (FLATTENED) - EQUIVARIANT
            edges: [row, col] - batched edge indices
            edge_attr: (n_edges, 1) - edge attributes
        """
        if isinstance(batch_obs_n, np.ndarray):
            batch_obs_n = torch.tensor(batch_obs_n, dtype=torch.get_default_dtype(), device=self.device)
        
        batch_size = batch_obs_n.shape[0]
        n_nodes = n_agents + n_landmarks  # 6
        
        # === Parse observations ===
        # obs per agent: [vel(2), pos(2), landmark_rel(6), other_agent_rel(4)]
        
        # Agent velocities - indices 0:2
        agent_vel = batch_obs_n[:, :, 0:2]  # (batch, 3, 2)
        
        # Agent positions (absolute) - indices 2:4
        agent_pos = batch_obs_n[:, :, 2:4]  # (batch, 3, 2)
        
        # Landmark positions (reconstruct from agent 0's relative observations)
        agent0_pos = agent_pos[:, 0:1, :]  # (batch, 1, 2)
        landmark_rel_to_agent0 = batch_obs_n[:, 0, 4:10].reshape(batch_size, 3, 2)
        landmark_pos = agent0_pos + landmark_rel_to_agent0  # (batch, 3, 2)
        
        # Landmark velocities (static, so zero)
        landmark_vel = torch.zeros(batch_size, n_landmarks, 2, device=self.device)
        
        # === Build coordinate tensor x (EQUIVARIANT) ===
        x = torch.cat([agent_pos, landmark_pos], dim=1)  # (batch, 6, 2)
        x = x.reshape(batch_size * n_nodes, 2)  # (batch*6, 2) FLATTENED
        
        # === Build velocity tensor v (EQUIVARIANT) ===
        v = torch.cat([agent_vel, landmark_vel], dim=1)  # (batch, 6, 2)
        v = v.reshape(batch_size * n_nodes, 2)  # (batch*6, 2) FLATTENED
        
        # === Build feature tensor h (INVARIANT) ===
        # Node type: 1 for agents, 0 for landmarks
        agent_type = torch.ones(batch_size, n_agents, 1, device=self.device)
        landmark_type = torch.zeros(batch_size, n_landmarks, 1, device=self.device)
        h = torch.cat([agent_type, landmark_type], dim=1)  # (batch, 6, 1)
        h = h.reshape(batch_size * n_nodes, 1)  # (batch*6, 1) FLATTENED
        
        # === Build batched edges ===
        edges, edge_attr = self._get_edges_batch(n_nodes, batch_size)
        
        return h, x, v, edges, edge_attr