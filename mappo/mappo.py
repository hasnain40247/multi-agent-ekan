import torch
import torch.nn as nn
import torch.optim as optim
import logging
from policy_network.policy_net import PolicyNet 

class MAPPO:
    """
    Pretty minimal MAPPO collector for setup.
    """

    def __init__(self, env, gamma=0.99, lr=3e-4, device=None):
        self.env = env
        self.gamma = gamma
        self.device = device if device else ('mps' if torch.backends.mps.is_available() else 'cpu')

        self.policy = PolicyNet(env.obs_dim, env.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.trajectory = []

        self.logger = logging.getLogger("MAPPO")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
        self.logger.info(f"MAPPO initialized on device: {self.device}")

    def collect_trajectories(self, steps=10):
        """
        I basically collect episodic experience to inspect policy network outputs here
        """
        obs, _ = self.env.reset()
        obs = {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k,v in obs.items()}

        self.logger.info(f"Starting trajectory collection for {steps} steps...")

        for step in range(steps):
            actions, log_probs, values = {}, {}, {}

            for agent in self.env.agents:
                a, logp = self.policy.get_action(obs[agent])
                _, _, v = self.policy.forward(obs[agent])

                actions[agent] = a.cpu().detach().numpy()
                log_probs[agent] = logp.cpu().detach().item()
                values[agent] = v.cpu().detach().item()

            next_obs, rewards, terminations, truncations, _ = self.env.step(actions)

            for agent in self.env.agents:
                self.trajectory.append({
                    'agent': agent,
                    'obs': obs[agent].cpu().numpy(),
                    'action': actions[agent],
                    'log_prob': log_probs[agent],
                    'value': values[agent],
                    'reward': rewards[agent],
                    'done': terminations[agent] or truncations[agent]
                })

            done = all(terminations.values()) or all(truncations.values())
            if done:
                next_obs, _ = self.env.reset()

            obs = {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k,v in next_obs.items()}

        self.logger.info("Trajectory collection complete. Sample of collected steps:")

        for step_data in self.trajectory[:min(5, len(self.trajectory))]:
            self.logger.info(step_data)

        return self.trajectory
