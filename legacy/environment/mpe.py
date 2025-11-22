import numpy as np
import logging
from pettingzoo.mpe import simple_spread_v3
import supersuit as ss
import time

class MPEEnvironment:
    """
    This is basically a wrapper forout PettingZoo MPE Co-op env (simple_spread_v3 for the purposes of our project atleast for nwo).
    Handles basic setup, reset, action stepping, environment specs and rendering stuff.
    """

    def __init__(self, n_agents=3, max_cycles=25, continuous_actions=True, render_mode=None):
        """
        Initialization of the multi-agent environment.
        
        Args:
            n_agents (int): Number of agents in the environment that we wish to have.
            max_cycles (int): Maximum number of cycles per episode.
            continuous_actions (bool): Whether to use continuous actions.
            render_mode (str, optional): Rendering mode ('human' or 'rgb_array').
        """
        self.logger = logging.getLogger("MPEEnvironment")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')

        self.n_agents = n_agents
        self.max_cycles = max_cycles


        self.env = simple_spread_v3.parallel_env(
            N=n_agents,
            local_ratio=0.5,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode
        )

        self.env = ss.clip_reward_v0(self.env, lower_bound=-10, upper_bound=10)

    
        self.agents = self.env.possible_agents
        self.num_agents = len(self.agents)
        sample_agent = self.agents[0]
        self.observation_space = self.env.observation_space(sample_agent)
        self.action_space = self.env.action_space(sample_agent)
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.logger.info("Environment initialized")
        self.logger.info(f"  Agents: {self.num_agents}")
        self.logger.info(f"  Obs dim: {self.obs_dim}")
        self.logger.info(f"  Action dim: {self.action_dim}")

    def reset(self, seed=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility.
        
        Returns:
            dict: Initial observations for each agent.
        """
        return self.env.reset(seed=seed)

    def step(self, actions):
        """
        Take a step in the environment using agent actions.
        
        Args:
            actions (dict): A dictionary mapping agent IDs to their actions.
        
        Returns:
            tuple: (next_observations, rewards, terminations, truncations, infos)
        """
        return self.env.step(actions)

    def close(self):
        """Close env"""
        self.env.close()

    def get_specs(self):
        """
        Helper method to view the specifications of our env.
        
        Returns:
            dict: Dictionary containing agent count, observation/action dimensions, and agent IDs.
        """
        return {
            'n_agents': self.num_agents,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'agents': self.agents
        }

    def visualize(self, steps=50, random_actions=True, policy_fn=None, manual=True):
        """
        Run one rollout episode for visualization with per-frame info and optional manual stepping.

        Args:
            steps (int): Max number of steps.
            random_actions (bool): Use random actions if policy_fn not given.
            policy_fn (callable, optional): Function mapping obs to actions per agent.
            manual (bool): If True, wait for user input before next frame.
        """
        self.logger.info("Visualization started")
        obs = self.reset()

        for step in range(steps):
            
            if policy_fn:
                actions = {agent: policy_fn(obs[agent]) for agent in self.agents}
            elif random_actions:
                actions = {agent: self.action_space.sample() for agent in self.agents}
            else:
                self.logger.warning("No policy or random action mode provided — skipping step.")
                break

            obs, rewards, terminations, truncations, infos = self.step(actions)

            self.env.render()

            print(f"\n--- Step {step + 1} ---")
            for agent in self.agents:
                print(f"{agent}: action={actions[agent]}, reward={rewards[agent]}")
            
            base_env = self.env
            while hasattr(base_env, "env"):
                base_env = base_env.env

            if hasattr(base_env, 'world') and hasattr(base_env.world, 'landmarks'):
                landmarks = base_env.world.landmarks
                landmark_pos = [lm.state.p_pos for lm in landmarks]
                print("Landmark positions:", landmark_pos)


            if manual:
                user_input = input("next frame or 'q' to exit: ")
                if user_input.lower() == 'q':
                    break

            if (terminations and all(terminations.values())) or (truncations and all(truncations.values())):
                print("Episode finished")
                break

        self.logger.info("Visualization complete.")
        self.close()

