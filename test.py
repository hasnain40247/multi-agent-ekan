"""
E-KAN Multi-Agent Project
Author: Hasnain Sikora

Component: Environment setup + MAPPO baseline framework
"""

# ============================================================================
# requirements.txt
# ============================================================================
"""
torch>=2.0.0
numpy>=1.24.0
pettingzoo[mpe]>=1.24.0
supersuit>=3.9.0
gymnasium>=0.29.0
"""

# ============================================================================
# 1. Environment Setup (environment.py)
# ============================================================================

import numpy as np
from pettingzoo.mpe import simple_spread_v3
import supersuit as ss


class MPEEnvironment:
    """
    Wrapper for PettingZoo MPE Cooperative Navigation environment.
    Handles creation, reset, stepping, and environment specs.
    """
    
    def __init__(self, n_agents=3, max_cycles=25, continuous_actions=True, render_mode=None):
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        
        # Base environment
        self.env = simple_spread_v3.parallel_env(
            N=n_agents,
            local_ratio=0.5,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode
        )
        
        # Reward clipping
        self.env = ss.clip_reward_v0(self.env, lower_bound=-10, upper_bound=10)
        
        # Agent setup
        self.agents = self.env.possible_agents
        self.num_agents = len(self.agents)
        
        # Observation & action specs
        sample_agent = self.agents[0]
        self.observation_space = self.env.observation_space(sample_agent)
        self.action_space = self.env.action_space(sample_agent)
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        
        print(f"[Environment initialized]")
        print(f"  Agents: {self.num_agents}")
        print(f"  Obs dim: {self.obs_dim}")
        print(f"  Action dim: {self.action_dim}")
    
    def reset(self, seed=None):
        """Reset environment"""
        return self.env.reset(seed=seed)
    
    def step(self, actions):
        """Step environment"""
        return self.env.step(actions)
    
    def close(self):
        """Close environment"""
        self.env.close()
    
    def get_specs(self):
        """Return environment specs"""
        return {
            'n_agents': self.num_agents,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'agents': self.agents
        }


def test_environment():
    """Quick environment validation"""
    print("\n" + "=" * 60)
    print("Testing PettingZoo MPE Environment")
    print("=" * 60 + "\n")
    
    env = MPEEnvironment(n_agents=3)
    specs = env.get_specs()
    
    print(f"\nSpecs:")
    for k, v in specs.items():
        print(f"  {k}: {v}")
    
    obs, _ = env.reset(seed=42)
    print("\n✓ Reset successful")
    print(f"  Obs keys: {list(obs.keys())}")
    print(f"  Obs shape (agent 0): {obs[specs['agents'][0]].shape}")
    
    print("\nRunning random rollout...")
    rewards = {agent: 0.0 for agent in env.agents}
    for t in range(env.max_cycles):
        actions = {a: env.action_space.sample() for a in env.agents}
        obs, r, done, trunc, info = env.step(actions)
        for a in env.agents:
            rewards[a] += r[a]
        if all(done.values()) or all(trunc.values()):
            break
    print("\nTotal rewards:")
    for a, val in rewards.items():
        print(f"  {a}: {val:.3f}")
    env.close()
    print("\n✓ Environment test complete!\n")


# ============================================================================
# 2. MAPPO Framework (mappo_framework.py)
# ============================================================================

# ============================================================================
# 3. Main Script
# ============================================================================

def main():
    print("=" * 60)
    print("E-KAN Multi-Agent Project Setup")
    print("Hasnain's Component: PettingZoo + MAPPO Baseline")
    print("=" * 60)
    
    # Test environment
    test_environment()
    
    # Initialize environment and MAPPO framework
    env = MPEEnvironment(n_agents=3)
    framework = MAPPOFramework(env)
    
    print("\nSetup Complete!")
    print("=" * 60)
    print("Next steps:")
    print("  - Add policy network implementation")
    print("  - Add buffer and training logic")
    print(f"\nEnv specs: {framework.specs}")
    print("Config:", framework.config)
    
    env.close()


if __name__ == "__main__":
    main()
