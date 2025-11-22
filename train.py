"""
Training script for MADDPG with PettingZoo
"""
import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from datetime import datetime



from configs.config import Config
from maddpg import MADDPG, ReplayBuffer
from utils.env import get_env_info, ENV_MAP, create_single_env
from utils.logger import Logger
from utils.utils import evaluate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actor",
        type=str,
        default="traditional",
        choices=["traditional", "rotational equivariant"],
        help="Choose actor architecture"
    )

    parser.add_argument(
        "--critic",
        type=str,
        default="traditional",
        choices=["traditional", "permutation invariant"],
        help="Choose critic architecture"
    )


    parser.add_argument("--render", action="store_true")
    parser.add_argument("--create-gif", action="store_true")

    parser.add_argument("--total-timesteps", type=int, default=None)

    return parser.parse_args()


def train(args):
    
    cfg = Config()
    cfg.apply_cli(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = (
        f"{cfg.env_name}"
        f"_actor{cfg.actor}"
        f"_critic{cfg.critic}"
        f"_b{cfg.batch_size}"
        f"_usteps{cfg.update_every}"
        f"_g{cfg.gamma}"
        f"_t{cfg.tau}"
        f"_alr{cfg.actor_lr}"
        f"_clr{cfg.critic_lr}"
        f"_n{cfg.noise_scale}"
        f"_minn{cfg.min_noise}"
        f"_h{cfg.hidden_sizes}"
        f"_{timestamp}"
    )

    logger = Logger(
        run_name=experiment_name,
        folder="runs",
        algo=cfg.algo,
        env=cfg.env_name
    )

    logger.log_all_hyperparameters(cfg.__dict__)
    

    # Get environment information
    agents, num_agents, action_sizes, action_low, action_high, state_sizes = get_env_info(
        env_name=cfg.env_name, 
        max_steps=cfg.max_steps,
        apply_padding=False  
    )
  

    # Create environment with appropriate render mode
    env = create_single_env(
        env_name=cfg.env_name,
        max_steps=cfg.max_steps,
        render_mode=cfg.render_mode,
        apply_padding=False
    )
    
    # Create evaluation environment
    env_evaluate = create_single_env(
        env_name=cfg.env_name,
        max_steps=cfg.max_steps,
        render_mode="rgb_array",
        apply_padding=False
    )
    
    # Model path
    model_path = os.path.join(logger.dir_name, "model.pt")
    best_model_path = os.path.join(logger.dir_name, "best_model.pt")
    best_score = -float('inf')
    
    # Parse hidden sizes
    hidden_sizes = tuple(map(int, cfg.hidden_sizes.split(',')))
    
    # Create MADDPG agent

    maddpg = MADDPG(
            state_sizes=state_sizes,
            action_sizes=action_sizes,
            hidden_sizes=hidden_sizes,
            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr,
            gamma=cfg.gamma,
            tau=cfg.tau,
            action_low=action_low,
            action_high=action_high
        )
    
    
    # Create replay buffer with the correct dimensions
    buffer = ReplayBuffer(
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        agents=agents,
        state_sizes=state_sizes,
        action_sizes=action_sizes
    )
    
    # Training loop
    noise_scale = cfg.noise_scale
    noise_decay = (cfg.noise_scale - cfg.min_noise) / min(cfg.noise_decay_steps, cfg.total_timesteps)
    print(f"Using linear noise decay: {cfg.noise_scale} to {cfg.min_noise} over {cfg.noise_decay_steps} steps")
    print(f"Noise will decrease by {noise_decay:.6f} per step")

    evaluate(env_evaluate, maddpg, logger, record_gif=cfg.create_gif, num_eval_episodes=10, global_step=0)
    
    # For tracking agent-specific rewards
    agent_rewards = [[] for _ in range(len(agents))]
    episode_rewards = np.zeros(len(agents))

    # Reset environment and agents
    observations, _ = env.reset()

    for global_step in tqdm(range(1, cfg.total_timesteps + 1), desc="Training"):
        
        # Get states for all agents
        states_list = [np.array(observations[agent], dtype=np.float32) for agent in agents]
        
        # Get actions for all agents
        actions_list = maddpg.act(states_list, add_noise=True, noise_scale=noise_scale)
        
        # Convert actions to dictionary for environment
        actions = {agent: action for agent, action in zip(agents, actions_list)}
        
        # Take a step in the environment
        next_observations, rewards, terminations, truncations, _ = env.step(actions)
        
        # Check if episode is done
        dones = [terminations[agent] or truncations[agent] for agent in agents]
        done = any(dones)
        
        # Prepare data for buffer (convert to NumPy once)
        rewards_array = np.array([rewards[agent] for agent in agents], dtype=np.float32)
        next_states_list = [np.array(next_observations[agent], dtype=np.float32) for agent in agents]
        # we care about the termination of the episode
        terminations_array = np.array([terminations[agent] for agent in agents], dtype=np.uint8)
        
        # Store experience in replay buffer
        buffer.add(
            states=states_list,
            actions=actions_list,
            rewards=rewards_array,
            next_states=next_states_list,
            dones=terminations_array
        )
        
        # Update observations and rewards
        observations = next_observations
        episode_rewards += np.array(list(rewards.values()))         
        
        # Learn if enough samples are available in memory
        if global_step > cfg.warmup_steps and global_step % cfg.update_every == 0:
            for i in range(len(agents)):
                experiences = buffer.sample()  # Now returns pre-combined states
                critic_loss, actor_loss = maddpg.learn(experiences, i)
                
                # Log losses to TensorBoard
                logger.add_scalar(f'{agents[i]}/critic_loss', critic_loss, global_step)
                logger.add_scalar(f'{agents[i]}/actor_loss', actor_loss, global_step)
                
            maddpg.update_targets()
        
        # Update noise scale based on iteration number
        if global_step > cfg.warmup_steps and cfg.use_noise_decay:
            noise_scale = max(
                cfg.min_noise,
                noise_scale - noise_decay
            )
        
        # Handle episode end
        if done or (global_step % cfg.max_steps == 0):  # Reset after max_steps if not done
            for i, reward in enumerate(episode_rewards):
                agent_rewards[i].append(reward)
                logger.add_scalar(f"{agents[i]}/episode_reward", reward, global_step)
            logger.add_scalar('train/total_reward', np.sum(episode_rewards), global_step)
            logger.add_scalar(f"noise/scale", noise_scale, global_step)
            observations, _ = env.reset()
            episode_rewards = np.zeros(len(agents))
        
        # Evaluate and save
        if global_step % cfg.eval_interval == 0 or global_step == cfg.total_timesteps:
            maddpg.save(model_path)
            avg_eval_rewards = evaluate(env_evaluate, maddpg, logger,
                    num_eval_episodes=10, record_gif=cfg.create_gif, global_step=global_step)
            np.save(os.path.join(logger.dir_name, "agent_rewards.npy"), agent_rewards)
            score = np.sum(avg_eval_rewards)
            if score > best_score:
                best_score = score
                maddpg.save(best_model_path)
    
    # Save final models
    maddpg.save(model_path)
    np.save(os.path.join(logger.dir_name, "agent_rewards.npy"), agent_rewards)
    
    # Close environment and TensorBoard writer
    env.close()
    env_evaluate.close()
    logger.close()
    
    # Return both the agent rewards and the experiment name
    return agent_rewards, experiment_name

if __name__ == "__main__":
    args = parse_args()
    train(args)
