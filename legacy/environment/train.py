import random
import numpy as np
import torch
import os
import time
import torch.multiprocessing as mp
from multiprocessing import Queue
from multiprocessing.sharedctypes import Value

from config import config
from utils import make_env, sample_action_n, gen_n_actions, gen_action_range, copy_actor_policy
from eval import eval_model_q
from e3_ddpg_vec import E3DDPG, hard_update
from replay_memory import MAReplayMemory


def train():
    """Main training loop"""
    cfg = config
    
    print("=================Configuration==================")
    for k, v in cfg.items():
        print(f'{k}: {v}')
    print("================================================")
    
    # Setup
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() and cfg['cuda'] else "cpu")
    print(f"Using device: {device}")
    
    # Initialize environment
    env_info = setup_environment(cfg)
    env = env_info['env']
    n_agent = env_info['n_agent']
    
    # Set seeds
    env.seed(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    
    # Create agents
    agent = create_agent(cfg, env_info, device)
    
    eval_agent = create_agent(cfg, env_info, 'cpu')
    eval_agent.scenario = None  # To be set in eval.py
    eval_agent.world = None
    
    # Replay buffer
    memory = MAReplayMemory(cfg['replay_size'])
    
    # Setup logging
    exp_name = build_exp_name(cfg)
    exp_save_dir = os.path.join(cfg['save_dir'], exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    print(f"Saving to: {exp_save_dir}")
    
    # Setup evaluation process
    test_q = Queue()
    done_training = Value('i', False)
    p = mp.Process(target=eval_model_q, args=(test_q, done_training, cfg))
    p.start()
    
    start_time = time.time()
    copy_actor_policy(agent, eval_agent)
    
    tr_log = {
        'exp_save_dir': exp_save_dir,
        'total_numsteps': 0,
        'i_episode': 0,
        'start_time': start_time,
        'value_loss': None,
        'policy_loss': None,
    }
    test_q.put([eval_agent, tr_log])
    
    # Training loop
    rewards = []
    total_numsteps = 0
    
    for i_episode in range(cfg['num_episodes']):
        obs_n, info = env.reset()
        episode_reward = 0
        episode_step = 0
        agents_rew = [[] for _ in range(n_agent)]
        
        # Reset exploration noise
        if cfg['continuous']:
            explr_pct_remaining = max(0, cfg['n_exploration_eps'] - i_episode) / cfg['n_exploration_eps']
            noise_scale = cfg['final_noise_scale'] + (cfg['init_noise_scale'] - cfg['final_noise_scale']) * explr_pct_remaining
            agent.scale_noise(noise_scale)
            agent.reset_noise()
        
        # Episode loop
        while True:
            action_n = agent.select_action(
                np.array(obs_n),
                action_noise=True,
                param_noise=False
            ).squeeze().cpu().numpy()
            
            state = info['state']
            next_obs_n, reward_n, done_n, info = env.step(action_n)
            next_state = info['state']
            
            total_numsteps += 1
            episode_step += 1
            terminal = (episode_step >= cfg['num_steps'])
            
            # Store transition
            memory.push(
                state,
                np.array(obs_n),
                action_n,
                np.array([not done for done in done_n]),
                next_state,
                np.array(next_obs_n),
                np.array(reward_n),
            )
            
            for i, r in enumerate(reward_n):
                agents_rew[i].append(r)
            episode_reward += np.sum(reward_n)
            obs_n = next_obs_n
            
            # Update networks
            if len(memory) > cfg['batch_size']:
                if total_numsteps % cfg['steps_per_actor_update'] == 0:
                    for _ in range(cfg['actor_updates_per_step']):
                        batch = memory.sample(cfg['batch_size'])
                        policy_loss = agent.update_actor_parameters(batch)
                    print(f"Episode {i_episode}, policy loss: {policy_loss:.4f}, "
                          f"actor_lr: {agent.actor_optim.param_groups[0]['lr']:.6f}")
                
                if total_numsteps % cfg['steps_per_critic_update'] == 0:
                    value_losses = []
                    for _ in range(cfg['critic_updates_per_step']):
                        batch = memory.sample(cfg['batch_size'])
                        value_losses.append(agent.update_critic_parameters(batch, i_episode))
                    value_loss = np.mean(value_losses)
                    print(f"Episode {i_episode}, value loss: {value_loss:.4f}, "
                          f"critic_lr: {agent.critic_optim.param_groups[0]['lr']:.6f}")
                    
                    if cfg['target_update_mode'] == 'episodic':
                        hard_update(agent.critic_target, agent.critic)
            
            if done_n[0] or terminal:
                episode_step = 0
                break
        
        # Adjust learning rate
        if not cfg['fixed_lr']:
            agent.adjust_lr(i_episode)
        
        rewards.append(episode_reward)
        
        # Evaluation
        if (i_episode + 1) % cfg['eval_freq'] == 0:
            copy_actor_policy(agent, eval_agent)
            tr_log = {
                'exp_save_dir': exp_save_dir,
                'total_numsteps': total_numsteps,
                'i_episode': i_episode,
                'start_time': start_time,
                'value_loss': value_loss,
                'policy_loss': policy_loss
            }
            test_q.put([eval_agent, tr_log])
            print(f"Episode {i_episode + 1}/{cfg['num_episodes']}, "
                  f"Avg Reward (last 100): {np.mean(rewards[-100:]):.2f}")
    
    # Cleanup
    env.close()
    time.sleep(5)
    done_training.value = True
    print("Training completed!")


if __name__ == '__main__':
    train()