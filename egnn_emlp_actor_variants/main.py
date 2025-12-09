import random
import numpy as np
import torch
import os
import time
import torch.multiprocessing as mp
from multiprocessing import Queue
from multiprocessing.sharedctypes import Value

from maddpg.utils import make_env, sample_action_n, gen_n_actions, gen_action_range
from maddpg.utils import copy_actor_policy
from maddpg.eval import eval_model_q
from maddpg.ddpg_vec import hard_update
from maddpg.replay_memory import MAReplayMemory
from maddpg.simple_ddpg import DDPG

from dataclasses import dataclass, field, asdict

import argparse
from dataclasses import dataclass, field, asdict

def get_parser():
    parser = argparse.ArgumentParser(
        description="Train MADDPG/DDPG agents with configurable options."
    )

    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread_n3")
    parser.add_argument("--continuous", action="store_true", default=True)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)

    # Actor
    parser.add_argument("--actor_type", type=str, default="mlp")
    parser.add_argument("--actor_hidden_size", type=int, default=128)

    # Critic
    parser.add_argument("--critic_type", type=str, default="mlp")
    parser.add_argument("--critic_hidden_size", type=int, default=128)
    parser.add_argument("--lmax_attr", type=int, default=3)
    parser.add_argument("--node_input_type", type=str, default="")

    # Optimization
    parser.add_argument("--train_noise", action="store_true", default=False)
    parser.add_argument("--target_update_mode", type=str, default="soft")
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--num_episodes", type=int, default=62000)
    parser.add_argument("--replay_size", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--steps_per_actor_update", type=int, default=100)
    parser.add_argument("--steps_per_critic_update", type=int, default=100)
    parser.add_argument("--actor_updates_per_step", type=int, default=8)
    parser.add_argument("--critic_updates_per_step", type=int, default=8)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--actor_clip_grad_norm", type=float, default=0.5)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--fixed_lr", action="store_true", default=True)

    # Exploration
    parser.add_argument("--n_exploration_eps", type=int, default=25000)
    parser.add_argument("--init_noise_scale", type=float, default=0.3)
    parser.add_argument("--final_noise_scale", type=float, default=0.0)

    # Eval
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--num_eval_runs", type=int, default=200)

    # Logging
    parser.add_argument("--save_dir", type=str, default="./ckpt_plot2")

    # Compute
    parser.add_argument("--cuda", action="store_true", default=True)

    return parser


def parse_args():
    parser = get_parser()
    cli_args = parser.parse_args()
    return Config(**vars(cli_args))


@dataclass
class Config:
    # Environment
    scenario: str = "simple_spread_n3"
    continuous: bool = True
    gamma: float = 0.95
    num_steps: int = 25
    seed: int = 1

    # Actor
    actor_type: str = "mlp"
    actor_hidden_size: int = 128

    # Critic
    critic_type: str = "mlp"
    critic_hidden_size: int = 128
    lmax_attr: int = 3
    node_input_type: str = ''

    # Optimization
    train_noise: bool = False
    target_update_mode: str = 'soft'
    tau: float = 0.01
    num_episodes: int = 62000
    replay_size: int = 1000000
    batch_size: int = 1024
    steps_per_actor_update: int = 100
    steps_per_critic_update: int = 100
    actor_updates_per_step: int = 8
    critic_updates_per_step: int = 8
    actor_lr: float = 1e-4
    actor_clip_grad_norm: float = 0.5
    critic_lr: float =  1e-3
    fixed_lr: bool = True

    # Exploration noise
    n_exploration_eps: int = 25000
    init_noise_scale: float = 0.3
    final_noise_scale: float = 0.0

    # Eval
    eval_freq: int = 1000
    num_eval_runs: int = 200

    # Logging
    save_dir: str = "./ckpt_plot2"

    # Compute
    cuda: bool = True

    # Auto-generated
    exp_name: str = field(init=False)

    def __post_init__(self):
   

        self.exp_name = (
            f"{self.scenario}_"
            f"{'continuous' if self.continuous else 'discrete'}"
            f"_actor_{self.actor_type}_lr_{self.actor_lr}"
            f"_critic_{self.critic_type}_lr_{self.critic_lr}"
            f"{'_fixed_lr' if self.fixed_lr else ''}"
            f"_batch_size_{self.batch_size}"
            f"_actor_clip_grad_norm_{self.actor_clip_grad_norm}"
            f"_seed{self.seed}"
        )

    def print(self):
        print("=================Arguments==================")
        for k, v in asdict(self).items():
            print(f"{k}: {v}")
        print("========================================")


if __name__ == '__main__':
    # args = Config()
    args = parse_args()

    print(args)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    # env
    env = make_env(args.scenario, args.continuous, arglist=None)
    scenario, world = env.scenario, env.world
    env.n_agent = n_agent = env.n
    env.n_actions = n_actions = gen_n_actions(env.action_space)
    n_action = n_actions[0]
    env.action_range = action_range = gen_action_range(env.action_space)
    obs_dims = [env.observation_space[i].shape[0] for i in range(n_agent)]
    obs_dim = obs_dims[0]
    obs_dims.insert(0, 0)
    obs_n, info = env.reset()
    sample_action_n = sample_action_n(env.action_space)
 
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # E3DDPG agent, eval_agent
    agent = DDPG(
        gamma=args.gamma,
        continuous=args.continuous,
        obs_dim=obs_dim,
        n_action=n_action,
        n_agent=n_agent,
        obs_dims=obs_dims,
        action_range=action_range,
        actor_type=args.actor_type, critic_type=args.critic_type,
        actor_hidden_size=args.actor_hidden_size,
        critic_hidden_size=args.critic_hidden_size,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        fixed_lr=args.fixed_lr, num_episodes=args.num_episodes,
        train_noise=args.train_noise,
        target_update_mode=args.target_update_mode, tau=args.tau,
        device=device,
    )
    if args.actor_type != 'emlp':
        eval_agent = DDPG(
            gamma=args.gamma,
            continuous=args.continuous,
            obs_dim=obs_dim,
            n_action=n_action,
            n_agent=n_agent,
            obs_dims=obs_dims,
            action_range=action_range,
            actor_type=args.actor_type, critic_type=args.critic_type,
            actor_hidden_size=args.actor_hidden_size,
            critic_hidden_size=args.critic_hidden_size,
            actor_lr=args.actor_lr, critic_lr=args.critic_lr,
            fixed_lr=args.fixed_lr, num_episodes=args.num_episodes,
            train_noise=args.train_noise,
            target_update_mode=args.target_update_mode, tau=args.tau,
            device='cpu'
        )
    else:
        eval_agent = None  # Will be created in eval process
   
    memory = MAReplayMemory(args.replay_size)

    rewards = []
    total_numsteps = 0
    updates = 0
    exp_save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)

    test_q = Queue()
    done_training = Value('i', False)
    p = mp.Process(target=eval_model_q, args=(test_q, done_training, args))
    p.start()

    start_time = time.time()

    if args.actor_type != 'emlp':
        copy_actor_policy(agent, eval_agent)
        tr_log = {
            'exp_save_dir': exp_save_dir,
            'total_numsteps': total_numsteps,
            'i_episode': 0, 'start_time': start_time,
            'value_loss': None, 'policy_loss': None,
        }
        test_q.put([eval_agent, tr_log])
    else:

        tr_log = {
            'exp_save_dir': exp_save_dir,
            'total_numsteps': total_numsteps,
            'i_episode': 0, 'start_time': start_time,
            'value_loss': None, 'policy_loss': None,
        }
        actor_path = os.path.join(exp_save_dir+"/temps/", f'temp_actor_ep{0}.pt')

        torch.save(agent.actor.state_dict(), actor_path)
        tr_log['actor_path'] = actor_path  
        test_q.put([None, tr_log])
 


    for i_episode in range(args.num_episodes):
        obs_n, info = env.reset()

        episode_reward = 0
        episode_step = 0
        agents_rew = [[] for _ in range(n_agent)]

        if env.continuous:
            explr_pct_remaining = max(0, args.n_exploration_eps - i_episode) / args.n_exploration_eps
            agent.scale_noise(args.final_noise_scale + (args.init_noise_scale - args.final_noise_scale) * explr_pct_remaining)
            agent.reset_noise()

        while True:
            action_n = agent.select_action(
                np.array(obs_n),
                action_noise=True, param_noise=False
            ).squeeze().cpu().numpy()
         
            state = info['state']
            next_obs_n, reward_n, done_n, info = env.step(action_n)
            next_state = info['state']
            total_numsteps += 1
            episode_step += 1
            terminal = (episode_step >= args.num_steps)

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

            if len(memory) > args.batch_size:
                if total_numsteps % args.steps_per_actor_update == 0:
                    for _ in range(args.actor_updates_per_step):
                        batch = memory.sample(args.batch_size)
                        policy_loss = agent.update_actor_parameters(batch)
                    print(f'episode {i_episode}, p loss {policy_loss}, p_lr {agent.actor_optim.param_groups[0]["lr"]}')
                if total_numsteps % args.steps_per_critic_update == 0:
                    value_losses = []
                    for _ in range(args.critic_updates_per_step):
                        batch = memory.sample(args.batch_size)
                        value_losses.append(agent.update_critic_parameters(batch, i))
                    value_loss = np.mean(value_losses)
                    print(f'episode {i_episode}, q loss {value_loss}, q_lr {agent.critic_optim.param_groups[0]["lr"]}')
                    if args.target_update_mode == 'episodic':
                        hard_update(agent.critic_target, agent.critic)

            if done_n[0] or terminal:
                episode_step = 0
                break

        if not args.fixed_lr:
            agent.adjust_lr(i_episode)

        rewards.append(episode_reward)

        if (i_episode + 1) % args.eval_freq == 0:
            tr_log = {
                'exp_save_dir': exp_save_dir,
                'total_numsteps': total_numsteps,
                'i_episode': i_episode, 'start_time': start_time,
                'value_loss': value_loss, 'policy_loss': policy_loss
            }
           
            if args.actor_type == 'emlp':
                actor_path = os.path.join(exp_save_dir+"/temps/", f'temp_actor_ep{i_episode}.pt')
                torch.save(agent.actor.state_dict(), actor_path)
                tr_log['actor_path'] = actor_path  
                test_q.put([None, tr_log])
            else:
                copy_actor_policy(agent, eval_agent)

                test_q.put([eval_agent, tr_log])
            

    env.close()
    time.sleep(5)
    done_training.value = True