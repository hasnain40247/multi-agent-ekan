
# import numpy as np
# import os
# import time
# import torch
# import random
# import torch.multiprocessing as mp
# from multiprocessing import Queue
# from multiprocessing.sharedctypes import Value


# from maddpg.ddpg_vec import DDPG
# from maddpg.replay_memory import ReplayMemory, Transition
# from maddpg.utils import *
# from maddpg.eval import eval_model_q
# from maddpg.utils import n_actions, copy_actor_policy
# from maddpg.ddpg_vec import hard_update


# # ==================== CONSTANTS ====================
# class Args:
#     scenario = 'simple_spread_n3'  
#     continuous = True

#     gamma = 0.95
#     tau = 0.01
#     ou_noise = True
#     param_noise = False
#     train_noise = False
#     noise_scale = 0.3
#     final_noise_scale = 0.3
#     exploration_end = 60000
#     seed = 9
#     batch_size = 1024
#     num_steps = 25
#     num_episodes = 60000
#     hidden_size = 128
#     updates_per_step = 8
#     critic_updates_per_step = 8
#     replay_size = 1000000
#     actor_lr = 1e-2
#     critic_lr = 1e-2
#     fixed_lr = False
#     num_eval_runs = 1000
#     exp_name = None  # Will be auto-generated if None
#     save_dir = "./ckpt_plot"
#     static_env = False
#     critic_type = 'mlp'  # Supports [mlp, gcn_mean, gcn_max]
#     actor_type = 'mlp'  # Supports [mlp, gcn_max]
#     critic_dec_cen = 'cen'
#     env_agent_ckpt = 'ckpt_plot/simple_tag_v5_al0a10_4/agents.ckpt'
#     shuffle = None  # None|shuffle|sort
#     episode_per_update = 4
#     episode_per_actor_update = 4
#     episode_per_critic_update = 4
#     steps_per_actor_update = 100
#     steps_per_critic_update = 100
#     target_update_mode = 'soft'  # soft | hard | episodic
#     cuda = False
#     eval_freq = 1000
# if __name__ == '__main__':
#     args = Args()
#     # ===================================================

#     if args.exp_name is None:
#         args.exp_name = args.scenario + '_' + args.critic_type + '_' + args.target_update_mode + '_hiddensize' \
#                         + str(args.hidden_size) + '_' + str(args.seed)

#     print("=================Arguments==================")
#     for k, v in vars(args).items():
#         if not k.startswith('_'):
#             print(f'{k}: {v}')
#     print("========================================")

#     torch.set_num_threads(1)
#     device = torch.device("cpu")

#     env = make_env(args.scenario, args.continuous, arglist = None)
#     print(f"Environment: {args.scenario}")
#     print(f"Continuous: {args.continuous}")

#     scenario, world = env.scenario, env.world
#     print(f"Scenario type: {type(scenario)}")
#     print(f"World type: {type(world)}")
#     print(f"World dim_p (physical): {world.dim_p}")
#     print(f"World dim_c (communication): {world.dim_c}")

#     env.n_agent = n_agent = env.n
#     print(f"Number of agents: {n_agent}")

#     env.n_actions = n_actions = gen_n_actions(env.action_space)
#     print(f"n_actions: {n_actions}")  # [2, 2, 2, 2, 2, 2] if continuous

#     n_action = n_actions[0]
#     print(f"n_action (per agent): {n_action}")

#     env.action_range = action_range = gen_action_range(env.action_space)
#     print(f"action_range: {action_range}")

#     obs_dims = [env.observation_space[i].shape[0] for i in range(n_agent)]
#     print(f"obs_dims: {obs_dims}")  # [26, 26, 26, 26, 26, 26]

#     obs_dim = obs_dims[0]
#     print(f"obs_dim (per agent): {obs_dim}")

#     obs_dims.insert(0, 0)
#     print(f"obs_dims after insert: {obs_dims}")  # [0, 26, 26, 26, 26, 26, 26]

#     obs_n, info = env.reset()
#     print(f"obs_n type: {type(obs_n)}")
#     print(f"obs_n length: {len(obs_n)}")
#     print(f"obs_n[0] shape: {np.array(obs_n[0]).shape}")
#     print(f"obs_n full shape: {np.array(obs_n).shape}")
#     print(f"info keys: {info.keys()}")
#     print(f"state shape: {np.array(info['state']).shape if info.get('state') is not None else None}")

#     sample_action_n = sample_action_n(env.action_space)
#     print(f"sample_action_n type: {type(sample_action_n)}")
#     print(f"sample_action_n length: {len(sample_action_n)}")
#     print(f"sample_action_n[0] shape: {np.array(sample_action_n[0]).shape}")
#     print(f"sample_action_n full shape: {np.array(sample_action_n).shape}")

#     # Bonus: inspect action space details
#     print(f"\n--- Action Space Details ---")
#     for i, action_space in enumerate(env.action_space):
#         print(f"Agent {i}: {action_space}, low={action_space.low}, high={action_space.high}")

#     print(f"\n--- Observation Space Details ---")
#     for i, obs_space in enumerate(env.observation_space):
#         print(f"Agent {i}: {obs_space}, shape={obs_space.shape}")


#     # env = make_env(args.scenario, None)
#     # n_agents = env.n
#     env.seed(args.seed)
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     num_adversary = 0

#     # n_actions = n_actions(env.action_space)
#     # obs_dims = [env.observation_space[i].shape[0] for i in range(n_agents)]
#     # obs_dims.insert(0, 0)

#     agent = DDPG(args.gamma, args.tau, args.hidden_size,
#                 env.observation_space[0].shape[0], n_actions[0], n_agent, obs_dims, 0,
#                 args.actor_lr, args.critic_lr,
#                 args.fixed_lr, args.critic_type, args.actor_type, args.train_noise, args.num_episodes,
#                 args.num_steps, args.critic_dec_cen, args.target_update_mode, device)
#     eval_agent = DDPG(args.gamma, args.tau, args.hidden_size,
#                 env.observation_space[0].shape[0], n_actions[0], n_agent, obs_dims, 0,
#                 args.actor_lr, args.critic_lr,
#                 args.fixed_lr, args.critic_type, args.actor_type, args.train_noise, args.num_episodes,
#                 args.num_steps, args.critic_dec_cen, args.target_update_mode, 'cpu')

#     memory = ReplayMemory(args.replay_size)
#     feat_dims = []
#     for i in range(n_agent):
#         feat_dims.append(env.observation_space[i].shape[0])

#     # Find main agents index
#     unique_dims = list(set(feat_dims))
#     agents0 = [i for i, feat_dim in enumerate(feat_dims) if feat_dim == unique_dims[0]]
#     if len(unique_dims) > 1:
#         agents1 = [i for i, feat_dim in enumerate(feat_dims) if feat_dim == unique_dims[1]]
#         main_agents = agents0 if len(agents0) >= len(agents1) else agents1
#     else:
#         main_agents = agents0

#     rewards = []
#     total_numsteps = 0
#     updates = 0
#     exp_save_dir = os.path.join(args.save_dir, args.exp_name)
#     os.makedirs(exp_save_dir, exist_ok=True)
#     best_eval_reward, best_good_eval_reward, best_adversary_eval_reward = -1000000000, -1000000000, -1000000000
#     start_time = time.time()
#     copy_actor_policy(agent, eval_agent)
#     torch.save({'agents': eval_agent}, os.path.join(exp_save_dir, 'agents_best.ckpt'))

#     # for mp test
#     test_q = Queue()
#     done_training = Value('i', False)
#     p = mp.Process(target=eval_model_q, args=(test_q, done_training, args))
#     p.start()

#     for i_episode in range(1):

#         obs_n,_ = env.reset()

#         episode_reward = 0
#         episode_step = 0
#         agents_rew = [[] for _ in range(n_agent)]
#         while True:
#             action_n = agent.select_action(torch.Tensor(obs_n).to(device), action_noise=True,
#                                         param_noise=False).squeeze().cpu().numpy()
#             print("actions")
#             print(action_n)
#             next_obs_n, reward_n, done_n, info = env.step(action_n)
#             total_numsteps += 1
#             episode_step += 1
#             terminal = (episode_step >= args.num_steps)

#             action = torch.Tensor(action_n).view(1, -1)
#             mask = torch.Tensor([[not done for done in done_n]])
#             next_x = torch.Tensor(np.concatenate(next_obs_n, axis=0)).view(1, -1)
#             reward = torch.Tensor([reward_n])
#             x = torch.Tensor(np.concatenate(obs_n, axis=0)).view(1, -1)
#             memory.push(x, action, mask, next_x, reward)
#             for i, r in enumerate(reward_n):
#                 agents_rew[i].append(r)
#             episode_reward += np.sum(reward_n)
#             obs_n = next_obs_n
#             n_update_iter = 5
#             if len(memory) > args.batch_size:
#                 if total_numsteps % args.steps_per_actor_update == 0:
#                     for _ in range(args.updates_per_step):
#                         transitions = memory.sample(args.batch_size)
#                         batch = Transition(*zip(*transitions))
#                         policy_loss = agent.update_actor_parameters(batch, i, args.shuffle)
#                         updates += 1
#                     print('episode {}, p loss {}, p_lr {}'.
#                         format(i_episode, policy_loss, agent.actor_lr))
#                 if total_numsteps % args.steps_per_critic_update == 0:
#                     value_losses = []
#                     for _ in range(args.critic_updates_per_step):
#                         transitions = memory.sample(args.batch_size)
#                         batch = Transition(*zip(*transitions))
#                         value_losses.append(agent.update_critic_parameters(batch, i, args.shuffle))
#                         updates += 1
#                     value_loss = np.mean(value_losses)
#                     print('episode {}, q loss {},  q_lr {}'.
#                         format(i_episode, value_loss, agent.critic_optim.param_groups[0]['lr']))
#                     if args.target_update_mode == 'episodic':
#                         hard_update(agent.critic_target, agent.critic)

#             if done_n[0] or terminal:
#                 print('train epidoe reward', episode_reward)
#                 episode_step = 0
#                 break
#         if not args.fixed_lr:
#             agent.adjust_lr(i_episode)
#         rewards.append(episode_reward)
#         if (i_episode + 1) % args.eval_freq == 0:
#             tr_log = {'num_adversary': 0,
#                     'best_good_eval_reward': best_good_eval_reward,
#                     'best_adversary_eval_reward': best_adversary_eval_reward,
#                     'exp_save_dir': exp_save_dir, 'total_numsteps': total_numsteps,
#                     'value_loss': value_loss, 'policy_loss': policy_loss,
#                     'i_episode': i_episode, 'start_time': start_time}
#             copy_actor_policy(agent, eval_agent)
#             test_q.put([eval_agent, tr_log])

#     env.close()
#     time.sleep(5)
#     done_training.value = True

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

APPENDIX_REF = "refer appendix"

# Allowed configurations
SCENARIOS = [
    "simple_spread_n3", "simple_spread_n6",
    "simple_coop_push_n3", "simple_coop_push_n6",
    "simple_tag_n3", "simple_tag_n6"
]

ACTOR_TYPES = ["mlp", "gcn_max", "segnn"]
CRITIC_TYPES = ["mlp", "segnn"]


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
        # Validate scenario
        if self.scenario not in SCENARIOS:
            raise ValueError(f"Invalid scenario: {self.scenario}")

        # # Enforce special rule for simple_spread_n3
        # if self.scenario == "simple_spread_n3":
        #     self.actor_type = "mlp"
        #     self.critic_type = "mlp"

        # Build experiment name (same as CLI version)
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
    args = Config()
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
    # eval_agent = DDPG(
    #     gamma=args.gamma,
    #     continuous=args.continuous,
    #     obs_dim=obs_dim,
    #     n_action=n_action,
    #     n_agent=n_agent,
    #     obs_dims=obs_dims,
    #     action_range=action_range,
    #     actor_type=args.actor_type, critic_type=args.critic_type,
    #     actor_hidden_size=args.actor_hidden_size,
    #     critic_hidden_size=args.critic_hidden_size,
    #     actor_lr=args.actor_lr, critic_lr=args.critic_lr,
    #     fixed_lr=args.fixed_lr, num_episodes=args.num_episodes,
    #     train_noise=args.train_noise,
    #     target_update_mode=args.target_update_mode, tau=args.tau,
    #     device='cpu'
    # )

    # replay
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
        # For EMLP, save state dict and send None
        tr_log = {
            'exp_save_dir': exp_save_dir,
            'total_numsteps': total_numsteps,
            'i_episode': 0, 'start_time': start_time,
            'value_loss': None, 'policy_loss': None,
        }
        actor_path = os.path.join(exp_save_dir+"/temps/", f'temp_actor_ep{0}.pt')

        torch.save(agent.actor.state_dict(), actor_path)
        tr_log['actor_path'] = actor_path  # Pass the path in tr_log
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
                tr_log['actor_path'] = actor_path  # Pass the path in tr_log
                test_q.put([None, tr_log])
            else:
                copy_actor_policy(agent, eval_agent)

                test_q.put([eval_agent, tr_log])
            

    env.close()
    time.sleep(5)
    done_training.value = True