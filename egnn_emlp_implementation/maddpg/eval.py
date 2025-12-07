from maddpg.utils import make_env, dict2csv
import numpy as np
import contextlib
import torch
from maddpg.ckpt_plot.plot_curve import plot_result
import os
import time


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def eval_model_q(test_q, done_training, args):
    plot = {'rewards': [], 'steps': [], 'q_loss': [],
            'p_loss': [], 'final': [], 'abs': []}
    best_eval_reward = -100000000
    agent = None
    
    # Create eval agent once for EMLP (can't receive through queue)
    if args.actor_type == 'emlp':
        from maddpg.simple_ddpg import DDPG
        from maddpg.utils import gen_n_actions, gen_action_range
        
        eval_env_init = make_env(args.scenario, args.continuous, arglist=args)
        n_agent = eval_env_init.n
        n_actions = gen_n_actions(eval_env_init.action_space)
        n_action = n_actions[0]
        action_range = gen_action_range(eval_env_init.action_space)
        obs_dims = [eval_env_init.observation_space[i].shape[0] for i in range(n_agent)]
        obs_dim = obs_dims[0]
        obs_dims.insert(0, 0)
        eval_env_init.close()
        
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
            device='cpu'
        )
    
    while True:
        if not test_q.empty():
            print('=================== start eval ===================')
            eval_env = make_env(args.scenario, args.continuous, arglist=args)
            eval_env.seed(args.seed + 10)
            eval_rewards = []
            good_eval_rewards = []
            
            received_agent, tr_log = test_q.get()
            
            # For EMLP, load state dict from file
            if args.actor_type == 'emlp':
                # actor_path = os.path.join(tr_log['exp_save_dir'], 'temp_actor.pt')
                # agent.actor.load_state_dict(torch.load(actor_path, map_location='cpu'))
                actor_path = tr_log['actor_path']  # Get the specific path
                agent.actor.load_state_dict(torch.load(actor_path, map_location='cpu'))
            else:
                agent = received_agent
            
            agent.world, agent.scenario = eval_env.world, eval_env.scenario
            action_noise = True if not eval_env.continuous else False
            
            with temp_seed(args.seed):
                for n_eval in range(args.num_eval_runs):
                    obs_n, info = eval_env.reset()
                    episode_reward = 0
                    episode_step = 0
                    n_agents = eval_env.n
                    agents_rew = [[] for _ in range(n_agents)]
                    while True:
                        action_n = agent.select_action(np.array(obs_n), 
                                                       action_noise=action_noise,
                                                       param_noise=False).squeeze().cpu().numpy()
                        next_obs_n, reward_n, done_n, _ = eval_env.step(action_n)
                        episode_step += 1
                        terminal = (episode_step >= args.num_steps)
                        episode_reward += np.sum(reward_n)
                        for i, r in enumerate(reward_n):
                            agents_rew[i].append(r)
                        obs_n = next_obs_n
                        if done_n[0] or terminal:
                            eval_rewards.append(episode_reward)
                            agents_rew = [np.sum(rew) for rew in agents_rew]
                            good_reward = np.sum(agents_rew)
                            good_eval_rewards.append(good_reward)
                            break
                
                if np.mean(eval_rewards) > best_eval_reward:
                    best_eval_reward = np.mean(eval_rewards)
                    
                    # For EMLP, only save state dicts (can't pickle the whole agent)
                    if args.actor_type == 'emlp':
                        torch.save({
                            'actor_state_dict': agent.actor.state_dict(),
                            'critic_state_dict': agent.critic.state_dict(),
                        }, os.path.join(tr_log['exp_save_dir'], 'agents_best.ckpt'))
                    else:
                        agent.world, agent.scenario = None, None
                        torch.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 'agents_best.ckpt'))
                        agent.world, agent.scenario = eval_env.world, eval_env.scenario

                plot['rewards'].append(np.mean(eval_rewards))
                plot['steps'].append(tr_log['total_numsteps'])
                plot['q_loss'].append(tr_log['value_loss'])
                plot['p_loss'].append(tr_log['policy_loss'])
                print("========================================================")
                print(
                    "Episode: {}, total numsteps: {}, {} eval runs, total time: {} s".
                        format(tr_log['i_episode'], tr_log['total_numsteps'], args.num_eval_runs,
                               time.time() - tr_log['start_time']))
                print("reward: avg {} std {}, average reward last 10: {}, best reward {}".format(
                    np.mean(eval_rewards),
                    np.std(eval_rewards),
                    np.mean(plot['rewards'][-10:]),
                    best_eval_reward))
                plot['final'].append(np.mean(plot['rewards'][-10:]))
                plot['abs'].append(best_eval_reward)
                dict2csv(plot, os.path.join(tr_log['exp_save_dir'], 'train_curve.csv'))
                
                # Save periodic checkpoint
                print(tr_log)
                if args.actor_type == 'emlp':
                    torch.save({
                        'actor_state_dict': agent.actor.state_dict(),
                        'critic_state_dict': agent.critic.state_dict(),
                    }, os.path.join(tr_log['exp_save_dir'], 'agents_ep{}.ckpt'.format(tr_log['i_episode'] + 1)))
                else:
                    agent.world, agent.scenario = None, None
                    torch.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 
                                                               'agents_ep{}.ckpt'.format(tr_log['i_episode'] + 1)))
                    agent.world, agent.scenario = eval_env.world, eval_env.scenario
                    
                eval_env.close()
                
        if done_training.value and test_q.empty():
            if agent is not None:
                # Final save
                if args.actor_type == 'emlp':
                    torch.save({
                        'actor_state_dict': agent.actor.state_dict(),
                        'critic_state_dict': agent.critic.state_dict(),
                    }, os.path.join(tr_log['exp_save_dir'], 'agents.ckpt'))
                else:
                    agent.world, agent.scenario = None, None
                    torch.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 'agents.ckpt'))
            break