
def build_exp_name(cfg):
    """Build experiment name from configuration"""
    exp_name = cfg['scenario']
    exp_name += '_' + ('continuous' if cfg['continuous'] else 'discrete')
    exp_name += f"_actor_{cfg['actor_type']}_lr_{cfg['actor_lr']}"
    exp_name += f"_critic_{cfg['critic_type']}_lr_{cfg['critic_lr']}"
    exp_name += '_fixed_lr' if cfg['fixed_lr'] else ''
    exp_name += f"_batch_size_{cfg['batch_size']}"
    exp_name += f"_actor_clip_grad_norm_{cfg['actor_clip_grad_norm']}"
    exp_name += f"_seed{cfg['seed']}"
    return exp_name


def setup_environment(cfg):
    """Initialize environment and extract relevant information"""
    env = make_env(cfg['scenario'], cfg['continuous'], arglist=None)
    scenario, world = env.scenario, env.world
    
    env.n_agent = n_agent = env.n
    env.n_actions = n_actions = gen_n_actions(env.action_space)
    n_action = n_actions[0]
    env.action_range = action_range = gen_action_range(env.action_space)
    
    obs_dims = [env.observation_space[i].shape[0] for i in range(n_agent)]
    obs_dim = obs_dims[0]
    obs_dims.insert(0, 0)
    
    obs_n, info = env.reset()
    sample_actions = sample_action_n(env.action_space)
    
    # Generate graph info for actor and critic
    _, obs_graph_info = scenario.gen_obs_graph(
        batch_obs=obs_n[0],
        lmax_attr=cfg['lmax_attr'],
        node_input_type=cfg['node_input_type'],
        world=world,
        gen_graph_info=True
    )
    
    _, state_action_n_graph_info = scenario.gen_state_action_n_graph(
        batch_s=info['state'],
        batch_action_n=sample_actions,
        lmax_attr=cfg['lmax_attr'],
        node_input_type=cfg['node_input_type'],
        world=world,
        gen_graph_info=True
    )
    
    return {
        'env': env,
        'scenario': scenario,
        'world': world,
        'n_agent': n_agent,
        'n_action': n_action,
        'obs_dim': obs_dim,
        'obs_dims': obs_dims,
        'action_range': action_range,
        'obs_graph_info': obs_graph_info,
        'state_action_n_graph_info': state_action_n_graph_info,
    }


def create_agent(cfg, env_info, device):
    """Create E3DDPG agent"""
    return E3DDPG(
        gamma=cfg['gamma'],
        continuous=cfg['continuous'],
        obs_dim=env_info['obs_dim'],
        n_action=env_info['n_action'],
        n_agent=env_info['n_agent'],
        obs_dims=env_info['obs_dims'],
        action_range=env_info['action_range'],
        actor_graph_info=env_info['obs_graph_info'],
        critic_graph_info=env_info['state_action_n_graph_info'],
        actor_type=cfg['actor_type'],
        critic_type=cfg['critic_type'],
        actor_hidden_size=cfg['actor_hidden_size'],
        critic_hidden_size=cfg['critic_hidden_size'],
        lmax_attr=cfg['lmax_attr'],
        node_input_type=cfg['node_input_type'],
        actor_lr=cfg['actor_lr'],
        critic_lr=cfg['critic_lr'],
        fixed_lr=cfg['fixed_lr'],
        num_episodes=cfg['num_episodes'],
        train_noise=cfg['train_noise'],
        target_update_mode=cfg['target_update_mode'],
        tau=cfg['tau'],
        device=device,
        scenario=env_info['scenario'],
        world=env_info['world'],
    )

