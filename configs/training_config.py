"""Configuration file for E3DDPG training on simple_spread_n3"""

config = {
    # Environment
    'scenario': 'simple_spread_n3',
    'continuous': True,
    'num_steps': 25,
    'seed': 1,
    
    # Network architecture
    'actor_type': 'mlp',  # Options: 'mlp', 'gcn_max', 'segnn'
    'actor_hidden_size': 128,
    'critic_type': 'mlp',  # Options: 'mlp', 'gcn_max', 'segnn'
    'critic_hidden_size': 128,
    
    # SEGNN specific (only used if actor_type or critic_type is 'segnn')
    'lmax_attr': 3,
    'node_input_type': '',  # Options: '', 'pos'
    
    # Training
    'gamma': 0.95,
    'train_noise': False,
    'target_update_mode': 'soft',  # Options: 'soft', 'hard', 'episodic'
    'tau': 0.01,
    'num_episodes': 62000,
    
    # Replay buffer
    'replay_size': 1000000,
    'batch_size': 1024,
    
    # Update frequencies
    'steps_per_actor_update': 100,
    'steps_per_critic_update': 100,
    'actor_updates_per_step': 8,
    'critic_updates_per_step': 8,
    
    # Learning rates
    'actor_lr': 1e-2,
    'actor_clip_grad_norm': 0.5,
    'critic_lr': 1e-2,
    'fixed_lr': True,
    
    # Exploration
    'n_exploration_eps': 25000,
    'init_noise_scale': 0.3,
    'final_noise_scale': 0.0,
    
    # Evaluation
    'eval_freq': 1000,
    'num_eval_runs': 200,
    
    # Logging
    'save_dir': './ckpt_plot',
    
    # Hardware
    'cuda': True,
}
