import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

from maddpg.utils import make_env, gen_n_actions, gen_action_range
from maddpg.simple_ddpg import DDPG
def load_agent(ckpt_path, env):
    """
    Loads an agent from a checkpoint.
    Automatically parses actor_type & critic_type,
    and handles reconstruction for EMLP.
    """

    actor_type, critic_type = parse_types_from_path2(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # EMLP = saved as state_dicts only
    if actor_type == "emlp":
        env = make_env("simple_spread_n3", True, arglist=None)
        env.n_agent = n_agent = env.n
        env.n_actions = n_actions = gen_n_actions(env.action_space)
        n_action = n_actions[0]
        env.action_range = action_range = gen_action_range(env.action_space)
        obs_dims = [env.observation_space[i].shape[0] for i in range(n_agent)]
        obs_dim = obs_dims[0]
        obs_dims.insert(0, 0)
        env.close()
    
        agent = DDPG(
                gamma=0.95,
                continuous=True,
                obs_dim=obs_dim,
                n_action=n_action,
                n_agent=n_agent,
                obs_dims=obs_dims,
                action_range=action_range,

                # Only these vary
                actor_type=actor_type,
                critic_type=critic_type,

                # Hardcoded settings
                actor_hidden_size=128,
                critic_hidden_size=128,
                actor_lr=0.0001,
                critic_lr=0.001,
                fixed_lr=True,
                num_episodes=62000,
                train_noise=False,
                target_update_mode='soft',
                tau=0.01,
                device='cpu'
            )

        agent.actor.load_state_dict(ckpt["actor_state_dict"])
        agent.critic.load_state_dict(ckpt["critic_state_dict"])
        return agent

    # EGNN / MLP checkpoints saved full agent → load directly
    return ckpt["agents"]


def rotation_matrix(angle_deg):
    theta = np.radians(angle_deg)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ], dtype=np.float32)


def rotate_obs(obs_n, R):
    obs_n = obs_n.copy()
    for agent in range(3):
        for vec in range(7):
            idx = vec * 2
            obs_n[agent, idx:idx+2] = R @ obs_n[agent, idx:idx+2]
    return obs_n


def test_equivariance_egnn(agent, obs_n, angle_deg=90):
    R = rotation_matrix(angle_deg)
    obs_n = np.array(obs_n, dtype=np.float32)
    
    obs_batch = obs_n[np.newaxis, ...]
    h, x, v, edges, edge_attr = agent.build_graph_from_obs_batch(obs_batch)
    
    with torch.no_grad():
        h_out, x_out, v_out = agent.actor(h, x, v, edges, edge_attr)
    action_original = x_out[:3].cpu().numpy()
    
    obs_rotated = rotate_obs(obs_n, R)
    obs_rotated_batch = obs_rotated[np.newaxis, ...]
    h_rot, x_rot, v_rot, edges_rot, edge_attr_rot = agent.build_graph_from_obs_batch(obs_rotated_batch)
    
    with torch.no_grad():
        h_out_rot, x_out_rot, v_out_rot = agent.actor(h_rot, x_rot, v_rot, edges_rot, edge_attr_rot)
    action_from_rotated = x_out_rot[:3].cpu().numpy()
    
    action_original_rotated = action_original @ R.T
    max_error = np.max(np.abs(action_original_rotated - action_from_rotated))
    
    return max_error


def test_equivariance_mlp(agent, obs_n, angle_deg=90):
    R = rotation_matrix(angle_deg)
    obs_n = np.array(obs_n, dtype=np.float32)
    
    obs_tensor = torch.tensor(obs_n, dtype=torch.float32)
    with torch.no_grad():
        action_original = agent.actor(obs_tensor).cpu().numpy()
    
    obs_rotated = rotate_obs(obs_n, R)
    obs_rotated_tensor = torch.tensor(obs_rotated, dtype=torch.float32)
    with torch.no_grad():
        action_from_rotated = agent.actor(obs_rotated_tensor).cpu().numpy()
    
    action_original_rotated = action_original @ R.T
    max_error = np.max(np.abs(action_original_rotated - action_from_rotated))
    
    return max_error

def rotate_obs_single(obs, R):
    """Rotate a single agent's observation (14-dim: 7 vectors of 2D each)"""
    obs = obs.copy()
    for vec in range(7):
        idx = vec * 2
        obs[idx:idx+2] = R @ obs[idx:idx+2]
    return obs


def test_equivariance_emlp(agent, obs_n, angle_deg=90):
    """
    Test equivariance for EMLP actor.
    EMLP processes each agent independently with input shape (B, 14).
    """
    R = rotation_matrix(angle_deg)
    obs_n = np.array(obs_n, dtype=np.float32)
    
    # EMLP expects (batch, 14) per agent - process all agents
    obs_tensor = torch.tensor(obs_n, dtype=torch.float32)  # (3, 14)
    with torch.no_grad():
        action_original = agent.actor(obs_tensor).cpu().numpy()  # (3, 2)
    
    # Rotate observations for all agents
    obs_rotated = np.array([rotate_obs_single(obs_n[i], R) for i in range(3)])
    obs_rotated_tensor = torch.tensor(obs_rotated, dtype=torch.float32)
    with torch.no_grad():
        action_from_rotated = agent.actor(obs_rotated_tensor).cpu().numpy()  # (3, 2)
    
    # Rotate original actions and compare
    action_original_rotated = action_original @ R.T
    max_error = np.max(np.abs(action_original_rotated - action_from_rotated))
    
    return max_error

def test_multiple_obs(agent, env, actor_type, n_samples=50, angle_deg=90):
    errors = []
    for _ in range(n_samples):
        obs_n, _ = env.reset()
        obs_n = np.array(obs_n, dtype=np.float32)
        
        if actor_type == 'egnn':
            error = test_equivariance_egnn(agent, obs_n, angle_deg)
        elif actor_type == 'emlp':
            error = test_equivariance_emlp(agent, obs_n, angle_deg)
        else:  # mlp
            error = test_equivariance_mlp(agent, obs_n, angle_deg)
        errors.append(error)
    
    return np.mean(errors), np.std(errors)
import re

def parse_types_from_path(ckpt_path):
    """
    Extract actor_type and critic_type from experiment directory name.

    Works for:
      actor_mlp
      actor_emlp
      actor_egnn
      actor_anything_with_underscores
      critic_mlp
      critic_gcn_max
      critic_anything_with_underscores
    """

    folder = os.path.basename(os.path.dirname(ckpt_path))

    # Regex examples:
    #   actor_(.*?)_lr
    #   critic_(.*?)_lr

    actor_match = re.search(r'actor_(.*?)_lr', folder)
    critic_match = re.search(r'critic_(.*?)_lr', folder)

    if actor_match is None:
        raise ValueError(f"Could not parse actor_type from: {folder}")
    if critic_match is None:
        raise ValueError(f"Could not parse critic_type from: {folder}")

    actor_type = actor_match.group(1)
    critic_type = critic_match.group(1)

    return actor_type, critic_type


def get_checkpoints(exp_dir):
    ckpt_files = sorted(glob.glob(os.path.join(exp_dir, 'agents_ep*.ckpt')))
    episodes = []
    for f in ckpt_files:
        ep = int(f.split('_ep')[-1].split('.ckpt')[0])
        episodes.append(ep)
    return ckpt_files, episodes
def plot_comparison_multi(exp_dirs, angles=[30, 45, 90, 180], n_samples=50):
    """
    Generalized equivariance-over-training plot:
      - Accepts a list of experiment directories
      - Parses actor and critic types automatically
      - Uses correct equivariance test per actor
      - Plots ONLY mean lines (NO std / variance shading)
    """

    # ---------------------------------------
    # Setup
    # ---------------------------------------
    env = make_env('simple_spread_n3', True, arglist=None)

    # Colors / markers
    pastel_colors = ['#F08787', '#96A78D', '#BAE1FF', '#696FC7']
    markers = ['o', 's', '^', 'D']

    # ---------------------------------------
    # Prepare figure
    # ---------------------------------------
    n_plots = len(exp_dirs)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    axes = np.array(axes).reshape(-1)

    fig.patch.set_facecolor('#FFF5F5')

    # ---------------------------------------
    # Storage of results for summary
    # ---------------------------------------
    all_results = {}

    # ---------------------------------------
    # Process each experiment directory
    # ---------------------------------------
    for idx, exp_dir in enumerate(exp_dirs):
        ax = axes[idx]

        # ----------------------------
        # Parse experiment name
        # ----------------------------
        actor_type, critic_type = parse_types_from_path2(
            os.path.join(exp_dir, "agents_best.ckpt")
        )

        exp_name = f"{actor_type.upper()} (critic_{critic_type})"
        print(f"\n=== Processing {exp_name} ===")

        # ----------------------------
        # Load checkpoints
        # ----------------------------
        ckpts, episodes = get_checkpoints(exp_dir)

        # Storage
        results = {angle: {'mean': []} for angle in angles}

        # Determine test function
        if actor_type == "egnn":
            test_fn = test_equivariance_egnn
        elif actor_type == "emlp":
            test_fn = test_equivariance_emlp
        elif actor_type.startswith("gcn"):
            test_fn = test_equivariance_gcn
        else:
            test_fn = test_equivariance_mlp

        # ----------------------------
        # Run tests across checkpoints
        # ----------------------------
        for ckpt_path, ep in zip(ckpts, episodes):
            print(f"  Episode {ep}")

            agent = load_agent(ckpt_path, env)

            for angle in angles:
                mean_err, _ = test_multiple_obs(
                    agent, env, actor_type, n_samples, angle
                )
                results[angle]["mean"].append(mean_err)

        all_results[exp_name] = (results, episodes)

        # ----------------------------
        # Plot this experiment
        # ----------------------------
        ax.set_facecolor('#FFFAF0')

        for angle, color, marker in zip(angles, pastel_colors, markers):
            means = np.array(results[angle]['mean'])
            means_plot = np.maximum(means, 1e-10)

            ax.plot(
                episodes, means_plot,
                '-', marker=marker, label=f'{angle}°',
                markersize=5, color=color, linewidth=2.5,
                markeredgecolor='black', markeredgewidth=0.5
            )

        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Max Equivariance Error', fontsize=12, fontweight='bold')
        ax.set_title(
            f'{actor_type.upper()} (critic={critic_type})',
            fontsize=14, fontweight='bold', color='#555555'
        )
        ax.set_yscale('log')
        ax.set_ylim(1e-11, 1e-5)
        ax.legend(fontsize=10, facecolor='white', edgecolor='#CCCCCC', framealpha=0.9)
        ax.grid(True, alpha=0.4, linestyle='--', color='#CCCCCC')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ---------------------------------------
    # Final layout
    # ---------------------------------------
    plt.tight_layout()
    plt.savefig('results/equivariance_over_training_multi_lines.png', dpi=150, facecolor=fig.get_facecolor())
    plt.show()

    env.close()

    # ---------------------------------------
    # Summary Printing
    # ---------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    header = f"{'Angle':<10}" + "".join([f"{name:<20}" for name in all_results.keys()])
    print(header)
    print("-" * len(header))

    for angle in angles:
        row = f"{angle}°{'':<6}"
        for name, (results, _) in all_results.items():
            err = np.mean(results[angle]['mean'])
            row += f"{err:<20.2e}"
        print(row)

    print("-" * len(header))
    row = f"{'Overall':<10}"
    for name, (results, _) in all_results.items():
        overall = np.mean([np.mean(results[a]['mean']) for a in angles])
        row += f"{overall:<20.2e}"
    print(row)

    return all_results

def plot_boxplot_best_models(exp_dirs, angle_deg=30, n_samples=100):
    """
    Compute and plot a boxplot of equivariance errors for 'best' checkpoints
    across a list of experiment directories.

    Labels include both actor and critic types.
    """

    env = make_env('simple_spread_n3', True, arglist=None)

    agents = {}
    labels = []

    # -------------------------------------------------------
    # Load agents + parse actor/critic types
    # -------------------------------------------------------
    for exp_dir in exp_dirs:
        ckpt_path = os.path.join(exp_dir, "agents_best.ckpt")
        if not os.path.exists(ckpt_path):
            print(f"Warning: missing best checkpoint in {exp_dir}")
            continue

        agent = load_agent(ckpt_path, env)

        actor_type, critic_type = parse_types_from_path2(ckpt_path)

        label = f"{actor_type.upper()} (critic: {critic_type})"

        agents[label] = (agent, actor_type)
        labels.append(label)

    # -------------------------------------------------------
    # Compute equivariance errors
    # -------------------------------------------------------
    errors = {label: [] for label in labels}

    for label, (agent, actor_type) in agents.items():
        for _ in range(n_samples):
            obs_n, _ = env.reset()
            obs_n = np.array(obs_n, dtype=np.float32)

            if actor_type == "egnn":
                err = test_equivariance_egnn(agent, obs_n, angle_deg)

            elif actor_type == "emlp":
                err = test_equivariance_emlp(agent, obs_n, angle_deg)

            elif actor_type == "mlp":
                err = test_equivariance_mlp(agent, obs_n, angle_deg)

            else:
                raise ValueError(f"Unknown actor_type '{actor_type}' parsed from path.")

            errors[label].append(err)

    env.close()

    # -------------------------------------------------------
    # Plot
    # -------------------------------------------------------
    data = [errors[k] for k in labels]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.yscale("log")
    plt.ylabel("Equivariance Error (log scale)")
    plt.title(f"Equivariance Error Distribution at {angle_deg}° (Best Models)", fontsize=13)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"results/boxplot_equivariance_{angle_deg}.png", dpi=150)
    plt.show()

    return errors



def parse_types_from_path(ckpt_path):
    """Extract actor_type and critic_type from experiment directory name."""
    folder = os.path.basename(ckpt_path.rstrip('/'))
    print("folder")
    print(folder)
    
    actor_match = re.search(r'actor_(.*?)_lr', folder)
    critic_match = re.search(r'critic_(.*?)_lr', folder)
    
    if actor_match is None:
        raise ValueError(f"Could not parse actor_type from: {folder}")
    if critic_match is None:
        raise ValueError(f"Could not parse critic_type from: {folder}")
    
    return actor_match.group(1), critic_match.group(1)


def parse_types_from_path2(ckpt_path):
    """Extract actor_type and critic_type from experiment directory name."""
    folder = os.path.basename(ckpt_path.rstrip('/'))

    actor_match = re.search(r'actor_(.*?)_lr', ckpt_path)
    critic_match = re.search(r'critic_(.*?)_lr', ckpt_path)
    
    if actor_match is None:
        raise ValueError(f"Could not parse actor_type from: {folder}")
    if critic_match is None:
        raise ValueError(f"Could not parse critic_type from: {folder}")
    
    return actor_match.group(1), critic_match.group(1)


def load_train_curve(exp_dir):
    """Load training curve from CSV file."""
    csv_path = os.path.join(exp_dir, 'train_curve.csv')
    
    rewards, steps = [], []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('rewards,'):
                parts = line.split(',')[1:]
                rewards = [float(x) for x in parts if x]
            elif line.startswith('steps,'):
                parts = line.split(',')[1:]
                steps = [int(x) for x in parts if x]
    
    return np.array(steps), np.array(rewards)


def plot_training_curves(exp_dirs, window=5, save_path='results/training_curves_comparison.png'):
    """
    Plot training curves comparison across multiple experiments.
    
    Args:
        exp_dirs: List of experiment directory paths
        window: Smoothing window size for moving average
        save_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#FFF5F5')
    ax.set_facecolor('#FFFAF0')
    
    colors = ['#F08787', '#96A78D', '#BAE1FF', '#696FC7', '#FFB347', '#87CEEB']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for idx, exp_dir in enumerate(exp_dirs):
        actor_type, critic_type = parse_types_from_path(exp_dir)
        label = f"{actor_type.upper()} (critic: {critic_type})"
        
        steps, rewards = load_train_curve(exp_dir)
        
        # Apply smoothing
        if window > 1 and len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            steps_smooth = steps[:len(smoothed)]
        else:
            smoothed, steps_smooth = rewards, steps
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # Plot raw data with transparency
        ax.plot(steps, rewards, alpha=0.3, color=color, linewidth=1)
        # Plot smoothed curve
        ax.plot(steps_smooth, smoothed, '-', marker=marker, label=label,
                markersize=4, color=color, linewidth=2.5,
                markeredgecolor='black', markeredgewidth=0.5,
                markevery=max(1, len(steps_smooth)//10))
    
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Episode Reward', fontsize=12, fontweight='bold')
    ax.set_title('Training Curves Comparison', fontsize=14, fontweight='bold', color='#555555')
    ax.legend(fontsize=10, facecolor='white', edgecolor='#CCCCCC', framealpha=0.9)
    ax.grid(True, alpha=0.4, linestyle='--', color='#CCCCCC')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.show()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"{'Model':<35} {'Final Reward':<15} {'Best Reward':<15}")
    print("-" * 60)
    
    for exp_dir in exp_dirs:
        actor_type, critic_type = parse_types_from_path(exp_dir)
        label = f"{actor_type.upper()} (critic: {critic_type})"
        steps, rewards = load_train_curve(exp_dir)
        print(f"{label:<35} {rewards[-1]:<15.2f} {np.max(rewards):<15.2f}")
    
    return fig, ax
if __name__ == '__main__':
    egnn_mlp = "ckpt_plot2/simple_spread_n3_continuous_actor_egnn_lr_0.0001_critic_mlp_lr_0.001_fixed_lr_batch_size_1024_actor_clip_grad_norm_0.5_seed1"
    egnn_gcn_max = "ckpt_plot2/simple_spread_n3_continuous_actor_egnn_lr_0.0001_critic_gcn_max_lr_0.001_fixed_lr_batch_size_1024_actor_clip_grad_norm_0.5_seed1"
    
    mlp_mlp = "ckpt_plot2/simple_spread_n3_continuous_actor_mlp_lr_0.0001_critic_mlp_lr_0.001_fixed_lr_batch_size_1024_actor_clip_grad_norm_0.5_seed1"
    mlp_gcn_max = "ckpt_plot2/simple_spread_n3_continuous_actor_mlp_lr_0.0001_critic_gcn_max_lr_0.001_fixed_lr_batch_size_1024_actor_clip_grad_norm_0.5_seed1"
    
    emlp_mlp = "ckpt_plot2/simple_spread_n3_continuous_actor_emlp_lr_0.0001_critic_mlp_lr_0.001_fixed_lr_batch_size_1024_actor_clip_grad_norm_0.5_seed1"
    emlp_gcn_max = "ckpt_plot2/simple_spread_n3_continuous_actor_emlp_lr_0.0001_critic_gcn_max_lr_0.001_fixed_lr_batch_size_1024_actor_clip_grad_norm_0.5_seed1"

    exp_dirs=[mlp_mlp,mlp_gcn_max,egnn_mlp,egnn_gcn_max,emlp_mlp,emlp_gcn_max]
    
    # plot_comparison_multi(
    #     exp_dirs,
    #     angles=[15],
    #     n_samples=50
    # )


    plot_boxplot_best_models(
   exp_dirs,
    angle_deg=15,
    n_samples=100
)
    plot_training_curves(exp_dirs, window=5)


