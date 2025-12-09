import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import permutations
from maddpg.utils import make_env
from maddpg.utils import make_env, gen_n_actions, gen_action_range
from maddpg.simple_ddpg import DDPG
import re

# ------------------------------------------------------------
# Permute observations + actions
# ------------------------------------------------------------

def permute_obs_action(obs_n, actions, perm):
    perm = list(perm)
    return obs_n[perm], actions[perm]


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

# ------------------------------------------------------------
# Load agent checkpoint
# ------------------------------------------------------------

# def load_agent(ckpt_path):
#     checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
#     return checkpoint['agents']
def load_agent(ckpt_path):
    """
    Loads an agent from a checkpoint.
    Automatically parses actor_type & critic_type,
    and handles reconstruction for EMLP.
    """

    actor_type, critic_type = parse_types_from_path(ckpt_path)
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

    return ckpt["agents"]


# ------------------------------------------------------------
# Collect full permutation invariance distribution
# ------------------------------------------------------------

def test_critic_full(agent, env, n_samples=100):
    n_agents = env.n
    variances, max_devs = [], []

    agent.actor.eval()
    agent.critic.eval()

    obs_n, info = env.reset()

    with torch.no_grad():
        for _ in range(n_samples):
            obs = np.array(obs_n)
            act = agent.select_action(obs, action_noise=False, param_noise=False)
            act = act.squeeze().cpu().numpy()

            # pick up to 24 random permutations
            perms = list(permutations(range(n_agents)))
            if len(perms) > 24:
                idx = np.random.choice(len(perms), 24, replace=False)
                perms = [perms[i] for i in idx]

            q_vals = []
            for perm in perms:
                p_obs, p_act = permute_obs_action(obs, act, perm)
                o = torch.FloatTensor(p_obs.flatten()).unsqueeze(0)
                a = torch.FloatTensor(p_act.flatten()).unsqueeze(0)
                q = agent.critic(o, a).cpu().numpy().squeeze()
                q_vals.append(q)

            q_vals = np.array(q_vals)
            variances.append(np.var(q_vals))
            max_devs.append(np.max(q_vals) - np.min(q_vals))

            obs_n, _, done_n, info = env.step(act)
            if done_n[0]:
                obs_n, info = env.reset()

    return variances, max_devs


# ------------------------------------------------------------
# Extract latest checkpoint
# ------------------------------------------------------------

def get_latest_checkpoint(exp_dir):
    eps = []
    for f in os.listdir(exp_dir):
        if f.startswith("agents_ep") and f.endswith(".ckpt"):
            ep = int(f.replace("agents_ep", "").replace(".ckpt", ""))
            eps.append(ep)
    if len(eps) == 0:
        return None
    best_ep = max(eps)
    return os.path.join(exp_dir, f"agents_ep{best_ep}.ckpt"), best_ep


# ------------------------------------------------------------
# Gather distributions from best checkpoints
# ------------------------------------------------------------

def collect_boxplot_data(exp_dirs, scenario, continuous, seed=0, n_samples=200):
    env = make_env(scenario, continuous, arglist=None)
    env.seed(seed)

    results = {}

    for name, exp_dir in exp_dirs.items():
        ckpt_path, ep = get_latest_checkpoint(exp_dir)
        print(f"[{name}] Using best ep {ep}")

        agent = load_agent(ckpt_path)
        vars_list, dev_list = test_critic_full(agent, env, n_samples)
        results[name] = {
            "variances": vars_list,
            "max_devs": dev_list,
        }

    env.close()
    return results


# ------------------------------------------------------------
# Plot boxplots
# ------------------------------------------------------------
def plot_boxplots(results, save_path="perm_boxplots"):
    """
    Plot separate boxplots for permutation variance and max deviation.
    
    Args:
        results: dict like {exp_name: {"variances": [...], "max_devs": [...]} }
        save_path: prefix for saving plots
    """
    names = list(results.keys())
    var_data = [results[n]["variances"] for n in names]
    dev_data = [results[n]["max_devs"] for n in names]

    # --------------------------
    # Variance boxplot
    # --------------------------
    fig_var, ax_var = plt.subplots(figsize=(8, 5))
    ax_var.boxplot(var_data, tick_labels=names, showfliers=True)
    ax_var.set_title("Permutation Variance")
    ax_var.set_yscale("log")
    ax_var.grid(True, alpha=0.3, linestyle="--")
    ax_var.set_ylabel("Variance")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig_var.savefig(f"{save_path}_variance.png", dpi=150)
    plt.close(fig_var)

    # --------------------------
    # Max deviation boxplot
    # --------------------------
    fig_dev, ax_dev = plt.subplots(figsize=(8, 5))
    ax_dev.boxplot(dev_data, tick_labels=names, showfliers=True)
    ax_dev.set_title("Permutation Max Deviation")
    ax_dev.set_yscale("log")
    ax_dev.grid(True, alpha=0.3, linestyle="--")
    ax_dev.set_ylabel("Max Deviation")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig_dev.savefig(f"{save_path}_max_dev.png", dpi=150)
    plt.close(fig_dev)






# ------------------------------------------------------------
# MAIN (boxplots only)
# ------------------------------------------------------------

if __name__ == "__main__":
    exp_dirs = {
        "MLP-MLP": "ckpt_plot2/simple_spread_n3_continuous_actor_mlp_lr_0.0001_critic_mlp_lr_0.001_fixed_lr_batch_size_1024_actor_clip_grad_norm_0.5_seed1",
        "MLP-GCN": "ckpt_plot2/simple_spread_n3_continuous_actor_mlp_lr_0.0001_critic_gcn_max_lr_0.001_fixed_lr_batch_size_1024_actor_clip_grad_norm_0.5_seed1",
        "EGNN-MLP": "ckpt_plot2/simple_spread_n3_continuous_actor_egnn_lr_0.0001_critic_mlp_lr_0.001_fixed_lr_batch_size_1024_actor_clip_grad_norm_0.5_seed1",
        "EGNN-GCN": "ckpt_plot2/simple_spread_n3_continuous_actor_egnn_lr_0.0001_critic_gcn_max_lr_0.001_fixed_lr_batch_size_1024_actor_clip_grad_norm_0.5_seed1",
        "EMLP-MLP": "ckpt_plot2/simple_spread_n3_continuous_actor_emlp_lr_0.0001_critic_mlp_lr_0.001_fixed_lr_batch_size_1024_actor_clip_grad_norm_0.5_seed1",
        "EMLP-GCN": "ckpt_plot2/simple_spread_n3_continuous_actor_emlp_lr_0.0001_critic_gcn_max_lr_0.001_fixed_lr_batch_size_1024_actor_clip_grad_norm_0.5_seed1",

    }

    scenario = "simple_spread_n3"
    continuous = True

    results = collect_boxplot_data(
        exp_dirs,
        scenario,
        continuous,
        seed=42,
        n_samples=200
    )

    plot_boxplots(results, save_path="results/permutation_invariance_boxplots")
