# test_invariances.py
import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from maddpg import MADDPG
from utils.env import create_single_env, get_env_info


"""
This script evaluates geometric invariance properties of two trained MADDPG models:
one using an EKAN actor with a permutation-invariant critic (Run A),
and one using an EKAN actor with a traditional critic (Run B).

The script performs the following:

1. Load Models
   - Loads checkpoints for Run A and Run B.
   - Infers EKAN hidden dimensions from checkpoint weights.
   - Builds corresponding MADDPG agents and loads parameters.

2. Define Rotation & Permutation Utilities
   - rot2(): constructs a 2D rotation matrix.
   - rotate_pairs(): rotates selected (x, y) index pairs in a vector.
   - rotate_batch_obs(), rotate_batch_actions(): apply rotation to observations/actions.
   - flatten_sa(): concatenates per-agent states and actions.
   - permute_blocks(): permutes agent state/action blocks to simulate reordering.

3. Actor SO(2) Equivariance Test
   - Rotates selected observation coordinates by angle θ.
   - Gets actions on original and rotated observations.
   - Rotates the original actions for comparison.
   - Measures equivariance error.
   - Returns mean, max, and per-batch errors.

4. Critic Permutation Invariance Test
   - Samples random joint states and actions.
   - Computes Q(s, a) and Q value of swapped agents.
   - Returns mean, max, and per-batch differences.

5. Output
   - Stores all results in a JSON file including:
     * actor equivariance metrics,
     * critic permutation invariance metrics,
     * inferred EKAN hidden sizes,
     * environment and test configuration.

"""


#  small linear algebra helpers 


def rot2(theta: float):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def rotate_pairs(vec: np.ndarray, pair_indices: List[Tuple[int, int]], theta: float):
    R = rot2(theta)
    out = vec.copy()
    for ix, iy in pair_indices:
        v = np.array([vec[ix], vec[iy]], dtype=np.float32)
        v2 = R @ v
        out[ix], out[iy] = v2[0], v2[1]
    return out


def rotate_batch_obs(
    obs_batch: List[np.ndarray],
    per_agent_xy: Dict[int, List[Tuple[int, int]]],
    theta: float,
) -> List[np.ndarray]:
    out = []
    for aid, obs in enumerate(obs_batch):
        out.append(rotate_pairs(obs, per_agent_xy.get(aid, []), theta))
    return out


def rotate_batch_actions(
    act_batch: List[np.ndarray],
    per_agent_xy: Dict[int, List[Tuple[int, int]]],
    theta: float,
) -> List[np.ndarray]:
    out = []
    for aid, act in enumerate(act_batch):
        out.append(rotate_pairs(act, per_agent_xy.get(aid, []), theta))
    return out


def flatten_sa(
    states_list: List[np.ndarray], actions_list: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    s_full = np.concatenate(states_list, axis=0)
    a_full = np.concatenate(actions_list, axis=0)
    return s_full, a_full


def permute_blocks(
    x: np.ndarray, block_sizes: List[int], perm: List[int]
) -> np.ndarray:
    blocks = []
    i = 0
    for sz in block_sizes:
        blocks.append(x[i : i + sz])
        i += sz
    blocks = [blocks[p] for p in perm]
    return np.concatenate(blocks, axis=0)


#  checkpoint inspection & arch inference 


def _get_actor_sd(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Support either a flat dict or a nested {'models': {...}} payload.
    """
    if "models" in ckpt and isinstance(ckpt["models"], dict):
        pool = ckpt["models"]
    else:
        pool = ckpt
    # prefer agent_0 as canonical (actors should share architecture)
    key = "agent_0_actor"
    if key not in pool:
        # fallback: find any *_actor
        for k in list(pool.keys()):
            if k.endswith("_actor"):
                key = k
                break
    if key not in pool:
        raise RuntimeError("No actor weights found in checkpoint.")
    return pool[key]


def infer_head_hidden_from_ckpt(ckpt_path: str) -> int:
    """
    Infer the first head hidden width from comm_head.0.weight shape.
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _get_actor_sd(payload)
    if "comm_head.0.weight" not in sd:
        # Fallback: try another common naming
        for k in sd.keys():
            if k.endswith("comm_head.0.weight"):
                w = sd[k]
                return int(w.shape[0])
        raise RuntimeError("Could not find 'comm_head.0.weight' in actor state_dict.")
    w = sd["comm_head.0.weight"]
    return int(w.shape[0])


def print_ekan_hints(ckpt_path: str, tag: str):
    """
    Print EKAN layer hints so you can align EKAN config (width/grid/k) if needed.
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _get_actor_sd(payload)
    print(f"\n== EKAN layer hints ({tag}) ==")
    for k, v in sd.items():
        if not hasattr(v, "shape"):
            continue
        if (
            k.startswith("ekan.act_fun.")
            and (
                "linear.weight" in k or "linear.bias" in k or "grid" in k or "bi_w" in k
            )
        ) or k.startswith("comm_head.0.weight"):
            print(f"{k:50s} {tuple(v.shape)}")
    print("== end EKAN hints ==\n")


#  MADDPG build/load wrappers 


def build_maddpg_for_run(
    state_sizes,
    action_sizes,
    action_low,
    action_high,
    hidden_sizes: Tuple[int, ...],
    actor_kind: str,
    critic_kind: str,
) -> MADDPG:
    """
    Build one MADDPG instance for a given run with per-run hidden sizes.
    """
    agent = MADDPG(
        state_sizes=state_sizes,
        action_sizes=action_sizes,
        hidden_sizes=hidden_sizes,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.01,
        action_low=action_low,
        action_high=action_high,
        actor=actor_kind,
        critic=critic_kind,
    )
    return agent


def load_into(maddpg: MADDPG, ckpt_path: str):
    maddpg.load(ckpt_path)


#  actor & critic tests 


def maddpg_act(
    maddpg: MADDPG, states_list: List[np.ndarray], noise=False
) -> List[np.ndarray]:
    return maddpg.act(states_list, add_noise=noise)


def maddpg_Q(maddpg, s_full: np.ndarray, a_full: np.ndarray) -> float:
    q_vals = []
    critic_dev = next(maddpg.agents[0].critic.parameters()).device
    s = torch.as_tensor(s_full, dtype=torch.float32, device=critic_dev).unsqueeze(0)
    a = torch.as_tensor(a_full, dtype=torch.float32, device=critic_dev).unsqueeze(0)
    with torch.no_grad():
        for ag in maddpg.agents:
            q = ag.critic(s, a)
            q_vals.append(float(q.squeeze().cpu().numpy()))
    return float(np.mean(q_vals))


def actor_equivariance_test(
    maddpg: MADDPG,
    env_name: str,
    per_agent_obs_xy: Dict[int, List[Tuple[int, int]]],
    per_agent_act_xy: Dict[int, List[Tuple[int, int]]],
    n_batches=64,
    theta_deg=15.0,
):
    theta = math.radians(theta_deg)
    env = create_single_env(
        env_name, max_steps=50, render_mode=None, apply_padding=False
    )
    obs, _ = env.reset()
    agents = list(obs.keys())
    errors = []
    with torch.no_grad():
        for _ in range(n_batches):
            states = [np.array(obs[a], dtype=np.float32) for a in agents]
            acts = maddpg_act(maddpg, states, noise=False)

            states_rot = rotate_batch_obs(states, per_agent_obs_xy, theta)
            acts_from_rot = maddpg_act(maddpg, states_rot, noise=False)

            acts_rot = rotate_batch_actions(acts, per_agent_act_xy, theta)

            err = np.mean(
                [np.linalg.norm(a1 - a2) for a1, a2 in zip(acts_from_rot, acts_rot)]
            )
            errors.append(err)

            rand_actions = {
                ag: np.random.uniform(0.0, 1.0, size=action_sizes[i]).astype(np.float32)
                for i, ag in enumerate(agents)
            }

            obs, *_ = env.step(rand_actions)
            terminations = _[0]
            truncations = _[1]
            if all(terminations.values()) or all(truncations.values()):
                obs, _ = env.reset()
            return {
                "theta_deg": theta_deg,
                "mean_L2": float(np.mean(errors)),
                "max_L2": float(np.max(errors)),
                "per_batch_L2": [float(e) for e in errors],
            }


def critic_permutation_test(
    maddpg: MADDPG,
    env_name: str,
    state_sizes: List[int],
    action_sizes: List[int],
    perm: List[int],
    n_batches=64,
):
    env = create_single_env(
        env_name, max_steps=50, render_mode=None, apply_padding=False
    )
    obs, _ = env.reset()
    agents = list(obs.keys())

    s_blk = state_sizes
    a_blk = action_sizes
    diffs = []

    with torch.no_grad():
        for _ in range(n_batches):
            states = [np.array(obs[a], dtype=np.float32) for a in agents]
            actions = [
                np.random.uniform(-1, 1, size=a_blk[i]).astype(np.float32)
                for i in range(len(agents))
            ]

            s_full, a_full = flatten_sa(states, actions)
            q0 = maddpg_Q(maddpg, s_full, a_full)

            s_full_p = permute_blocks(s_full, s_blk, perm)
            a_full_p = permute_blocks(a_full, a_blk, perm)
            q1 = maddpg_Q(maddpg, s_full_p, a_full_p)

            diffs.append(abs(q0 - q1))

            rand_actions = {
                ag: np.random.uniform(0.0, 1.0, size=action_sizes[i]).astype(np.float32)
                for i, ag in enumerate(agents)
            }
            obs, *_ = env.step(rand_actions)
            terminations = _[0]
            truncations = _[1]
            if all(terminations.values()) or all(truncations.values()):
                obs, _ = env.reset()

    return {
        "perm": perm,
        "mean_abs_diff": float(np.mean(diffs)),
        "max_abs_diff": float(np.max(diffs)),
        "per_batch_abs_diff": [float(d) for d in diffs],
    }


#  CLI & main 


def parse_pairs(spec: str, n_agents: int) -> Dict[int, List[Tuple[int, int]]]:
    out: Dict[int, List[Tuple[int, int]]] = {}
    if not spec.strip():
        for i in range(n_agents):
            out[i] = []
        return out
    per_agent = spec.strip().split(";")
    if len(per_agent) == 1 and n_agents > 1:
        pairs = []
        if per_agent[0].strip():
            for token in per_agent[0].split(","):
                x, y = token.split(":")
                pairs.append((int(x), int(y)))
        for i in range(n_agents):
            out[i] = list(pairs)
        return out
    for i, chunk in enumerate(per_agent):
        chunk = chunk.strip()
        pairs = []
        if chunk:
            for token in chunk.split(","):
                x, y = token.split(":")
                pairs.append((int(x), int(y)))
        out[i] = pairs
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", required=True)
    ap.add_argument(
        "--actor-xy",
        default="",
        help="Comma (x:y) pairs per agent; ';' between agents. E.g. '0:1;0:1;0:1'",
    )
    ap.add_argument("--obs-xy", default="", help="Same format for observations.")
    ap.add_argument("--device", default="cpu")

    # two runs:
    ap.add_argument(
        "--runA_dir",
        required=True,
        help="Folder with best_model.pt or model.pt for EKAN actor + PI critic",
    )
    ap.add_argument(
        "--runB_dir",
        required=True,
        help="Folder with best_model.pt or model.pt for EKAN actor + Trad critic",
    )

    # batches / rotation
    ap.add_argument("--n_actor_batches", type=int, default=64)
    ap.add_argument("--n_critic_batches", type=int, default=64)
    ap.add_argument("--theta_deg", type=float, default=15.0)

    ap.add_argument("--out", default="invariance_results.json")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agents, n_agents, action_sizes, action_low, action_high, state_sizes = get_env_info(
        env_name=args.env_name, apply_padding=False
    )

    # Resolve checkpoints
    def pick_ckpt(run_dir: str) -> str:
        c1 = os.path.join(run_dir, "best_model.pt")
        c2 = os.path.join(run_dir, "model.pt")
        if os.path.exists(c1):
            return c1
        if os.path.exists(c2):
            return c2
        raise FileNotFoundError(
            f"No checkpoint in {run_dir} (expected best_model.pt or model.pt)"
        )

    ckpt_A = pick_ckpt(args.runA_dir)
    ckpt_B = pick_ckpt(args.runB_dir)

    # Infer per-run hidden widths from checkpoints to avoid mismatches
    head_A = infer_head_hidden_from_ckpt(ckpt_A)  # e.g., 64
    head_B = infer_head_hidden_from_ckpt(ckpt_B)  # could differ

    # Print EKAN hints so you can confirm matching config
    print_ekan_hints(ckpt_A, "Run A (EKAN + PI critic)")
    print_ekan_hints(ckpt_B, "Run B (EKAN + Traditional critic)")

    # Build MADDPG for each run with matching hidden sizes
    maddpg_A = build_maddpg_for_run(
        state_sizes,
        action_sizes,
        action_low,
        action_high,
        hidden_sizes=(head_A, head_A),
        actor_kind="ekan",
        critic_kind="permutation invariant",
    )
    maddpg_B = build_maddpg_for_run(
        state_sizes,
        action_sizes,
        action_low,
        action_high,
        hidden_sizes=(head_B, head_B),
        actor_kind="ekan",
        critic_kind="traditional",
    )

    # Load checkpoints (strict)
    maddpg_A.load(ckpt_A)
    maddpg_B.load(ckpt_B)

    # Parse XY index pairs
    obs_xy = parse_pairs(args.obs_xy, n_agents)
    act_xy = parse_pairs(args.actor_xy, n_agents)

    # Actor SO(2) equivariance
    actorA = actor_equivariance_test(
        maddpg_A,
        args.env_name,
        obs_xy,
        act_xy,
        n_batches=args.n_actor_batches,
        theta_deg=args.theta_deg,
    )
    print("Actor A equivariance:", actorA)
    actorB = actor_equivariance_test(
        maddpg_B,
        args.env_name,
        obs_xy,
        act_xy,
        n_batches=args.n_actor_batches,
        theta_deg=args.theta_deg,
    )

    # Critic permutation invariance (swap first two agents if available)
    perm = list(range(n_agents))
    if n_agents >= 2:
        perm[0], perm[1] = perm[1], perm[0]

    criticA = critic_permutation_test(
        maddpg_A,
        args.env_name,
        state_sizes,
        action_sizes,
        perm,
        n_batches=args.n_critic_batches,
    )
    criticB = critic_permutation_test(
        maddpg_B,
        args.env_name,
        state_sizes,
        action_sizes,
        perm,
        n_batches=args.n_critic_batches,
    )

    results = {
        "config": {
            "env": args.env_name,
            "theta_deg": args.theta_deg,
            "perm": perm,
            "n_actor_batches": args.n_actor_batches,
            "n_critic_batches": args.n_critic_batches,
            "obs_xy": args.obs_xy,
            "actor_xy": args.actor_xy,
        },
        "EKAN+PIcritic": {
            "actor_SO2_equivariance": actorA,
            "critic_perm_invariance": criticA,
            "head_hidden_inferred": head_A,
        },
        "EKAN+Tradcritic": {
            "actor_SO2_equivariance": actorB,
            "critic_perm_invariance": criticB,
            "head_hidden_inferred": head_B,
        },
    }

    print(json.dumps(results, indent=2))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {args.out}")
