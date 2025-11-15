import os, sys, json
import logging
import argparse
import yaml
import torch
import wandb

from environment.mpe import MPEEnvironment
from mappo.mappo import MAPPO
from policy_networks.registry import POLICY_REGISTRY

from utils import (
    load_config,
    parse_overrides,
    deep_update,
    pick_device,
    set_global_seed,
    setup_logging,
)


def build_env_from_cfg(cfg_env: dict, render_mode=None):
    """Builds the environment, from the config file"""
    return MPEEnvironment(
        n_agents=cfg_env["n_agents"],
        max_cycles=cfg_env["max_cycles"],
        continuous_actions=cfg_env["continuous_actions"],
        render_mode=render_mode,
    )


def build_policy_from_cfg(model_cfg: dict, obs_dim: int, act_dim: int):
    """Builds the policy from the config ffile

    Args:
        model_cfg (dict): config dict of the mode
        obs_dim (int): state dimensions of the environment
        act_dim (int): action dimensions of the environment

    Raises:
        ValueError: If model name is invalid

    Returns:
        _type_:
    """
    name = model_cfg["name"]
    if name not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(POLICY_REGISTRY.keys())}"
        )
    shared = {
        k: v for k, v in model_cfg.items() if not isinstance(v, dict) and k != "name"
    }
    sub = model_cfg.get(name) or {}
    # sub overrides shared
    merged = {**shared, **sub}
    return POLICY_REGISTRY[name](obs_dim, act_dim, merged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPE trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--override", action="append", help="Dot-list overrides: key.subkey=val"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    # loading config from yaml
    cfg = load_config(args.config)
    overrides = parse_overrides(args.override)
    cfg = deep_update(cfg, overrides)

    # Seed
    set_global_seed(cfg.get("run", {}).get("seed", None))

    mode = cfg["run"]["mode"]
    env_cfg = cfg["env"]
    trn_cfg = cfg["training"]
    mdl_cfg = cfg["model"]
    wb_cfg = cfg.get("wandb", {"enabled": False})

    if mode == "visualize":
        env = build_env_from_cfg(env_cfg, render_mode="human")
        specs = env.get_specs()
        print("\nSpecs:")
        for k, v in specs.items():
            print(f"{k}: {v}")

        vis = env_cfg.get("visualize", {})
        env.visualize(
            steps=vis.get("steps", 50),
            random_actions=not vis.get("manual", False),
            manual=vis.get("manual", False),
            delay=vis.get("delay", 0.0),
        )
        env.close()
        sys.exit(0)

    device = pick_device()

    # train
    env = build_env_from_cfg(env_cfg, render_mode=None)

    # Build the chosen backbone
    policy = build_policy_from_cfg(mdl_cfg, env.obs_dim, env.action_dim).to(device)

    # wandb
    run = None
    if wb_cfg.get("enabled", False):
        run = wandb.init(
            project=wb_cfg.get("project", "multi-agent-ekan"),
            name=wb_cfg.get("name", None),
            entity=wb_cfg.get("entity", None),
            config=cfg,
            save_code=True,
        )
        try:
            wandb.watch(policy, log="all", log_freq=100)
        except Exception:
            pass

    agent = MAPPO(
        env,
        gamma=trn_cfg["gamma"],
        lr=trn_cfg["lr"],
        policy=policy,
    )

    train_kwargs = dict(
        epochs=trn_cfg["epochs"],
        steps_per_epoch=trn_cfg["steps_per_epoch"],
        value_coef=trn_cfg["value_coef"],
        entropy_coef=trn_cfg["entropy_coef"],
        max_grad_norm=trn_cfg["max_grad_norm"],
    )

    if "log_to_wandb" in MAPPO.train.__code__.co_varnames:
        train_kwargs["log_to_wandb"] = wb_cfg.get("enabled", False)
    if "wandb_run" in MAPPO.train.__code__.co_varnames:
        train_kwargs["wandb_run"] = run

    agent.train(**train_kwargs)

    if run is not None:
        run.finish()
    env.close()
