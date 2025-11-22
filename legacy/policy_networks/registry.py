from typing import Callable, Dict
import torch.nn as nn

from policy_networks.mlp_policy_net import MLPPolicyNet
from policy_networks.kan_policy_net import KANPolicyNet
from policy_networks.ekan_policy_net import EKANPolicyNet

from utils import build_ekan_parts

RegistryType = Dict[str, Callable[..., nn.Module]]


def _build_ekan(obs_dim: int, act_dim: int, cfg: dict):
    # Build EKAN parts from YAML (rep_in, rep_out, group, kwargs)
    rep_in, rep_out, group, ekan_kwargs = build_ekan_parts(cfg, obs_dim)
    return EKANPolicyNet(
        rep_in=rep_in,
        rep_out=rep_out,
        group=group,
        action_dim=act_dim,
        ekan_kwargs=ekan_kwargs,
    )


POLICY_REGISTRY: RegistryType = {
    "mlp": lambda obs_dim, act_dim, cfg: MLPPolicyNet(
        obs_dim, act_dim, hidden_dim=cfg.get("hidden_dim", 64)
    ),
    "kan": lambda obs_dim, act_dim, cfg: KANPolicyNet(
        obs_dim,
        act_dim,
        hidden_dim=cfg.get("hidden_dim", 64),
        grid=cfg.get("grid", 16),
        k=cfg.get("k", 3),
        kw=cfg.get("kw", {}),
    ),
    "ekan": _build_ekan,
}
