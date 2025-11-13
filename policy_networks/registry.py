from typing import Callable, Dict
import torch.nn as nn

from policy_networks.mlp_policy_net import MLPPolicyNet
from policy_networks.kan_policy_net import KANPolicyNet

RegistryType = Dict[str, Callable[..., nn.Module]]

POLICY_REGISTRY: RegistryType = {
    "mlp": lambda obs_dim, act_dim, cfg: MLPPolicyNet(
        obs_dim,
        act_dim,
        hidden_dim=cfg.get("hidden_dim", 64)
    ),
    
    "kan": lambda obs_dim, act_dim, cfg: KANPolicyNet(
        obs_dim,
        act_dim,
        hidden_dim=cfg.get("hidden_dim", 64),
        grid=cfg.get("grid", 16),
        k=cfg.get("k", 3),
        kw=cfg.get("kw", {}),
    ),
}