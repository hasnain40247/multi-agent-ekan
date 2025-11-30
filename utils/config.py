import json
import yaml



def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_overrides(pairs):
    """
    Override values in config from CLI: --override training.lr=0.0001
    """
    result = {}
    for p in pairs or []:
        if "=" not in p:
            continue
        key, sval = p.split("=", 1)
        try:
            val = json.loads(sval)
        except Exception:
            val = sval
        cur = result
        parts = key.split(".")
        for subk in parts[:-1]:
            cur = cur.setdefault(subk, {})
        cur[parts[-1]] = val
    return result


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base
