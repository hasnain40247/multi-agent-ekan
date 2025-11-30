from legacy.ekan.groups import SO, O, SO13, SO13p, Lorentz
from legacy.ekan.representation import Vector, Scalar

def _make_group(name: str):
    """Map YAML string to EKAN group object."""
    key = (name or "").upper()
    if key == "SO2":
        return SO(2)
    if key == "O2":
        return O(2)
    if key == "SO13":
        return SO13()
    if key == "SO13P":
        return SO13p()
    if key == "LORENTZ":
        return Lorentz()
    raise ValueError(
        f"Unknown EKAN group '{name}'. Choose one of SO2|O2|SO13|SO13p|Lorentz."
    )


def _rep_size(rep) -> int:
    """Return the real-space dimension of a representation."""
    if hasattr(rep, "size"):
        return int(rep.size())
    raise TypeError("EKAN representation does not expose .size().")


def _build_rep_from_counts(vectors: int, scalars: int):
    """Build a representation: vectors * Vector  +  scalars * Scalar."""
    rep = None
    if vectors and vectors > 0:
        rep = vectors * Vector
    if scalars and scalars > 0:
        rep = (rep + scalars * Scalar) if rep is not None else scalars * Scalar
    if rep is None:
        # fall back to a 0-dim scalar space if both counts are zero
        rep = 0 * Scalar
    return rep


def build_ekan_parts(cfg: dict, obs_dim: int):
    """
    Parse EKAN config from YAML:

    model:
      name: ekan
      hidden_dim: 64
      ekan:
        group: "SO2"
        rep_in:
          vectors: 9
          scalars: 0
        rep_out:
          scalars: 64
        ekan_kwargs:
          width: 64
          grid: 16
          k: 3

    Returns:
        (rep_in, rep_out, group, ekan_kwargs) where rep_in/rep_out are *concrete* (bound to group).
    """
    # group
    group_name = cfg.get("group", "SO2")
    group = _make_group(group_name)

    # rep_in, must match obs_dim
    rin = cfg.get("rep_in", {}) or {}
    rin_vec = int(rin.get("vectors", 0))
    rin_sca = int(rin.get("scalars", 0))
    rep_in_sym = _build_rep_from_counts(rin_vec, rin_sca)
    if rep_in_sym is None:
        raise ValueError(
            "EKAN rep_in is empty (vectors=0 and scalars=0). "
            "Set rep_in so that rep_in.size() equals env obs_dim."
        )
    rep_in = rep_in_sym(group) 
    rin_size = _rep_size(rep_in)
    if rin_size != obs_dim:
        raise ValueError(
            f"EKAN rep_in.size()={rin_size} does not match env obs_dim={obs_dim}. "
            f"Adjust YAML rep_in (vectors/scalars) so sizes match."
        )

    # rep_out,default to hidden_dim * Scalar if not specified
    rout = cfg.get("rep_out", None)
    if rout is None:
        hidden_dim = int(cfg.get("hidden_dim", 64))
        rep_out_sym = hidden_dim * Scalar
    else:
        # Support both vectors and scalars in rep_out
        rout_vec = int(rout.get("vectors", 0))
        rout_sca = int(rout.get("scalars", 0))
        rep_out_sym = _build_rep_from_counts(rout_vec, rout_sca)
        if rep_out_sym is None:
            # fall back to hidden_dim if user mistakenly set both to 0
            hidden_dim = int(cfg.get("hidden_dim", 64))
            rep_out_sym = hidden_dim * Scalar
    rep_out = rep_out_sym(group)

    # kwargs passed to EKAN(...)
    ekan_kwargs = dict(cfg.get("ekan_kwargs", {}) or {})

    # normalize width: allow int or list as input
    if "width" in ekan_kwargs:
        w = ekan_kwargs["width"]
        if isinstance(w, int):
            ekan_kwargs["width"] = [w]
        elif isinstance(w, (list, tuple)):
            ekan_kwargs["width"] = list(w)
        else:
            raise TypeError(f"ekan_kwargs.width must be int|list|tuple, got {type(w)}")

    return rep_in, rep_out, group, ekan_kwargs
