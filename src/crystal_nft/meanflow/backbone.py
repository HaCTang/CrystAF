"""DiT backbone presets for MeanFlow (Clari-compatible, optional slim variants)."""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)

# Presets mirror Clari train scripts; MeanFlow-S targets ~40M params vs Clari-M ~88M.
DIT_PRESETS: dict[str, dict[str, Any]] = {
    "clari_m": dict(
        dim=512,
        dim_pair=64,
        dim_cond=512,
        num_heads=8,
        expand=2.666,
        depth=16,
        lattice_nodes=True,
        self_cond=True,
        use_mpa=False,
    ),
    "meanflow_s": dict(
        dim=384,
        dim_pair=48,
        dim_cond=384,
        num_heads=6,
        expand=2.666,
        depth=12,
        lattice_nodes=True,
        self_cond=True,
        use_mpa=False,
    ),
    "meanflow_s_wide": dict(
        dim=512,
        dim_pair=64,
        dim_cond=512,
        num_heads=8,
        expand=2.666,
        depth=12,
        lattice_nodes=True,
        self_cond=True,
        use_mpa=False,
    ),
}


def build_dit_from_preset(name: str) -> nn.Module:
    from clari.models import DiT

    if name not in DIT_PRESETS:
        raise KeyError(f"Unknown dit_preset={name!r}; choose from {list(DIT_PRESETS)}")
    return DiT(**DIT_PRESETS[name])


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def transfer_clari_dit_weights(target: nn.Module, source: nn.Module) -> dict[str, int]:
    """
    Copy matching tensors from a Clari DiT into another DiT (e.g. shallow or narrow student).

    MeanFlow delta-time embed is applied after wrapping; only the base DiT is transferred here.
    """
    tgt = target.state_dict()
    src = source.state_dict()
    loaded = {k: v for k, v in src.items() if k in tgt and v.shape == tgt[k].shape}
    missing = [k for k in tgt if k not in loaded]
    skipped = [k for k in src if k not in loaded]
    target.load_state_dict(loaded, strict=False)
    logger.info(
        "DiT weight transfer: %d tensors copied, %d target-only, %d source-skipped",
        len(loaded),
        len(missing),
        len(skipped),
    )
    return {"copied": len(loaded), "target_only": len(missing), "source_skipped": len(skipped)}
