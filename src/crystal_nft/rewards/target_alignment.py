"""Lightweight target-condition alignment proxies for crystal reward training."""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np


def periodic_pair_distance_signature(
    atoms,
    *,
    bins: int = 64,
    max_distance: float = 8.0,
) -> np.ndarray:
    """Normalized PBC pair-distance histogram (PDD-like packing proxy)."""
    distances = np.asarray(atoms.get_all_distances(mic=True), dtype=np.float64)
    values = distances[np.triu_indices(len(atoms), k=1)]
    values = values[np.isfinite(values) & (values > 0.0)]
    hist, _ = np.histogram(values, bins=int(bins), range=(0.0, float(max_distance)))
    out = hist.astype(np.float64)
    return out / max(out.sum(), 1.0)


def reciprocal_powder_signature(
    atoms,
    *,
    max_hkl: int = 3,
    bins: int = 64,
    q_max: float = 8.0,
) -> np.ndarray:
    """Cell reciprocal-vector histogram used as an inexpensive XRD proxy."""
    reciprocal = np.asarray(atoms.cell.reciprocal(), dtype=np.float64)
    hkl = np.asarray(
        [
            h
            for h in itertools.product(
                range(-int(max_hkl), int(max_hkl) + 1), repeat=3
            )
            if h != (0, 0, 0)
        ],
        dtype=np.float64,
    )
    q = 2.0 * np.pi * np.linalg.norm(hkl @ reciprocal, axis=1)
    q = q[np.isfinite(q) & (q > 0.0)]
    hist, _ = np.histogram(q, bins=int(bins), range=(0.0, float(q_max)))
    out = hist.astype(np.float64)
    return out / max(out.sum(), 1.0)


def target_alignment_rewards(
    candidates: Sequence,
    target,
    *,
    pdd_weight: float = 1.0,
    xrd_weight: float = 1.0,
    volume_weight: float = 1.0,
) -> np.ndarray:
    """Return higher-is-better PDD/XRD/volume alignment rewards."""
    target_pdd = periodic_pair_distance_signature(target)
    target_xrd = reciprocal_powder_signature(target)
    target_volume = max(float(target.get_volume()), 1e-8)
    rewards = np.empty(len(candidates), dtype=np.float64)
    for i, atoms in enumerate(candidates):
        try:
            pdd = np.abs(periodic_pair_distance_signature(atoms) - target_pdd).mean()
            xrd = np.abs(reciprocal_powder_signature(atoms) - target_xrd).mean()
            volume = abs(float(atoms.get_volume()) - target_volume) / target_volume
            rewards[i] = -(
                float(pdd_weight) * pdd
                + float(xrd_weight) * xrd
                + float(volume_weight) * volume
            )
        except Exception:
            rewards[i] = -1e6
    return rewards
