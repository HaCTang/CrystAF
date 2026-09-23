#!/usr/bin/env python
"""Calibrate `CRYSTAF_VOL_SCALE` on the TRAIN split.

CrystAF over-predicts cell volume (signed rel. error +3.2%, bias/SE 9.8 on
val). Clari keeps the lattice and Cartesian coordinates separate, so a pure
cell rescale fixes the offset without moving any atom -- PoseBusters is
untouched, only the periodic quantities (clash, PDD) move.

The scale must come from TRAIN families; reading it off the evaluation targets
would be fitting the metric. Reports the volume factor that minimises the
scored statistic (bootstrap min-of-5 of |V_pred*s/V_true - 1| per family).

    python scripts/analysis/calibrate_vol_scale.py [--families 120] [--samples 5]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.linalg as LA

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", type=int, default=120)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--config", default="configs/rl/uma_seed929.yaml")
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--rho", default="0.75")
    args = ap.parse_args()

    os.environ.update(
        MEANFLOW_SAMPLER_MODE="interval",
        MEANFLOW_INTERVAL_RHO=str(args.rho),
        MEANFLOW_INTERVAL_SCHEDULE="power",
    )
    os.environ.pop("CRYSTAF_VOL_SCALE", None)  # calibrate the RAW model

    from crystal_nft.meanflow.adapter import (
        load_meanflow_bundle,
        load_meanflow_checkpoint,
        sample_from_crystal_meanflow,
    )
    from crystal_nft.meanflow.tune import apply_student_tune
    from crystal_nft.train import train_meanflow_nft as T

    cfg = T._load_config(args.config)
    saved = T._checkpoint_config(cfg["meanflow_ckpt"])
    arch = T._load_architecture(cfg, saved)
    bundle = load_meanflow_bundle(
        cfg["checkpoint"], device=torch.device("cuda:0"),
        meanflow_steps=args.nfe, enable_chiral=False,
        gate_value=float(arch["gate_value"]),
        conditioning_mode=str(arch["conditioning_mode"]),
        time_parameterization=str(arch["time_parameterization"]),
        dual_time_feature_mode=str(arch.get("dual_time_feature_mode", "both")),
        dual_gate_value=float(arch.get("dual_gate_value", 1.0)),
        keep_nft_copies=False,
    )
    apply_student_tune(bundle["net"], saved)
    load_meanflow_checkpoint(cfg["meanflow_ckpt"], bundle, load_ema=True)
    bundle["old_net"] = bundle["net"]

    ds = T._load_csd_train_dataset(cfg)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=args.families, replace=False)
    per_family: list[np.ndarray] = []
    for k, i in enumerate(idx):
        tmpl = ds[int(i)]
        cands, _ = sample_from_crystal_meanflow(
            bundle, tmpl, samples=args.samples, use_old_net=True,
            batch_size=args.samples,
        )
        v_true = float(LA.det(tmpl.lattice).abs())
        per_family.append(
            np.array([float(LA.det(c.lattice).abs()) / v_true for c in cands])
        )
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{args.families} families", flush=True)

    ratios = np.concatenate(per_family)
    print(f"\nTRAIN split, {args.families} families x {args.samples} samples, NFE={args.nfe} rho={args.rho}")
    print(f"  V_pred/V_true: mean={ratios.mean():.4f} median={np.median(ratios):.4f} std={ratios.std(ddof=1):.4f}")
    print(f"  signed rel err: mean={100*(ratios.mean()-1):+.3f}%  median={100*(np.median(ratios)-1):+.3f}%")

    boot = np.random.default_rng(1)

    def scored(s: float) -> float:
        # Bootstrap min-of-5 per family, matching summary.paper_bootstrap.
        out = []
        for r in per_family:
            e = np.abs(r * s - 1.0) * 100
            for _ in range(400):
                out.append(e[boot.choice(len(e), 5, replace=True)].min())
        return float(np.mean(out))

    grid = np.linspace(0.93, 1.03, 41)
    vals = [scored(float(s)) for s in grid]
    best = int(np.argmin(vals))
    print(f"\n  scored min-of-5 at s=1.000 (no correction): {scored(1.0):.4f}")
    print(f"  best volume factor s = {grid[best]:.4f}  -> scored {vals[best]:.4f}")
    print(f"  1/median heuristic   = {1.0/np.median(ratios):.4f}  -> scored {scored(1.0/np.median(ratios)):.4f}")
    print(f"\n  export CRYSTAF_VOL_SCALE={grid[best]:.4f}")


if __name__ == "__main__":
    main()
