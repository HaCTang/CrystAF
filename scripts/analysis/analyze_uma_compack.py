#!/usr/bin/env python
"""Quantify UMA-energy enrichment of experimental COMPACK matches.

The Clari Table-3 protocol generates ``n_s`` candidates for each deposited
crystal, ranks them by UMA single-point energy, and runs COMPACK on the top
``k`` candidates.  This script measures, within that already evaluated set,
whether lower UMA energy enriches candidates that reproduce the deposited
packing (at least 8 matched molecules and RMSD < 2 Angstrom).  It deliberately
does not call this a higher-fidelity energy calculation: COMPACK is an
external experimental-structure target, while the ranking potential remains
UMA.

Example (run inside the Clari environment)::

  CrystalGenModel/clari/.venv/bin/python \
    scripts/analysis/analyze_uma_compack.py \
    runs/table3/crystaf-uma_oxtal \
    runs/table3/clari-m_oxtal \
    --output runs/table3/uma_compack_enrichment.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def _load(experiment: Path) -> pl.DataFrame:
    pred = pl.read_parquet(experiment / "predictions.parquet").select("sample_idx", "id")
    energy = pl.read_csv(experiment / "energies.csv").select(
        "sample_idx", pl.col("energies").alias("energy")
    )
    compack = pl.read_csv(experiment / "compack.csv").select("sample_idx", "nmatched", "rmsd")
    collision = pl.read_csv(experiment / "collision.csv").select("sample_idx", "collision")
    return pred.join(energy, on="sample_idx").join(compack, on="sample_idx").join(collision, on="sample_idx")


def _topk_and_label(df: pl.DataFrame, k: int) -> pl.DataFrame:
    return (
        df.sort("id", "energy", "sample_idx")
        .with_columns(pl.int_range(1, pl.len() + 1).over("id").alias("energy_rank"))
        .filter(pl.col("energy_rank") <= k)
        .with_columns(
            ((pl.col("nmatched") >= 8) & (pl.col("rmsd") < 2.0) & ~pl.col("collision"))
            .fill_null(False)
            .alias("solc_pass")
        )
    )


def _family_statistics(df: pl.DataFrame) -> list[dict[str, float | int | str]]:
    """Return one rank-enrichment statistic per target with a positive and negative."""
    rows: list[dict[str, float | int | str]] = []
    for target, group in df.group_by("id", maintain_order=True):
        group = group.sort("energy_rank")
        energies = group.get_column("energy").to_numpy()
        positive = group.get_column("solc_pass").to_numpy().astype(bool)
        p = int(positive.sum())
        n = int((~positive).sum())
        if p and n:
            # Probability that a random experimental match has a lower UMA
            # energy than a random non-match from the same target.  Ties get
            # half credit, i.e. this is an AUC with -energy as the score.
            e_pos, e_neg = energies[positive], energies[~positive]
            auc = ((e_pos[:, None] < e_neg[None, :]).sum() + 0.5 * (e_pos[:, None] == e_neg[None, :]).sum()) / (p * n)
            mean_rank = float(group.filter(pl.col("solc_pass")).get_column("energy_rank").mean())
            best_rank = int(group.filter(pl.col("solc_pass")).get_column("energy_rank").min())
            rows.append(
                {
                    "id": target[0] if isinstance(target, tuple) else target,
                    "n_success": p,
                    "auc_lower_energy": float(auc),
                    "mean_success_rank": mean_rank,
                    "best_success_rank": best_rank,
                }
            )
    return rows


def _bootstrap_mean(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _analyse(experiment: Path, k: int, n_boot: int, seed: int) -> dict:
    topk = _topk_and_label(_load(experiment), k)
    per_target = _family_statistics(topk)
    auc = np.asarray([r["auc_lower_energy"] for r in per_target], dtype=float)
    best_rank = np.asarray([r["best_success_rank"] for r in per_target], dtype=float)
    solved = (
        topk.group_by("id")
        .agg(pl.col("solc_pass").sum().alias("n_success"))
        .get_column("n_success")
        .to_numpy()
    )
    auc_mean, auc_lo, auc_hi = _bootstrap_mean(auc, n_boot, seed)
    rank_mean, rank_lo, rank_hi = _bootstrap_mean(best_rank, n_boot, seed + 1)
    return {
        "experiment": experiment.name,
        "candidates_per_target_ranked": k,
        "n_targets": int(topk.select("id").n_unique()),
        "n_solved_targets": int((solved > 0).sum()),
        "n_targets_with_success_and_failure": len(per_target),
        "lower_energy_match_auc": {
            "mean": auc_mean,
            "ci95": [auc_lo, auc_hi],
            "null": 0.5,
        },
        "best_match_energy_rank": {
            "mean": rank_mean,
            "ci95": [rank_lo, rank_hi],
        },
        "per_target": per_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", type=Path)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = [_analyse(path, args.k, args.n_boot, args.seed) for path in args.experiments]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"protocol": "UMA-energy-ranked top-k candidates; COMPACK target: collision-free, >=8/15 molecules, and RMSD <2 A", "results": results}, indent=2) + "\n")
    for row in results:
        auc = row["lower_energy_match_auc"]
        rank = row["best_match_energy_rank"]
        print(
            f"{row['experiment']}: solved={row['n_solved_targets']}/{row['n_targets']}; "
            f"AUC={auc['mean']:.3f} [{auc['ci95'][0]:.3f}, {auc['ci95'][1]:.3f}]; "
            f"best-match rank={rank['mean']:.1f} [{rank['ci95'][0]:.1f}, {rank['ci95'][1]:.1f}]"
        )


if __name__ == "__main__":
    main()
