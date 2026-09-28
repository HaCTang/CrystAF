#!/usr/bin/env python
"""Table 3 Sol@k with the paper's bootstrap statistic.

clari's `summarize` reports the plain fraction solved_targets/n_targets, but the
paper's Table 3 reports "bootstrap means over 5000 resamples" -- which is why its
numbers are not fractions of 5, 6 or 8 (CSP6 = 0.729 over five targets).

Resampling targets would leave the mean unchanged, so the resampling has to be on
the *sample* axis, matching `_paper_bootstrap_summary` for Table 1: draw k samples
with replacement per target, count the target solved if any drawn sample passes,
average over targets. A target carried by a single passing sample survives only
1-(1-1/k)^k ~ 63% of resamples, which is exactly what pulls the mean below the
fraction. That reconstruction reproduces the published values:

    CSP6 Clari-L  4/5 with one m=1 target -> 0.7266   (paper 0.729)
    CSP5 Clari-L  6/6 with one m=2 target -> 0.9777   (paper 0.975)

Resampling happens *within* the deterministic top-k, so it stays exact when
COMPACK was only run on the top-k (see compack_shard.py --topk).

    python table3_bootstrap.py clari-m_oxtal_ns400 [...] -k 200
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "CrystalGenModel" / "clari"))

from clari.csd import AVAILABLE_CSD_SUBSETS, csd_fam  # noqa: E402
from clari.evaluation.results_utils import (  # noqa: E402
    annotate_compack_results,
    select_topk_ranked_per_id,
)
from clari.paths import resolve_results_path  # noqa: E402

OXTAL_GROUPS = ("rigid", "flexible", "csp5", "csp6", "csp7")


def _load(exp: Path) -> pl.DataFrame:
    """Same join, in the same row order, as clari's summarize._load_results."""
    pred = pl.read_parquet(exp / "predictions.parquet").select("sample_idx", "id")
    energies = pl.read_csv(exp / "energies.csv").select(
        "sample_idx", pl.col("energies").alias("energy")
    )
    compack = pl.read_csv(exp / "compack.csv")
    collision = pl.read_csv(exp / "collision.csv").select("sample_idx", "collision")
    return (
        pred.join(energies, on="sample_idx", how="left")
        .join(compack.drop("id", "energy", strict=False), on="sample_idx", how="left")
        .join(collision, on="sample_idx", how="left")
        .with_columns(pl.col("collision").cast(pl.Boolean, strict=False))
    )


def _sol_stats(
    df: pl.DataFrame, k: int, n_boot: int, seed: int
) -> tuple[float, float, float, float, int]:
    """(mean, SE over samples, SE over targets, plain fraction, n_targets)."""
    topk = select_topk_ranked_per_id(df, k=k)
    ann = annotate_compack_results(topk)
    per = (
        ann.group_by("id")
        .agg(
            pl.col("solc_pass").fill_null(False).sum().alias("m"),
            pl.len().alias("n"),
        )
        .sort("id")
    )
    m = per.get_column("m").to_numpy().astype(np.int64)
    n = per.get_column("n").to_numpy().astype(np.int64)
    n_targets = len(m)
    if n_targets == 0:
        return float("nan"), float("nan"), float("nan"), 0

    # P(target survives one resample of its own n samples, drawn with replacement)
    p = 1.0 - np.power((n - m) / n, n)
    rng = np.random.default_rng(seed)

    # Two SEs, because neither single axis reproduces the paper's.
    #   se_samp  resample the k samples per target. Its MEAN is the statistic the
    #            paper reports (verified: CSP6 0.7266 vs 0.729, CSP5 0.9777 vs
    #            0.975), but its SE collapses toward zero on the small subsets,
    #            where every target is solved by many samples or by none.
    #   se_tgt   resample the targets. Overshoots: Rigid at ns=30 gives 0.066
    #            against the paper's 0.039, and a plain target bootstrap cannot
    #            go below sqrt(p(1-p)/n)=0.064 there, so the paper is not doing
    #            this either.
    # The published SE sits between the two and its definition is not recoverable
    # from the paper, so both are printed rather than tuning a knob until one
    # matches -- that would be fitting the statistic.
    sol_s = (rng.random((n_boot, n_targets)) < p).mean(axis=1)
    tgt = rng.integers(0, n_targets, size=(n_boot, n_targets))
    sol_t = (rng.random((n_boot, n_targets)) < p[tgt]).mean(axis=1)
    return (
        float(sol_s.mean()),
        float(sol_s.std(ddof=1)),
        float(sol_t.std(ddof=1)),
        float((m > 0).mean()),
        n_targets,
    )


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("-k", type=int, default=200)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teaching", action="store_true", help="one Sol column over the teaching set")
    args = ap.parse_args()

    for name in args.experiments:
        exp = resolve_results_path(name)
        if not (exp / "compack.csv").is_file():
            print(f"SKIP {name}: no compack.csv")
            continue
        df = _load(exp)
        fam = pl.col("id").map_elements(csd_fam, return_dtype=pl.Utf8)
        groups = ("teaching",) if args.teaching else OXTAL_GROUPS
        print(f"\n{exp.name}  (n_s={df.height // df.select('id').n_unique()}, k={args.k})")
        print(f"  {'subset':<10}{'mean':>9} {'±se_samp':<7}{'±se_tgt':<8}{'fraction':>10}{'targets':>9}")
        for g in groups:
            fams = {csd_fam(c) for c in AVAILABLE_CSD_SUBSETS[g]}
            sub = df.filter(fam.is_in(fams))
            if sub.is_empty():
                print(f"  {g:<10}{'--':>9}")
                continue
            mean, se_s, se_t, frac, nt = _sol_stats(sub, args.k, args.n_boot, args.seed)
            print(
                f"  {g:<10}{mean:>9.3f}±{se_s:<6.3f}±{se_t:<7.3f}{frac:>10.3f}{nt:>9}"
            )


if __name__ == "__main__":
    main()
