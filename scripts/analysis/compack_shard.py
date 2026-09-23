#!/usr/bin/env python
"""Run one shard of an official COMPACK job so several SLURM tasks can split it.

clari's compack.py already writes one CSV per CSD id under ``<exp>/compack/`` and
skips ids whose shard is already complete, so the id axis is a safe way to split
the work: two tasks never touch the same file. This script does the shard; the
merge is just ``compack.py <exp>`` afterwards with no ``--overwrite`` (it finds
every shard complete, does no comparisons, and writes ``compack.csv``).

Shards are balanced by pair count, not id count -- ids differ by ~10x in how many
GT polymorphs they have, and round-robin leaves the last task running for hours.

    python compack_shard.py <exp> --shard 0 --num-shards 8 --num-processes 28
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "CrystalGenModel" / "clari"))

from clari.evaluation.compack import (  # noqa: E402
    RESULTS_DIR,
    _cid,
    _compare_key,
    _load_completed_shards,
    _load_gt,
    _run_rows,
)
from clari.evaluation.results_utils import select_topk_ranked_per_id  # noqa: E402


def _below_topk(exp: Path, k: int) -> set[int]:
    """sample_idx that Sol@k would discard, so COMPACK can skip them.

    Mirrors what summarize does: the frame is predictions joined with energies in
    predictions order, and select_topk_ranked_per_id sorts by (id, energy, row
    order). Reusing that function rather than re-deriving the sort keeps ties
    resolved identically -- otherwise a handful of rows near the cut would differ.
    """
    frame = pl.read_parquet(exp / "predictions.parquet").select("sample_idx", "id")
    energies = pl.read_csv(exp / "energies.csv").select(
        "sample_idx", pl.col("energies").alias("energy")
    )
    frame = frame.join(energies, on="sample_idx", how="left")
    keep = set(select_topk_ranked_per_id(frame, k=k).get_column("sample_idx").to_list())
    return set(frame.get_column("sample_idx").to_list()) - keep


def _balanced_shard(ids_with_cost: list[tuple[str, int]], shard: int, num_shards: int) -> set[str]:
    """Greedy longest-processing-time bin packing: heaviest id to the lightest bin."""
    bins: list[tuple[int, int, list[str]]] = [(0, i, []) for i in range(num_shards)]
    for cid, cost in sorted(ids_with_cost, key=lambda x: -x[1]):
        bins.sort(key=lambda b: (b[0], b[1]))
        load, idx, members = bins[0]
        bins[0] = (load + cost, idx, members + [cid])
    return set(next(members for load, idx, members in bins if idx == shard))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--num_processes", type=int, default=28)
    parser.add_argument("--timeout_seconds", type=float, default=100.0)
    parser.add_argument("--exact_ids", action="store_true")
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Only compare the top-k by UMA energy per id (exact for Sol@k, wrong for larger k).",
    )
    args = parser.parse_args()

    exp = Path(args.experiment_dir)
    if not exp.is_absolute() and not exp.exists() and len(exp.parts) == 1:
        exp = RESULTS_DIR / exp

    rows = (
        pl.read_parquet(exp / "predictions.parquet")
        .select("sample_idx", "id", "cif")
        .to_dicts()
    )
    gt_by_key = _load_gt(exp / "config.json", args.exact_ids)

    # cost(id) = n_predictions(id) * n_GT_polymorphs(id) -- the pair count
    n_pred: dict[str, int] = {}
    for row in rows:
        n_pred[_cid(row["id"])] = n_pred.get(_cid(row["id"]), 0) + 1
    cost = [
        (cid, n * len(gt_by_key.get(_compare_key(cid, args.exact_ids), [])))
        for cid, n in n_pred.items()
    ]

    skip = _below_topk(exp, args.topk) if args.topk else None
    if skip:
        # rebalance on the work that will actually run
        keep_per_id: dict[str, int] = {}
        for row in rows:
            if row["sample_idx"] not in skip:
                keep_per_id[_cid(row["id"])] = keep_per_id.get(_cid(row["id"]), 0) + 1
        cost = [
            (cid, keep_per_id.get(cid, 0) * len(gt_by_key.get(_compare_key(cid, args.exact_ids), [])))
            for cid, _ in cost
        ]
    mine = _balanced_shard(cost, args.shard, args.num_shards)
    shard_dir = exp / "compack"

    rows_by_id: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_id.setdefault(_cid(row["id"]), []).append(row)
    _, done = _load_completed_shards(shard_dir, rows_by_id, args.exact_ids)

    todo = [r for r in rows if _cid(r["id"]) in mine and _cid(r["id"]) not in done]
    pairs = sum(
        len(gt_by_key.get(_compare_key(r["id"], args.exact_ids), []))
        for r in todo
        if skip is None or r["sample_idx"] not in skip
    )
    print(
        f"shard {args.shard}/{args.num_shards}: {len(mine)} ids "
        f"({len(mine & done)} already done), {len(todo)} rows, {pairs} pairs, "
        f"{args.num_processes} workers",
        flush=True,
    )
    if not todo:
        return
    _run_rows(
        todo,
        gt_by_key,
        args.num_processes,
        args.timeout_seconds,
        args.exact_ids,
        shard_dir,
        skip_sample_idx=skip,
    )


if __name__ == "__main__":
    main()
