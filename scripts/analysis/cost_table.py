"""Inference-cost columns of the main table (UMA calls and ms per sample).

Reads the per-rank ``metrics.shard*.json`` written by ``eval_clari_table1`` and drops
the first sampler call of each rank (CUDA warm-up, UMA load).

    python scripts/analysis/cost_table.py runs/cost/base_none runs/cost/base_relax10 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cost(run_dir: Path) -> tuple[float, float, int]:
    sec, n_timed, uma_structs, n_all = 0.0, 0, 0, 0
    shards = sorted(run_dir.glob("metrics.shard*.json"))
    if not shards:
        raise FileNotFoundError(f"no metrics.shard*.json in {run_dir}")
    for path in shards:
        d = json.loads(path.read_text())
        calls = d.get("gen_calls_local") or []
        for dt, n in calls[1:]:
            sec += float(dt)
            n_timed += int(n)
        n_all += sum(int(n) for _, n in calls)
        uma_structs += int((d.get("physics_stats_local") or {}).get("uma_structs", 0))
    if n_timed == 0:
        raise ValueError(f"{run_dir}: need at least two sampler calls per rank")
    return uma_structs / max(n_all, 1), 1000.0 * sec / n_timed, n_timed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()
    print(f"{'run':32s} {'UMA/sample':>10s} {'ms/sample':>10s} {'timed':>6s}")
    for run in args.runs:
        uma, ms, n = cost(run)
        print(f"{run.name:32s} {uma:10.1f} {ms:10.0f} {n:6d}")


if __name__ == "__main__":
    main()
