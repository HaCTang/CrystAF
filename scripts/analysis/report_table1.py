#!/usr/bin/env python
"""Print a Table1 `metrics.json` on the paper protocol, optionally vs a baseline.

    python scripts/analysis/report_table1.py <eval_dir_or_metrics.json> [more ...] \
        [--label NAME] [--baseline <dir_or_metrics.json>]

Reads `summary.paper_bootstrap` -- the bootstrap aggregate (5000 draws of 5;
quality metrics mean-of-5, reconstruction metrics min-of-5) that is the
decision metric for this repo. Deliberately reports **all four** columns plus
`mean_pb_valid`: raw `mean_pb_score` is invalid (unbuildable crystals score
pb_score=1 and inflate it), and PB alone has repeatedly moved in the opposite
direction to clash / volume / PDD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLUMNS = (
    ("pb_score_pct", "PB %", 1),
    ("clash_rate_pct", "clash %", -1),
    ("volume_error_pct", "Vol.Err %", -1),
    ("dist_pdd", "EMD PDD", -1),
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_file() else p / "metrics.json"


def _load(path: str | Path) -> tuple[dict, dict]:
    blob = json.loads(_resolve(path).read_text())
    summary = blob.get("summary") or {}
    boot = summary.get("paper_bootstrap") or {}
    if not boot:
        raise SystemExit(f"{_resolve(path)}: no summary.paper_bootstrap")
    return summary, boot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--label", default=None, help="Label for a single path")
    ap.add_argument("--baseline", default=None, help="Compare every path against this")
    args = ap.parse_args()

    base = _load(args.baseline)[1] if args.baseline else None
    if base is not None:
        print(f"baseline: {args.baseline}")

    for path in args.paths:
        summary, boot = _load(path)
        label = args.label if (args.label and len(args.paths) == 1) else str(path)
        print(f"\n=== {label}")
        print(f"    mean_pb_valid={float(summary.get('mean_pb_valid', float('nan'))):.4f} "
              f"(expect ~0.695; a drop means fewer crystals could even be built)")
        for key, name, better in COLUMNS:
            val = float(boot.get(key, float("nan")))
            se = float(boot.get(f"{key}_se", float("nan")))
            line = f"    {name:<10} {val:8.3f} +/- {se:.3f}"
            if base is not None:
                delta = val - float(base.get(key, float("nan")))
                # `better` is +1 when higher is better, -1 when lower is.
                mark = "better" if delta * better > 0 else "worse"
                if abs(delta) < 1e-9:
                    mark = "same"
                line += f"   delta {delta:+8.3f}  {mark}"
            print(line)


if __name__ == "__main__":
    main()
