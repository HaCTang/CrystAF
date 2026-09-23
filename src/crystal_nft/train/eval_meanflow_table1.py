"""Table 1 evaluation entrypoint for MeanFlow crystal checkpoints."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from crystal_nft.train.eval_clari_table1 import run_eval  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    p = argparse.ArgumentParser(description="MeanFlow Table 1 val evaluation")
    p.add_argument("--meanflow-ckpt", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="Base Clari ckpt used for interface")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--meanflow-steps", type=int, default=4)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--clari-data-dir", type=str, default=str(_REPO_ROOT / "dataset" / "clari"))
    p.add_argument("--max-crystals", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--pack-size", type=int, default=1)
    p.add_argument(
        "--sample-chunk",
        type=int,
        default=1,
        help="How many samples to draw before assessing/freeing (keep 1 to bound RAM).",
    )
    p.add_argument("--amd-metric", type=str, default="cityblock")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--metric-workers",
        type=int,
        default=1,
        help="Keep 1 for MeanFlow Table1 (AMD/PDD is not thread-safe; also bounds RAM).",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing metrics.shard*.jsonl (default true).",
    )
    p.add_argument("--assess-timeout-sec", type=float, default=180.0)
    p.add_argument(
        "--assess-mem-gb",
        type=float,
        default=32.0,
        help="Address-space headroom for assess child (GiB, added on top of current VmSize).",
    )
    p.add_argument(
        "--assess-in-subprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--overlap-assess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assess chunk i on CPU while GPU samples chunk i+1",
    )
    p.add_argument(
        "--meanflow-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load EMA student weights (default true). Use --no-meanflow-ema to probe live weights.",
    )
    args = p.parse_args()

    run_eval(
        checkpoint=args.checkpoint,
        output_dir=Path(args.output_dir),
        samples=args.samples,
        n_steps=int(args.meanflow_steps),
        clari_data_dir=Path(args.clari_data_dir),
        max_crystals=args.max_crystals,
        device=args.device,
        seed=int(args.seed),
        meanflow_ckpt=args.meanflow_ckpt,
        meanflow_steps=int(args.meanflow_steps),
        pack_size=int(args.pack_size),
        sample_chunk=max(1, int(args.sample_chunk)),
        amd_metric=str(args.amd_metric),
        metric_workers=1,
        resume=bool(args.resume),
        assess_timeout_sec=float(args.assess_timeout_sec),
        assess_mem_gb=float(args.assess_mem_gb),
        assess_in_subprocess=bool(args.assess_in_subprocess),
        overlap_assess=bool(args.overlap_assess),
        use_meanflow_ema=bool(args.meanflow_ema),
    )


if __name__ == "__main__":
    main()
