"""Export NFT weights into a Lightning .ckpt and evaluate Fig 3c (lattice RMAD).

Figure 3c in MolCrystalFlow reports lattice volume RMAD ≈ 3.86% on Thurlemann
test (10 samples). Lower is better.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MCF_ROOT = _REPO_ROOT / "CrystalGenModel" / "MolCrystalFlow"
logger = logging.getLogger(__name__)


def export_lightning_ckpt(nft_pt: Path, base_ckpt: Path, out_ckpt: Path) -> Path:
    """Replace FlowModule.model weights in a Lightning checkpoint."""
    payload = torch.load(nft_pt, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict") or payload.get("net_state_dict")
    if state is None:
        raise KeyError(f"No model_state_dict in {nft_pt}")

    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    n_replaced = 0
    for k, v in state.items():
        # Lightning keys are typically model.*
        candidates = [f"model.{k}", k]
        for ck in candidates:
            if ck in sd and sd[ck].shape == v.shape:
                sd[ck] = v
                n_replaced += 1
                break
    if n_replaced == 0:
        # Try stripping a leading 'model.' from nft keys
        for k, v in state.items():
            kk = k[6:] if k.startswith("model.") else k
            ck = f"model.{kk}"
            if ck in sd and sd[ck].shape == v.shape:
                sd[ck] = v
                n_replaced += 1
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    # Copy companion config.yaml next to out_ckpt for FlowModule.load_from_checkpoint
    cfg_src = base_ckpt.parent / "config.yaml"
    cfg_dst = out_ckpt.parent / "config.yaml"
    if cfg_src.is_file():
        cfg_dst.write_bytes(cfg_src.read_bytes())
    torch.save(ckpt, out_ckpt)
    logger.info("Exported %s (%d tensors replaced)", out_ckpt, n_replaced)
    meta = {
        "nft_pt": str(nft_pt),
        "base_ckpt": str(base_ckpt),
        "out_ckpt": str(out_ckpt),
        "n_replaced": n_replaced,
        "n_nft_keys": len(state),
        "n_lightning_keys": len(sd),
    }
    (out_ckpt.parent / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if n_replaced == 0:
        raise RuntimeError("Failed to map any NFT weights into Lightning state_dict")
    return out_ckpt


def run_inference(
    *,
    ckpt: Path,
    cache_dir: Path,
    out_dir: Path,
    num_samples: int,
    num_gpus: int,
    num_timesteps: int,
    scaling: float,
    exp_rate: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_MCF_ROOT / "molcrystalflow" / "experiments" / "inference.py"),
        f"inference.ckpt_path={ckpt}",
        f"inference.inference_dir={out_dir}",
        f"inference.output_dir={out_dir}",
        f"inference.num_samples={num_samples}",
        f"inference.num_gpus={num_gpus}",
        f"inference.save_trajectories=false",
        f"data.cache_dir={cache_dir}",
        f"interpolant.sampling.num_timesteps={num_timesteps}",
        f"interpolant.trans.scaling={scaling}",
        f"interpolant.rots.exp_rate={exp_rate}",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{_MCF_ROOT}:{_MCF_ROOT / 'csp-pipeline'}:{env.get('PYTHONPATH', '')}"
    )
    env["PYTHONUNBUFFERED"] = "1"
    # Ensure logical device 0 exists for the child process
    if not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = "0"
    logger.info(
        "Running inference CUDA_VISIBLE_DEVICES=%s | %s",
        env.get("CUDA_VISIBLE_DEVICES"),
        " ".join(cmd),
    )
    subprocess.run(cmd, check=True, cwd=str(_MCF_ROOT), env=env)
    return out_dir


def run_volume_analysis(
    *,
    gt_xyz: Path,
    pred_xyz: Path,
    num_samples: int,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_MCF_ROOT / "molcrystalflow" / "experiments" / "run_lattice_volume_analysis.py"),
        "--gt_file",
        str(gt_xyz),
        "--pred_file",
        str(pred_xyz),
        "--num_samples",
        str(num_samples),
        "--output_dir",
        str(out_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_MCF_ROOT}:{env.get('PYTHONPATH', '')}"
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_MCF_ROOT), env=env)

    # Prefer JSON summary if produced; else compute RMAD ourselves
    summaries = list(out_dir.glob("*.json"))
    if summaries:
        data = json.loads(summaries[0].read_text())
        return data

    from ase.io import read

    gt = read(str(gt_xyz), index=":")
    if not isinstance(gt, list):
        gt = [gt]
    pred = read(str(pred_xyz), index=":")
    if not isinstance(pred, list):
        pred = [pred]
    n_gt = len(gt)
    expected = n_gt * num_samples
    if len(pred) != expected:
        raise ValueError(f"Expected {expected} preds, got {len(pred)}")
    v_ref = np.array([a.get_volume() for a in gt], dtype=float)
    rel = []
    for s in range(num_samples):
        for i in range(n_gt):
            v_p = pred[i + s * n_gt].get_volume()
            rel.append(abs(v_p - v_ref[i]) / max(abs(v_ref[i]), 1e-12))
    rmad = float(np.mean(rel) * 100.0)
    return {"rmad_percent": rmad, "n_gt": n_gt, "num_samples": num_samples}


def run_structure_matching(
    *,
    pt_file: Path,
    num_samples: int,
    stol: float,
    out_json: Path,
    num_cpus: int = 8,
) -> dict:
    cmd = [
        sys.executable,
        str(_MCF_ROOT / "molcrystalflow" / "experiments" / "run_structure_matching.py"),
        "--pt_file",
        str(pt_file),
        "--num_samples",
        str(num_samples),
        "--stol",
        str(stol),
        "--num_cpus",
        str(num_cpus),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_MCF_ROOT}:{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, check=True, cwd=str(_MCF_ROOT), env=env)
    # Official script writes under results/; copy key fields if found
    candidates = list((_MCF_ROOT / "results").glob(f"*matching_summary*{stol}*.json"))
    if not candidates:
        candidates = list(Path(".").glob(f"*matching_summary*{stol}*.json"))
    if candidates:
        data = json.loads(candidates[0].read_text())
        out_json.write_text(json.dumps(data, indent=2) + "\n")
        return data
    return {}


def _rmad_with_uncertainty(volume_summary: dict) -> dict:
    """Paper-style RMAD as mean±std over independent samples (Fig 3c uses 10)."""
    per = volume_summary.get("per_sample_stats") or []
    mads = []
    for row in per:
        for key in ("mean_abs_deviation", "rmad_percent", "mad_percent"):
            if key in row:
                mads.append(float(row[key]))
                break
    if not mads and "rmad_percent" in volume_summary:
        mean = float(volume_summary["rmad_percent"])
        return {
            "rmad_mean_percent": mean,
            "rmad_std_across_samples": None,
            "rmad_sem_across_samples": None,
            "rmad_pm_std": f"{mean:.2f}",
            "n_samples_for_uncertainty": 0,
        }
    arr = np.asarray(mads, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sem = std / np.sqrt(len(arr)) if len(arr) else 0.0
    return {
        "rmad_mean_percent": mean,
        "rmad_std_across_samples": std,
        "rmad_sem_across_samples": sem,
        "rmad_pm_std": f"{mean:.2f}±{std:.2f}",
        "n_samples_for_uncertainty": int(len(arr)),
        "per_sample_mad_percent": mads,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MCF NFT vs Fig 3c")
    p.add_argument("--nft-ckpt", type=str, default=None, help="NFT .pt (optional for baseline)")
    p.add_argument(
        "--base-ckpt",
        type=str,
        default=str(_MCF_ROOT / "model-checkpoints" / "thurlemann23" / "best.ckpt"),
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=str(
            _REPO_ROOT
            / "dataset"
            / "molcrystalflow"
            / "thurlemann23"
            / "preprocessed"
            / "normalized"
        ),
    )
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--num-timesteps", type=int, default=50)
    p.add_argument("--scaling", type=float, default=9.0)
    p.add_argument("--exp-rate", type=float, default=3.0)
    p.add_argument("--stol", type=float, default=0.8)
    p.add_argument("--skip-matching", action="store_true")
    p.add_argument("--paper-rmad", type=float, default=3.86, help="Fig 3c paper RMAD %")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.nft_ckpt:
        export_ckpt = out_dir / "lightning_nft.ckpt"
        ckpt = export_lightning_ckpt(
            Path(args.nft_ckpt), Path(args.base_ckpt), export_ckpt
        )
    else:
        ckpt = Path(args.base_ckpt)

    infer_dir = out_dir / "inference"
    run_inference(
        ckpt=ckpt,
        cache_dir=Path(args.cache_dir),
        out_dir=infer_dir,
        num_samples=args.num_samples,
        num_gpus=args.num_gpus,
        num_timesteps=args.num_timesteps,
        scaling=args.scaling,
        exp_rate=args.exp_rate,
    )

    # Convert predictions_*.pt -> XYZ via official helper
    pts = list(infer_dir.rglob(f"predictions_{args.num_samples}.pt"))
    if not pts:
        pts = list(infer_dir.rglob("predictions_*.pt"))
    if not pts:
        # inference.yaml may write under inference.output_dir
        pts = list((_MCF_ROOT).rglob(f"predictions_{args.num_samples}.pt"))
    xyz_dir = out_dir / "xyz"
    gt_xyz = xyz_dir / "ground_truth.xyz"
    pred_xyz = xyz_dir / "predictions.xyz"
    if pts:
        sys.path.insert(0, str(_MCF_ROOT))
        from molcrystalflow.data.utils import visualize_predictions

        visualize_predictions(
            str(pts[0]),
            save_dir=str(xyz_dir),
            gt_xyz="ground_truth.xyz",
            pred_xyz="predictions.xyz",
        )
        logger.info("Wrote XYZ from %s", pts[0])
    else:
        logger.warning("No predictions_*.pt found under %s", infer_dir)

    volume_summary = {}
    if gt_xyz.is_file() and pred_xyz.is_file():
        vol_dir = out_dir / "volume"
        volume_summary = run_volume_analysis(
            gt_xyz=gt_xyz,
            pred_xyz=pred_xyz,
            num_samples=args.num_samples,
            out_dir=vol_dir,
        )
    else:
        logger.warning("Missing gt/pred xyz at %s", xyz_dir)

    matching = {}
    if not args.skip_matching and pts:
        matching = run_structure_matching(
            pt_file=pts[0],
            num_samples=args.num_samples,
            stol=args.stol,
            out_json=out_dir / f"matching_stol{args.stol}.json",
        )

    rmad = None
    for key in ("rmad_percent", "RMAD", "rmad", "mean_rmad_percent"):
        if key in volume_summary:
            rmad = float(volume_summary[key])
            break
    unc = _rmad_with_uncertainty(volume_summary) if volume_summary else {}
    if rmad is None and unc.get("rmad_mean_percent") is not None:
        rmad = float(unc["rmad_mean_percent"])
    summary = {
        "ckpt": str(ckpt),
        "nft_ckpt": args.nft_ckpt,
        "num_samples": args.num_samples,
        "volume": volume_summary,
        "matching": matching,
        "paper_fig3c_rmad_percent": args.paper_rmad,
        "paper_fig3c_rmad_pm_std": "3.86±0.07",
        "measured_rmad_percent": rmad,
        "measured_rmad_pm_std": unc.get("rmad_pm_std"),
        "rmad_mean_percent": unc.get("rmad_mean_percent"),
        "rmad_std_across_samples": unc.get("rmad_std_across_samples"),
        "rmad_sem_across_samples": unc.get("rmad_sem_across_samples"),
        "beats_fig3c": (rmad is not None and rmad < args.paper_rmad),
        "beats_target_3p2": (rmad is not None and rmad < 3.2),
    }
    (out_dir / "fig3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Summary -> %s", out_dir / "fig3_summary.json")
    if rmad is not None:
        logger.info(
            "RMAD=%s%% | paper=%s%% | beats_fig3c=%s | beats_3.2=%s",
            summary.get("measured_rmad_pm_std") or f"{rmad:.4f}",
            summary["paper_fig3c_rmad_pm_std"],
            summary["beats_fig3c"],
            summary["beats_target_3p2"],
        )


if __name__ == "__main__":
    main()
