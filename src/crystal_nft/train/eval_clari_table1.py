"""Table 1 validation metrics for Clari (clash, PoseBusters, volume, PDD EMD)."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from crystal_nft.adapters.clari_adapter import (  # noqa: E402
    load_clari_bundle,
)

logger = logging.getLogger(__name__)

# amd.PDD uses a tee'd generator that is not re-entrant across threads.
_ASSESS_LOCK = threading.Lock()

TABLE1_PAPER_TARGETS = {
    "clari-m": {
        "clash_rate_pct": 9.56,
        "pb_score_pct": 87.34,
        "volume_error": 1.59,
        "dist_pdd": 9.56,
    },
    "clari-l": {
        "clash_rate_pct": 7.69,
        "pb_score_pct": 85.89,
        "volume_error": 1.50,
        "dist_pdd": 9.28,
    },
}

TABLE1_VAL_SPLIT_OPTS = {
    "train": dict(group_by_fam=True, random_repr=True),
    "val": dict(group_by_fam=True, random_repr=False, augment=False),
    "predict": dict(group_by_fam=True, random_repr=False, augment=False),
    "test": dict(group_by_fam=False, augment=False),
}


def _resolve_model_key(checkpoint: str) -> str | None:
    name = Path(checkpoint).stem.lower()
    for key in ("clari-m", "clari-l", "clari-h"):
        if key.replace("-", "") in name.replace("-", "").replace("_", ""):
            return key
    lowered = checkpoint.strip().lower()
    if lowered in TABLE1_PAPER_TARGETS:
        return lowered
    return None


def _load_val_dataset(clari_data_dir: Path, seed: int = 0):
    os.environ["CLARI_DATA_DIR"] = str(clari_data_dir.resolve())
    csd_root = clari_data_dir.resolve() / "csd"
    for name in ("config.json", "val.pt"):
        path = csd_root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing Clari CSD artifact: {path}. "
                "Place train/val/test.pt under dataset/clari/csd/ "
                "(see the Data section of README.md)."
            )
    from clari.datamodules.csd import CrystalDataModule

    dm = CrystalDataModule(
        seed=seed,
        batch_size=1,
        num_workers=0,
        split_opts=TABLE1_VAL_SPLIT_OPTS,
    )
    return dm.datasets["val"]


def _aggregate_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = (
        "clash_rate",
        "pb_score",
        "volume_error",
        "dist_pdd",
        "dist_amd",
        "pb_valid",
        "stereo_agreement",
        "stereo_cell_correct",
        "stereo_agreement_defined",
        "stereo_cell_correct_defined",
    )
    out = {}
    for key in keys:
        vals = [r[key] for r in rows if key in r and r[key] is not None]
        if vals:
            out[f"mean_{key}"] = float(np.mean(vals))
    if "mean_clash_rate" in out:
        out["mean_clash_rate_pct"] = 100.0 * out["mean_clash_rate"]
    if "mean_pb_score" in out:
        out["mean_pb_score_pct"] = 100.0 * out["mean_pb_score"]
    if "mean_volume_error" in out:
        # Paper Table 1 reports relative volume error in percent.
        out["mean_volume_error_pct"] = 100.0 * out["mean_volume_error"]
    if "mean_stereo_agreement" in out:
        out["mean_stereo_agreement_pct"] = 100.0 * out["mean_stereo_agreement"]
        out["mean_stereo_cell_correct_pct"] = 100.0 * out.get("mean_stereo_cell_correct", 0.0)
        chiral_rows = [r for r in rows if (r.get("stereo_n_centers") or 0) > 0]
        out["n_stereo_samples"] = len(chiral_rows)
        out["n_stereo_families"] = len({r.get("family_idx") for r in chiral_rows})
    if "mean_stereo_agreement_defined" in out:
        out["mean_stereo_agreement_defined_pct"] = (
            100.0 * out["mean_stereo_agreement_defined"]
        )
        out["mean_stereo_cell_correct_defined_pct"] = 100.0 * out.get(
            "mean_stereo_cell_correct_defined", 0.0
        )
    out["n_samples"] = len(rows)
    out["n_families"] = len({r.get("family_idx") for r in rows})
    out["n_samples_per_crystal"] = rows[0].get("n_samples") if rows else 0
    out.update(_paper_bootstrap_summary(rows))
    return out


def _paper_bootstrap_summary(
    rows: list[dict],
    *,
    n_boot: int = 5000,
    draw: int = 5,
    seed: int = 0,
) -> dict:
    """Clari paper Table 1 aggregation (Sec 4.2 / Appendix C).

    For each bootstrap resample: draw ``draw`` samples/crystal with replacement;
    quality metrics use the mean of draws, reconstruction metrics use the min;
    then average across crystals. Volume is reported in percent.
    """
    if not rows:
        return {}
    fams = sorted({r["family_idx"] for r in rows})
    fam_to_i = {f: i for i, f in enumerate(fams)}
    n_samp = max(sum(1 for r in rows if r["family_idx"] == f) for f in fams)
    n_fam = len(fams)
    clash = np.full((n_fam, n_samp), np.nan)
    pb = np.full((n_fam, n_samp), np.nan)
    vol = np.full((n_fam, n_samp), np.nan)
    pdd = np.full((n_fam, n_samp), np.nan)
    pb_valid = np.zeros((n_fam, n_samp))
    seen = {f: 0 for f in fams}
    for r in rows:
        f = r["family_idx"]
        j = seen[f]
        seen[f] += 1
        i = fam_to_i[f]
        clash[i, j] = r.get("clash_rate", np.nan)
        pb[i, j] = r.get("pb_score", np.nan)
        vol[i, j] = r.get("volume_error", np.nan)
        pdd[i, j] = r.get("dist_pdd", np.nan)
        pb_valid[i, j] = float(r.get("pb_valid") or 0.0)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_samp, size=(n_boot, n_fam, draw))
    ii = np.arange(n_fam)[None, :, None]
    clash_b = np.nanmean(clash[ii, idx], axis=-1).mean(axis=-1) * 100.0
    vol_b = np.nanmin(vol[ii, idx], axis=-1).mean(axis=-1) * 100.0
    pdd_b = np.nanmin(pdd[ii, idx], axis=-1).mean(axis=-1)
    pb_s = pb[ii, idx]
    valid_s = pb_valid[ii, idx]
    with np.errstate(invalid="ignore"):
        num = np.nansum(np.where(valid_s > 0, pb_s, np.nan), axis=-1)
        den = np.sum(valid_s > 0, axis=-1)
        fam_pb = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        pb_b = np.nanmean(fam_pb, axis=-1) * 100.0
    return {
        "paper_bootstrap": {
            "n_boot": n_boot,
            "draw": draw,
            "seed": seed,
            "clash_rate_pct": float(clash_b.mean()),
            "clash_rate_pct_se": float(clash_b.std(ddof=1)),
            "pb_score_pct": float(np.nanmean(pb_b)),
            "pb_score_pct_se": float(np.nanstd(pb_b, ddof=1)),
            "volume_error_pct": float(vol_b.mean()),
            "volume_error_pct_se": float(vol_b.std(ddof=1)),
            "dist_pdd": float(pdd_b.mean()),
            "dist_pdd_se": float(pdd_b.std(ddof=1)),
            "note": (
                "quality=mean-of-draws; reconstruction=min-of-draws; "
                "volume in percent; matches Clari paper Table 1 protocol"
            ),
        }
    }


def _compare_to_paper(summary: dict, model_key: str | None) -> dict:
    if model_key is None or model_key not in TABLE1_PAPER_TARGETS:
        return {}
    targets = TABLE1_PAPER_TARGETS[model_key]
    boot = summary.get("paper_bootstrap") or {}
    compare = {"model": model_key, "paper_targets": targets, "delta": {}}
    mapping = {
        "clash_rate_pct": "clash_rate_pct",
        "pb_score_pct": "pb_score_pct",
        "volume_error_pct": "volume_error",
        "dist_pdd": "dist_pdd",
    }
    for ours, paper_key in mapping.items():
        if ours in boot and paper_key in targets:
            compare["delta"][paper_key] = boot[ours] - targets[paper_key]
    return compare


def _active_stereo_tags(tags):
    from crystal_nft.meanflow.stereo import active_stereo_tags

    return active_stereo_tags(tags)


_OFF = ("", "0", "false", "no", "off")


def _pcfm_enabled() -> bool:
    """True when either inference-time chirality corrector is switched on.

    ``CRYSTAF_MIRROR_FIX`` takes ``cell`` / ``body`` as well as ``1``, so this
    must test for "not off" rather than for a fixed set of truthy spellings.
    """
    for var in ("CRYSTAF_MIRROR_FIX", "CRYSTAF_PCFM", "CRYSTAF_PCFM_PARITY",
                "CRYSTAF_PCFM_BOND"):
        if str(os.environ.get(var, "")).strip().lower() not in _OFF:
            return True
    return False


def _bond_enabled() -> bool:
    return str(os.environ.get("CRYSTAF_PCFM_BOND", "")).strip().lower() not in _OFF


def _active_bond(batch):
    """Publish RDKit DG bond bounds so the sampler can project onto them."""
    from contextlib import nullcontext

    if not _bond_enabled():
        return nullcontext()
    from crystal_nft.meanflow.pcfm import active_bond_constraint
    from crystal_nft.meanflow.stereo import build_bond_constraint

    return active_bond_constraint(build_bond_constraint(batch, device=batch.x.device))


def _active_pcfm(spec, body_ids=None):
    """Publish the R/S constraint for the inference-time chirality correctors.

    The target at each centre is the handedness of the *reference molecule* in
    its canonical CIP frame — one bit per stereocentre, i.e. the R/S label.
    That is molecular identity, the same class of input Clari already gets from
    bond orders and formal charges; it says nothing about the target packing.
    """
    from contextlib import nullcontext

    from crystal_nft.meanflow.pcfm import active_constraint, constraint_from_batch_stereo

    if spec is None or not _pcfm_enabled():
        return nullcontext()
    margin = float(os.environ.get("CRYSTAF_PCFM_MARGIN", "0.62"))
    return active_constraint(
        constraint_from_batch_stereo(spec, margin=margin, body_ids=body_ids)
    )


def _conformer_for(batch, mode: str):
    """Conformer coords (in x-units) plus a per-atom validity mask.

    ``oracle``/``relax`` take the reference crystal's own internal geometry --
    a ceiling. ``etkdg``/``etkdg_relax`` take an RDKit ETKDGv3 embedding of the
    molecular graph carrying the requested R/S bits and no reference
    coordinates, which is MolCrystalFlow's actual setting.
    """
    x = batch.x[:, 3:, :].detach().clone()
    if mode in ("oracle", "1", "true", "relax"):
        return x, None, None
    want_tors = "torsion" in mode
    from clari.chem import Crystal

    from crystal_nft.rigid.conformer import STATS, etkdg_conformer_coords

    scale = float(Crystal.COORD_NORM)
    conf = x.clone()
    ok = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
    tors_all = [] if want_tors else None
    for i, single in enumerate(batch.cpu().unbatch()):
        res = etkdg_conformer_coords(single, want_torsions=want_tors)
        pos, good = res[0], res[1]
        if want_tors:
            tors_all.append(res[2])
        n = min(pos.shape[0], x.shape[1])
        conf[i, :n] = (pos[:n] / scale).to(conf.dtype).to(x.device)
        ok[i, :n] = good[:n].to(x.device)
    logger.info(
        "rigid conformer source=%s bodies=%d embedded=%d failed=%d",
        mode, STATS.bodies, STATS.embedded, STATS.failed,
    )
    return conf, ok, tors_all


def _active_rigid(batch):
    """Publish a conformer for the MolCrystalFlow-style rigid projection.

    ``CRYSTAF_RIGID=oracle`` uses the reference crystal's own INTERNAL geometry.
    Packing (lattice, centroids, orientations) still comes entirely from the
    model -- only each molecule's internal shape is replaced. This is a CEILING
    measurement and its numbers are not comparable to the CrystAF table; see
    crystal_nft/rigid/projector.py.
    """
    from contextlib import nullcontext

    mode = str(os.environ.get("CRYSTAF_RIGID", "")).strip().lower()
    if mode in _OFF:
        return nullcontext()
    if mode not in ("oracle", "1", "true", "relax", "etkdg", "etkdg_relax",
                    "etkdg_torsion", "etkdg_torsion_relax"):
        raise ValueError(
            f"CRYSTAF_RIGID={mode!r}: use oracle|relax|etkdg|etkdg_relax|"
            f"etkdg_torsion|etkdg_torsion_relax"
        )
    from crystal_nft.rigid.projector import RigidReference, active_rigid

    body_ids = getattr(batch, "body_ids", None)
    if body_ids is None:
        return nullcontext()
    conformer, conformer_ok, torsions = _conformer_for(batch, mode)
    return active_rigid(
        RigidReference(
            conformer=conformer,
            body_ids=body_ids,
            mask=getattr(batch, "mask", None),
            source="oracle",
            atom_nums=getattr(batch, "atom_nums", None),
            conformer_ok=conformer_ok,
            torsions=torsions,
            relax=mode.endswith("relax"),
            relax_steps=int(os.environ.get("CRYSTAF_RIGID_RELAX_STEPS", "250")),
            relax_margin=float(os.environ.get("CRYSTAF_RIGID_MARGIN", "0.05")),
        )
    )


def _stereo_metrics(pred, true_crystal) -> dict:
    """Per-crystal tetrahedral-stereo retention of a prediction vs its reference.

    ``pred`` and ``true_crystal`` share atom ordering (the prediction is the
    reference crystal with new coordinates), so the index-ordered chiral volume
    at each centre is directly comparable and needs no canonical frame.
    """
    empty = {
        "stereo_n_centers": 0,
        "stereo_agreement": None,
        "stereo_cell_correct": None,
        "stereo_n_defined": 0,
        "stereo_agreement_defined": None,
        "stereo_cell_correct_defined": None,
    }
    try:
        from crystal_nft.meanflow.stereo import (
            TAG_R,
            TAG_S,
            chiral_signs,
            find_stereo_centers,
        )

        centers, tags = find_stereo_centers(true_crystal)
        if centers is None or centers.numel() == 0:
            return dict(empty)
        ref = true_crystal.coords.detach().float().cpu()
        got = pred.coords.detach().float().cpu()
        if got.shape[0] < int(centers.max()) + 1:
            return dict(empty)
        match = chiral_signs(ref, centers) == chiral_signs(got, centers)
        n = int(centers.shape[0])
        ok = int(match.sum().item())
        out = {
            "stereo_n_centers": n,
            "stereo_agreement": ok / n,
            "stereo_cell_correct": 1.0 if ok == n else 0.0,
            **{k: empty[k] for k in ("stereo_n_defined", "stereo_agreement_defined",
                                     "stereo_cell_correct_defined")},
        }
        # RDKit's `includeUnassigned` centres also cover atoms whose substituents
        # tie on CIP rank -- those are not stereogenic at all, and their
        # "handedness" is noise pinned at chance.  Report the restriction to
        # genuinely R/S-labelled centres alongside the raw number.
        if tags is not None:
            defined = (tags[centers[:, 0]] == TAG_R) | (tags[centers[:, 0]] == TAG_S)
            n_def = int(defined.sum().item())
            if n_def:
                ok_def = int((match & defined).sum().item())
                out["stereo_n_defined"] = n_def
                out["stereo_agreement_defined"] = ok_def / n_def
                out["stereo_cell_correct_defined"] = 1.0 if ok_def == n_def else 0.0
        return out
    except Exception as exc:  # noqa: BLE001 — metric must never break eval
        return {**empty, "stereo_error": f"{type(exc).__name__}: {exc}"}


def _freeze_crystal(crystal):
    """Detach (+CPU) all tensor fields so metrics can call .numpy() safely."""
    import dataclasses

    updates = {}
    for field in dataclasses.fields(type(crystal)):
        value = getattr(crystal, field.name)
        if torch.is_tensor(value):
            updates[field.name] = value.detach().cpu()
    return crystal.replace(**updates) if updates else crystal


def _sample_microbatch(lit, crystals: list) -> list:
    """Sample one prediction per crystal; on CUDA OOM, split until size-1 works."""
    from clari.chem import Crystal

    if not crystals:
        return []
    device = next(lit.net.parameters()).device
    use_bf16 = device.type == "cuda"
    # CrystAF stereo conditioning: only pay the RDKit cost when the loaded
    # student actually has a stereo token head.
    stereo_on = (
        getattr(lit.net, "_stereo_tokens", None) is not None
        or getattr(lit.net, "_stereo_pairs", None) is not None
        or _pcfm_enabled()
    )
    out: list = []
    i = 0
    micro = len(crystals)
    while i < len(crystals):
        n_try = min(micro, len(crystals) - i)
        try:
            batch_gpu = Crystal.collate(crystals[i : i + n_try]).to(device)
            stereo_tags = None
            spec = None
            if stereo_on:
                from crystal_nft.meanflow.stereo import (
                    build_batch_stereo,
                    build_stereo_conditioning,
                )

                spec = build_batch_stereo(batch_gpu, device=device)
                if spec is not None:
                    stereo_tags = build_stereo_conditioning(
                        spec,
                        int(batch_gpu.x.shape[1] - 3),
                        pair_edges=getattr(lit.net, "_stereo_pairs", None) is not None,
                    )
            with torch.inference_mode(), _active_stereo_tags(stereo_tags), _active_pcfm(
                spec, getattr(batch_gpu, "body_ids", None)
            ), _active_bond(batch_gpu), _active_rigid(batch_gpu):
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    pred_batch = lit.sampler.sample(
                        lit.interface,
                        lit.net,
                        batch_gpu,
                        pbar=False,
                    )
                preds = [_freeze_crystal(p) for p in pred_batch.cpu().unbatch()]
            del batch_gpu, pred_batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if len(preds) != n_try:
                raise RuntimeError(f"Expected {n_try} samples, got {len(preds)}")
            out.extend(preds)
            i += n_try
            # Slowly grow microbatch after successes (capped by remaining).
            micro = min(len(crystals) - i, max(n_try, micro))
        except RuntimeError as exc:
            if device.type != "cuda" or "out of memory" not in str(exc).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if n_try == 1:
                raise
            micro = max(1, n_try // 2)
            logger.warning("GPU OOM; retrying sample microbatch size=%d", micro)
    return out


def _json_safe_metrics(metrics: dict) -> dict:
    clean = {}
    for k, v in metrics.items():
        if torch.is_tensor(v):
            v = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().tolist()
        elif isinstance(v, (np.floating, np.integer)):
            v = v.item()
        clean[k] = v
    return clean


def _null_metric_row(
    *,
    csd_id: str,
    family_idx: int,
    sample_idx: int,
    samples: int,
    assess_error: str,
) -> dict:
    return {
        "csd_id": csd_id,
        "family_idx": family_idx,
        "sample_idx": sample_idx,
        "n_samples": samples,
        "volume_error": None,
        "clash_rate": None,
        "dist_amd": None,
        "dist_pdd": None,
        "pb_score": None,
        "pb_valid": None,
        "assess_error": assess_error,
    }


def _assess_subprocess_entry(
    result_queue,
    pred,
    true_crystal,
    amd_metric: str,
    mem_limit_bytes: int,
) -> None:
    """Deprecated wrapper. Spawn target is assess_subprocess.assess_subprocess_entry."""
    from crystal_nft.train.assess_subprocess import assess_subprocess_entry

    assess_subprocess_entry(
        result_queue, pred, true_crystal, amd_metric, mem_limit_bytes
    )


def _assess_one(
    pred,
    true_crystal,
    **kwargs,
) -> dict:
    """``_assess_one_impl`` plus the CrystAF stereo-retention columns.

    Stereo is computed in-process (RDKit centre lookup is cached per ``csd_id``
    and the rest is a 3x3 determinant), so it cannot be lost to an assess-child
    timeout or OOM the way the PoseBusters columns can.
    """
    row = _assess_one_impl(pred, true_crystal, **kwargs)
    stereo = _stereo_metrics(pred, true_crystal)
    return {**row, **stereo}


def _assess_one_impl(
    pred,
    true_crystal,
    *,
    csd_id: str,
    family_idx: int,
    sample_idx: int,
    samples: int,
    assess_fn=None,
    amd_metric: str = "cityblock",
    assess_timeout_sec: float = 180.0,
    assess_mem_gb: float = 32.0,
    assess_in_subprocess: bool = True,
) -> dict:
    """Score one prediction; returns JSON-safe scalars only.

    By default runs AMD/PDD/PoseBusters in a spawn'd subprocess with an address
    space cap + join timeout. If the child is killed (OOM) or times out, the
    parent keeps running and records null metrics + ``assess_error``.
    """
    pred = _freeze_crystal(pred)
    true_crystal = _freeze_crystal(true_crystal)
    base = {
        "csd_id": csd_id,
        "family_idx": family_idx,
        "sample_idx": sample_idx,
        "n_samples": samples,
    }

    if not assess_in_subprocess:
        # Legacy in-process path (unsafe for pathological PDD cases).
        if assess_fn is None:
            from clari.pipelines.utils.metrics import assess_crystals_eval as assess_fn
        with _ASSESS_LOCK:
            with torch.inference_mode():
                metrics = assess_fn(pred, true_crystal, amd_metric=amd_metric)
        return {**base, **_json_safe_metrics(metrics)}

    from crystal_nft.train.assess_subprocess import assess_subprocess_entry

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(1)
    mem_bytes = int(float(assess_mem_gb) * (1024**3)) if assess_mem_gb and assess_mem_gb > 0 else 0
    proc = ctx.Process(
        target=assess_subprocess_entry,
        args=(result_queue, pred, true_crystal, amd_metric, mem_bytes),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=float(assess_timeout_sec))
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=30)
        logger.warning(
            "Assess timeout family=%s sample=%d (>%ss) — recording null metrics",
            csd_id,
            sample_idx,
            assess_timeout_sec,
        )
        return _null_metric_row(
            csd_id=csd_id,
            family_idx=family_idx,
            sample_idx=sample_idx,
            samples=samples,
            assess_error=f"timeout>{assess_timeout_sec}s",
        )

    try:
        status, payload = result_queue.get(timeout=5)
    except Exception:  # noqa: BLE001
        exitcode = proc.exitcode
        err = f"child_exitcode={exitcode}"
        if exitcode == -9:
            err = "child_sigkill_oom_or_external"
        logger.warning(
            "Assess child failed family=%s sample=%d (%s)",
            csd_id,
            sample_idx,
            err,
        )
        return _null_metric_row(
            csd_id=csd_id,
            family_idx=family_idx,
            sample_idx=sample_idx,
            samples=samples,
            assess_error=err,
        )
    finally:
        try:
            result_queue.close()
        except Exception:  # noqa: BLE001
            pass

    if status != "ok":
        logger.warning(
            "Assess error family=%s sample=%d: %s", csd_id, sample_idx, payload
        )
        return _null_metric_row(
            csd_id=csd_id,
            family_idx=family_idx,
            sample_idx=sample_idx,
            samples=samples,
            assess_error=str(payload),
        )
    return {**base, **payload}


def _assess_chunk_rows(
    preds: list,
    chunk: list[tuple[int, int]],
    *,
    pack_crystals: list,
    ids: list[str],
    pack_indices: list[int],
    samples: int,
    amd_metric: str,
    assess_timeout_sec: float,
    assess_mem_gb: float,
    assess_in_subprocess: bool,
) -> list[dict]:
    rows: list[dict] = []
    for pred, (pack_i, sample_idx) in zip(preds, chunk):
        rows.append(
            _assess_one(
                pred,
                pack_crystals[pack_i],
                csd_id=ids[pack_i],
                family_idx=pack_indices[pack_i],
                sample_idx=sample_idx,
                samples=samples,
                amd_metric=amd_metric,
                assess_timeout_sec=assess_timeout_sec,
                assess_mem_gb=assess_mem_gb,
                assess_in_subprocess=assess_in_subprocess,
            )
        )
    return rows


def _append_metric_rows(
    shard_jsonl: Path,
    rows: list[dict],
    *,
    pack_indices: list[int],
    chunk: list[tuple[int, int]],
    done_keys: set[tuple[int, int]],
) -> int:
    n_fail = 0
    with shard_jsonl.open("a", encoding="utf-8") as fh:
        for row, (pack_i, sample_idx) in zip(rows, chunk):
            if row.get("assess_error"):
                n_fail += 1
            fh.write(json.dumps(row) + "\n")
            done_keys.add((pack_indices[pack_i], sample_idx))
    return n_fail


def _completed_keys_from_jsonl(path: Path) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    for row in _rows_from_jsonl(path):
        if row.get("assess_error"):
            continue
        try:
            done.add((int(row["family_idx"]), int(row["sample_idx"])))
        except Exception:  # noqa: BLE001
            continue
    return done


def _setup_ranks():
    """Read torchrun env; bind CUDA device to LOCAL_RANK."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def _rows_from_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON via temp file + rename so peers never read a partial shard."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def _wait_for_shard_files(
    output_dir: Path,
    world_size: int,
    *,
    timeout_sec: float = 48 * 3600,
    poll_sec: float = 5.0,
) -> list[Path]:
    """Poll until metrics.shard{0..world_size-1}.json all exist (no process group)."""
    expected = [output_dir / f"metrics.shard{i}.json" for i in range(world_size)]
    deadline = time.time() + timeout_sec
    last_log = 0.0
    while True:
        missing = [p.name for p in expected if not p.is_file()]
        if not missing:
            return expected
        now = time.time()
        if now >= deadline:
            raise TimeoutError(
                f"Timed out waiting for shard files under {output_dir}; missing={missing}"
            )
        if now - last_log >= 60.0:
            logger.info(
                "Waiting for %d/%d shard files (missing=%s)",
                world_size - len(missing),
                world_size,
                missing,
            )
            last_log = now
        time.sleep(poll_sec)


def merge_shard_metrics(output_dir: Path, *, expected_shards: int | None = None) -> dict:
    """Merge metrics.shard*.json into metrics.json (recompute summary)."""
    output_dir = Path(output_dir)
    if expected_shards is not None:
        shard_paths = _wait_for_shard_files(output_dir, int(expected_shards))
    else:
        shard_paths = sorted(output_dir.glob("metrics.shard*.json"))
        # Ignore in-progress *.tmp.* leftovers from atomic writes.
        shard_paths = [p for p in shard_paths if ".tmp." not in p.name]
    if not shard_paths:
        raise FileNotFoundError(f"No metrics.shard*.json under {output_dir}")

    per_crystal: list[dict] = []
    meta: dict | None = None
    for path in shard_paths:
        payload = json.loads(path.read_text())
        if meta is None:
            meta = {
                k: v
                for k, v in payload.items()
                if k not in ("summary", "paper_compare", "per_sample")
            }
        per_crystal.extend(payload.get("per_sample") or [])

    per_crystal.sort(key=lambda r: (r.get("family_idx", 0), r.get("sample_idx", 0)))
    summary = _aggregate_metrics(per_crystal)
    model_key = (meta or {}).get("model_key") or _resolve_model_key(
        (meta or {}).get("checkpoint", "")
    )
    paper_compare = _compare_to_paper(summary, model_key)
    out = {
        **(meta or {}),
        "model_key": model_key,
        "summary": summary,
        "paper_compare": paper_compare,
        "per_sample": per_crystal,
        "n_shards_merged": len(shard_paths),
    }
    _write_json_atomic(output_dir / "metrics.json", out)
    logger.info("Merged %d shards -> %s", len(shard_paths), output_dir / "metrics.json")
    if summary:
        logger.info(
            "Summary | clash=%.2f%% pb=%.2f%% vol=%.4f amd=%.4f emd=%.4f (n=%d)",
            summary.get("mean_clash_rate_pct", float("nan")),
            summary.get("mean_pb_score_pct", float("nan")),
            summary.get("mean_volume_error", float("nan")),
            summary.get("mean_dist_amd", float("nan")),
            summary.get("mean_dist_pdd", float("nan")),
            summary.get("n_families", 0),
        )
    return out


def run_eval(
    *,
    checkpoint: str,
    output_dir: Path,
    samples: int,
    n_steps: int,
    clari_data_dir: Path,
    max_crystals: int | None,
    device: str,
    seed: int,
    nft_ckpt: str | None = None,
    meanflow_ckpt: str | None = None,
    meanflow_steps: int = 4,
    use_meanflow_ema: bool = True,
    shard_id: int | None = None,
    num_shards: int | None = None,
    pack_size: int = 1,
    metric_workers: int = 1,
    amd_metric: str = "cityblock",
    sample_chunk: int = 1,
    resume: bool = True,
    assess_timeout_sec: float = 180.0,
    assess_mem_gb: float = 32.0,
    assess_in_subprocess: bool = True,
    overlap_assess: bool = True,
) -> dict:
    # AMD/PDD is not thread-safe; force sync assess to avoid RAM backlog + tee bugs.
    if metric_workers != 1:
        logger.warning("Forcing metric_workers=1 (requested %d)", metric_workers)
        metric_workers = 1

    rank, world_size, local_rank = _setup_ranks()
    # torchrun DDP takes precedence over manual --shard-id/--num-shards.
    if world_size > 1:
        shard_id = rank
        num_shards = world_size
        if device == "cuda" or device.startswith("cuda"):
            device = f"cuda:{local_rank}"
    else:
        shard_id = 0 if shard_id is None else shard_id
        num_shards = 1 if num_shards is None else num_shards

    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    if pack_size < 1:
        raise ValueError(f"pack_size must be >= 1, got {pack_size}")
    sample_chunk = max(1, int(sample_chunk))

    output_dir.mkdir(parents=True, exist_ok=True)
    val_ds = _load_val_dataset(clari_data_dir, seed=seed)
    n_eval = len(val_ds) if max_crystals is None else min(max_crystals, len(val_ds))
    indices = [i for i in range(n_eval) if i % num_shards == shard_id]
    logger.info(
        "Evaluating %d/%d val families on rank/shard %d/%d "
        "(device=%s pack=%d sample_chunk=%d)",
        len(indices),
        n_eval,
        shard_id,
        num_shards,
        device,
        pack_size,
        sample_chunk,
    )

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    bundle = load_clari_bundle(
        checkpoint,
        device=device,
        use_ema=True,
        n_steps=n_steps,
    )
    lit = bundle["lit"]
    if nft_ckpt:
        from crystal_nft.adapters.clari_adapter import load_clari_nft_state

        missing, unexpected = load_clari_nft_state(lit.net, nft_ckpt)
        logger.info(
            "Loaded NFT weights from %s (missing=%d unexpected=%d)",
            nft_ckpt,
            len(missing),
            len(unexpected),
        )
    if meanflow_ckpt:
        from crystal_nft.meanflow.adapter import apply_meanflow_checkpoint_to_lit

        payload = apply_meanflow_checkpoint_to_lit(
            lit, meanflow_ckpt, meanflow_steps=int(meanflow_steps), load_ema=use_meanflow_ema
        )
        logger.info(
            "Loaded MeanFlow weights from %s (steps=%d ema=%s)",
            meanflow_ckpt,
            meanflow_steps,
            bool(use_meanflow_ema and payload.get("ema_state_dict")),
        )
    lit.eval()

    sample_t0 = time.perf_counter()
    sample_n = 0
    gen_sec = [0.0]
    gen_calls: list[tuple[float, int]] = []

    def _timed_sample(lit_obj, replicas):
        # Generation-only wall clock (network + any sampling-time correction),
        # excluding metric assessment, for the cost column of the paper.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = _sample_microbatch(lit_obj, replicas)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_call = time.perf_counter() - t0
        gen_sec[0] += dt_call
        gen_calls.append((dt_call, len(replicas)))
        return out
    # Stream scalar metric rows only — never all_gather Crystal objects.
    shard_jsonl = output_dir / f"metrics.shard{rank}.jsonl"
    done_keys: set[tuple[int, int]] = set()
    if resume and shard_jsonl.is_file():
        done_keys = _completed_keys_from_jsonl(shard_jsonl)
        logger.info(
            "[rank %d] resume: %d completed (family,sample) keys in %s",
            rank,
            len(done_keys),
            shard_jsonl.name,
        )
    elif shard_jsonl.is_file():
        shard_jsonl.unlink()

    n_skipped = 0
    n_assess_fail = 0
    for start in range(0, len(indices), pack_size):
        pack_indices = indices[start : start + pack_size]
        pack_crystals = [_freeze_crystal(val_ds[idx]) for idx in pack_indices]
        ids = [
            getattr(c, "csd_id", f"val_{idx}")
            for c, idx in zip(pack_crystals, pack_indices)
        ]
        pending: list[tuple[int, int]] = []  # (pack_i, sample_idx)
        for pack_i, fam_idx in enumerate(pack_indices):
            for sample_idx in range(samples):
                if (fam_idx, sample_idx) in done_keys:
                    n_skipped += 1
                    continue
                pending.append((pack_i, sample_idx))
        if not pending:
            continue

        logger.info(
            "[rank %d | pack %d-%d / %d] %d samples left (chunk=%d timeout=%ss mem=%.0fGiB)",
            shard_id,
            start + 1,
            start + len(pack_indices),
            len(indices),
            len(pending),
            sample_chunk,
            assess_timeout_sec,
            assess_mem_gb,
        )

        assess_kw = dict(
            pack_crystals=pack_crystals,
            ids=ids,
            pack_indices=pack_indices,
            samples=samples,
            amd_metric=amd_metric,
            assess_timeout_sec=assess_timeout_sec,
            assess_mem_gb=assess_mem_gb,
            assess_in_subprocess=assess_in_subprocess,
        )
        chunk_list = [
            pending[p0 : p0 + sample_chunk]
            for p0 in range(0, len(pending), sample_chunk)
        ]

        def _flush_assess(preds, chunk) -> None:
            nonlocal n_assess_fail
            rows = _assess_chunk_rows(preds, chunk, **assess_kw)
            if os.environ.get("CRYSTAF_EVAL_UMA_SCORE", "0") not in ("", "0"):
                # UMA single-point diagnostics of the generated structure. Runs
                # after (and outside) the timed sampler call, so it never enters
                # the cost column.
                from crystal_nft.meanflow.physics_injection import uma_diagnostics

                for row, diag in zip(rows, uma_diagnostics(preds)):
                    row.update(diag)
            n_assess_fail += _append_metric_rows(
                shard_jsonl,
                rows,
                pack_indices=pack_indices,
                chunk=chunk,
                done_keys=done_keys,
            )

        if overlap_assess and len(chunk_list) > 1:
            # GPU samples chunk i+1 while CPU assesses chunk i (PDD is not
            # thread-safe, so only one assess worker).
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="assess") as pool:
                prev_fut = None
                prev_preds = None
                prev_chunk = None
                for chunk in chunk_list:
                    replicas = [pack_crystals[pack_i] for pack_i, _ in chunk]
                    preds = _timed_sample(lit, replicas)
                    sample_n += len(preds)
                    if prev_fut is not None:
                        prev_fut.result()
                    prev_fut = pool.submit(_flush_assess, preds, chunk)
                    prev_preds, prev_chunk = preds, chunk
                    del replicas
                if prev_fut is not None:
                    prev_fut.result()
                del prev_preds, prev_chunk
        else:
            for chunk in chunk_list:
                replicas = [pack_crystals[pack_i] for pack_i, _ in chunk]
                preds = _timed_sample(lit, replicas)
                sample_n += len(preds)
                _flush_assess(preds, chunk)
                del preds, replicas, chunk

        del pack_crystals, ids, pending
        if (start // max(pack_size, 1)) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    sample_elapsed = time.perf_counter() - sample_t0
    model_key = _resolve_model_key(checkpoint)
    meta = {
        "checkpoint": checkpoint,
        "nft_ckpt": nft_ckpt,
        "meanflow_ckpt": meanflow_ckpt,
        "meanflow_steps": meanflow_steps,
        "sampling_wall_sec_local": sample_elapsed,
        "sampling_n_local": sample_n,
        "ms_per_sample_local": 1000.0 * sample_elapsed / max(sample_n, 1),
        "gen_sec_local": gen_sec[0],
        "gen_ms_per_sample_local": 1000.0 * gen_sec[0] / max(sample_n, 1),
        # per sampler call (seconds, structures); drop the first for warm-up
        "gen_calls_local": gen_calls,
        "physics_stats_local": dict(__import__("crystal_nft.meanflow.physics_injection", fromlist=["STATS"]).STATS),
        "model_key": model_key,
        "clari_data_dir": str(clari_data_dir),
        "samples": samples,
        "n_steps": n_steps,
        "max_crystals": max_crystals,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "world_size": world_size,
        "pack_size": pack_size,
        "sample_chunk": sample_chunk,
        "metric_workers": metric_workers,
        "amd_metric": amd_metric,
        "resume": resume,
        "assess_timeout_sec": assess_timeout_sec,
        "assess_mem_gb": assess_mem_gb,
        "assess_in_subprocess": assess_in_subprocess,
        "overlap_assess": overlap_assess,
        "n_skipped_resume": n_skipped,
        "n_assess_fail": n_assess_fail,
    }
    logger.info(
        "[rank %d] finished local loop skipped=%d assess_fail=%d sampled=%d",
        rank,
        n_skipped,
        n_assess_fail,
        sample_n,
    )

    local_rows = _rows_from_jsonl(shard_jsonl)
    local_summary = _aggregate_metrics(local_rows)
    # Atomic write: peers must never observe a half-written shard JSON.
    _write_json_atomic(
        output_dir / f"metrics.shard{rank}.json",
        {
            **meta,
            "summary": local_summary,
            "paper_compare": {},
            "per_sample": local_rows,
        },
    )
    logger.info(
        "[rank %d] wrote shard (%d rows, %.1f min sampling+metrics)",
        rank,
        len(local_rows),
        sample_elapsed / 60.0,
    )
    n_local_rows = len(local_rows)
    del local_rows
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # File rendezvous only — no process-group barrier (avoids hang if a peer dies
    # after writing its shard, and avoids gloo/NCCL conflicts with concurrent jobs).
    summary: dict = {}
    paper_compare: dict = {}
    if rank == 0:
        merged = merge_shard_metrics(output_dir, expected_shards=world_size)
        summary = merged.get("summary") or {}
        paper_compare = merged.get("paper_compare") or {}
    return {
        **meta,
        "summary": summary,
        "paper_compare": paper_compare,
        "per_sample": [],
        "n_local_rows": n_local_rows,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Clari Table 1 val-set evaluation")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="clari-m, clari-l, or path to .ckpt (not required with --merge-shards)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for metrics.json",
    )
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument(
        "--clari-data-dir",
        type=str,
        default=str(_REPO_ROOT / "dataset" / "clari"),
    )
    p.add_argument("--max-crystals", type=int, default=None)
    p.add_argument(
        "--sample-chunk",
        type=int,
        default=1,
        help="Samples per sampler call (the CrystAF evals use 20)",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--nft-ckpt",
        type=str,
        default=None,
        help="Optional NFT checkpoint with net_state_dict to load on top of --checkpoint",
    )
    p.add_argument("--shard-id", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument(
        "--pack-size",
        type=int,
        default=1,
        help="Crystals packed into one GPU sampling batch",
    )
    p.add_argument(
        "--metric-workers",
        type=int,
        default=1,
        help="CPU threads for metrics (amd.PDD is not thread-safe; keep 1 unless locked)",
    )
    p.add_argument(
        "--amd-metric",
        type=str,
        default="cityblock",
        help="Row distance for EMD(PDD); paper uses L1/cityblock (code default was chebyshev)",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from metrics.shard{rank}.jsonl if present (default: true)",
    )
    p.add_argument(
        "--assess-timeout-sec",
        type=float,
        default=180.0,
        help="Kill a single AMD/PDD/PB assess after this many seconds",
    )
    p.add_argument(
        "--assess-mem-gb",
        type=float,
        default=32.0,
        help="Address-space headroom for assess child (GiB, added on top of current VmSize); 0 disables",
    )
    p.add_argument(
        "--assess-in-subprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run assess in a spawn'd child so OOM/timeout cannot kill the sampler",
    )
    p.add_argument(
        "--overlap-assess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assess chunk i on CPU while GPU samples chunk i+1",
    )
    p.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge metrics.shard*.json in --output-dir into metrics.json",
    )
    p.add_argument(
        "--reaggregate",
        action="store_true",
        help="Recompute paper bootstrap summary from existing metrics.json (no resampling)",
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.merge_shards:
        merge_shard_metrics(Path(args.output_dir))
        return
    if args.reaggregate:
        path = Path(args.output_dir) / "metrics.json"
        payload = json.loads(path.read_text())
        summary = _aggregate_metrics(payload.get("per_sample") or [])
        model_key = payload.get("model_key") or _resolve_model_key(payload.get("checkpoint", ""))
        payload["summary"] = summary
        payload["paper_compare"] = _compare_to_paper(summary, model_key)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        boot = summary.get("paper_bootstrap") or {}
        logger.info(
            "Reaggregated %s | clash=%.2f±%.2f pb=%.2f±%.2f vol%%=%.3f±%.3f pdd=%.3f±%.3f",
            path,
            boot.get("clash_rate_pct", float("nan")),
            boot.get("clash_rate_pct_se", float("nan")),
            boot.get("pb_score_pct", float("nan")),
            boot.get("pb_score_pct_se", float("nan")),
            boot.get("volume_error_pct", float("nan")),
            boot.get("volume_error_pct_se", float("nan")),
            boot.get("dist_pdd", float("nan")),
            boot.get("dist_pdd_se", float("nan")),
        )
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --merge-shards/--reaggregate")
    run_eval(
        checkpoint=args.checkpoint,
        output_dir=Path(args.output_dir),
        samples=args.samples,
        n_steps=args.n_steps,
        clari_data_dir=Path(args.clari_data_dir),
        max_crystals=args.max_crystals,
        device=args.device,
        seed=args.seed,
        nft_ckpt=args.nft_ckpt,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        pack_size=args.pack_size,
        metric_workers=args.metric_workers,
        amd_metric=args.amd_metric,
        resume=bool(args.resume),
        assess_timeout_sec=float(args.assess_timeout_sec),
        assess_mem_gb=float(args.assess_mem_gb),
        assess_in_subprocess=bool(args.assess_in_subprocess),
        overlap_assess=bool(args.overlap_assess),
        sample_chunk=max(1, int(args.sample_chunk)),
    )


if __name__ == "__main__":
    main()
