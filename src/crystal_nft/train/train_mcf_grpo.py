"""Online Flow-GRPO-style fine-tuning for MolCrystalFlow.

Fair counterpart to train_mcf_nft.py: same UMA/volume rewards, sampling, and
multi-GPU filesystem sync; replaces NFT mixture loss with clipped GRPO on a
continuous -MSE log-prob proxy (Flow-GRPO style without SDE logprobs).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from crystal_nft.adapters.mcf_adapter import (  # noqa: E402
    extract_features_from_xyz,
    grpo_train_step_mcf,
    load_mcf_bundle,
    sample_packings,
)
from crystal_nft.nft.loss import sync_old_policy  # noqa: E402
from crystal_nft.rewards.advantages import (  # noqa: E402
    compute_group_advantages,
)
from crystal_nft.rewards.uma_scorer import UMAScorer  # noqa: E402

logger = logging.getLogger(__name__)


def _default_ckpt() -> str:
    cand = (
        _REPO_ROOT
        / "CrystalGenModel"
        / "MolCrystalFlow"
        / "model-checkpoints"
        / "thurlemann23"
        / "best.ckpt"
    )
    if cand.is_file():
        return str(cand)
    cand2 = (
        _REPO_ROOT
        / "CrystalGenModel"
        / "MolCrystalFlow"
        / "model-checkpoints"
        / "omc25-mcf"
        / "best.ckpt"
    )
    return str(cand2)


def _setup_ranks():
    """Read torchrun / manual rank env without initializing NCCL."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def _average_state_dicts(dicts: list[dict]) -> dict:
    keys = dicts[0].keys()
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts]
        if not torch.is_floating_point(vals[0]):
            out[k] = vals[0]
            continue
        acc = vals[0].float().clone()
        for v in vals[1:]:
            acc += v.float()
        acc /= float(len(vals))
        out[k] = acc.to(dtype=vals[0].dtype)
    return out


def _sync_weights_filesystem(
    save_dir: Path,
    epoch: int,
    rank: int,
    world_size: int,
    train_module: torch.nn.Module,
    old_net: torch.nn.Module,
    timeout_s: float = 86400.0,
) -> None:
    """Average net/old_net across ranks via shard files (no NCCL)."""
    if world_size <= 1:
        return
    shard = save_dir / f".shard_rank{rank}_epoch{epoch}.pt"
    torch.save(
        {
            "net_state_dict": {
                k: v.detach().cpu() for k, v in train_module.state_dict().items()
            },
            "old_state_dict": {
                k: v.detach().cpu() for k, v in old_net.state_dict().items()
            },
        },
        shard,
    )
    (save_dir / f".shard_rank{rank}_epoch{epoch}.done").write_text("ok\n")

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if all(
            (save_dir / f".shard_rank{r}_epoch{epoch}.done").exists()
            for r in range(world_size)
        ):
            break
        time.sleep(2.0)
    else:
        raise TimeoutError(f"Timed out waiting for epoch {epoch} shards")

    avg_path = save_dir / f".avg_epoch{epoch}.pt"
    if rank == 0:
        dicts_net = []
        dicts_old = []
        for r in range(world_size):
            payload = torch.load(
                save_dir / f".shard_rank{r}_epoch{epoch}.pt",
                map_location="cpu",
                weights_only=False,
            )
            dicts_net.append(payload["net_state_dict"])
            dicts_old.append(payload["old_state_dict"])
        torch.save(
            {
                "net_state_dict": _average_state_dicts(dicts_net),
                "old_state_dict": _average_state_dicts(dicts_old),
            },
            avg_path,
        )
        (save_dir / f".avg_epoch{epoch}.done").write_text("ok\n")

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if (save_dir / f".avg_epoch{epoch}.done").exists() and avg_path.exists():
            break
        time.sleep(2.0)
    else:
        raise TimeoutError(f"Timed out waiting for averaged epoch {epoch}")

    time.sleep(0.5 * rank)
    avg = torch.load(avg_path, map_location="cpu", weights_only=False)
    train_module.load_state_dict(avg["net_state_dict"], strict=False)
    old_net.load_state_dict(avg["old_state_dict"], strict=False)
    (save_dir / f".loaded_rank{rank}_epoch{epoch}.done").write_text("ok\n")

    if rank == 0:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if all(
                (save_dir / f".loaded_rank{r}_epoch{epoch}.done").exists()
                for r in range(world_size)
            ):
                break
            time.sleep(1.0)
        # Extra settle so slow ranks finish reading avg before deletion
        time.sleep(2.0)
        for r in range(world_size):
            for p in (
                save_dir / f".shard_rank{r}_epoch{epoch}.pt",
                save_dir / f".shard_rank{r}_epoch{epoch}.done",
                save_dir / f".loaded_rank{r}_epoch{epoch}.done",
            ):
                p.unlink(missing_ok=True)
        (save_dir / f".avg_epoch{epoch}.done").unlink(missing_ok=True)
        avg_path.unlink(missing_ok=True)


def _load_molecules(cfg: dict) -> list[dict]:
    molecules = list(cfg.get("molecules") or [])
    manifest = cfg.get("molecules_manifest")
    if manifest:
        data = json.loads(Path(manifest).read_text())
        if isinstance(data, list):
            molecules.extend(data)
        else:
            molecules.extend(data.get("molecules") or [])
    mol_dir = cfg.get("molecule_dir")
    if mol_dir:
        for xyz in sorted(Path(mol_dir).glob("*.xyz")):
            molecules.append(
                {
                    "id": xyz.stem,
                    "xyz": str(xyz.resolve()),
                    "z_value": int(cfg.get("z_value", 4)),
                    "has_axis_flip": bool(cfg.get("has_axis_flip", False)),
                }
            )
    # Deduplicate by id
    seen = set()
    unique = []
    for m in molecules:
        mid = m.get("id") or Path(m["xyz"]).stem
        if mid in seen:
            continue
        seen.add(mid)
        m = dict(m)
        m["id"] = mid
        unique.append(m)
    return unique


def _load_config(path: str | None) -> dict:
    defaults = {
        "ckpt_path": _default_ckpt(),
        "device": "cuda",
        "molecules": [],
        "molecules_manifest": None,
        "molecule_dir": None,
        "num_epochs": 2,
        "steps_per_epoch": 0,
        "samples_per_mol": 8,
        "inner_epochs": 1,
        "train_batch_size": 2,
        "z_value": 4,
        "has_axis_flip": False,
        "num_timesteps": 50,
        "scaling": 9.0,
        "exp_rate": 3.0,
        "sampling_batch_size": 1,
        "async_sample_prefetch": False,
        "lr": 3.0e-7,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "beta": 0.1,
        "kl_coef": 0.01,
        "clip_range": 0.2,
        "grpo_tau": 1.0,
        "decay_type": 1,
        "ef_lambda": 0.5,
        "advantage_clip": 5.0,
        "adv_mode": "continuous",
        "w_clash": 1.0,
        "w_fmax": 0.0,
        "w_stress": 0.0,
        "w_volume": 0.0,
        "project_volume_targets": False,
        "volume_projection_strength": 1.0,
        "projection_positive_only": True,
        "train_lattice_only": False,
        "uma_model": "uma-s-1p1",
        "uma_ckpt_path": str(_REPO_ROOT / "checkpoints" / "uma" / "uma-s-1p1.pt"),
        "uma_batch_size": 4,
        "use_ultrafast": False,
        "save_dir": "logs/grpo/molcrystalflow",
        "save_every": 1,
        "seed": 0,
        "sync_timeout_s": 86400.0,
        "start_epoch": 0,
        "resume_ckpt": None,
    }
    if path is None:
        return defaults
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    defaults.update(user)
    if not defaults.get("ckpt_path"):
        defaults["ckpt_path"] = _default_ckpt()
    return defaults


def parse_args():
    p = argparse.ArgumentParser(description="MolCrystalFlow Flow-GRPO fine-tuning")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--xyz", type=str, default=None, help="Override single-molecule XYZ")
    p.add_argument("--z", type=int, default=None)
    return p.parse_args()


def _sample_molecule_records(
    *,
    mol: dict,
    cfg: dict,
    flow,
    old_model,
    device: str,
    epoch: int,
) -> tuple[dict, dict, list[dict], dict]:
    """Feature extract + packing sample for one molecule (GPU producer stage)."""
    mol_id = mol.get("id", Path(mol["xyz"]).stem)
    z_value = int(mol.get("z_value", cfg["z_value"]))
    has_flip = bool(mol.get("has_axis_flip", cfg["has_axis_flip"]))
    K = int(cfg["samples_per_mol"])
    t0 = time.perf_counter()
    mol_features = extract_features_from_xyz(mol["xyz"], verbose=False)
    t_feat = time.perf_counter()
    records = sample_packings(
        flow,
        old_model,
        mol_features,
        z_value=z_value,
        num_samples=K,
        has_axis_flip=has_flip,
        num_timesteps=cfg["num_timesteps"],
        scaling=cfg["scaling"],
        exp_rate=cfg["exp_rate"],
        sampling_batch_size=int(cfg.get("sampling_batch_size", 1)),
        device=device,
        seed=cfg["seed"] + epoch * 10007 + hash(mol_id) % 100000,
    )
    t_sample = time.perf_counter()
    meta = {
        "mol_id": mol_id,
        "z_value": z_value,
        "has_flip": has_flip,
        "feature_s": t_feat - t0,
        "sample_s": t_sample - t_feat,
    }
    return mol, mol_features, records, meta


def _train_one_molecule(
    *,
    mol: dict,
    cfg: dict,
    flow,
    model,
    old_model,
    ref_model,
    optimizer,
    scorer: UMAScorer,
    device: str,
    epoch: int,
    global_step: int,
    prefetched: tuple | None = None,
) -> tuple[list[dict], int]:
    t_start = time.perf_counter()
    mol_id = mol.get("id", Path(mol["xyz"]).stem)
    z_value = int(mol.get("z_value", cfg["z_value"]))
    has_flip = bool(mol.get("has_axis_flip", cfg["has_axis_flip"]))
    K = int(cfg["samples_per_mol"])
    stats: list[dict] = []

    logger.info("Epoch %d | %s Z=%d K=%d", epoch, mol_id, z_value, K)
    if prefetched is not None:
        _mol, mol_features, records, sample_meta = prefetched
        pref_id = _mol.get("id", Path(_mol["xyz"]).stem)
        if pref_id != mol_id:
            raise RuntimeError(
                f"Async prefetch mismatch: expected mol={mol_id}, got={pref_id}"
            )
        z_value = int(_mol.get("z_value", z_value))
        has_flip = bool(_mol.get("has_axis_flip", has_flip))
    else:
        _mol, mol_features, records, sample_meta = _sample_molecule_records(
            mol=mol,
            cfg=cfg,
            flow=flow,
            old_model=old_model,
            device=device,
            epoch=epoch,
        )
    feature_s = float(sample_meta.get("feature_s", 0.0))
    sample_s = float(sample_meta.get("sample_s", 0.0))
    if not records:
        logger.warning("No samples for %s", mol_id)
        return stats, global_step

    atoms_list = [r["atoms"] for r in records]
    scores = scorer.score_atoms(atoms_list, n_mols=[z_value] * len(atoms_list))
    t_scored = time.perf_counter()

    # Per-candidate cell volume from the sampled lattice (Angstrom^3).
    w_volume = float(cfg.get("w_volume", 0.0) or 0.0)
    volumes = None
    volume_targets = None
    gt_volume = mol.get("gt_volume")
    if w_volume != 0.0 and gt_volume is not None:
        vols = []
        for rec in records:
            lat = rec["lattice_1"]
            lat = lat.detach().cpu().numpy() if hasattr(lat, "detach") else np.asarray(lat)
            vols.append(float(abs(np.linalg.det(lat.reshape(3, 3)))))
        volumes = vols
        volume_targets = [float(gt_volume)] * len(records)

    advantages = compute_group_advantages(
        scores,
        [mol_id] * len(scores),
        ef_lambda=cfg["ef_lambda"],
        advantage_clip=cfg["advantage_clip"],
        adv_mode=cfg["adv_mode"],
        w_clash=cfg["w_clash"],
        w_fmax=cfg["w_fmax"],
        w_stress=cfg["w_stress"],
        w_volume=w_volume,
        volumes=volumes,
        volume_targets=volume_targets,
    )
    if volumes is not None:
        rel = np.abs(np.array(volumes) - float(gt_volume)) / max(abs(float(gt_volume)), 1e-8)
        logger.info(
            "  vol rel-dev to GT: mean=%.3f min=%.3f (GT=%.1f)",
            float(np.mean(rel)), float(np.min(rel)), float(gt_volume),
        )

        # Keep optional volume projection for ablations; default off for fair
        # GRPO vs NFT-v2 (neither uses projection).
        if bool(cfg.get("project_volume_targets", False)):
            strength = float(cfg.get("volume_projection_strength", 1.0))
            positive_only = bool(cfg.get("projection_positive_only", True))
            projected = 0
            for i, rec in enumerate(records):
                if positive_only and advantages[i] <= 0.0:
                    continue
                vol = max(float(volumes[i]), 1e-8)
                linear_scale = (float(gt_volume) / vol) ** (strength / 3.0)
                rec["lattice_1"] = rec["lattice_1"] * linear_scale
                projected += 1
            logger.info(
                "  projected lattice targets=%d/%d strength=%.2f",
                projected,
                len(records),
                strength,
            )
    valid_e = [s.energy_per_mol for s in scores if s.valid]
    logger.info(
        "  scored %d | valid=%d | mean E/Z=%.4f | mean_adv=%.3f",
        len(scores),
        sum(1 for s in scores if s.valid),
        float(np.mean(valid_e)) if valid_e else float("nan"),
        float(np.mean(advantages)),
    )

    model.train()
    bs = int(cfg["train_batch_size"])
    for _inner in range(int(cfg["inner_epochs"])):
        perm = np.random.permutation(len(records))
        for start in range(0, len(records), bs):
            sel = perm[start : start + bs]
            batch_recs = [records[i] for i in sel]
            batch_adv = torch.tensor(advantages[sel], dtype=torch.float32, device=device)
            loss_dict = grpo_train_step_mcf(
                flow,
                model,
                old_model,
                ref_model,
                mol_features,
                batch_recs,
                batch_adv,
                clip_range=float(cfg.get("clip_range", 0.2)),
                tau=float(cfg.get("grpo_tau", 1.0)),
                kl_coef=cfg["kl_coef"],
                has_axis_flip=has_flip,
                lattice_only=bool(cfg.get("train_lattice_only", False)),
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            global_step += 1
            stats.append(
                {
                    "epoch": epoch,
                    "mol": mol_id,
                    "step": global_step,
                    "loss": float(loss_dict["loss"].detach()),
                    "policy_loss": float(loss_dict["policy_loss"]),
                    "kl_loss": float(loss_dict["kl_loss"]),
                    "clipfrac": float(loss_dict["clipfrac"]),
                }
            )
            logger.info(
                "  step %d loss=%.4f policy=%.4f kl=%.4f clipfrac=%.3f",
                global_step,
                stats[-1]["loss"],
                stats[-1]["policy_loss"],
                stats[-1]["kl_loss"],
                stats[-1]["clipfrac"],
            )
    t_updated = time.perf_counter()
    # If samples were prefetched, t_start is the consumer start (score/update).
    if prefetched is not None:
        score_s = t_scored - t_start
    else:
        score_s = t_scored - (t_start + feature_s + sample_s)
    update_s = t_updated - t_scored
    wall_s = t_updated - t_start
    pipeline_s = feature_s + sample_s + score_s + update_s
    logger.info(
        "  perf feature=%.2fs sample=%.2fs score=%.2fs update=%.2fs "
        "sample_throughput=%.2f structures/s pipeline=%.2fs wall=%.2fs prefetch=%s",
        feature_s,
        sample_s,
        score_s,
        update_s,
        len(records) / max(sample_s, 1e-8),
        pipeline_s,
        wall_s,
        bool(prefetched is not None),
    )
    return stats, global_step


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [r%(process)d] %(levelname)s %(message)s",
        force=True,
    )
    # Ensure INFO reaches redirected nohup logs under torchrun
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    cfg = _load_config(args.config)
    rank, world_size, local_rank = _setup_ranks()
    logger.info("rank=%d world=%d local_rank=%d", rank, world_size, local_rank)

    if args.xyz:
        cfg["molecules"] = [
            {
                "id": Path(args.xyz).stem,
                "xyz": args.xyz,
                "z_value": args.z or cfg["z_value"],
            }
        ]
    if args.smoke:
        cfg["num_epochs"] = 1
        cfg["samples_per_mol"] = 4
        cfg["train_batch_size"] = 2
        cfg["inner_epochs"] = 1
        cfg["num_timesteps"] = 20
        cfg["steps_per_epoch"] = max(2, int(cfg.get("steps_per_epoch") or 2))

    molecules = _load_molecules(cfg)
    if not molecules:
        raise SystemExit(
            "No molecules configured. Pass --config with molecules_manifest / "
            "molecules[] / molecule_dir, or --xyz path."
        )

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device
    torch.manual_seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (save_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        (save_dir / "molecules_used.json").write_text(
            json.dumps({"n": len(molecules), "molecules": molecules}, indent=2) + "\n"
        )

    logger.info(
        "Loading MCF on %s (world=%d) | molecules=%d", device, world_size, len(molecules)
    )
    bundle = load_mcf_bundle(cfg["ckpt_path"], device=device)
    flow, model, old_model, ref_model = (
        bundle["flow"],
        bundle["model"],
        bundle["old_model"],
        bundle["ref_model"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    global_step = 0
    start_epoch = int(cfg.get("start_epoch") or 0)
    resume_ckpt = cfg.get("resume_ckpt")
    if resume_ckpt:
        payload = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=False)
        if payload.get("old_state_dict"):
            old_model.load_state_dict(payload["old_state_dict"], strict=False)
        if payload.get("optimizer"):
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except Exception as exc:
                logger.warning("Could not load optimizer state (%s)", exc)
        global_step = int(payload.get("global_step") or 0)
        if start_epoch == 0 and "epoch" in payload:
            start_epoch = int(payload["epoch"]) + 1
        logger.info(
            "Resumed from %s | start_epoch=%d global_step=%d",
            resume_ckpt,
            start_epoch,
            global_step,
        )

    scorer = UMAScorer(
        cfg["uma_model"],
        device=device,
        use_ultrafast=cfg["use_ultrafast"],
        batch_size=cfg["uma_batch_size"],
        checkpoint_path=cfg.get("uma_ckpt_path"),
    )

    metrics_log = []
    steps_per_epoch = int(cfg.get("steps_per_epoch") or 0)
    sync_timeout_s = float(cfg.get("sync_timeout_s") or 86400.0)

    for epoch in range(start_epoch, cfg["num_epochs"]):
        if steps_per_epoch > 0:
            rng = np.random.default_rng(cfg["seed"] + epoch * 17 + rank)
            order = rng.permutation(len(molecules))
            # Each rank takes a strided subset of a shared epoch schedule
            schedule = [
                molecules[int(order[i % len(order)])]
                for i in range(steps_per_epoch)
                if i % world_size == rank
            ]
        else:
            schedule = [m for i, m in enumerate(molecules) if i % world_size == rank]

        logger.info(
            "Epoch %d rank %d local molecules=%d", epoch, rank, len(schedule)
        )
        logger.info("[rank%d] epoch=%d local_mols=%d", rank, epoch, len(schedule))
        epoch_stats = []
        use_prefetch = bool(cfg.get("async_sample_prefetch", False)) and len(schedule) > 0
        if use_prefetch:
            # Double-buffer: while scoring/updating molecule i on this thread,
            # sample molecule i+1 in a worker. Both use the same CUDA device;
            # PyTorch releases the GIL around CUDA kernels so Python-side
            # overhead and some kernel queuing can overlap. A lock is avoided
            # to maximize occupancy; correctness relies on old_model being
            # frozen and scorer/model not sharing mutable buffers across threads.
            logger.info("Async sample prefetch enabled (double-buffer)")
            with ThreadPoolExecutor(max_workers=1) as pool:
                next_fut: Future | None = pool.submit(
                    _sample_molecule_records,
                    mol=schedule[0],
                    cfg=cfg,
                    flow=flow,
                    old_model=old_model,
                    device=device,
                    epoch=epoch,
                )
                for mi, mol in enumerate(schedule):
                    assert next_fut is not None
                    prefetched = next_fut.result()
                    if mi + 1 < len(schedule):
                        next_fut = pool.submit(
                            _sample_molecule_records,
                            mol=schedule[mi + 1],
                            cfg=cfg,
                            flow=flow,
                            old_model=old_model,
                            device=device,
                            epoch=epoch,
                        )
                    else:
                        next_fut = None
                    stats, global_step = _train_one_molecule(
                        mol=mol,
                        cfg=cfg,
                        flow=flow,
                        model=model,
                        old_model=old_model,
                        ref_model=ref_model,
                        optimizer=optimizer,
                        scorer=scorer,
                        device=device,
                        epoch=epoch,
                        global_step=global_step,
                        prefetched=prefetched,
                    )
                    epoch_stats.extend(stats)
                    if (mi + 1) % 5 == 0 or (mi + 1) == len(schedule):
                        logger.info(
                            "[rank%d] epoch=%d mol %d/%d step=%d",
                            rank,
                            epoch,
                            mi + 1,
                            len(schedule),
                            global_step,
                        )
                        (save_dir / f"progress_rank{rank}.json").write_text(
                            json.dumps(
                                {
                                    "epoch": epoch,
                                    "rank": rank,
                                    "mol_i": mi + 1,
                                    "mol_total": len(schedule),
                                    "global_step": global_step,
                                }
                            )
                            + "\n"
                        )
        else:
            for mi, mol in enumerate(schedule):
                stats, global_step = _train_one_molecule(
                    mol=mol,
                    cfg=cfg,
                    flow=flow,
                    model=model,
                    old_model=old_model,
                    ref_model=ref_model,
                    optimizer=optimizer,
                    scorer=scorer,
                    device=device,
                    epoch=epoch,
                    global_step=global_step,
                )
                epoch_stats.extend(stats)
                if (mi + 1) % 5 == 0 or (mi + 1) == len(schedule):
                    logger.info(
                        "[rank%d] epoch=%d mol %d/%d step=%d",
                        rank,
                        epoch,
                        mi + 1,
                        len(schedule),
                        global_step,
                    )
                    (save_dir / f"progress_rank{rank}.json").write_text(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "rank": rank,
                                "mol_i": mi + 1,
                                "mol_total": len(schedule),
                                "global_step": global_step,
                            }
                        )
                        + "\n"
                    )

        decay = sync_old_policy(
            model, old_model, step=global_step, decay_type=cfg["decay_type"]
        )
        logger.info(
            "Epoch %d rank %d local done | decay=%.4f | syncing weights",
            epoch,
            rank,
            decay,
        )
        logger.info("[rank%d] epoch=%d syncing weights...", rank, epoch)
        _sync_weights_filesystem(
            save_dir,
            epoch,
            rank,
            world_size,
            model,
            old_model,
            timeout_s=sync_timeout_s,
        )
        metrics_log.extend(epoch_stats)

        if rank == 0 and (epoch + 1) % int(cfg["save_every"]) == 0:
            ckpt_path = save_dir / f"checkpoint-epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "old_state_dict": old_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "source_checkpoint": bundle["ckpt_path"],
                },
                ckpt_path,
            )
            logger.info("Saved %s", ckpt_path)
            logger.info("Saved %s", ckpt_path)

        # Barrier via filesystem so ranks stay aligned.
        # Three-phase: done -> ack (observed by all) -> release; only then
        # rank0 deletes. Deleting after ack alone races: slow ranks never see
        # all acks at once and spin until timeout (GPU util drops to 0).
        if world_size > 1:
            done = save_dir / f".epoch{epoch}_rank{rank}.done"
            done.write_text("ok\n")
            t0 = time.time()
            while time.time() - t0 < sync_timeout_s:
                if all(
                    (save_dir / f".epoch{epoch}_rank{r}.done").exists()
                    for r in range(world_size)
                ):
                    break
                time.sleep(1.0)
            ack = save_dir / f".epoch{epoch}_rank{rank}.ack"
            ack.write_text("ok\n")
            t0 = time.time()
            while time.time() - t0 < sync_timeout_s:
                if all(
                    (save_dir / f".epoch{epoch}_rank{r}.ack").exists()
                    for r in range(world_size)
                ):
                    break
                time.sleep(1.0)
            # Each rank records that it has observed the full ack set.
            release = save_dir / f".epoch{epoch}_rank{rank}.release"
            release.write_text("ok\n")
            if rank == 0:
                t0 = time.time()
                while time.time() - t0 < sync_timeout_s:
                    if all(
                        (save_dir / f".epoch{epoch}_rank{r}.release").exists()
                        for r in range(world_size)
                    ):
                        break
                    time.sleep(1.0)
                time.sleep(2.0)
                for r in range(world_size):
                    for suffix in ("done", "ack", "release"):
                        (save_dir / f".epoch{epoch}_rank{r}.{suffix}").unlink(
                            missing_ok=True
                        )

    if rank == 0:
        (save_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2) + "\n")
        logger.info("Finished. Metrics -> %s", save_dir / "metrics.json")
    else:
        logger.info("Rank %d finished", rank)


if __name__ == "__main__":
    main()
