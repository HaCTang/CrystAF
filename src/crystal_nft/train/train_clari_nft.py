"""Online DiffusionNFT fine-tuning for Clari (single-GPU or multi-GPU shard).

Multi-GPU uses torchrun env vars for per-rank family steps + filesystem weight
averaging. NCCL/DDP is intentionally avoided (NCCL barriers hang on this host).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from crystal_nft.adapters.clari_adapter import (  # noqa: E402
    apply_clari_student_tune,
    crystal_n_mols,
    crystals_to_ase,
    load_clari_bundle,
    merged_clari_state_dict,
    nft_train_step,
    resume_clari_nft_weights,
    sample_candidates,
    sample_from_crystal,
)
from crystal_nft.nft.loss import sync_old_policy  # noqa: E402
from crystal_nft.rewards.advantages import (  # noqa: E402
    advantage_to_nft_weight,
    compute_group_advantages,
    compute_pb_rank_advantages,
    pb_has_signal,
)
from crystal_nft.rewards.posebusters import score_crystals_pb, scores_from_pb_rows  # noqa: E402
from crystal_nft.rewards.uma_scorer import UMAScorer  # noqa: E402

logger = logging.getLogger(__name__)

CSD_TRAIN_SPLIT_OPTS = {
    "train": dict(group_by_fam=True, random_repr=True),
    "val": dict(group_by_fam=True, random_repr=True),
    "predict": dict(group_by_fam=True, random_repr=False, augment=False),
    "test": dict(group_by_fam=False, augment=False),
}


def _load_config(path: str | None) -> dict:
    defaults = {
        "checkpoint": "clari-h",
        "device": "cuda",
        "use_ema": True,
        "n_steps": 50,
        "molecules": [{"id": "ethanol", "smiles": "CCO", "copies": 4}],
        "num_epochs": 2,
        "steps_per_epoch": 1000,
        "data_mode": "molecules",
        "use_csd_train": False,
        "require_csd": False,
        "clari_data_dir": str(_REPO_ROOT / "dataset" / "clari"),
        "samples_per_mol": 8,
        "inner_epochs": 1,
        "train_batch_size": 4,
        "lr": 3.0e-7,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "beta": 0.1,
        "kl_coef": 0.01,
        "decay_type": 1,
        "ef_lambda": 0.5,
        "advantage_clip": 5.0,
        "adv_mode": "continuous",
        "reward_mode": "uma",
        "student_tune": "full",
        "lora_rank": 32,
        "lora_alpha": 64.0,
        "w_e": 1.0,
        "w_f": 1.0,
        "w_clash": 1.0,
        "w_fmax": 0.0,
        "w_stress": 0.0,
        "w_pb": 0.0,
        "top_frac": 0.25,
        "bottom_frac": 0.25,
        "clash_veto_positive": True,
        "skip_flat_pb": True,
        "min_pb_range": 1.0e-6,
        "max_family_tries": 4,
        "drop_neutral_r": False,
        "uma_model": "uma-s-1p1",
        "uma_ckpt_path": str(_REPO_ROOT / "checkpoints" / "uma" / "uma-s-1p1.pt"),
        "uma_batch_size": 8,
        "use_ultrafast": False,
        "filter_clashing": False,
        "save_dir": "logs/nft/clari",
        "save_every": 1,
        "save_full_checkpoint": False,
        "resume_nft": None,
        "seed": 0,
    }
    if path is None:
        return defaults
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    defaults.update(user)
    return defaults


def _use_csd_train(cfg: dict) -> bool:
    return cfg.get("data_mode") == "csd_train" or bool(cfg.get("use_csd_train"))


def _csd_data_ready(clari_data_dir: str | Path) -> bool:
    root = Path(clari_data_dir) / "csd"
    required = ["config.json", "train.pt", "val.pt", "test.pt"]
    return all((root / name).is_file() for name in required)


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
    timeout_s: float = 7200.0,
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
        for r in range(world_size):
            for p in (
                save_dir / f".shard_rank{r}_epoch{epoch}.pt",
                save_dir / f".shard_rank{r}_epoch{epoch}.done",
                save_dir / f".loaded_rank{r}_epoch{epoch}.done",
            ):
                p.unlink(missing_ok=True)
        (save_dir / f".avg_epoch{epoch}.done").unlink(missing_ok=True)
        avg_path.unlink(missing_ok=True)


def _load_csd_train_dataset(cfg: dict):
    clari_data_dir = Path(cfg["clari_data_dir"]).resolve()
    os.environ["CLARI_DATA_DIR"] = str(clari_data_dir)
    from clari.datamodules.csd import CrystalDataModule

    dm = CrystalDataModule(
        seed=int(cfg["seed"]),
        batch_size=1,
        num_workers=0,
        split_opts=CSD_TRAIN_SPLIT_OPTS,
    )
    return dm.datasets["train"]


def _trainable_params(module):
    return [p for p in module.parameters() if p.requires_grad]


def _uses_pb_reward(cfg: dict) -> bool:
    return str(cfg.get("reward_mode", "uma")).lower().strip() == "pb"


def _score_candidates(cfg, candidates, template, mol_id, scorer, n_mols):
    extra = {}
    if _uses_pb_reward(cfg):
        rows = score_crystals_pb(candidates, true=template)
        scores = scores_from_pb_rows(rows)
        pb_used = [float(row["pb_used"]) for row in rows]
        clash = [float(row["clash_rate"]) for row in rows]
        vol = [float(row.get("volume_error", float("nan"))) for row in rows]
        if str(cfg.get("adv_mode", "continuous")) == "pb_rank":
            advantages = compute_pb_rank_advantages(
                pb_used,
                clash=clash,
                clash_weight=float(cfg.get("clash_rank_weight", 0.0)),
                volume_error=vol,
                volume_weight=float(cfg.get("vol_rank_weight", 0.0)),
                top_frac=float(cfg.get("top_frac", 0.25)),
                bottom_frac=float(cfg.get("bottom_frac", 0.25)),
                clash_veto_positive=bool(cfg.get("clash_veto_positive", True)),
                min_range=float(cfg.get("min_pb_range", 1e-6)),
            )
        else:
            advantages = compute_group_advantages(
                scores,
                [mol_id] * len(scores),
                ef_lambda=0.0,
                advantage_clip=cfg["advantage_clip"],
                adv_mode=cfg["adv_mode"],
                w_e=0.0,
                w_f=0.0,
                w_fmax=0.0,
                w_stress=0.0,
                w_clash=cfg["w_clash"],
                w_pb=float(cfg.get("w_pb", 1.0)),
                pb_rewards=pb_used,
                # Volume too: `score_crystal_pb` already returns `volume_error`
                # against the reference cell, so ranking on PB + clash + volume
                # costs nothing extra and targets three Table-1 columns at once.
                # Previously only PB and clash reached the advantage, which is
                # why volume drifted the wrong way (2.09 -> 2.58, 2.9 sigma)
                # while PB stayed flat.
                w_volume=float(cfg.get("w_volume", 0.0)),
                volumes=[-v if v == v else float("nan") for v in vol],
                volume_targets=[0.0] * len(vol),
                top_frac=float(cfg.get("top_frac", 0.2)),
                bottom_frac=float(cfg.get("bottom_frac", 0.3)),
            )
        finite_vol = [v for v in vol if v == v]
        extra = {
            "mean_pb": float(np.mean(pb_used)),
            "pb_std": float(np.std(pb_used)),
            "mean_vol_err": float(np.mean(finite_vol)) if finite_vol else float("nan"),
            "pb_valid_frac": float(np.mean([row["pb_valid"] for row in rows])),
            "mean_clash": float(np.mean(clash)),
            "n_pos": int(np.sum(advantages > 0)),
            "n_neg": int(np.sum(advantages < 0)),
            "pb_used": pb_used,
        }
        return scores, advantages, extra

    atoms_list = crystals_to_ase(candidates)
    scores = scorer.score_atoms(atoms_list, n_mols=[n_mols] * len(atoms_list))
    advantages = compute_group_advantages(
        scores,
        [mol_id] * len(scores),
        ef_lambda=cfg["ef_lambda"],
        advantage_clip=cfg["advantage_clip"],
        adv_mode=cfg["adv_mode"],
        w_e=float(cfg.get("w_e", 1.0)),
        w_f=float(cfg.get("w_f", 1.0)),
        w_clash=cfg["w_clash"],
        w_fmax=cfg["w_fmax"],
        w_stress=cfg["w_stress"],
        w_pb=float(cfg.get("w_pb", 0.0)),
    )
    extra = {
        "mean_energy_per_mol": float(
            np.mean([s.energy_per_mol for s in scores if s.valid])
        )
        if any(s.valid for s in scores)
        else None,
        "n_valid": int(sum(1 for s in scores if s.valid)),
    }
    return scores, advantages, extra


def _nft_update_batch(
    *,
    lit,
    train_module,
    old_net,
    ref_net,
    optimizer,
    candidates,
    r,
    device,
    cfg,
    global_step,
    mol_id,
    epoch,
    rank,
    epoch_stats,
):
    train_module.train()
    lit.net = train_module
    bs = int(cfg["train_batch_size"])
    for _inner in range(int(cfg["inner_epochs"])):
        perm = np.random.permutation(len(candidates))
        for start in range(0, len(candidates), bs):
            sel = perm[start : start + bs]
            if len(sel) == 0:
                continue
            batch_crystals = [candidates[i] for i in sel]
            batch_r = torch.tensor(r[sel], dtype=torch.float32, device=device)
            loss_dict = nft_train_step(
                lit,
                train_module,
                old_net,
                ref_net,
                batch_crystals,
                batch_r,
                beta=cfg["beta"],
                kl_coef=cfg["kl_coef"],
                adv_clip_max=cfg["advantage_clip"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                _trainable_params(train_module), cfg["max_grad_norm"]
            )
            optimizer.step()
            global_step += 1
            stat = {
                "epoch": epoch,
                "mol": mol_id,
                "rank": rank,
                "step": global_step,
                "loss": float(loss_dict["loss"].detach()),
                "policy_loss": float(loss_dict["policy_loss"]),
                "kl_loss": float(loss_dict["kl_loss"]),
            }
            epoch_stats.append(stat)
            logger.info(
                "  step %d loss=%.4f policy=%.4f kl=%.4f",
                global_step,
                stat["loss"],
                stat["policy_loss"],
                stat["kl_loss"],
            )
    return global_step


def _run_csd_epoch(
    *,
    cfg,
    lit,
    train_module,
    old_net,
    ref_net,
    optimizer,
    scorer,
    train_ds,
    device,
    epoch,
    rank,
    global_step,
):
    steps = int(cfg["steps_per_epoch"])
    K = int(cfg["samples_per_mol"])
    epoch_stats = []
    n_families = len(train_ds)
    skip_flat = bool(cfg.get("skip_flat_pb", False)) and _uses_pb_reward(cfg)
    max_tries = max(1, int(cfg.get("max_family_tries", 1)))
    min_pb_range = float(cfg.get("min_pb_range", 1e-6))
    drop_neutral = bool(cfg.get("drop_neutral_r", False))
    n_skipped_flat = 0
    useful = 0
    attempts = 0
    max_attempts = steps * max_tries

    while useful < steps and attempts < max_attempts:
        attempts += 1
        fam_idx = random.randrange(n_families)
        template = train_ds[fam_idx]
        mol_id = getattr(template, "csd_id", None) or f"fam_{fam_idx}"
        n_mols = crystal_n_mols(template)

        logger.info(
            "Epoch %d useful %d/%d try %d | family %s n_mols=%d K=%d",
            epoch,
            useful + 1,
            steps,
            attempts,
            mol_id,
            n_mols,
            K,
        )
        candidates, _template = sample_from_crystal(
            lit,
            template,
            samples=K,
            filter_clashing=cfg["filter_clashing"],
            use_old_net=True,
            old_net=old_net,
            pbar=False,
        )
        if len(candidates) == 0:
            logger.warning("No candidates for %s; retrying", mol_id)
            continue

        scores, advantages, extra = _score_candidates(
            cfg, candidates, template, mol_id, scorer, n_mols
        )
        if skip_flat and not pb_has_signal(
            extra.get("pb_used", []), min_range=min_pb_range
        ):
            n_skipped_flat += 1
            logger.info("  flat PB (std=%.4f); skipping family", extra.get("pb_std", 0.0))
            continue
        r = advantage_to_nft_weight(advantages, adv_clip_max=cfg["advantage_clip"])
        if drop_neutral:
            keep = np.abs(r - 0.5) > 1e-6
            if int(keep.sum()) < 2:
                n_skipped_flat += 1
                logger.info("  fewer than 2 non-neutral ranks; skipping family")
                continue
            candidates = [c for c, flag in zip(candidates, keep) if flag]
            r = r[keep]
            advantages = advantages[keep]
        log_extra = {k: v for k, v in extra.items() if k != "pb_used"}
        logger.info(
            "  scored %d | mean adv=%.4f | extra=%s",
            len(scores),
            float(np.mean(advantages)),
            log_extra,
        )

        step_before = global_step
        global_step = _nft_update_batch(
            lit=lit,
            train_module=train_module,
            old_net=old_net,
            ref_net=ref_net,
            optimizer=optimizer,
            candidates=candidates,
            r=r,
            device=device,
            cfg=cfg,
            global_step=global_step,
            mol_id=mol_id,
            epoch=epoch,
            rank=rank,
            epoch_stats=epoch_stats,
        )
        log_extra = {k: v for k, v in extra.items() if k != "pb_used"}
        for stat in epoch_stats:
            if stat.get("step", 0) > step_before and stat.get("mol") == mol_id:
                stat.update(log_extra)
        useful += 1

    if rank == 0 or n_skipped_flat:
        logger.info(
            "Epoch %d useful=%d skipped_flat=%d attempts=%d",
            epoch,
            useful,
            n_skipped_flat,
            attempts,
        )
    return global_step, epoch_stats


def _run_molecules_epoch(
    *,
    cfg,
    lit,
    train_module,
    old_net,
    ref_net,
    optimizer,
    scorer,
    my_mols,
    device,
    epoch,
    rank,
    global_step,
):
    epoch_stats = []
    K = int(cfg["samples_per_mol"])

    for mol in my_mols:
        mol_id = mol.get("id", mol["smiles"])
        smiles = mol["smiles"]
        copies = int(mol.get("copies", 4))

        logger.info("Epoch %d | sampling %s K=%d", epoch, mol_id, K)
        candidates, _template = sample_candidates(
            lit,
            smiles,
            copies=copies,
            samples=K,
            mol_id=mol_id,
            filter_clashing=cfg["filter_clashing"],
            use_old_net=True,
            old_net=old_net,
            pbar=False,
        )
        if len(candidates) == 0:
            logger.warning("No candidates for %s; skipping", mol_id)
            continue

        scores, advantages, extra = _score_candidates(
            cfg, candidates, _template, mol_id, scorer, copies
        )
        r = advantage_to_nft_weight(advantages, adv_clip_max=cfg["advantage_clip"])
        logger.info(
            "  scored %d | mean adv=%.4f | extra=%s",
            len(scores),
            float(np.mean(advantages)),
            extra,
        )

        step_before = global_step
        global_step = _nft_update_batch(
            lit=lit,
            train_module=train_module,
            old_net=old_net,
            ref_net=ref_net,
            optimizer=optimizer,
            candidates=candidates,
            r=r,
            device=device,
            cfg=cfg,
            global_step=global_step,
            mol_id=mol_id,
            epoch=epoch,
            rank=rank,
            epoch_stats=epoch_stats,
        )
        for stat in epoch_stats:
            if stat.get("step", 0) > step_before and stat.get("mol") == mol_id:
                stat.update(extra)

    return global_step, epoch_stats


def parse_args():
    p = argparse.ArgumentParser(description="Clari DiffusionNFT fine-tuning")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--smoke", action="store_true", help="Tiny run for sanity check")
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = _setup_ranks()
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [r{rank}] %(levelname)s %(message)s",
        force=True,
    )

    cfg = _load_config(args.config)
    want_csd = _use_csd_train(cfg)
    csd_ready = _csd_data_ready(cfg["clari_data_dir"]) if want_csd else False

    if args.smoke:
        cfg["num_epochs"] = 1
        cfg["samples_per_mol"] = 4
        cfg["train_batch_size"] = 2
        cfg["inner_epochs"] = 1
        if want_csd and csd_ready:
            cfg["steps_per_epoch"] = 2
        else:
            cfg["data_mode"] = "molecules"
            cfg["use_csd_train"] = False
            cfg["molecules"] = [{"id": "ethanol", "smiles": "CCO", "copies": 4}]

    use_csd = _use_csd_train(cfg) and csd_ready
    if _use_csd_train(cfg) and not csd_ready:
        msg = (
            f"CSD train requested but {cfg['clari_data_dir']}/csd is incomplete "
            "(need config.json, train.pt, val.pt, test.pt). "
            "See scripts/check_clari_csd_status.py --scan-parts."
        )
        if bool(cfg.get("require_csd", False)) and not args.smoke:
            raise SystemExit(msg)
        if rank == 0:
            logger.warning("%s Falling back to molecules list.", msg)
        use_csd = False

    torch.manual_seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    random.seed(cfg["seed"] + rank)

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (save_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    logger.info("Loading Clari on %s (world=%d)", device, world_size)
    bundle = load_clari_bundle(
        cfg["checkpoint"],
        device=device,
        use_ema=cfg["use_ema"],
        n_steps=cfg["n_steps"],
    )
    tune_stats = apply_clari_student_tune(bundle, cfg)
    resume_path = cfg.get("resume_nft")
    if resume_path:
        missing, unexpected = resume_clari_nft_weights(bundle, resume_path)
        logger.info(
            "Resumed NFT weights from %s (missing=%d unexpected=%d)",
            resume_path,
            len(missing),
            len(unexpected),
        )
    lit, net, old_net, ref_net = (
        bundle["lit"],
        bundle["net"],
        bundle["old_net"],
        bundle["ref_net"],
    )
    train_module = net
    if rank == 0:
        logger.info("Student tune stats: %s", tune_stats)

    trainable = _trainable_params(train_module)
    if not trainable:
        raise SystemExit("No trainable parameters after student_tune")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scorer = None
    if not _uses_pb_reward(cfg):
        scorer = UMAScorer(
            cfg["uma_model"],
            device=str(device),
            use_ultrafast=cfg["use_ultrafast"],
            batch_size=cfg["uma_batch_size"],
            checkpoint_path=cfg.get("uma_ckpt_path"),
        )

    train_ds = None
    my_mols = []
    if use_csd:
        train_ds = _load_csd_train_dataset(cfg)
        if rank == 0:
            logger.info(
                "NFT CSD train | world=%d epochs=%d steps/epoch=%d families=%d K=%d save=%s",
                world_size,
                cfg["num_epochs"],
                cfg["steps_per_epoch"],
                len(train_ds),
                cfg["samples_per_mol"],
                save_dir,
            )
    else:
        molecules = list(cfg["molecules"])
        my_mols = [m for i, m in enumerate(molecules) if i % world_size == rank]
        if rank == 0:
            logger.info(
                "NFT molecules | world=%d epochs=%d mols=%d K=%d save=%s",
                world_size,
                cfg["num_epochs"],
                len(molecules),
                cfg["samples_per_mol"],
                save_dir,
            )
        logger.info(
            "Local molecules (%d): %s",
            len(my_mols),
            [m.get("id", m["smiles"]) for m in my_mols],
        )

    global_step = 0
    metrics_log = []

    for epoch in range(cfg["num_epochs"]):
        if use_csd:
            global_step, epoch_stats = _run_csd_epoch(
                cfg=cfg,
                lit=lit,
                train_module=train_module,
                old_net=old_net,
                ref_net=ref_net,
                optimizer=optimizer,
                scorer=scorer,
                train_ds=train_ds,
                device=device,
                epoch=epoch,
                rank=rank,
                global_step=global_step,
            )
        else:
            global_step, epoch_stats = _run_molecules_epoch(
                cfg=cfg,
                lit=lit,
                train_module=train_module,
                old_net=old_net,
                ref_net=ref_net,
                optimizer=optimizer,
                scorer=scorer,
                my_mols=my_mols,
                device=device,
                epoch=epoch,
                rank=rank,
                global_step=global_step,
            )

        sync_old_policy(
            train_module, old_net, step=max(global_step, 1), decay_type=cfg["decay_type"]
        )
        metrics_log.extend(epoch_stats)

        logger.info(
            "Epoch %d local done; syncing weights across %d ranks", epoch, world_size
        )
        _sync_weights_filesystem(
            save_dir, epoch, rank, world_size, train_module, old_net
        )

        if rank == 0 and (epoch + 1) % int(cfg["save_every"]) == 0:
            ckpt_path = save_dir / f"checkpoint-epoch{epoch}.pt"
            payload = {
                "epoch": epoch,
                "global_step": global_step,
                "net_state_dict": {
                    k: v.detach().cpu() for k, v in train_module.state_dict().items()
                },
                "config": cfg,
                "source_checkpoint": bundle["checkpoint"],
            }
            if cfg.get("save_full_checkpoint", False):
                payload["old_state_dict"] = {
                    k: v.detach().cpu() for k, v in old_net.state_dict().items()
                }
                payload["optimizer"] = optimizer.state_dict()
            torch.save(payload, ckpt_path)
            logger.info("Saved %s", ckpt_path)
            if str(cfg.get("student_tune", "full")).lower().strip() == "attn_lora":
                eval_path = save_dir / f"checkpoint-epoch{epoch}-eval.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "net_state_dict": merged_clari_state_dict(train_module),
                        "config": {**cfg, "student_tune": "full"},
                        "source_checkpoint": bundle["checkpoint"],
                        "merged_lora": True,
                    },
                    eval_path,
                )
                logger.info("Saved merged eval checkpoint %s", eval_path)

    shard_metrics = save_dir / f"metrics_rank{rank}.json"
    shard_metrics.write_text(json.dumps(metrics_log, indent=2) + "\n")
    if rank == 0:
        merged = []
        for r in range(world_size):
            p = save_dir / f"metrics_rank{r}.json"
            for _ in range(120):
                if p.exists():
                    break
                time.sleep(1)
            if p.exists():
                merged.extend(json.loads(p.read_text()))
        (save_dir / "metrics.json").write_text(json.dumps(merged, indent=2) + "\n")
        logger.info("Training finished. Metrics -> %s", save_dir / "metrics.json")


if __name__ == "__main__":
    main()
