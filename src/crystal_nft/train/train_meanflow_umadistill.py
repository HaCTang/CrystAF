"""UMA distillation for the CrystAF student: relax, then regress.

Distillation, not RL. Each family's generated candidates are pulled downhill on
the UMA potential surface (`meanflow/uma_relax.py`) and the *relaxed* geometry
becomes a supervised flow-map target. There is no reward, no advantage, and no
old/ref policy in the objective -- the student is simply taught to emit what UMA
says the crystal should have been.

This is training-time only: the inference chain is unchanged, which is what
separates it from the `CRYSTAF_RELAX_CLASH` repair chain (that relaxes at
sampling time, and so cannot be reported as a model result).

It targets the columns the PB-rank reward cannot reach. UMA relaxation removes
close contacts (clash), the stress term sizes the cell (Vol.Err), and better
packing shows up in PDD -- none of which appear in a PoseBusters pass rate.

Derived from train_meanflow_nft.py: identical setup, EMA, filesystem weight
averaging and checkpointing; only the per-family inner block differs.
"""

from __future__ import annotations

import argparse
import copy
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

from crystal_nft.adapters.clari_adapter import crystal_n_mols, crystals_to_ase  # noqa: E402
from crystal_nft.meanflow.adapter import (  # noqa: E402
    load_meanflow_bundle,
    load_meanflow_checkpoint,
    meanflow_nft_train_step,
    sample_from_crystal_meanflow,
    save_meanflow_checkpoint,
)
from crystal_nft.nft.loss import sync_old_policy  # noqa: E402
from crystal_nft.rewards.advantages import (  # noqa: E402
    advantage_to_nft_weight,
    compute_group_advantages,
    pb_has_signal,
)
from crystal_nft.rewards.target_alignment import target_alignment_rewards  # noqa: E402
from crystal_nft.meanflow.adapter import meanflow_distill_train_step  # noqa: E402
from crystal_nft.meanflow.uma_relax import atoms_to_state_x, relax_atoms_uma  # noqa: E402
from crystal_nft.rewards.uma_scorer import UMAScorer  # noqa: E402
from crystal_nft.train.train_clari_nft import (  # noqa: E402
    _load_csd_train_dataset,
    _score_candidates,
    _setup_ranks,
    _sync_weights_filesystem,
    _uses_pb_reward,
)

logger = logging.getLogger(__name__)


def _load_config(path: str | None) -> dict:
    defaults = {
        "checkpoint": "clari-h",
        "meanflow_ckpt": None,
        "resume_nft": None,
        "device": "cuda",
        "enable_chiral": True,
        "clari_data_dir": str(_REPO_ROOT / "dataset" / "clari"),
        "require_csd": True,
        "num_epochs": 4,
        "steps_per_epoch": 250,
        "samples_per_mol": 8,
        "train_batch_size": 4,
        "inner_epochs": 1,
        "lr": 3.0e-7,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "beta": 0.1,
        "kl_coef": 0.01,
        # Abort when KL to the frozen ref runs away. `crystaf_mf_big` diverged
        # exponentially (kl 0.005 -> 428 over 16 epochs, PB 92.9 -> 0.07); eval
        # quality held while kl <~ 0.2 and was gone by kl 8.7. With kl_coef 1e-4
        # the KL term contributes ~1e-8 to the loss, so nothing restrains this
        # but lr -- hence a hard stop rather than a penalty. 0 disables.
        "kl_stop": 0.0,
        # --- UMA distillation ---
        "relax_steps": 25,
        "relax_max_disp": 0.05,
        "relax_cell": True,
        "relax_cell_step": 0.05,
        "relax_max_strain": 0.01,
        "relax_lattice_scale": 1.0,
        "distill_batch_size": 8,
        "distill_min_improve": 0.0,
        "meanflow_steps": 16,
        "sampler_mode": "interval",
        "sampler_rho": 0.75,
        "sampler_schedule": "power",
        "sample_batch_size": 0,
        "nft_velocity_mode": "instantaneous",
        "cd_velocity_source": "noise_minus_data",
        # MeanFlowNFT (t, r) recipe. `nft_diffusion_ratio` is the fraction of a
        # batch pinned to r = t (pure instantaneous velocity); the remainder
        # draws adjacent jumps on the `nft_nfe_steps` power grid.
        "nft_diffusion_ratio": 0.5,
        "nft_consistency_ratio": 0.0,
        "nft_nfe_steps": 16,
        "nft_grid_rho": 0.75,
        "nft_cd_eps": 5.0e-3,
        "nft_share_cd_with_old": True,
        "nft_student_tune": None,
        "lora_rank": 32,
        "lora_alpha": 64.0,
        # Off by default: the trust region for the distilled student is the
        # KL to `ref_net` (frozen CrystAF). Set > 0 to also anchor the flow
        # map on the ground-truth family crystal.
        "forward_anchor_weight": 0.0,
        # Generator EMA. BOTH reference implementations keep one and report the
        # EMA weights (DiffusionNFT `train.ema = True`; MeanFlowNFT
        # `use_ema: true, ema_decay: 0.9`). This trainer never did, which
        # matters here: every arm peaks at its first checkpoint and then
        # degrades, and an EMA holds near the peak instead of following the
        # drift.
        "use_ema": True,
        "ema_decay": 0.99,
        "anyflow_use_aux_losses": False,
        "anyflow_aux_vol_weight": 0.0,
        "anyflow_aux_ldd_weight": 0.0,
        "uma_model": "uma-s-1p1",
        "uma_ckpt_path": str(_REPO_ROOT / "checkpoints" / "uma" / "uma-s-1p1.pt"),
        "use_ultrafast": False,
        "ef_lambda": 0.5,
        "advantage_clip": 5.0,
        "adv_mode": "continuous",
        "reward_mode": "uma",
        "top_frac": 0.25,
        "bottom_frac": 0.25,
        "clash_veto_positive": True,
        "skip_flat_pb": True,
        "min_pb_range": 1.0e-6,
        "max_family_tries": 4,
        "drop_neutral_r": False,
        "w_clash": 1.0,
        "w_fmax": 0.0,
        "w_stress": 0.25,
        "w_alignment": 1.0,
        # Dedicated volume advantage (see _score_uma_group). 0 keeps the old
        # behaviour; > 0 is the lever for the Vol.Err column.
        "w_volume": 0.0,
        "w_pb": 1.0,
        "align_pdd_weight": 1.0,
        "align_xrd_weight": 1.0,
        "align_volume_weight": 1.0,
        "uma_batch_size": 8,
        "save_dir": None,
        "save_every": 1,
        "seed": 42,
    }
    if path is None:
        return defaults
    with open(path) as f:
        defaults.update(yaml.safe_load(f) or {})
    return defaults


def _checkpoint_config(path: str | None) -> dict:
    if not path:
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload.get("meta") or {}
    saved = meta.get("cfg") if isinstance(meta.get("cfg"), dict) else {}
    del payload
    return dict(saved)


def _build_forward_anchor(cfg: dict):
    weight = float(cfg.get("forward_anchor_weight", 0.0))
    if weight <= 0:
        return None
    from crystal_nft.meanflow.loss_anyflow import AnyFlowLossConfig, CrystalAnyFlowLoss

    loss_cfg = AnyFlowLossConfig(
        variant="official",
        diffusion_ratio=float(cfg.get("anyflow_diffusion_ratio", 0.5)),
        consistency_ratio=float(cfg.get("anyflow_consistency_ratio", 0.25)),
        weight_type=str(cfg.get("anyflow_weight_type", "beta08")),
        weight_grid_size=int(cfg.get("anyflow_weight_grid_size", 1000)),
        cd_eps=float(cfg.get("anyflow_cd_eps", 0.005)),
        v_target_source="noise_minus_data",
        cd_velocity_source="noise_minus_data",
        # Supervised auxiliary losses on the family's ground-truth crystal.
        #
        # This is the only term that can correct a *systematic* volume offset.
        # NFT's volume advantage is `_group_standardize(-rel_dev)`, which is
        # zero-mean by construction: it ranks a family's candidates against
        # each other but gives no pull toward the true volume when all of them
        # are off in the same direction. `interface._vol_losses(pred_x1, x1)`
        # regresses the predicted endpoint's cell volume onto the real one.
        # Volume error is a Table-1 column and the GT cell is training data, so
        # this is supervision, not metric leakage at inference.
        use_aux_losses=bool(cfg.get("anyflow_use_aux_losses", False)),
        aux_vol_weight=float(cfg.get("anyflow_aux_vol_weight", 0.0)),
        aux_ldd_weight=float(cfg.get("anyflow_aux_ldd_weight", 0.0)),
        chiral_loss_weight=0.0,
    )
    return CrystalAnyFlowLoss(loss_cfg, teacher_net=None)


def pb_is_whole_rank_key(cfg: dict) -> bool:
    """True when PoseBusters pass rate is the *entire* `pb_rank` ranking key.

    `compute_pb_rank_advantages` ranks on ``pb - clash_w * clash - vol_w * vol``.
    Whenever either extra weight is on, a family whose PB is flat can still be
    ranked by that term -- and for clash those are exactly the families that
    carry the clash signal. Only skip flat-PB families when neither is on.
    """
    extra = float(cfg.get("vol_rank_weight", 0.0)) + float(
        cfg.get("clash_rank_weight", 0.0)
    )
    return extra <= 0.0


class _ParamEMA:
    """EMA over the trainable parameters, saved alongside the live weights.

    `save_meanflow_checkpoint` already has an `ema_state` slot and
    `load_meanflow_checkpoint(load_ema=True)` prefers it, so evaluation picks
    the EMA up automatically once it is populated.
    """

    def __init__(self, module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone().float()
            for name, param in module.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, module) -> None:
        d = self.decay
        for name, param in module.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(d).add_(param.detach().float(), alpha=1.0 - d)

    def state_dict(self, module) -> dict:
        """Full state dict with EMA values substituted for the trained params."""
        out = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        for name, shadow in self.shadow.items():
            if name in out:
                out[name] = shadow.detach().cpu().to(out[name].dtype).clone()
        return out


def _train_candidate_group(
    *,
    cfg: dict,
    bundle: dict,
    optimizer,
    candidates,
    weights: np.ndarray,
    template,
    smiles: str | None,
    forward_anchor,
    global_step: int,
) -> tuple[int, list[dict]]:
    net = bundle["net"]
    old_net = bundle["old_net"]
    ref_net = bundle["ref_net"]
    interface = bundle["interface"]
    chiral = bundle["chiral"]
    stats = []
    batch_size = int(cfg["train_batch_size"])
    for _ in range(int(cfg["inner_epochs"])):
        order = np.random.permutation(len(candidates))
        for start in range(0, len(candidates), batch_size):
            index = order[start : start + batch_size]
            if len(index) == 0:
                continue
            batch = [candidates[i] for i in index]
            r = torch.tensor(weights[index], dtype=torch.float32, device=bundle["device"])
            optimizer.zero_grad(set_to_none=True)
            metrics = meanflow_nft_train_step(
                interface,
                net,
                old_net,
                ref_net,
                batch,
                r,
                smiles=smiles,
                chiral=chiral,
                beta=float(cfg["beta"]),
                kl_coef=float(cfg["kl_coef"]),
                adv_clip_max=float(cfg["advantage_clip"]),
                nft_velocity_mode=str(cfg["nft_velocity_mode"]),
                cd_velocity_source=str(cfg["cd_velocity_source"]),
                diffusion_ratio=float(cfg["nft_diffusion_ratio"]),
                consistency_ratio=float(cfg["nft_consistency_ratio"]),
                nfe_steps=int(cfg["nft_nfe_steps"]),
                grid_rho=float(cfg["nft_grid_rho"]),
                cd_eps=float(cfg["nft_cd_eps"]),
                share_cd_with_old=bool(cfg["nft_share_cd_with_old"]),
            )
            loss = metrics["loss"]
            anchor_value = torch.zeros((), device=loss.device)
            if forward_anchor is not None:
                C0, C1 = interface.collate_fn([template for _ in range(len(batch))])
                C0, C1 = C0.to(bundle["device"]), C1.to(bundle["device"])
                anchor = forward_anchor(net, interface, C0, C1)
                anchor_value = anchor["loss"]
                loss = loss + float(cfg["forward_anchor_weight"]) * anchor_value
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.requires_grad],
                float(cfg["max_grad_norm"]),
            )
            optimizer.step()
            global_step += 1
            sync_old_policy(net, old_net, step=global_step, decay_type=1)
            ema = bundle.get("param_ema")
            if ema is not None:
                ema.update(net)
            stats.append(
                {
                    "step": global_step,
                    "loss": float(loss.detach()),
                    "policy_loss": float(metrics["policy_loss"]),
                    "kl_loss": float(metrics["kl_loss"]),
                    "forward_anchor": float(anchor_value.detach()),
                    "positive_loss": float(metrics["positive_loss"]),
                    "negative_loss": float(metrics["negative_loss"]),
                    "old_deviate": float(metrics["old_deviate"]),
                    "mean_r": float(metrics["mean_r"]),
                    "mean_t": float(metrics["mean_t"]),
                    "frac_r_eq_t": float(metrics["frac_r_eq_t"]),
                }
            )
    return global_step, stats


def _distill_targets(
    *, cfg: dict, bundle: dict, optimizer, targets, smiles, global_step: int
) -> tuple[int, list[dict]]:
    """Regress the student onto UMA-relaxed targets. No reward, no old policy."""
    net = bundle["net"]
    interface = bundle["interface"]
    chiral = bundle["chiral"]
    stats: list[dict] = []
    batch_size = int(cfg["distill_batch_size"])
    for _ in range(int(cfg["inner_epochs"])):
        order = np.random.permutation(len(targets))
        for start in range(0, len(targets), batch_size):
            index = order[start : start + batch_size]
            batch = [targets[int(j)] for j in index]
            if len(batch) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            metrics = meanflow_distill_train_step(
                interface,
                net,
                batch,
                smiles=smiles,
                chiral=chiral,
                nfe_steps=int(cfg["nft_nfe_steps"]),
                grid_rho=float(cfg["nft_grid_rho"]),
                diffusion_ratio=float(cfg["nft_diffusion_ratio"]),
                consistency_ratio=float(cfg["nft_consistency_ratio"]),
            )
            loss = metrics["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.requires_grad],
                float(cfg["max_grad_norm"]),
            )
            optimizer.step()
            global_step += 1
            ema = bundle.get("param_ema")
            if ema is not None:
                ema.update(net)
            stats.append(
                {
                    "step": global_step,
                    "loss": float(loss.detach()),
                    "mse": float(metrics["mse"]),
                    "target_norm": float(metrics["target_norm"]),
                    "mean_r": float(metrics["mean_r"]),
                    "mean_t": float(metrics["mean_t"]),
                    # kept so the shared epoch summary in main() works unchanged
                    "policy_loss": float(metrics["mse"]),
                    "kl_loss": 0.0,
                    "old_deviate": 0.0,
                    "frac_r_eq_t": 0.0,
                }
            )
    return global_step, stats


def _load_architecture(cfg: dict, saved_cfg: dict) -> dict:
    architecture = dict(saved_cfg)
    for key in (
        "conditioning_mode",
        "time_parameterization",
        "dual_time_feature_mode",
        "dual_gate_value",
        "gate_value",
        "enable_chiral",
    ):
        if key in cfg:
            architecture[key] = cfg[key]
    return architecture


def _score_uma_group(cfg: dict, candidates, template, family_id: str, scorer):
    candidate_atoms = crystals_to_ase(candidates)
    target_atoms = crystals_to_ase([template])[0]
    n_mol = crystal_n_mols(template)
    scores = scorer.score_atoms(
        candidate_atoms,
        n_mols=[n_mol] * len(candidate_atoms),
    )
    alignment = target_alignment_rewards(
        candidate_atoms,
        target_atoms,
        pdd_weight=float(cfg["align_pdd_weight"]),
        xrd_weight=float(cfg["align_xrd_weight"]),
        volume_weight=float(cfg["align_volume_weight"]),
    )
    # Dedicated lattice-volume advantage, separate from `alignment`. The
    # alignment term buries volume behind two normalised 64-bin histograms
    # (PDD / XRD proxies), whose mean-abs differences are the same ~0.01-0.03
    # scale as the relative volume error -- so volume only gets ~1/3 of that
    # signal. `w_volume` instead group-standardises -|V - V_true|/V_true on its
    # own, which is what drove MolCrystalFlow's volume RMAD 3.88% -> 3.16%,
    # and `compute_group_advantages` deliberately
    # does not gate it on UMA validity because volume error IS a Table-1 column.
    volumes = [float(a.get_volume()) for a in candidate_atoms]
    target_volume = float(target_atoms.get_volume())
    advantages = compute_group_advantages(
        scores,
        [family_id] * len(scores),
        ef_lambda=float(cfg["ef_lambda"]),
        advantage_clip=float(cfg["advantage_clip"]),
        adv_mode=str(cfg["adv_mode"]),
        w_clash=float(cfg["w_clash"]),
        w_fmax=float(cfg["w_fmax"]),
        w_stress=float(cfg["w_stress"]),
        w_alignment=float(cfg["w_alignment"]),
        alignment_rewards=alignment,
        w_volume=float(cfg.get("w_volume", 0.0)),
        volumes=volumes,
        volume_targets=[target_volume] * len(volumes),
    )
    rel_dev = [abs(v - target_volume) / max(target_volume, 1e-8) for v in volumes]
    extra = {
        "mean_alignment": float(np.mean(alignment)),
        "mean_vol_rel_dev": float(np.mean(rel_dev)),
        "best_vol_rel_dev": float(np.min(rel_dev)),
        "valid_fraction": float(np.mean([s.valid for s in scores])),
    }
    return scores, advantages, extra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = _load_config(args.config)
    rank, world_size, local_rank = _setup_ranks()
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [r{rank}] %(levelname)s %(message)s",
        force=True,
    )
    if not cfg.get("meanflow_ckpt"):
        raise SystemExit("meanflow_ckpt is required for family-level MeanFlow NFT")
    if not cfg.get("save_dir"):
        raise SystemExit("save_dir is required")
    if args.smoke:
        cfg["num_epochs"] = 1
        cfg["steps_per_epoch"] = 1
        cfg["samples_per_mol"] = 4
        cfg["train_batch_size"] = 2

    seed = int(cfg["seed"]) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["CLARI_DATA_DIR"] = str(Path(cfg["clari_data_dir"]).resolve())
    # `MeanFlowCrystalSampler` is configured from the environment inside
    # `load_meanflow_bundle`. The rollout must be the interval flow map on the
    # report grid -- Clari's Heun sampler silently drops the second time
    # argument, so NFT would then optimise a policy that is never sampled.
    os.environ["MEANFLOW_SAMPLER_MODE"] = str(cfg["sampler_mode"])
    os.environ["MEANFLOW_INTERVAL_RHO"] = str(cfg["sampler_rho"])
    os.environ["MEANFLOW_INTERVAL_SCHEDULE"] = str(cfg["sampler_schedule"])
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    saved_cfg = _checkpoint_config(cfg["meanflow_ckpt"])
    architecture_cfg = _load_architecture(cfg, saved_cfg)
    bundle = load_meanflow_bundle(
        cfg["checkpoint"],
        device=device,
        meanflow_steps=int(cfg["meanflow_steps"]),
        enable_chiral=bool(architecture_cfg.get("enable_chiral", cfg.get("enable_chiral", True))),
        gate_value=float(architecture_cfg.get("gate_value", 0.25)),
        conditioning_mode=str(architecture_cfg.get("conditioning_mode", "mix")),
        time_parameterization=str(architecture_cfg.get("time_parameterization", "legacy")),
        dual_time_feature_mode=str(architecture_cfg.get("dual_time_feature_mode", "both")),
        dual_gate_value=float(architecture_cfg.get("dual_gate_value", 1.0)),
        keep_nft_copies=False,
    )
    from crystal_nft.meanflow.tune import apply_student_tune

    apply_student_tune(bundle["net"], saved_cfg or {"student_tune": "full"})
    load_meanflow_checkpoint(cfg["meanflow_ckpt"], bundle, load_ema=True)
    nft_tune = dict(saved_cfg)
    nft_tune["student_tune"] = str(
        cfg.get("nft_student_tune") or saved_cfg.get("student_tune", "delta_lora")
    )
    nft_tune["lora_rank"] = int(cfg.get("lora_rank", saved_cfg.get("lora_rank", 32)))
    nft_tune["lora_alpha"] = float(cfg.get("lora_alpha", saved_cfg.get("lora_alpha", 64.0)))
    nft_tune["cond_lora_rank"] = int(
        cfg.get("cond_lora_rank", saved_cfg.get("cond_lora_rank", nft_tune["lora_rank"]))
    )
    nft_tune["cond_lora_alpha"] = float(
        cfg.get("cond_lora_alpha", saved_cfg.get("cond_lora_alpha", nft_tune["lora_alpha"]))
    )
    apply_student_tune(bundle["net"], nft_tune)
    bundle["ref_net"] = copy.deepcopy(bundle["net"]).eval()
    resume_nft = cfg.get("resume_nft")
    if resume_nft:
        resume_cfg = _checkpoint_config(resume_nft)
        apply_student_tune(bundle["net"], resume_cfg or nft_tune)
        load_meanflow_checkpoint(resume_nft, bundle, load_ema=False)
        logger.info("Resumed MeanFlow NFT weights from %s", resume_nft)
    bundle["old_net"] = copy.deepcopy(bundle["net"]).eval()
    for frozen in (bundle["old_net"], bundle["ref_net"]):
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)

    net = bundle["net"]
    trainable = [p for p in net.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters after MeanFlow NFT student_tune")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    bundle["param_ema"] = (
        _ParamEMA(net, float(cfg["ema_decay"])) if bool(cfg["use_ema"]) else None
    )
    if bundle["param_ema"] is not None:
        logger.info(
            "Generator EMA on (decay=%.4f, %d tensors)",
            float(cfg["ema_decay"]), len(bundle["param_ema"].shadow),
        )
    use_pb = _uses_pb_reward(cfg)
    # Distillation always needs UMA -- it *is* the target source, regardless of
    # what reward_mode the inherited config says (which is unused here).
    scorer = None
    if True:
        scorer = UMAScorer(
            model_name=cfg["uma_model"],
            device=str(device),
            checkpoint_path=cfg.get("uma_ckpt_path"),
            use_ultrafast=bool(cfg.get("use_ultrafast", False)),
            batch_size=int(cfg["uma_batch_size"]),
        )
    train_ds = _load_csd_train_dataset(cfg)
    forward_anchor = _build_forward_anchor(cfg)
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (save_dir / "config_resolved.json").write_text(json.dumps(cfg, indent=2) + "\n")

    skip_flat = bool(cfg.get("skip_flat_pb", False)) and use_pb
    max_tries = max(1, int(cfg.get("max_family_tries", 1)))
    min_pb_range = float(cfg.get("min_pb_range", 1e-6))
    drop_neutral = bool(cfg.get("drop_neutral_r", False))
    global_step = 0
    all_stats: list[dict] = []
    for epoch in range(int(cfg["num_epochs"])):
        if (save_dir / "STOP_KL").exists():
            logger.error(
                "STOP_KL present (%s); ending before epoch %d",
                (save_dir / "STOP_KL").read_text().strip(), epoch,
            )
            break
        useful = 0
        attempts = 0
        n_skipped_flat = 0
        target_useful = int(cfg["steps_per_epoch"])
        max_attempts = target_useful * max_tries
        while useful < target_useful and attempts < max_attempts:
            attempts += 1
            family_idx = random.randrange(len(train_ds))
            template = train_ds[family_idx]
            family_id = str(getattr(template, "csd_id", None) or f"fam_{family_idx}")
            smiles = getattr(template, "smiles", None)
            n_mol = crystal_n_mols(template)
            logger.info(
                "epoch=%d useful=%d/%d try=%d family=%s n_mols=%d",
                epoch,
                useful + 1,
                target_useful,
                attempts,
                family_id,
                n_mol,
            )
            candidates, _ = sample_from_crystal_meanflow(
                bundle,
                template,
                samples=int(cfg["samples_per_mol"]),
                smiles=smiles,
                use_old_net=True,
                batch_size=int(cfg["sample_batch_size"]) or None,
            )
            if not candidates:
                logger.warning("No MeanFlow candidates for %s; retrying", family_id)
                continue
            atoms = crystals_to_ase(candidates)
            # Sampling 24 candidates at NFE=32 leaves the allocator holding
            # ~43.9 of 44.5 GiB, so UMA's forward then OOMs mid-relaxation
            # (measured: ~4% of chunk calls, which relax_atoms_uma skips and
            # logs). Releasing cached blocks first costs one sync and removes
            # the contention; the candidates are already on CPU as ASE Atoms.
            torch.cuda.empty_cache()
            relaxed, rstats = relax_atoms_uma(
                scorer,
                atoms,
                steps=int(cfg["relax_steps"]),
                max_disp=float(cfg["relax_max_disp"]),
                relax_cell=bool(cfg["relax_cell"]),
                cell_step=float(cfg["relax_cell_step"]),
                max_strain=float(cfg["relax_max_strain"]),
                lattice_scale=float(cfg["relax_lattice_scale"]),
                batch_size=int(cfg["uma_batch_size"]),
            )
            targets = []
            for cand, relaxed_atoms in zip(candidates, relaxed):
                if relaxed_atoms is None:
                    continue
                try:
                    x_star = atoms_to_state_x(relaxed_atoms).to(cand.x)
                except Exception as exc:
                    logger.warning("  target build failed: %s", exc)
                    continue
                if not bool(torch.isfinite(x_star).all()):
                    continue
                targets.append(cand.replace(x=x_star))
            if len(targets) < 2:
                logger.info("  fewer than 2 usable relaxed targets; skipping family")
                n_skipped_flat += 1
                continue
            global_step, stats = _distill_targets(
                cfg=cfg,
                bundle=bundle,
                optimizer=optimizer,
                targets=targets,
                smiles=smiles,
                global_step=global_step,
            )
            advantages = np.zeros(len(targets), dtype=np.float64)
            extra = {
                "n_targets": len(targets),
                "relax_fmax_before": rstats.fmax_before,
                "relax_fmax_after": rstats.fmax_after,
                "relax_disp": rstats.mean_disp,
                "relax_vol_ratio": rstats.mean_vol_ratio,
            }
            log_extra = {k: v for k, v in extra.items() if k != "pb_used"}
            for stat in stats:
                stat.update(
                    {
                        "epoch": epoch,
                        "rank": rank,
                        "family": family_id,
                        "mean_advantage": float(np.mean(advantages)),
                        **log_extra,
                    }
                )
            all_stats.extend(stats)
            useful += 1
            logger.info(
                "  relaxed %d useful=%d step=%d extra=%s",
                len(targets),
                useful,
                global_step,
                log_extra,
            )

        # The optimiser-side signals (`old_deviate`, `kl_loss`, `policy_loss`)
        # used to reach disk only after the whole run finished, so a multi-hour
        # arm could not be told apart from a no-op until it was over. Summarise
        # them per epoch and flush this rank's stats now.
        ep_stats = [st for st in all_stats if st.get("epoch") == epoch]
        if ep_stats:
            def _mean(key: str) -> float:
                vals = [float(st[key]) for st in ep_stats if key in st]
                return sum(vals) / len(vals) if vals else float("nan")

            # Rollout columns must be de-duplicated by family: the scoring
            # metrics are copied onto every gradient step of that family, so a
            # plain mean weights families by how many steps they produced.
            # Families are also freshly drawn each epoch (no repeats), and
            # per-family mean_pb spans 0.02-0.99, so read these only as a trend
            # over several epochs -- `pb_valid` and `hard_frac` say how the
            # draw itself differed.
            by_family = {}
            for st in ep_stats:
                by_family.setdefault(st.get("family"), st)
            fams = list(by_family.values())

            def _fmean(key: str, default: float = float("nan")) -> float:
                vals = [float(f[key]) for f in fams if key in f]
                return sum(vals) / len(vals) if vals else default

            logger.info(
                "Epoch %d steps=%d fams=%d loss=%.5f policy=%.5f kl=%.5f "
                "old_deviate=%.3e rollout_pb=%.4f rollout_clash=%.4f "
                "pb_valid=%.4f hard_frac=%.3f",
                epoch,
                len(ep_stats),
                len(fams),
                _mean("loss"),
                _mean("policy_loss"),
                _mean("kl_loss"),
                _mean("old_deviate"),
                _fmean("mean_pb"),
                _fmean("mean_clash"),
                _fmean("pb_valid_frac"),
                (sum(float(f.get("mean_pb", 1.0)) < 0.1 for f in fams) / len(fams))
                if fams
                else float("nan"),
            )
        (save_dir / f"metrics_rank{rank}.partial.json").write_text(
            json.dumps(all_stats) + "\n"
        )
        kl_stop = float(cfg.get("kl_stop", 0.0) or 0.0)
        if kl_stop > 0.0 and ep_stats:
            ep_kl = sum(float(st["kl_loss"]) for st in ep_stats) / len(ep_stats)
            if ep_kl > kl_stop:
                logger.error(
                    "Epoch %d: kl=%.5f exceeded kl_stop=%.5f -- policy is running "
                    "away; stopping before the checkpoint degrades further",
                    epoch, ep_kl, kl_stop,
                )
                # A rank-local flag deadlocks: ranks trip on their own kl, so
                # they break at different epochs and `_sync_weights_filesystem`
                # waits forever on files the departed ranks never write (this
                # hung 2 of crystaf_mf_rerun's 4 ranks for 1.7h). Publish a
                # sentinel instead and let every rank act on it at the next
                # epoch boundary, after this epoch's sync and save complete.
                (save_dir / "STOP_KL").write_text(
                    f"rank={rank} epoch={epoch} kl={ep_kl:.6f}\n"
                )
        logger.info(
            "Epoch %d useful=%d skipped_flat=%d attempts=%d",
            epoch,
            useful,
            n_skipped_flat,
            attempts,
        )
        _sync_weights_filesystem(
            save_dir,
            epoch,
            rank,
            world_size,
            net,
            bundle["old_net"],
        )
        if rank == 0 and (epoch + 1) % int(cfg["save_every"]) == 0:
            save_meanflow_checkpoint(
                save_dir / f"checkpoint-nft-epoch{epoch + 1}.pt",
                net=net,
                chiral=bundle["chiral"],
                ema_state=(
                    bundle["param_ema"].state_dict(net)
                    if bundle.get("param_ema") is not None
                    else None
                ),
                meta={
                    "cfg": {**saved_cfg, **cfg, "student_tune": nft_tune["student_tune"]},
                    "steps": global_step,
                    "stage": "meanflow_nft_csd",
                },
            )

    metrics_path = save_dir / f"metrics_rank{rank}.json"
    metrics_path.write_text(json.dumps(all_stats, indent=2) + "\n")
    if rank == 0:
        merged = []
        for worker in range(world_size):
            path = save_dir / f"metrics_rank{worker}.json"
            for _ in range(7200):
                if path.exists():
                    break
                time.sleep(1.0)
            if path.exists():
                merged.extend(json.loads(path.read_text()))
        (save_dir / "metrics.json").write_text(json.dumps(merged, indent=2) + "\n")
    logger.info("MeanFlow CSD family NFT complete at step=%d", global_step)


if __name__ == "__main__":
    main()
