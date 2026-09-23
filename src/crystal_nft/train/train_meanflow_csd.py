"""Multi-GPU CSD MeanFlow pretraining (DDP, high throughput)."""

from __future__ import annotations

import argparse
import copy
import subprocess
import json
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from crystal_nft.meanflow.adapter import (  # noqa: E402
    enantiomer_augment_x1,
    load_meanflow_bundle,
    make_chiral_bias_for_batch,
    save_meanflow_checkpoint,
)
from crystal_nft.meanflow.chirality import enantiomer_consistency_loss  # noqa: E402
from crystal_nft.meanflow.stereo import (  # noqa: E402
    active_stereo_tags,
    build_batch_stereo,
    build_stereo_conditioning,
    chiral_hinge_loss,
    mirror_coords_in_place_x,
    stereo_agreement,
    swap_rs_tags,
)
from crystal_nft.meanflow.combined_data import build_train_dataset  # noqa: E402
from crystal_nft.meanflow.loss import CrystalMeanFlowLoss, MeanFlowLossConfig  # noqa: E402
from crystal_nft.meanflow.loss_anyflow import (  # noqa: E402
    AnyFlowLossConfig,
    CrystalAnyFlowLoss,
    nfe_grid_choices,
    parse_nfe_steps_list,
)
from crystal_nft.meanflow.loss_imf import CrystalIMFLoss, IMFLossConfig  # noqa: E402
from crystal_nft.meanflow.loss_anyflow_onpolicy import (  # noqa: E402
    AnyFlowOnPolicyConfig,
    CrystalAnyFlowOnPolicyLoss,
)
from crystal_nft.meanflow.loss_sample_align import (  # noqa: E402
    CrystalSampleAlignLoss,
    SampleAlignLossConfig,
)
from crystal_nft.meanflow.sampler import (  # noqa: E402
    allreduce_grads,
    heun_instantaneous_rollout,
)
from crystal_nft.meanflow.net import (  # noqa: E402
    get_time_proxy,
    is_time_mix_buffer_key,
    set_time_mix,
    time_mix_from_cfg,
)
from crystal_nft.meanflow.tune import set_nfe8_lora_mix  # noqa: E402
from crystal_nft.meanflow.chirality import enantiomer_consistency_loss  # noqa: E402
from crystal_nft.meanflow.stereo import (  # noqa: E402
    active_stereo_tags,
    build_batch_stereo,
    build_stereo_conditioning,
    chiral_hinge_loss,
    mirror_coords_in_place_x,
    stereo_agreement,
    swap_rs_tags,
)
from crystal_nft.meanflow.combined_data import build_train_dataset  # noqa: E402
from crystal_nft.meanflow.loss import CrystalMeanFlowLoss, MeanFlowLossConfig  # noqa: E402
from crystal_nft.meanflow.loss_anyflow import (  # noqa: E402
    AnyFlowLossConfig,
    CrystalAnyFlowLoss,
    nfe_grid_choices,
    parse_nfe_steps_list,
)
from crystal_nft.meanflow.loss_imf import CrystalIMFLoss, IMFLossConfig  # noqa: E402
from crystal_nft.meanflow.loss_anyflow_onpolicy import (  # noqa: E402
    AnyFlowOnPolicyConfig,
    CrystalAnyFlowOnPolicyLoss,
)
from crystal_nft.meanflow.loss_sample_align import (  # noqa: E402
    CrystalSampleAlignLoss,
    SampleAlignLossConfig,
)
from torch.utils.data import DataLoader  # noqa: E402

logger = logging.getLogger(__name__)

# Clari-M Table1 reference (cityblock AMD); see result/clari/table1/clari-m/metrics.json
CLARI_M_BASELINE_AMD = 8.57
CLARI_M_BASELINE_PB = 88.43

CSD_TRAIN_SPLIT_OPTS = {
    "train": dict(group_by_fam=True, random_repr=True),
    "val": dict(group_by_fam=True, random_repr=True),
    "predict": dict(group_by_fam=True, random_repr=False, augment=False),
    "test": dict(group_by_fam=False, augment=False),
}


def _load_config(path: str | None) -> dict:
    defaults = {
        "checkpoint": str(_REPO_ROOT / "checkpoints" / "clari-med.ckpt"),
        "clari_nft_init": None,
        "use_ema_init": True,
        "meanflow_steps": 4,
        "enable_chiral": False,
        "enantiomer_flip_p": 0.0,
        "chiral_loss_weight": 0.1,
        # CrystAF stereochemistry: per-atom CIP conditioning + chiral-volume hinge.
        "enable_stereo": False,
        "stereo_loss_weight": 0.0,
        "stereo_margin": 0.62,
        "stereo_lr_mult": 1.0,
        "stereo_mirror_p": 0.5,
        "train_index_json": "",
        "rigid_target_cache": "",
        "stereo_pair_edges": True,
        "stereo_hinge_r_max": 1.0,
        "stereo_hinge_gamma": 0.0,
        "stereo_hinge_t_max": 1.0,
        "stereo_hinge_soft": True,
        "stereo_gain": 1.0,
        "anyflow_v_data_r_max": 0.0,
        "stereo_cond_path": False,
        "stereo_node_mod": False,
        "stereo_chiral_branch": False,
        "stereo_chiral_branch_scale": 1.0,
        "stereo_chiral_branch_gate": 0.0,
        "stereo_chiral_branch_endpoint_geom": False,
        "stereo_chiral_branch_bond_preserving": False,
        "stereo_chiral_branch_t_max": 1.0,
        "stereo_chiral_branch_t_min": 0.0,
        "stereo_head_t_min": 0.0,
        "stereo_head_t_max": 1.0,
        "stereo_low_t_frac": 0.0,
        "stereo_low_t_max": 0.25,
        "stereo_mismatch_frac": 0.0,
        "stereo_mismatch_t_lo": 0.35,
        "stereo_mismatch_t_hi": 0.85,
        "stereo_flow_map_hinge": False,
        "stereo_self_state_frac": 0.0,
        "stereo_self_state_steps": 4,
        "stereo_self_state_denom_min": 0.2,
        "stereo_self_state_hinge_only": False,
        "stereo_branch_l2_weight": 0.0,
        "clari_data_dir": str(_REPO_ROOT / "dataset" / "clari"),
        "require_csd": True,
        "batch_size": 16,
        "dit_preset": None,
        "init_dit_from_checkpoint": True,
        "num_workers": 16,
        "prefetch_factor": 4,
        "grad_accum": 1,
        "num_epochs": 150,
        "steps_per_epoch": 1000,
        "max_steps": 0,
        "mcf_cache": str(_REPO_ROOT / "dataset" / "meanflow" / "mcf_train_crystals.pt"),
        "mcf_pkl": str(
            _REPO_ROOT
            / "dataset/molcrystalflow/thurlemann23/preprocessed/normalized/train_molcrystal_normalized.pkl.gz"
        ),
        "mcf_extxyz": None,
        "mcf_sample_prob": 0.25,
        "resume_from": None,
        "reset_step_on_resume": False,
        "use_mcf": True,
        "save_every_epoch": 1,
        "lr": 4.0e-5,
        "weight_decay": 0.0,
        "lr_warmup_steps": 2000,
        "max_grad_norm": 1.0,
        "fm_warmup_steps": 800,
        "ratio_r_not_equal_t": 0.75,
        "aux_vol_weight": 1.0,
        "aux_ldd_weight": 5.0,
        "use_aux_losses": True,
        "ema_decay": 0.9999,
        "log_every": 50,
        "save_every_steps": 0,
        "save_every_n_epochs": 20,
        "eval_every_n_epochs": 0,
        "eval_at_epochs": [],
        "eval_after_train": True,
        "keep_last_n_step_ckpts": 0,
        "keep_last_n_epoch_ckpts": 5,
        "eval_table1_nproc": 4,
        "eval_table1_pack_size": 4,
        "eval_table1_max_crystals": None,
        "eval_table1_samples": 20,
        "eval_mcf_max_structures": 10,
        "eval_mcf_samples": 1,
        "baseline_eval_every_steps": 0,
        "loss_type": "meanflow",
        "imf_data_proportion": 0.5,
        "imf_cd_velocity_source": "u_self",
        "imf_loss_v_weight": 1.0,
        "anyflow_data_proportion": 0.5,
        "anyflow_variant": "legacy",
        "anyflow_diffusion_ratio": 0.5,
        "anyflow_consistency_ratio": 0.25,
        "anyflow_weight_type": "beta08",
        "anyflow_weight_grid_size": 1000,
        "anyflow_cd_eps": 1.0e-3,
        "anyflow_teacher_ckpt": "",
        "anyflow_v_target_source": "teacher",
        "anyflow_cd_velocity_source": "teacher",
        "anyflow_x1_source": "data",
        "anyflow_jvp_source": "student",
        "anyflow_loss_clip": 0.0,
        "anyflow_nfe_steps": 0,
        "anyflow_nfe_steps_list": [],
        "anyflow_nfe_homogeneous": False,
        "anyflow_nfe_grid_rho": 1.0,
        "anyflow_nfe_k_power": 1.0,
        "anyflow_nfe_k_fixed": -1,
        "anyflow_large_jump_dt": 0.0,
        "anyflow_large_jump_substeps": 6,
        "anyflow_large_jump_method": "euler",
        "anyflow_large_jump_loss_mult": 1.0,
        "anyflow_large_jump_r_max": 1.0,
        "anyflow_compose_fine_steps": 0,
        "anyflow_compose_from": "live",
        "anyflow_rollout_from": "",
        "anyflow_compose_nfe_max": 0,
        "anyflow_loss_v_weight": 1.0,
        "anyflow_full_jump_prob": 0.2,
        "teacher_steps": 50,
        "student_steps_list": [4, 8],
        "align_to": "teacher",
        "align_from": "teacher",
        "teacher_sampler": "euler",
        "sample_align_detach_between_jumps": False,
        "sample_align_checkpoint_jumps": False,
        "sample_align_lattice_weight": 1.0,
        "sample_align_coord_weight": 1.0,
        "sample_align_vol_weight": 0.5,
        "sample_align_ldd_weight": 1.0,
        "sample_align_use_adaptive": False,
        "sample_align_periodic_coord": True,
        "sample_align_rollout_mode": "shortcut",
        "sample_align_geom_scale": 8.0,
        "anyflow_cotrain_weight": 1.0,
        "dmd_weight": 1.0,
        "dmd_t_min": 0.02,
        "dmd_t_max": 0.98,
        "dmd_gradient_normalization": True,
        "dmd_rollout_mode": "shortcut",
        "dmd_periodic_coord": True,
        "dmd_lattice_weight": 1.0,
        "dmd_coord_weight": 1.0,
        "discriminator_update_ratio": 1,
        "fake_score_lr": 1.0e-5,
        "real_score_fp32": True,
        "student_tune": "full",
        "lora_rank": 32,
        "lora_alpha": 64.0,
        "gate_value": 0.25,
        "conditioning_mode": "mix",
        "time_parameterization": "legacy",
        "dual_time_feature_mode": "both",
        "dual_gate_value": 1.0,
        "dual_lora_start_step": 0,
        "score_conditioning_mode": "mix",
        "fake_score_tune": "delta_lora",
        "gate_anneal_start": None,
        "gate_anneal_end": None,
        "gate_anneal_steps": 0,
        "time_residual": 0.0,
        "time_residual_start": None,
        "time_residual_end": None,
        "time_residual_anneal_steps": 0,
        "endpoint_mix": 0.0,
        "endpoint_mix_start": None,
        "endpoint_mix_end": None,
        "endpoint_mix_anneal_steps": 0,
        "interval_embed_mix": 0.0,
        "reset_step_on_resume": False,
        "baseline_clari_metrics_json": str(
            _REPO_ROOT / "result/clari/table1/clari-m/metrics.json"
        ),
        "save_dir": str(_REPO_ROOT / "runs" / "distill"),
        "val_quick_batch_size": 4,
        "seed": 0,
        "bf16": True,
    }
    if path is None:
        return defaults
    with open(path) as f:
        defaults.update(yaml.safe_load(f) or {})
    return defaults


def _anyflow_cfg_from_dict(cfg: dict, *, enable_chiral: bool = True) -> AnyFlowLossConfig:
    flip = float(cfg.get("enantiomer_flip_p", 0.0)) if enable_chiral else 0.0
    chiral_w = float(cfg.get("chiral_loss_weight", 0.1)) if enable_chiral else 0.0
    # `enable_stereo` drives the same loss slot with the chiral-volume hinge.
    if enable_chiral and bool(cfg.get("enable_stereo", False)):
        chiral_w = float(cfg.get("stereo_loss_weight", 0.0))
    return AnyFlowLossConfig(
        ratio_r_not_equal_t=float(cfg["ratio_r_not_equal_t"]),
        aux_vol_weight=float(cfg["aux_vol_weight"]),
        aux_ldd_weight=float(cfg["aux_ldd_weight"]),
        use_aux_losses=bool(cfg["use_aux_losses"]),
        variant=str(cfg.get("anyflow_variant", "legacy")),
        diffusion_ratio=float(cfg.get("anyflow_diffusion_ratio", 0.5)),
        consistency_ratio=float(cfg.get("anyflow_consistency_ratio", 0.25)),
        weight_type=str(cfg.get("anyflow_weight_type", "beta08")),
        weight_grid_size=int(cfg.get("anyflow_weight_grid_size", 1000)),
        data_proportion=float(cfg.get("anyflow_data_proportion", 0.5)),
        full_jump_prob=float(cfg.get("anyflow_full_jump_prob", 0.2)),
        v_target_source=str(cfg.get("anyflow_v_target_source", "teacher")),
        cd_velocity_source=str(cfg.get("anyflow_cd_velocity_source", "teacher")),
        cd_eps=float(cfg.get("anyflow_cd_eps", 1.0e-3)),
        jvp_source=str(cfg.get("anyflow_jvp_source", "student")),
        loss_clip=float(cfg.get("anyflow_loss_clip", 0.0)),
        nfe_steps=int(cfg.get("anyflow_nfe_steps", 0)),
        nfe_steps_list=parse_nfe_steps_list(cfg.get("anyflow_nfe_steps_list")),
        nfe_homogeneous=bool(cfg.get("anyflow_nfe_homogeneous", False)),
        nfe_grid_rho=float(cfg.get("anyflow_nfe_grid_rho", 1.0) or 1.0),
        nfe_k_power=float(cfg.get("anyflow_nfe_k_power", 1.0) or 1.0),
        nfe_k_fixed=int(cfg.get("anyflow_nfe_k_fixed", -1)),
        large_jump_dt=float(cfg.get("anyflow_large_jump_dt", 0.0) or 0.0),
        large_jump_substeps=int(cfg.get("anyflow_large_jump_substeps", 6) or 6),
        large_jump_method=str(cfg.get("anyflow_large_jump_method", "euler") or "euler"),
        large_jump_loss_mult=float(cfg.get("anyflow_large_jump_loss_mult", 1.0) or 1.0),
        large_jump_r_max=float(cfg.get("anyflow_large_jump_r_max", 1.0)),
        compose_fine_steps=int(cfg.get("anyflow_compose_fine_steps", 0) or 0),
        compose_from=str(cfg.get("anyflow_compose_from", "live") or "live"),
        rollout_from=str(cfg.get("anyflow_rollout_from", "") or ""),
        compose_nfe_max=int(cfg.get("anyflow_compose_nfe_max", 0) or 0),
        loss_v_weight=float(cfg.get("anyflow_loss_v_weight", 1.0)),
        enantiomer_flip_p=flip,
        chiral_loss_weight=chiral_w,
        chiral_on_flow_map=bool(cfg.get("stereo_flow_map_hinge", False)),
        v_data_r_max=float(cfg.get("anyflow_v_data_r_max", 0.0)),
    )


def _csd_ready(root: Path) -> bool:
    csd = root / "csd"
    return all((csd / n).is_file() for n in ("config.json", "train.pt", "val.pt", "test.pt"))


def _setup_dist():
    if "RANK" not in os.environ:
        return 0, 1, 0, False
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return rank, world, local, True


def _build_optimizer(
    net: torch.nn.Module,
    lr: float,
    wd: float,
    *,
    extra_modules: list[torch.nn.Module] | None = None,
    embed_deltat_lr_mult: float = 1.0,
    stereo_lr_mult: float = 1.0,
    betas: tuple[float, float] = (0.9, 0.95),
):
    from clari.pipelines.utils import build_muon_optimizer

    params_adam, params_muon, params_delta = [], [], []
    params_stereo: list = []
    seen: set[int] = set()
    for name, p in net.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        # Zero-init stereo tag embedding: needs a much larger lr than the
        # already-converged backbone to leave 0 within a short continuation.
        if "_stereo_" in name:
            params_stereo.append(p)
            continue
        # AnyFlow time-embedding delta: embed_timestep.delta.* (new arch).
        # Also match legacy keys for backward-compat.
        if (
            "embed_timestep.delta" in name
            or "_time_proxy.delta" in name
            or "stem_cond.conditioner." in name
            or "_dual_time." in name
            or "_output_head" in name
            or "_delta_head" in name
            or "delta_mlp" in name
            or "embed_deltat" in name
            or "interval_proj" in name
            or "interval_mlp" in name
        ):
            params_delta.append(p)
        elif "lora_A" in name or "lora_B" in name:
            params_adam.append(p)
        elif ("dit.trunk" in name or "dit.stem" in name) and p.ndim >= 2:
            params_muon.append(p)
        else:
            params_adam.append(p)
    if extra_modules:
        for mod in extra_modules:
            for p in mod.parameters():
                if p.requires_grad:
                    params_adam.append(p)
    # Δt head uses a separate AdamW (Muon aux-Adam under DDP was leaving Δt
    # grads unused / weights stuck at 0). Backbone stays on Muon when present.
    delta_opt = None
    if params_adam or params_muon:
        if params_muon:
            opt = build_muon_optimizer(params_adam, params_muon, lr, wd, lr_warmup=0)
        else:
            opt = torch.optim.AdamW(params_adam, lr=lr, weight_decay=wd, betas=betas)
        for pg in opt.param_groups:
            pg.setdefault("lr_mult", 1.0)
        if params_delta:
            delta_opt = torch.optim.AdamW(
                params_delta,
                lr=lr * float(embed_deltat_lr_mult),
                weight_decay=wd,
                betas=betas,
            )
            for pg in delta_opt.param_groups:
                pg["lr_mult"] = float(embed_deltat_lr_mult)
    else:
        if not params_delta:
            if params_stereo:
                # student_tune=stereo_only: the stereo heads are the ONLY
                # trainable tensors. Below, params_stereo is attached with
                # add_param_group to an optimizer built from the backbone /
                # delta params -- but here there is no such optimizer to attach
                # to, so seed one from the stereo params directly instead of
                # declaring the run untrainable.
                opt = torch.optim.AdamW(
                    params_stereo,
                    lr=lr * float(stereo_lr_mult),
                    weight_decay=0.0,
                    betas=betas,
                )
                for pg in opt.param_groups:
                    pg["lr_mult"] = float(stereo_lr_mult)
                opt._mf_stereo_params = list(params_stereo)  # type: ignore[attr-defined]
                opt._mf_delta_opt = None  # type: ignore[attr-defined]
                opt._mf_delta_params = []  # type: ignore[attr-defined]
                opt._mf_delta_lr_mult = float(embed_deltat_lr_mult)  # type: ignore[attr-defined]
                return opt
            raise RuntimeError("No trainable parameters after student_tune freeze")
        opt = torch.optim.AdamW(
            params_delta,
            lr=lr * float(embed_deltat_lr_mult),
            weight_decay=wd,
            betas=betas,
        )
        for pg in opt.param_groups:
            pg["lr_mult"] = float(embed_deltat_lr_mult)
        params_delta = list(params_delta)
    if params_stereo:
        host = delta_opt if delta_opt is not None else opt
        host.add_param_group(
            {
                "params": params_stereo,
                "lr": lr * float(stereo_lr_mult),
                "weight_decay": 0.0,
                "lr_mult": float(stereo_lr_mult),
            }
        )
    opt._mf_stereo_params = list(params_stereo)  # type: ignore[attr-defined]
    opt._mf_delta_opt = delta_opt  # type: ignore[attr-defined]
    opt._mf_delta_params = list(params_delta)  # type: ignore[attr-defined]
    opt._mf_delta_lr_mult = float(embed_deltat_lr_mult)  # type: ignore[attr-defined]
    return opt


def _stereo_embed_l2(net: torch.nn.Module) -> float:
    """L2 of the stereo token embedding; 0 at init, grows once it is learning."""
    total = 0.0
    found = False
    for name in ("_stereo_tokens", "_stereo_pairs"):
        mod = getattr(net, name, None)
        if mod is not None:
            found = True
            total += float(mod.emb.weight.detach().float().norm()) ** 2
    return total**0.5 if found else float("nan")


@torch.no_grad()
def _ema_update(ema: dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    msd = model.state_dict()
    for k, v in msd.items():
        if not torch.is_floating_point(v):
            continue
        if is_time_mix_buffer_key(k):
            continue
        if k not in ema:
            ema[k] = v.detach().clone()
            continue
        cur = ema[k]
        if cur.device != v.device or cur.dtype != v.dtype:
            cur = cur.to(device=v.device, dtype=v.dtype)
            ema[k] = cur
        cur.mul_(decay).add_(v.detach(), alpha=1.0 - decay)


def _suppress_lora_grads(model: torch.nn.Module, step: int, start_step: int) -> None:
    if int(step) >= int(start_step):
        return
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.grad = None


def _lr_scale(step: int, warmup: int, base_lr: float) -> float:
    if warmup <= 0:
        return base_lr
    return base_lr * min(1.0, (step + 1) / float(warmup))


def _prune_epoch_checkpoints(save_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    ckpts = sorted(save_dir.glob("checkpoint-epoch*.pt"), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-keep_last]:
        try:
            old.unlink()
        except OSError:
            pass


def _run_periodic_evals(
    *,
    ckpt_path: Path,
    epoch: int,
    cfg: dict,
    repo_root: Path,
    venv_python: Path,
) -> None:
    """Clari Table1 + MCF volume RMAD (rank-0 subprocess; Table1 may use multi-GPU torchrun)."""
    ep = epoch + 1
    eval_root = Path(cfg["save_dir"]) / f"eval_epoch{ep:03d}"
    table1_out = eval_root / "table1"
    mcf_out = eval_root / "mcf_fig3"
    table1_out.mkdir(parents=True, exist_ok=True)
    mcf_out.mkdir(parents=True, exist_ok=True)

    cuda_dev = str(cfg.get("eval_cuda_devices", "0,1,2,3"))
    env = os.environ.copy()
    env["AMD_METRIC"] = env.get("AMD_METRIC", "cityblock")
    env["CUDA_VISIBLE_DEVICES"] = cuda_dev
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    table1_nproc = int(cfg.get("eval_table1_nproc", 1))
    pack_size = int(cfg.get("eval_table1_pack_size", 4))
    table1_args = [
        "--meanflow-ckpt",
        str(ckpt_path),
        "--meanflow-steps",
        str(int(cfg.get("meanflow_steps", 4))),
        "--output-dir",
        str(table1_out),
        "--checkpoint",
        str(cfg["checkpoint"]),
        "--samples",
        str(int(cfg.get("eval_table1_samples", 10))),
        "--clari-data-dir",
        str(cfg["clari_data_dir"]),
        "--seed",
        str(int(cfg.get("seed", 42))),
        "--amd-metric",
        env.get("AMD_METRIC", "cityblock"),
        "--pack-size",
        str(pack_size),
    ]
    max_c = cfg.get("eval_table1_max_crystals")
    if max_c is not None and int(max_c) > 0:
        table1_args.extend(["--max-crystals", str(int(max_c))])

    if table1_nproc > 1:
        table1_cmd = [
            str(venv_python),
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={table1_nproc}",
            "-m",
            "crystal_nft.train.eval_meanflow_table1",
            *table1_args,
        ]
    else:
        table1_cmd = [
            str(venv_python),
            "-m",
            "crystal_nft.train.eval_meanflow_table1",
            *table1_args,
            "--device",
            f"cuda:{cuda_dev.split(',')[0]}",
        ]

    logger.info(
        "Periodic Table1 eval epoch=%d nproc=%d max_crystals=%s -> %s",
        ep,
        table1_nproc,
        max_c,
        table1_out,
    )
    subprocess.run(table1_cmd, check=False, cwd=str(repo_root), env=env)

    mcf_gpu = cuda_dev.split(",")[0]
    mcf_cmd = [
        str(venv_python),
        "-m",
        "crystal_nft.train.eval_meanflow_mcf_rmad",
        "--meanflow-ckpt",
        str(ckpt_path),
        "--mcf-cache",
        str(cfg["mcf_cache"]),
        "--output-dir",
        str(mcf_out),
        "--checkpoint",
        str(cfg["checkpoint"]),
        "--meanflow-steps",
        str(int(cfg.get("meanflow_steps", 4))),
        "--max-structures",
        str(int(cfg.get("eval_mcf_max_structures", 5))),
        "--samples",
        str(int(cfg.get("eval_mcf_samples", 1))),
        "--device",
        f"cuda:{mcf_gpu}",
    ]
    logger.info("Periodic MCF RMAD eval epoch=%d -> %s", ep, mcf_out)
    subprocess.run(mcf_cmd, check=False, cwd=str(repo_root), env=env)


def _eval_subprocess_env(cfg: dict) -> dict:
    env = os.environ.copy()
    env["EVAL_TABLE1_NPROC"] = str(int(cfg.get("eval_table1_nproc", 4)))
    env["EVAL_TABLE1_SAMPLES"] = str(int(cfg.get("eval_table1_samples", 20)))
    env["EVAL_TABLE1_PACK_SIZE"] = str(int(cfg.get("eval_table1_pack_size", 2)))
    env["EVAL_MCF_MAX_STRUCTURES"] = str(int(cfg.get("eval_mcf_max_structures", 10)))
    max_c = cfg.get("eval_table1_max_crystals")
    if max_c is not None and int(max_c) > 0:
        env["EVAL_TABLE1_MAX_CRYSTALS"] = str(int(max_c))
    else:
        env.pop("EVAL_TABLE1_MAX_CRYSTALS", None)
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.get("eval_cuda_devices", "0,1,2,3"))
    env["EVAL_SKIP_GPU_WAIT"] = "1"
    return env


def _run_post_train_evals(*, cfg: dict, repo_root: Path, venv_python: Path) -> None:
    """Run Table1 + MCF for each eval_at_epochs checkpoint after training finishes."""
    epochs = sorted(int(x) for x in (cfg.get("eval_at_epochs") or []))
    if not epochs:
        return
    save_dir = Path(cfg["save_dir"])
    script = repo_root / "experiment/meanflow/run_periodic_eval_background.sh"
    env = _eval_subprocess_env(cfg)
    for ep_idx in epochs:
        ckpt = save_dir / f"checkpoint-epoch{ep_idx}.pt"
        if not ckpt.is_file():
            logger.warning("Post-train eval: missing %s, skip", ckpt)
            continue
        ep_tag = ep_idx + 1
        out_dir = save_dir / f"eval_epoch{ep_tag:03d}"
        logger.info("Post-train eval epoch=%d ckpt=%s -> %s", ep_tag, ckpt, out_dir)
        if script.is_file():
            subprocess.run(
                ["bash", str(script), str(ckpt), f"{ep_tag:03d}", str(out_dir)],
                cwd=str(repo_root),
                env=env,
                check=False,
            )
        else:
            _run_periodic_evals(
                ckpt_path=ckpt,
                epoch=ep_idx,
                cfg=cfg,
                repo_root=repo_root,
                venv_python=venv_python,
            )


def _prune_step_checkpoints(save_dir: Path, keep_last: int, pattern: str = "checkpoint-step*.pt") -> None:
    if keep_last <= 0:
        return
    ckpts = sorted(save_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-keep_last]:
        try:
            old.unlink()
        except OSError:
            pass


def _load_clari_baseline_amd(metrics_path: str) -> float:
    p = Path(metrics_path)
    if not p.is_file():
        return CLARI_M_BASELINE_AMD
    try:
        data = json.loads(p.read_text())
        summary = data.get("summary") or {}
        return float(summary.get("mean_dist_amd", CLARI_M_BASELINE_AMD))
    except (json.JSONDecodeError, TypeError, ValueError):
        return CLARI_M_BASELINE_AMD


@torch.no_grad()
def _quick_val_loss(
    net: torch.nn.Module,
    interface,
    loss_fn,
    val_loader: DataLoader,
    device: torch.device,
    *,
    use_fm: bool,
) -> dict[str, float]:
    try:
        batch = next(iter(val_loader))
    except StopIteration:
        return {}
    C0, C1 = batch
    C0 = C0.to(device)
    C1 = C1.to(device)
    net.eval()
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        if use_fm:
            out = interface.fm_supervision_loss(net, (C0, C1), chiral_bias=None)
        else:
            out = loss_fn(net, interface, C0, C1, chiral_bias=None)
    net.train()
    return {k: float(v.detach()) for k, v in out.items() if torch.is_tensor(v)}


def _write_baseline_eval(
    save_dir: Path,
    step: int,
    val_metrics: dict[str, float],
    clari_amd: float,
) -> None:
    eval_dir = save_dir / "baseline_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "val": val_metrics,
        "clari_m_baseline_amd": clari_amd,
        "note": "Quick val loss on a fixed mini-batch; not full Table1 AMD.",
    }
    (eval_dir / f"step_{step:06d}.json").write_text(json.dumps(payload, indent=2))



_FOLLOW_BATCH: dict = {}


@torch.no_grad()
def _stereo_follow_probe(net, interface, C1, stereo_spec, *, min_centres: int = 10):
    """Leak-free follow rate: from PURE NOISE at t=0, does the CIP tag steer handedness?

    This is the only instrument that says whether the conditioning is being
    learned when the hinge is off, and the hinge has to be off -- all 16 failed
    configurations had one, the probe that worked had none.

    Measured on a **fixed cached batch**, not the live one. With batch_size 8 the
    number of chiral centres in a batch swings between 4 and 52, and a rate over
    4 centres is noise -- the curve has to be comparable across steps to be read
    at all.

    Must be evaluated at t=0 from noise: at any t>0 the state ``x_t`` is built
    from the true ``x1`` and already encodes the handedness, so the model scores
    high by denoising the leak and the tag is irrelevant.
    """
    from crystal_nft.meanflow.stereo import chiral_signs

    cached = _FOLLOW_BATCH.get("batch")
    if cached is None:
        if stereo_spec is None or stereo_spec.label_sign is None:
            return float("nan"), 0
        act = (stereo_spec.label_sign != 0) & stereo_spec.valid
        if int(act.sum()) < min_centres:
            return float("nan"), 0
        _FOLLOW_BATCH["batch"] = C1.replace(x=C1.x.detach().clone())
        _FOLLOW_BATCH["spec"] = stereo_spec
        _FOLLOW_BATCH["cond"] = build_stereo_conditioning(
            stereo_spec, int(C1.x.shape[1] - 3), pair_edges=True
        )
        cached = _FOLLOW_BATCH["batch"]
        logger.info(
            "follow-probe batch cached: %d stereocentres (fixed from here on)",
            int(act.sum()),
        )
    spec = _FOLLOW_BATCH["spec"]
    cond = _FOLLOW_BATCH["cond"]
    active = (spec.label_sign != 0) & spec.valid
    n = int(active.sum())
    if n == 0:
        return float("nan"), 0
    was = net.training
    net.eval()
    try:
        with active_stereo_tags(cond):
            z0 = interface.sample_prior(cached.replace(x=torch.zeros_like(cached.x))).x
            t0 = torch.zeros(cached.x.shape[0], device=cached.x.device)
            pred = interface.forward(net=net, xt=z0, xsc=None, t=t0, r=t0, f=cached)
            x1_hat = interface.estimate_x1(z0, t0, pred)
        got = chiral_signs(x1_hat[:, 3:].float(), spec.centers)
        agree = int(((got == spec.label_sign) & active).sum())
    except Exception:  # noqa: BLE001
        return float("nan"), 0
    finally:
        net.train(was)
    return agree / max(n, 1), n


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = _load_config(args.config)

    rank, world, local, distributed = _setup_dist()

    clari_data_dir = Path(cfg["clari_data_dir"]).resolve()
    os.environ["CLARI_DATA_DIR"] = str(clari_data_dir)
    if cfg.get("require_csd") and not _csd_ready(clari_data_dir):
        raise FileNotFoundError(f"CSD not ready under {clari_data_dir}/csd")

    if args.smoke:
        cfg["batch_size"] = 2
        cfg["steps_per_epoch"] = 4
        cfg["num_epochs"] = 1
        cfg["num_workers"] = 0
        cfg["save_every_steps"] = 4
        cfg["log_every"] = 1
        cfg["save_dir"] = str(Path(cfg["save_dir"]).parent / "smoke_train")
        if str(cfg.get("loss_type", "")).lower() in ("sample_align", "anyflow_onpolicy"):
            cfg["teacher_steps"] = 2
            cfg["student_steps_list"] = [2]

    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg["seed"]) + rank)

    loss_type = str(cfg.get("loss_type", "meanflow")).lower()
    bundle = load_meanflow_bundle(
        cfg["checkpoint"],
        device=device,
        use_ema=bool(cfg.get("use_ema_init", True)),
        meanflow_steps=int(cfg["meanflow_steps"]),
        enable_chiral=bool(cfg.get("enable_chiral", False)),
        enable_stereo=bool(cfg.get("enable_stereo", False)),
        stereo_pair_edges=bool(cfg.get("stereo_pair_edges", True)),
        stereo_gain=float(cfg.get("stereo_gain", 1.0)),
        stereo_cond_path=bool(cfg.get("stereo_cond_path", False)),
        stereo_node_mod=bool(cfg.get("stereo_node_mod", False)),
        stereo_chiral_branch=bool(cfg.get("stereo_chiral_branch", False)),
        stereo_chiral_branch_scale=float(cfg.get("stereo_chiral_branch_scale", 1.0)),
        stereo_chiral_branch_gate=float(cfg.get("stereo_chiral_branch_gate", 0.0)),
        stereo_chiral_branch_endpoint_geom=bool(
            cfg.get("stereo_chiral_branch_endpoint_geom", False)
        ),
        stereo_chiral_branch_bond_preserving=bool(
            cfg.get("stereo_chiral_branch_bond_preserving", False)
        ),
        stereo_chiral_branch_t_max=float(cfg.get("stereo_chiral_branch_t_max", 1.0)),
        stereo_chiral_branch_t_min=float(cfg.get("stereo_chiral_branch_t_min", 0.0)),
        stereo_head_t_min=float(cfg.get("stereo_head_t_min", 0.0)),
        stereo_head_t_max=float(cfg.get("stereo_head_t_max", 1.0)),
        compile_model=False,
        dit_preset=cfg.get("dit_preset"),
        init_dit_from_checkpoint=bool(cfg.get("init_dit_from_checkpoint", True)),
        load_teacher=(
            loss_type in ("anyflow", "anyflow_onpolicy")
            or (
                loss_type == "sample_align"
                and float(cfg.get("anyflow_cotrain_weight", 1.0)) > 0
            )
        ),
        load_fake_score=(loss_type == "anyflow_onpolicy"),
        gate_value=float(cfg.get("gate_value", 0.25)),
        conditioning_mode=str(cfg.get("conditioning_mode", "mix")),
        dual_time_feature_mode=str(cfg.get("dual_time_feature_mode", "both")),
        dual_gate_value=float(cfg.get("dual_gate_value", 1.0)),
        time_parameterization=str(cfg.get("time_parameterization", "legacy")),
        score_flowmap=bool(cfg.get("score_flowmap", False)),
        score_conditioning_mode=str(cfg.get("score_conditioning_mode", "mix")),
        score_use_nft_init=bool(cfg.get("score_use_nft_init", True)),
        dmd_real_from=str(cfg.get("dmd_real_from", "teacher")),
        fake_from=str(cfg.get("fake_from", cfg.get("dmd_real_from", "teacher"))),
        align_from=str(cfg.get("align_from", "teacher")),
        keep_nft_copies=False,
        clari_nft_init=cfg.get("clari_nft_init"),
    )
    interface = bundle["interface"]
    net = bundle["net"].to(device)
    teacher_net = bundle.get("teacher_net")
    if teacher_net is not None:
        teacher_net = teacher_net.to(device).eval()
    _tck = str(cfg.get("anyflow_teacher_ckpt", "") or "").strip()
    if teacher_net is not None and _tck:
        # Stage B: distil from a CHIRALITY-AWARE teacher.
        #
        # The AnyFlow target is ~94% frozen-teacher velocity, and the default
        # teacher has no stereo head -- so distillation structurally teaches the
        # student to IGNORE the tag. Measured (v16): with a parity branch on the
        # student but a blind teacher, the branch's residual is counted as
        # regression error, af went 0.19 -> 1.20 and sampled chirality stayed at
        # 0.50/fr0.24, versus 0.655/fr0.46 by step100 on the pure-FM line where
        # the target (x1 - x0) carries handedness.
        #
        # Loading a stereo-aware checkpoint here makes the teacher's velocity
        # carry handedness, so the distillation target and the chirality
        # objective finally point the same way instead of fighting.
        from crystal_nft.meanflow.adapter import build_meanflow_teacher_from_ckpt

        teacher_net = build_meanflow_teacher_from_ckpt(
            _tck, base_lit=bundle["lit"], device=device, cfg=cfg
        )
        if rank == 0:
            logger.info("AnyFlow teacher replaced by stereo-aware ckpt: %s", _tck)
    align_net = bundle.get("align_net")
    if align_net is not None:
        align_net = align_net.to(device).eval()
        for p in align_net.parameters():
            p.requires_grad_(False)
    fake_score_net = bundle.get("fake_score_net")
    if fake_score_net is not None:
        fake_score_net = fake_score_net.to(device)
    real_score_net = bundle.get("real_score_net")
    if real_score_net is not None:
        real_score_net = real_score_net.to(device).eval()
        for p in real_score_net.parameters():
            p.requires_grad_(False)
    chiral = bundle["chiral"]
    if chiral is not None:
        chiral = chiral.to(device)

    from crystal_nft.meanflow.tune import apply_student_tune

    tune_stats = apply_student_tune(net, cfg)
    if fake_score_net is not None:
        fake_cfg = dict(cfg)
        fake_cfg["student_tune"] = str(cfg.get("fake_score_tune", "delta_lora"))
        fake_cfg["lora_rank"] = int(cfg.get("fake_score_lora_rank", cfg.get("lora_rank", 32)))
        fake_cfg["lora_alpha"] = float(
            cfg.get("fake_score_lora_alpha", cfg.get("lora_alpha", 64.0))
        )
        fake_tune_stats = apply_student_tune(fake_score_net, fake_cfg)
        if rank == 0:
            logger.info(
                "Fake score tune=%s lora=%d trainable=%d/%d",
                fake_cfg["student_tune"],
                int(fake_tune_stats.get("n_lora", 0)),
                int(fake_tune_stats.get("n_trainable", 0)),
                int(fake_tune_stats.get("n_all", 0)),
            )
    if rank == 0:
        logger.info(
            "Student tune=%s lora=%d trainable=%d/%d",
            str(cfg.get("student_tune", "full")),
            int(tune_stats.get("n_lora", 0)),
            int(tune_stats.get("n_trainable", 0)),
            int(tune_stats.get("n_all", 0)),
        )

    if distributed:
        net = DDP(
            net,
            device_ids=[local],
            output_device=local,
            broadcast_buffers=False,
            find_unused_parameters=bool(cfg.get("enable_chiral", False))
            or bool(cfg.get("enable_stereo", False))
            or loss_type == "anyflow_onpolicy",
        )
        if fake_score_net is not None:
            fake_score_net = DDP(
                fake_score_net,
                device_ids=[local],
                output_device=local,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

    flip_p = float(cfg.get("enantiomer_flip_p", 0.0))

    def _chirality_fn(x1, crystal):
        return enantiomer_augment_x1(x1, crystal, flip_p)

    enable_stereo = bool(cfg.get("enable_stereo", False))
    stereo_margin = float(cfg.get("stereo_margin", 0.62))
    stereo_mirror_p = float(cfg.get("stereo_mirror_p", 0.0)) if enable_stereo else 0.0
    stereo_pair_edges = bool(cfg.get("stereo_pair_edges", True))
    stereo_hinge_r_max = float(cfg.get("stereo_hinge_r_max", 1.0))
    stereo_hinge_gamma = float(cfg.get("stereo_hinge_gamma", 0.0))
    stereo_hinge_t_max = float(cfg.get("stereo_hinge_t_max", 1.0))
    stereo_hinge_soft = bool(cfg.get("stereo_hinge_soft", True))
    # Per-batch stereocentre bookkeeping, refreshed right before each loss call.
    stereo_state: dict[str, object] = {"spec": None, "agree": float("nan"), "n": 0}

    def _chiral_consistency_fn(pred_x1, target_x1, C0, r=None):
        # target_x1 is post-enantiomer-augment packed coords from the loss.
        if enable_stereo:
            spec = stereo_state.get("spec")
            if spec is None:
                return pred_x1.new_zeros(())
            # Concentrate the hinge near r=0. At moderate r the noised state
            # already encodes the handedness, so the model satisfies the hinge by
            # reading x_r and the gradient into the CIP-tag embedding vanishes --
            # that is exactly why cont4 trained to chance-level control.
            w = None
            if r is not None and stereo_hinge_gamma > 0.0:
                # w(t) = t^gamma, i.e. UP-weight toward data (Clari t=1). The
                # `clamp(1 - r/r_max)` schedule below does the opposite: it
                # concentrates the hinge at r~0 to stop the model reading the
                # answer out of a noised state that already leaks it. But the
                # measured trajectory says the hinge is already satisfied there
                # and fails at the other end:
                #   t          0.20   0.50   0.70   0.80   0.90
                #   chir(x1^)  0.915  0.855  0.735  0.605  0.540
                # so the r~0 schedule puts ~all the weight where the gradient is
                # ~0 and none where the failure is. It also fought the
                # self-state rows, which exist precisely at larger t.
                r_f = r.detach().reshape(-1).float().clamp(min=0.0)
                w = r_f ** float(stereo_hinge_gamma)
                if stereo_hinge_t_max < 1.0:
                    # Stop pressing once the crystal is essentially finished.
                    # Attribution (60 fam, Heun-50): the branch itself is nearly
                    # free -- branch + self-state with a NOISE-end hinge scores
                    # PB 78.55 / clash 10.14 vs the 82.26 / 10.59 baseline, i.e.
                    # clash even improves. Switching that hinge to the DATA end
                    # is what costs PB (78.55 -> 69.00), and it is also what buys
                    # sampled chirality (0.47 -> 0.86).
                    # But the trajectory trace says handedness is decided in
                    # t in [0.5, 0.9], not at t -> 1. Pressing |tau| >= margin on
                    # an almost-final structure distorts bonds/angles (what PB
                    # measures) for no chirality gain, so cut the weight above
                    # t_max and keep the part that does the work.
                    w = w * (r_f <= float(stereo_hinge_t_max)).float()
            elif r is not None and stereo_hinge_r_max < 1.0:
                r_flat = r.detach().reshape(-1).float()
                if stereo_hinge_soft:
                    w = torch.clamp(
                        1.0 - r_flat / max(stereo_hinge_r_max, 1e-6), min=0.0
                    )
                else:
                    w = (r_flat <= stereo_hinge_r_max).float()
            loss_st = chiral_hinge_loss(
                pred_x1[:, 3:],
                target_x1[:, 3:],
                spec,
                margin=stereo_margin,
                sample_weight=w,
            )
            agree, n = stereo_agreement(
                pred_x1[:, 3:].detach().float(), target_x1[:, 3:].float(), spec
            )
            stereo_state["agree"] = agree
            stereo_state["n"] = n
            return loss_st
        mask = getattr(C0, "mask", None)
        return enantiomer_consistency_loss(pred_x1, target_x1, mask)

    loss_cfg = MeanFlowLossConfig(
        ratio_r_not_equal_t=float(cfg["ratio_r_not_equal_t"]),
        aux_vol_weight=float(cfg["aux_vol_weight"]),
        aux_ldd_weight=float(cfg["aux_ldd_weight"]),
        use_aux_losses=bool(cfg["use_aux_losses"]),
        enantiomer_flip_p=float(cfg.get("enantiomer_flip_p", 0.0)),
    )
    if loss_type == "imf":
        imf_cfg = IMFLossConfig(
            ratio_r_not_equal_t=float(cfg["ratio_r_not_equal_t"]),
            aux_vol_weight=float(cfg["aux_vol_weight"]),
            aux_ldd_weight=float(cfg["aux_ldd_weight"]),
            use_aux_losses=bool(cfg["use_aux_losses"]),
            data_proportion=float(cfg.get("imf_data_proportion", 0.5)),
            cd_velocity_source=str(cfg.get("imf_cd_velocity_source", "u_self")),
            loss_v_weight=float(cfg.get("imf_loss_v_weight", 1.0)),
            enantiomer_flip_p=float(cfg.get("enantiomer_flip_p", 0.0)),
        )
        mf_loss = CrystalIMFLoss(imf_cfg)
    elif loss_type == "anyflow":
        if teacher_net is None:
            raise RuntimeError("loss_type=anyflow requires a frozen teacher_net")
        af_cfg = _anyflow_cfg_from_dict(cfg, enable_chiral=True)
        mf_loss = CrystalAnyFlowLoss(af_cfg, teacher_net=teacher_net)
        if rank == 0:
            nfe_choices, nfe_rho = nfe_grid_choices(af_cfg)
            logger.info(
                "AnyFlow loss: variant=%s v_target=%s cd=%s jvp=%s clip=%.3g nfe=%s homog=%s rho=%.3f k_pow=%.2f k_fixed=%d "
                "large_dt=%.3f large_rmax=%.3f large_sub=%d large_m=%s compose_from=%s rollout_from=%s fine=%d compose_nmax=%d large_w=%.2f eps=%.4g "
                "chiral_w=%.3f flip_p=%.2f data_p=%.2f full_jump_p=%.2f "
                "diffusion=%.2f consistency=%.2f weight=%s delta_lr_x=%.1f",
                af_cfg.variant,
                af_cfg.v_target_source,
                af_cfg.cd_velocity_source,
                af_cfg.jvp_source,
                af_cfg.loss_clip,
                list(nfe_choices) if nfe_choices else int(af_cfg.nfe_steps),
                str(bool(af_cfg.nfe_homogeneous)),
                nfe_rho,
                float(af_cfg.nfe_k_power),
                int(af_cfg.nfe_k_fixed),
                float(af_cfg.large_jump_dt),
                float(af_cfg.large_jump_r_max),
                int(af_cfg.large_jump_substeps),
                str(af_cfg.large_jump_method),
                str(af_cfg.compose_from),
                str(af_cfg.rollout_from) or str(af_cfg.compose_from),
                int(af_cfg.compose_fine_steps),
                int(af_cfg.compose_nfe_max),
                float(af_cfg.large_jump_loss_mult),
                af_cfg.cd_eps,
                af_cfg.chiral_loss_weight,
                af_cfg.enantiomer_flip_p,
                af_cfg.data_proportion,
                af_cfg.full_jump_prob,
                af_cfg.diffusion_ratio,
                af_cfg.consistency_ratio,
                af_cfg.weight_type,
                float(cfg.get("embed_deltat_lr_mult", 1.0)),
            )
            logger.info(
                "AnyFlow x1_source=%s align_from=%s teacher_sampler=%s teacher_steps=%d",
                str(cfg.get("anyflow_x1_source", "data")),
                str(cfg.get("align_from", "teacher")),
                str(cfg.get("teacher_sampler", "heun")),
                int(cfg.get("teacher_steps", 50)),
            )
    elif loss_type == "sample_align":
        align_from = str(cfg.get("align_from", "teacher")).lower()
        if (
            teacher_net is None
            and str(cfg.get("align_to", "teacher")) == "teacher"
            and align_from != "student"
        ):
            raise RuntimeError("loss_type=sample_align requires a frozen teacher_net")
        student_steps = cfg.get("student_steps_list") or [int(cfg.get("meanflow_steps", 8))]
        if isinstance(student_steps, int):
            student_steps = [student_steps]
        sa_cfg = SampleAlignLossConfig(
            teacher_steps=int(cfg.get("teacher_steps", 50)),
            student_steps_list=[int(s) for s in student_steps],
            detach_between_jumps=bool(cfg.get("sample_align_detach_between_jumps", False)),
            checkpoint_jumps=bool(cfg.get("sample_align_checkpoint_jumps", False)),
            lattice_weight=float(cfg.get("sample_align_lattice_weight", 1.0)),
            coord_weight=float(cfg.get("sample_align_coord_weight", 1.0)),
            vol_weight=float(cfg.get("sample_align_vol_weight", 0.5)),
            ldd_weight=float(cfg.get("sample_align_ldd_weight", 1.0)),
            align_to=str(cfg.get("align_to", "teacher")),
            anyflow_cotrain_weight=float(cfg.get("anyflow_cotrain_weight", 1.0)),
            use_adaptive=bool(cfg.get("sample_align_use_adaptive", False)),
            periodic_coord=bool(cfg.get("sample_align_periodic_coord", True)),
            rollout_mode=str(cfg.get("sample_align_rollout_mode", "shortcut")),
            geom_scale=float(cfg.get("sample_align_geom_scale", 8.0)),
            teacher_sampler=str(cfg.get("teacher_sampler", "euler")),
            teacher_self_cond=bool(cfg.get("teacher_self_cond", True)),
            student_self_cond=bool(cfg.get("student_self_cond", False)),
            interval_rho=float(
                cfg.get("sample_align_interval_rho", cfg.get("anyflow_nfe_grid_rho", 0.75))
            ),
            interval_schedule=str(
                cfg.get("sample_align_interval_schedule", "power")
            ),
        )
        af_cfg = None
        if sa_cfg.anyflow_cotrain_weight > 0:
            af_cfg = _anyflow_cfg_from_dict(cfg, enable_chiral=False)
        mf_loss = CrystalSampleAlignLoss(
            sa_cfg,
            teacher_net=teacher_net,
            align_net=align_net if align_net is not None else teacher_net,
            anyflow_cfg=af_cfg,
        )
        if rank == 0:
            logger.info(
                "SampleAlign loss: teacher_steps=%d teacher_sampler=%s student_steps=%s "
                "align_to=%s align_from=%s mode=%s rho=%.3f detach=%s checkpoint=%s adaptive=%s "
                "periodic=%s geom_scale=%.1f cotrain_w=%.3f nfe=%d jvp=%s clip=%.3g "
                "resume_ema=%s grad_allreduce=%s delta_lr_x=%.1f",
                sa_cfg.teacher_steps,
                sa_cfg.teacher_sampler,
                sa_cfg.student_steps_list,
                sa_cfg.align_to,
                str(cfg.get("align_from", "teacher")),
                sa_cfg.rollout_mode,
                float(sa_cfg.interval_rho),
                sa_cfg.detach_between_jumps,
                sa_cfg.checkpoint_jumps,
                sa_cfg.use_adaptive,
                sa_cfg.periodic_coord,
                sa_cfg.geom_scale,
                sa_cfg.anyflow_cotrain_weight,
                int(cfg.get("anyflow_nfe_steps", 0)),
                str(cfg.get("anyflow_jvp_source", "student")),
                float(cfg.get("anyflow_loss_clip", 0.0)),
                str(bool(cfg.get("resume_student_from_ema", False))),
                str(bool(distributed)),
                float(cfg.get("embed_deltat_lr_mult", 1.0)),
            )
    elif loss_type == "anyflow_onpolicy":
        if teacher_net is None or fake_score_net is None:
            raise RuntimeError(
                "loss_type=anyflow_onpolicy requires teacher_net and fake_score_net"
            )
        student_steps = cfg.get("student_steps_list") or [int(cfg.get("meanflow_steps", 4))]
        if isinstance(student_steps, int):
            student_steps = [student_steps]
        op_cfg = AnyFlowOnPolicyConfig(
            student_steps_list=[int(s) for s in student_steps],
            rollout_mode=str(cfg.get("dmd_rollout_mode", cfg.get("sample_align_rollout_mode", "shortcut"))),
            n_jumps=int(cfg.get("dmd_n_jumps", 8)),
            detach_between_jumps=bool(cfg.get("sample_align_detach_between_jumps", False)),
            checkpoint_jumps=bool(cfg.get("sample_align_checkpoint_jumps", False)),
            student_self_cond=bool(cfg.get("student_self_cond", False)),
            dmd_weight=float(cfg.get("dmd_weight", 1.0)),
            dmd_t_min=float(cfg.get("dmd_t_min", 0.02)),
            dmd_t_max=float(cfg.get("dmd_t_max", 0.98)),
            gradient_normalization=bool(cfg.get("dmd_gradient_normalization", True)),
            anyflow_cotrain_weight=float(cfg.get("anyflow_cotrain_weight", 1.0)),
            discriminator_update_ratio=int(cfg.get("discriminator_update_ratio", 1)),
            real_score_fp32=bool(cfg.get("real_score_fp32", True)),
            periodic_coord=bool(cfg.get("dmd_periodic_coord", True)),
            lattice_weight=float(cfg.get("dmd_lattice_weight", 1.0)),
            coord_weight=float(cfg.get("dmd_coord_weight", 1.0)),
        )
        af_cfg = None
        if op_cfg.anyflow_cotrain_weight > 0:
            af_cfg = _anyflow_cfg_from_dict(cfg, enable_chiral=False)
        mf_loss = CrystalAnyFlowOnPolicyLoss(
            op_cfg,
            teacher_net=teacher_net,
            fake_score_net=fake_score_net,
            real_score_net=real_score_net if real_score_net is not None else teacher_net,
            anyflow_cfg=af_cfg,
        )
        if rank == 0:
            logger.info(
                "AnyFlow on-policy DMD: steps=%s mode=%s n_jumps=%d detach=%s ckpt=%s "
                "dmd_w=%.3f cotrain_w=%.3f nfe=%d jvp=%s periodic=%s disc_ratio=%d "
                "real_fp32=%s fake_lr=%.2e score_flowmap=%s score_use_nft_init=%s "
                "dmd_real_from=%s fake_from=%s gen_start=%d resume_ema=%s",
                op_cfg.student_steps_list,
                op_cfg.rollout_mode,
                op_cfg.n_jumps,
                op_cfg.detach_between_jumps,
                op_cfg.checkpoint_jumps,
                op_cfg.dmd_weight,
                op_cfg.anyflow_cotrain_weight,
                int(cfg.get("anyflow_nfe_steps", 0)),
                str(cfg.get("anyflow_jvp_source", "student")),
                op_cfg.periodic_coord,
                op_cfg.discriminator_update_ratio,
                op_cfg.real_score_fp32,
                float(cfg.get("fake_score_lr", cfg["lr"])),
                str(bool(cfg.get("score_flowmap", False))),
                str(bool(cfg.get("score_use_nft_init", True))),
                str(cfg.get("dmd_real_from", "teacher")),
                str(cfg.get("fake_from", cfg.get("dmd_real_from", "teacher"))),
                int(cfg.get("dmd_generator_start_step", 0)),
                str(bool(cfg.get("resume_student_from_ema", False))),
            )
    else:
        mf_loss = CrystalMeanFlowLoss(loss_cfg)

    ema_state: dict[str, torch.Tensor] = {}
    raw_net = net.module if isinstance(net, DDP) else net

    resume = cfg.get("resume_from")
    start_epoch = 0
    global_step = 0
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        sd = dict(payload["net_state_dict"])
        # Drop legacy output-correction head keys (old architecture).  The new
        # AnyFlow time-embedding uses embed_timestep.delta.* keys; old
        # _output_head.* / _delta_head.* / embed_deltat.* are incompatible.
        for k in list(sd.keys()):
            if (
                k.startswith("_output_head.")
                or k.startswith("_delta_head.")
                or k.startswith("embed_deltat.")
            ):
                del sd[k]
        incompatible = raw_net.load_state_dict(sd, strict=False)
        if rank == 0 and (incompatible.missing_keys or incompatible.unexpected_keys):
            logger.info(
                "resume state_dict compat: missing=%d unexpected=%d",
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )
        if chiral is not None and payload.get("chiral_state_dict"):
            chiral.load_state_dict(payload["chiral_state_dict"], strict=True)
        if fake_score_net is not None and payload.get("fake_score_state_dict"):
            fake_raw = fake_score_net.module if isinstance(fake_score_net, DDP) else fake_score_net
            inc_f = fake_raw.load_state_dict(payload["fake_score_state_dict"], strict=False)
            if rank == 0:
                logger.info(
                    "resume fake_score: missing=%d unexpected=%d",
                    len(inc_f.missing_keys),
                    len(inc_f.unexpected_keys),
                )
        if payload.get("ema_state_dict"):
            ema_state = {}
            for k, v in payload["ema_state_dict"].items():
                if is_time_mix_buffer_key(k) or not torch.is_tensor(v):
                    continue
                ema_state[k] = v.to(device=device, copy=True)
        meta = payload.get("meta") or {}
        global_step = int(meta.get("step", 0))
        start_epoch = int(meta.get("epoch", -1)) + 1
        spe = max(int(cfg.get("steps_per_epoch", 1000)), 1)
        if start_epoch <= 0 and global_step > 0:
            start_epoch = global_step // spe
        if rank == 0:
            logger.info("Resumed from %s step=%d epoch=%d", resume, global_step, start_epoch)
        if bool(cfg.get("reset_step_on_resume", False)):
            global_step = 0
            start_epoch = 0
            if rank == 0:
                logger.info("reset_step_on_resume: counters set to step=0 epoch=0")
        # Optional: wipe polluted Δt embedding (e.g. after failed dt-gating run).
        if bool(cfg.get("reset_embed_deltat_on_resume", False)):
            # Re-init the delta embedder to match the base (so u = v at init).
            proxy = getattr(raw_net, "_time_proxy", None)
            if proxy is not None and hasattr(proxy, "delta"):
                proxy.delta.load_state_dict(proxy.base.state_dict())
                if ema_state:
                    for k in list(ema_state.keys()):
                        if "embed_timestep.delta" in k and torch.is_tensor(ema_state[k]):
                            base_key = k.replace("embed_timestep.delta", "embed_timestep.base")
                            if base_key in ema_state:
                                ema_state[k] = ema_state[base_key].clone()
                if rank == 0:
                    logger.info("reset_embed_deltat_on_resume: reinit delta embedder (+EMA)")
        # Zero inherited 16-path interval_proj before opening dt_min=0, otherwise
        # the first-jump residual fires on every 8-grid jump and packing collapses.
        if bool(cfg.get("reset_interval_residual_on_resume", False)):
            proxy = get_time_proxy(raw_net)
            if proxy is not None:
                proj = getattr(proxy, "interval_proj", None)
                if proj is not None and getattr(proj, "weight", None) is not None:
                    nn.init.zeros_(proj.weight)
                mlp = getattr(proxy, "interval_mlp", None)
                last = mlp[-1] if mlp is not None else None
                if last is not None and getattr(last, "weight", None) is not None:
                    nn.init.zeros_(last.weight)
                if last is not None and getattr(last, "bias", None) is not None:
                    nn.init.zeros_(last.bias)
                if ema_state:
                    live_sd = raw_net.state_dict()
                    for k in list(ema_state.keys()):
                        if (
                            ("interval_proj" in k or "interval_mlp" in k)
                            and k in live_sd
                            and torch.is_tensor(ema_state[k])
                        ):
                            ema_state[k] = live_sd[k].detach().clone()
                if rank == 0:
                    logger.info(
                        "reset_interval_residual_on_resume: zeroed interval_proj and MLP last layer"
                    )

        if bool(cfg.get("init_nfe8_lora_from_shared", False)):
            from crystal_nft.meanflow.tune import init_nfe8_lora_from_shared

            n_copied = init_nfe8_lora_from_shared(raw_net)
            if ema_state:
                live_sd = raw_net.state_dict()
                for k, v in live_sd.items():
                    if ("lora8_A" in k or "lora8_B" in k) and torch.is_tensor(v):
                        ema_state[k] = v.detach().clone()
            if rank == 0:
                logger.info(
                    "init_nfe8_lora_from_shared: copied %d LoRA modules into lora8 (+EMA)",
                    n_copied,
                )

        if bool(cfg.get("resume_student_from_ema", False)) and ema_state:
            from crystal_nft.meanflow.net import load_ema_preserving_time_mix

            load_ema_preserving_time_mix(
                raw_net,
                {
                    "ema_state_dict": ema_state,
                    "net_state_dict": sd,
                    "meta": meta,
                },
            )
            if rank == 0:
                logger.info(
                    "resume_student_from_ema: copied EMA weights into the live student"
                )

        if bool(cfg.get("reset_step_on_resume", False)):
            global_step = 0
            start_epoch = 0
            if rank == 0:
                logger.info("reset_step_on_resume: global_step=0 start_epoch=0")

    if (
        loss_type == "sample_align"
        and str(cfg.get("align_from", "teacher")).lower() == "student"
    ):
        frozen = copy.deepcopy(raw_net).to(device)
        frozen.eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
        set_nfe8_lora_mix(frozen, 0.0)
        set_time_mix(frozen, interval=0.0)
        align_net = frozen
        if isinstance(mf_loss, CrystalSampleAlignLoss):
            mf_loss.set_align_net(frozen)
        if rank == 0:
            logger.info(
                "align_from=student: froze MeanFlow copy after resume as interval teacher "
                "(nfe8_mix=0 interval=0)"
            )

    # Build optimizer AFTER resume/load so param references match the live modules.
    optimizer = _build_optimizer(
        net,
        float(cfg["lr"]),
        float(cfg["weight_decay"]),
        embed_deltat_lr_mult=float(cfg.get("embed_deltat_lr_mult", 1.0)),
        stereo_lr_mult=float(cfg.get("stereo_lr_mult", 1.0)),
        betas=(
            float(cfg.get("adam_beta1", 0.9)),
            float(cfg.get("adam_beta2", 0.95)),
        ),
    )
    fake_score_opt = None
    if fake_score_net is not None:
        fake_params = [p for p in fake_score_net.parameters() if p.requires_grad]
        fake_score_opt = torch.optim.AdamW(
            fake_params,
            lr=float(cfg.get("fake_score_lr", cfg["lr"])),
            weight_decay=float(cfg["weight_decay"]),
            betas=(
                float(cfg.get("fake_score_beta1", cfg.get("adam_beta1", 0.0))),
                float(cfg.get("fake_score_beta2", cfg.get("adam_beta2", 0.999))),
            ),
            eps=1e-8,
        )
    if rank == 0:
        logger.info(
            "Optimizer groups=%d delta_params=%d delta_lr_mult=%.1f fake_score_opt=%s",
            len(optimizer.param_groups),
            len(getattr(optimizer, "_mf_delta_params", []) or []),
            float(getattr(optimizer, "_mf_delta_lr_mult", 1.0)),
            fake_score_opt is not None,
        )

    gate0, res0, ep0 = time_mix_from_cfg(cfg, global_step)
    set_time_mix(
        raw_net,
        gate=gate0,
        residual=res0,
        endpoint=ep0,
        interval=float(cfg.get("interval_embed_mix", 0.0) or 0.0),
        interval_dt_min=float(cfg.get("interval_dt_min", 0.0) or 0.0),
        interval_r_max=float(cfg.get("interval_r_max", 1.0)),
        interval_r_min=float(cfg.get("interval_r_min", 0.0) or 0.0),
    )
    if rank == 0:
        logger.info(
            "Time mix init step=%d gate=%.4f residual=%.4f endpoint=%.4f interval=%.4f dt_min=%.3f r_max=%.3f r_min=%.3f",
            global_step,
            gate0,
            res0,
            ep0,
            float(cfg.get("interval_embed_mix", 0.0) or 0.0),
            float(cfg.get("interval_dt_min", 0.0) or 0.0),
            float(cfg.get("interval_r_max", 1.0)),
            float(cfg.get("interval_r_min", 0.0) or 0.0),
        )
    set_nfe8_lora_mix(raw_net, float(cfg.get("nfe8_lora_mix", 0.0) or 0.0))
    if rank == 0:
        logger.info(
            "nfe8_lora mix=%.3f train=%s",
            float(cfg.get("nfe8_lora_mix", 0.0) or 0.0),
            bool(cfg.get("train_nfe8_lora", False)),
        )

    if args.smoke:
        # Resume can leave start_epoch >= smoke num_epochs (empty loop). Force one short epoch.
        cfg["steps_per_epoch"] = 4
        cfg["num_epochs"] = start_epoch + 1
        cfg["num_workers"] = 0
        cfg["save_every_steps"] = 4
        cfg["log_every"] = 1
        if rank == 0:
            logger.info(
                "Smoke overrides: epochs=%d steps/epoch=%d (start_epoch=%d global_step=%d)",
                cfg["num_epochs"],
                cfg["steps_per_epoch"],
                start_epoch,
                global_step,
            )

    train_ds = build_train_dataset(
        clari_data_dir,
        split_opts=CSD_TRAIN_SPLIT_OPTS,
        mcf_cache=cfg.get("mcf_cache") if cfg.get("use_mcf", True) else None,
        mcf_prob=float(cfg.get("mcf_sample_prob", 0.25)),
    )
    if rank == 0:
        logger.info("Train dataset size (index space): %d", len(train_ds))
    # Optional: restrict training to crystals that actually carry stereocentres.
    # Only 27.4% do (178995/653761), so by default ~73% of every batch spends a
    # full forward/backward contributing zero chirality gradient. Restricting
    # concentrates the signal ~3.6x per step while still leaving 179k distinct
    # molecules, so this stays in the generalisation regime.
    _idx_json = str(cfg.get("train_index_json", "") or "")
    if _idx_json:
        import json as _json

        from torch.utils.data import Subset

        _idx = _json.loads(Path(_idx_json).read_text())["indices"]
        train_ds = Subset(train_ds, _idx)
        if rank == 0:
            logger.info(
                "restricted train set to %d chiral crystals via %s", len(train_ds), _idx_json
            )

    # Optional (off-trunk): replace every molecule in the training target by the
    # rigidly-fitted ETKDG conformer, so the model learns packings for the SAME
    # conformers it is handed at inference. See crystal_nft/rigid/dataset.py.
    _rigid_cache = str(cfg.get("rigid_target_cache", "") or "")
    if _rigid_cache:
        from crystal_nft.rigid.dataset import RigidTargetDataset

        train_ds = RigidTargetDataset(train_ds, _rigid_cache,
                                      logger=logger if rank == 0 else None)

    clari_amd = _load_clari_baseline_amd(str(cfg.get("baseline_clari_metrics_json", "")))
    import torch_geometric as pyg
    from clari.datamodules.csd import CrystalDataset

    val_root = clari_data_dir / "csd"
    val_crystals, val_slices, _ = pyg.io.fs.torch_load(val_root / "val.pt")
    val_crystals = pyg.data.Data.from_dict(val_crystals)
    val_ds = CrystalDataset(val_crystals, val_slices, **CSD_TRAIN_SPLIT_OPTS["val"])
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("val_quick_batch_size", 4)),
        shuffle=False,
        num_workers=0,
        collate_fn=interface.collate_fn,
    )
    baseline_eval_every = int(cfg.get("baseline_eval_every_steps", 0))
    keep_last_ckpts = int(cfg.get("keep_last_n_step_ckpts", 0))
    save_every_n_epochs = int(cfg.get("save_every_n_epochs", 0))
    eval_every_n_epochs = int(cfg.get("eval_every_n_epochs", 0))
    eval_at_epochs = [int(x) for x in (cfg.get("eval_at_epochs") or [])]
    eval_after_train = bool(cfg.get("eval_after_train", True))
    keep_last_epoch_ckpts = int(cfg.get("keep_last_n_epoch_ckpts", 5))
    save_every_steps = int(cfg.get("save_every_steps", 0))
    venv_python = Path(sys.executable)

    sampler = (
        DistributedSampler(train_ds, shuffle=True, seed=int(cfg["seed"])) if distributed else None
    )
    nw = int(cfg["num_workers"])
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        num_workers=nw,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        collate_fn=interface.collate_fn,
        persistent_workers=nw > 0,
        prefetch_factor=int(cfg.get("prefetch_factor", 4)) if nw > 0 else None,
    )

    save_dir = Path(cfg["save_dir"])
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "config_resolved.json").write_text(json.dumps(cfg, indent=2))

    if enable_stereo:
        # Fail fast on a dead stereo path.  A silent no-op here looks *exactly*
        # like a healthy run -- loss falls, GPUs are busy -- and simply trains
        # cont3 again.  Two bugs have already produced that: a CUDA tensor
        # reaching `Tensor.numpy()`, and the CIP labeller rejecting aromatics.
        probe = next(iter(loader))
        probe_spec = build_batch_stereo(probe[1].to(device), device=device)
        n_probe = 0 if probe_spec is None else probe_spec.n_centers
        if rank == 0:
            logger.info(
                "stereo self-check: %d stereocentres in the first batch "
                "(pair_edges=%s, mirror_p=%.2f, loss_weight=%.3g, gain=%.1f, r_max=%.2f)",
                n_probe,
                stereo_pair_edges,
                stereo_mirror_p,
                float(cfg.get("stereo_loss_weight", 0.0)),
                float(cfg.get("stereo_gain", 1.0)),
                stereo_hinge_r_max,
            )
        if n_probe == 0 and not bool(cfg.get("allow_empty_stereo", False)):
            raise RuntimeError(
                "enable_stereo=true but the first batch yielded no stereocentres. "
                "~35% of CSD training crystals have one, so this is a broken "
                "perception path, not an unlucky batch. Check the "
                "'stereo perception failed' warning above; set "
                "allow_empty_stereo=true to override."
            )
        del probe, probe_spec

    t0 = time.time()
    max_steps = int(cfg["max_steps"]) if int(cfg.get("max_steps", 0)) > 0 else None

    for epoch in range(start_epoch, int(cfg["num_epochs"])):
        if sampler is not None:
            sampler.set_epoch(epoch)
        loader_iter = iter(loader)
        for step_in_epoch in range(int(cfg["steps_per_epoch"])):
            if max_steps is not None and global_step >= max_steps:
                break

            use_fm = global_step < int(cfg["fm_warmup_steps"])
            lr = _lr_scale(global_step, int(cfg["lr_warmup_steps"]), float(cfg["lr"]))
            for pg in optimizer.param_groups:
                pg["lr"] = lr * float(pg.get("lr_mult", 1.0))
            delta_opt = getattr(optimizer, "_mf_delta_opt", None)
            if delta_opt is not None:
                for pg in delta_opt.param_groups:
                    pg["lr"] = lr * float(pg.get("lr_mult", 1.0))
            if fake_score_opt is not None:
                fake_lr = _lr_scale(
                    global_step,
                    int(cfg["lr_warmup_steps"]),
                    float(cfg.get("fake_score_lr", cfg["lr"])),
                )
                for pg in fake_score_opt.param_groups:
                    pg["lr"] = fake_lr

            gate_now, res_now, ep_now = time_mix_from_cfg(cfg, global_step)
            set_time_mix(
                raw_net,
                gate=gate_now,
                residual=res_now,
                endpoint=ep_now,
                interval=float(cfg.get("interval_embed_mix", 0.0) or 0.0),
                interval_dt_min=float(cfg.get("interval_dt_min", 0.0) or 0.0),
                interval_r_max=float(cfg.get("interval_r_max", 1.0)),
                interval_r_min=float(cfg.get("interval_r_min", 0.0) or 0.0),
            )
            set_nfe8_lora_mix(raw_net, float(cfg.get("nfe8_lora_mix", 0.0) or 0.0))

            accum = int(cfg["grad_accum"])
            loss_sum = 0.0
            out = None
            if loss_type == "anyflow_onpolicy":
                gen_start = int(cfg.get("dmd_generator_start_step", 0))
                do_gen = global_step >= gen_start
                disc_ratio = int(
                    getattr(getattr(mf_loss, "cfg", None), "discriminator_update_ratio", 1)
                    or 1
                )
                if not do_gen:
                    disc_ratio = max(
                        disc_ratio, int(cfg.get("dmd_warmup_disc_ratio", 2))
                    )
                out = {
                    "loss": torch.zeros((), device=device),
                    "loss_dmd": torch.zeros((), device=device),
                    "loss_anyflow": torch.zeros((), device=device),
                    "loss_disc": torch.zeros((), device=device),
                    "student_steps": torch.zeros((), device=device),
                    "grad_timestep": torch.zeros((), device=device),
                }
                gnorms = []
                if do_gen:
                    net.train()
                    for micro in range(accum):
                        try:
                            batch = next(loader_iter)
                        except StopIteration:
                            loader_iter = iter(loader)
                            batch = next(loader_iter)
                        C0, C1 = batch
                        C0 = C0.to(device)
                        C1 = C1.to(device)
                        is_last = micro == accum - 1
                        sync_ctx = (
                            net.no_sync()
                            if isinstance(net, DDP) and not is_last
                            else nullcontext()
                        )
                        with sync_ctx:
                            with torch.autocast(
                                device_type="cuda",
                                dtype=torch.bfloat16,
                                enabled=bool(cfg["bf16"]) and device.type == "cuda",
                            ):
                                chiral_bias = make_chiral_bias_for_batch(
                                    chiral, C1, C1.batch_size, device
                                )
                                out = mf_loss.generator_loss(
                                    net, interface, C0, C1, chiral_bias=chiral_bias
                                )
                            loss = out["loss"] / accum
                            loss.backward()
                            loss_sum += float(out["loss"].detach())
                    if distributed:
                        allreduce_grads(raw_net)
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), float(cfg["max_grad_norm"])
                    )
                    for p in getattr(optimizer, "_mf_delta_params", []) or []:
                        gnorms.append(
                            float("nan")
                            if p.grad is None
                            else float(p.grad.detach().float().norm())
                        )
                    _suppress_lora_grads(
                        raw_net, global_step, int(cfg.get("dual_lora_start_step", 0))
                    )
                    optimizer.step()
                    delta_opt = getattr(optimizer, "_mf_delta_opt", None)
                    if delta_opt is not None:
                        delta_opt.step()
                        delta_opt.zero_grad(set_to_none=True)
                    optimizer.zero_grad(set_to_none=True)
                else:
                    net.eval()

                disc_out = None
                for _ in range(max(1, disc_ratio)):
                    if fake_score_opt is None:
                        break
                    for micro in range(accum):
                        try:
                            batch = next(loader_iter)
                        except StopIteration:
                            loader_iter = iter(loader)
                            batch = next(loader_iter)
                        C0, C1 = batch
                        C0 = C0.to(device)
                        C1 = C1.to(device)
                        is_last = micro == accum - 1
                        sync_ctx = (
                            fake_score_net.no_sync()
                            if isinstance(fake_score_net, DDP) and not is_last
                            else nullcontext()
                        )
                        with sync_ctx:
                            with torch.autocast(
                                device_type="cuda",
                                dtype=torch.bfloat16,
                                enabled=bool(cfg["bf16"]) and device.type == "cuda",
                            ):
                                chiral_bias = make_chiral_bias_for_batch(
                                    chiral, C1, C1.batch_size, device
                                )
                                disc_out = mf_loss.discriminator_loss(
                                    net, interface, C0, C1, chiral_bias=chiral_bias
                                )
                            (disc_out["loss"] / accum).backward()
                    torch.nn.utils.clip_grad_norm_(
                        fake_score_net.parameters(), float(cfg["max_grad_norm"])
                    )
                    fake_score_opt.step()
                    fake_score_opt.zero_grad(set_to_none=True)
                net.train()
                if disc_out is not None:
                    out["loss_disc"] = disc_out["loss_disc"]
                    if not do_gen:
                        out["loss"] = disc_out["loss_disc"]
                        loss_sum = float(disc_out["loss"].detach())
            else:
                for micro in range(accum):
                    # Fetch a fresh batch for each micro-step (proper grad accum).
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(loader)
                        batch = next(loader_iter)
                    C0, C1 = batch
                    C0 = C0.to(device)
                    C1 = C1.to(device)
                    if (
                        loss_type == "anyflow"
                        and str(cfg.get("anyflow_x1_source", "data")).lower()
                        in ("teacher", "teacher_heun")
                    ):
                        if align_net is None:
                            raise RuntimeError(
                                "anyflow_x1_source=teacher requires align_net "
                                "(set align_from=raw for Clari-M Heun targets)"
                            )
                        x1_bias = make_chiral_bias_for_batch(
                            chiral, C1, C1.batch_size, device
                        )
                        with torch.no_grad():
                            C1 = C1.replace(
                                x=heun_instantaneous_rollout(
                                    interface,
                                    align_net,
                                    C0,
                                    x_init=C0.x,
                                    num_steps=int(cfg.get("teacher_steps", 50)),
                                    chiral_bias=x1_bias,
                                )
                            )

                    is_last = micro == accum - 1
                    sync_ctx = (
                        net.no_sync()
                        if isinstance(net, DDP) and not is_last
                        else nullcontext()
                    )
                    # CrystAF stereo: CIP tags come from the *reference* crystal
                    # (molecular identity, like bond orders), and stay published
                    # for every DiT forward inside the loss.
                    stereo_spec = (
                        build_batch_stereo(C1, device=device) if enable_stereo else None
                    )
                    if stereo_spec is not None and stereo_mirror_p > 0:
                        # Enantiomer augmentation: mirror the crystal *and* swap
                        # R<->S. Without it the model can memorise graph->handedness
                        # instead of reading the tag, and the conditioning is never
                        # causal.  Inverting the coordinates through the origin at
                        # fixed lattice preserves every interatomic distance, so the
                        # frozen teacher's pair features are unchanged.
                        flip = (
                            torch.rand(C1.x.shape[0], device=device) < stereo_mirror_p
                        )
                        if bool(flip.any()):
                            C1 = C1.replace(x=mirror_coords_in_place_x(C1.x, flip))
                            stereo_spec.atom_tags = swap_rs_tags(
                                stereo_spec.atom_tags, flip
                            )
                            if stereo_spec.label_sign is not None:
                                stereo_spec.label_sign = torch.where(
                                    flip.view(-1, 1),
                                    -stereo_spec.label_sign,
                                    stereo_spec.label_sign,
                                )
                    stereo_state["spec"] = stereo_spec
                    if not enable_stereo:
                        stereo_cond = None
                    elif stereo_spec is not None:
                        stereo_cond = build_stereo_conditioning(
                            stereo_spec,
                            int(C1.x.shape[1] - 3),
                            pair_edges=stereo_pair_edges,
                        )
                    else:
                        # No stereocentre in this batch: publish all-zero tags so the
                        # (masked, no-op) embedding stays in the DDP autograd graph.
                        stereo_cond = torch.zeros(
                            C1.x.shape[0], C1.x.shape[1] - 3, dtype=torch.long, device=device
                        )
                    with sync_ctx, active_stereo_tags(stereo_cond):
                        with torch.autocast(
                            device_type="cuda",
                            dtype=torch.bfloat16,
                            enabled=bool(cfg["bf16"]) and device.type == "cuda",
                        ):
                            chiral_bias = make_chiral_bias_for_batch(
                                chiral, C1, C1.batch_size, device
                            )
                            if use_fm:
                                # Pass the hinge through. Without it this branch
                                # ignores stereo_loss_weight entirely and the run
                                # is a silent no-op (stereo_n=0 in the log).
                                out = interface.fm_supervision_loss(
                                    net,
                                    (C0, C1),
                                    chiral_bias=chiral_bias,
                                    chiral_consistency_fn=(
                                        _chiral_consistency_fn
                                        if enable_stereo
                                        and float(cfg.get("stereo_loss_weight", 0)) > 0
                                        else None
                                    ),
                                    chiral_loss_weight=float(
                                        cfg.get("stereo_loss_weight", 0.0)
                                    ),
                                    low_t_frac=float(cfg.get("stereo_low_t_frac", 0.0)),
                                    low_t_max=float(cfg.get("stereo_low_t_max", 0.25)),
                                    mismatch_frac=float(cfg.get("stereo_mismatch_frac", 0.0)),
                                    mismatch_t_lo=float(
                                        cfg.get("stereo_mismatch_t_lo", 0.35)
                                    ),
                                    mismatch_t_hi=float(
                                        cfg.get("stereo_mismatch_t_hi", 0.85)
                                    ),
                                    chiral_on_flow_map=bool(
                                        cfg.get("stereo_flow_map_hinge", False)
                                    ),
                                    self_state_frac=float(
                                        cfg.get("stereo_self_state_frac", 0.0)
                                    ),
                                    self_state_steps=int(
                                        cfg.get("stereo_self_state_steps", 4)
                                    ),
                                    self_state_denom_min=float(
                                        cfg.get("stereo_self_state_denom_min", 0.2)
                                    ),
                                    self_state_hinge_only=bool(
                                        cfg.get("stereo_self_state_hinge_only", False)
                                    ),
                                    branch_l2_weight=float(
                                        cfg.get("stereo_branch_l2_weight", 0.0)
                                    ),
                                )
                            elif loss_type in ("anyflow", "sample_align"):
                                out = mf_loss(
                                    net,
                                    interface,
                                    C0,
                                    C1,
                                    chiral_bias=chiral_bias,
                                    chirality_fn=_chirality_fn if flip_p > 0 else None,
                                    chiral_consistency_fn=(
                                        _chiral_consistency_fn
                                        if float(cfg.get("chiral_loss_weight", 0)) > 0
                                        or (
                                            enable_stereo
                                            and float(cfg.get("stereo_loss_weight", 0)) > 0
                                        )
                                        else None
                                    ),
                                )
                            else:
                                out = mf_loss(
                                    net,
                                    interface,
                                    C0,
                                    C1,
                                    chiral_bias=chiral_bias,
                                    chirality_fn=_chirality_fn if flip_p > 0 else None,
                                )
                        loss = out["loss"] / accum
                        loss.backward()
                        loss_sum += float(out["loss"].detach())

                if distributed and (
                    loss_type == "sample_align" or bool(out.get("need_grad_allreduce"))
                ):
                    allreduce_grads(raw_net)
                torch.nn.utils.clip_grad_norm_(net.parameters(), float(cfg["max_grad_norm"]))
                gnorms = []
                for p in getattr(optimizer, "_mf_delta_params", []) or []:
                    gnorms.append(float("nan") if p.grad is None else float(p.grad.detach().float().norm()))
                _suppress_lora_grads(
                    raw_net, global_step, int(cfg.get("dual_lora_start_step", 0))
                )
                optimizer.step()
                delta_opt = getattr(optimizer, "_mf_delta_opt", None)
                if delta_opt is not None:
                    delta_opt.step()
                    delta_opt.zero_grad(set_to_none=True)
                optimizer.zero_grad(set_to_none=True)

            skip_ema = (
                loss_type == "anyflow_onpolicy"
                and global_step < int(cfg.get("dmd_generator_start_step", 0))
            )
            if not skip_ema:
                _ema_update(ema_state, raw_net, float(cfg["ema_decay"]))
            global_step += 1

            if rank == 0 and (global_step % int(cfg["log_every"]) == 0) and out is not None:
                elapsed = time.time() - t0
                ed = getattr(raw_net, "embed_deltat", None)
                delta_l2 = (
                    float(ed.proj.weight.detach().float().norm())
                    if ed is not None
                    else float("nan")
                )
                # Mix: delta embedder L2. Dual residual: adapter last-layer L2
                # (zero at init; should grow if the interval path is learning).
                proxy = getattr(raw_net, "_time_proxy", None)
                dual = getattr(raw_net, "_dual_time", None)
                if proxy is not None and hasattr(proxy, "delta"):
                    delta_mlp_l2 = float(
                        proxy.delta.proj.weight.detach().float().norm()
                    )
                elif dual is not None and hasattr(dual, "adapter"):
                    delta_mlp_l2 = float(
                        dual.adapter[-1].weight.detach().float().norm()
                    )
                else:
                    delta_mlp_l2 = float("nan")
                # raw_net, NOT net: this block is rank-0 only, and running a
                # DDP-wrapped module's forward on a single rank desyncs the
                # reducer (find_unused_parameters walks the autograd graph) --
                # observed as a SIGSEGV that killed a run at step 2660.
                follow_rate, follow_n = (
                    _stereo_follow_probe(raw_net, interface, C1, stereo_state.get("spec"))
                    if enable_stereo
                    else (float("nan"), 0)
                )
                logger.info(
                    "[rank0] step=%d epoch=%d loss=%.4f dmd=%.4f disc=%.4f vol=%.4f ldd=%.4f chiral=%.4f "
                    "lat=%.4f coord=%.4f geom=%.4f af=%.4f mse_lat=%.4f mse_coord=%.4f "
                    "nfe=%s k=%s fm=%s gate=%.4f tres=%.4f epmix=%.4f "
                    "delta_l2=%.6f delta_mlp_l2=%.6f delta_gnorm=%s "
                    "stereo_agree=%.4f stereo_n=%d stereo_l2=%.6f follow=%.4f follow_n=%d steps/s=%.2f",
                    global_step,
                    epoch,
                    loss_sum,
                    float(out.get("loss_dmd", 0)),
                    float(out.get("loss_disc", 0)),
                    float(out.get("loss_vol", 0)),
                    float(out.get("loss_ldd", 0)),
                    float(out.get("loss_chiral", 0)),
                    float(out.get("loss_lattice", 0)),
                    float(out.get("loss_coord", 0)),
                    float(out.get("loss_geom", 0)),
                    float(out.get("loss_anyflow", 0)),
                    float(out.get("mse_lat_raw", out.get("loss_lattice", 0))),
                    float(out.get("mse_coord_raw", out.get("loss_coord", 0))),
                    int(float(out.get("student_steps", 0))),
                    int(float(out.get("grad_timestep", -1))),
                    use_fm,
                    gate_now,
                    res_now,
                    ep_now,
                    delta_l2,
                    delta_mlp_l2,
                    gnorms,
                    float(stereo_state.get("agree", float("nan"))),
                    int(stereo_state.get("n", 0) or 0),
                    _stereo_embed_l2(raw_net),
                    follow_rate,
                    follow_n,
                    global_step / max(elapsed, 1e-6),
                )

            if (
                rank == 0
                and save_every_steps > 0
                and global_step % save_every_steps == 0
            ):
                ckpt = save_dir / f"checkpoint-step{global_step}.pt"
                save_meanflow_checkpoint(
                    ckpt,
                    net=raw_net,
                    chiral=chiral,
                    ema_state=ema_state,
                    meta={"step": global_step, "epoch": epoch, "cfg": cfg, "loss_type": loss_type},
                    fake_score_net=fake_score_net,
                )
                logger.info("saved %s", ckpt)
                _prune_step_checkpoints(save_dir, keep_last_ckpts)

            if (
                baseline_eval_every > 0
                and global_step > 0
                and global_step % baseline_eval_every == 0
            ):
                if distributed:
                    dist.barrier()
                if rank == 0:
                    use_fm_eval = global_step < int(cfg["fm_warmup_steps"])
                    val_m = _quick_val_loss(
                        raw_net,
                        interface,
                        mf_loss,
                        val_loader,
                        device,
                        use_fm=use_fm_eval,
                    )
                    _write_baseline_eval(save_dir, global_step, val_m, clari_amd)
                    logger.info(
                        "[rank0] baseline_eval step=%d val_loss=%.4f clari_m_amd_ref=%.2f",
                        global_step,
                        val_m.get("loss", float("nan")),
                        clari_amd,
                    )
                if distributed:
                    dist.barrier()

        if max_steps is not None and global_step >= max_steps:
            break

        epoch_done = epoch + 1
        should_save_epoch = save_every_n_epochs > 0 and epoch_done % save_every_n_epochs == 0
        should_save_ckpt = should_save_epoch or (
            bool(eval_at_epochs) and epoch in eval_at_epochs
        )

        if should_save_ckpt:
            if distributed:
                dist.barrier()
            if rank == 0:
                ckpt = save_dir / f"checkpoint-epoch{epoch}.pt"
                save_meanflow_checkpoint(
                    ckpt,
                    net=raw_net,
                    chiral=chiral,
                    ema_state=ema_state,
                    meta={
                        "step": global_step,
                        "epoch": epoch,
                        "cfg": cfg,
                        "loss_type": loss_type,
                    },
                    fake_score_net=fake_score_net,
                )
                logger.info("epoch %d done -> %s (step=%d)", epoch, ckpt, global_step)
                _prune_epoch_checkpoints(save_dir, keep_last_epoch_ckpts)
            if distributed:
                dist.barrier()

    if rank == 0:
        final = save_dir / "checkpoint-final.pt"
        save_meanflow_checkpoint(
            final,
            net=raw_net,
            chiral=chiral,
            ema_state=ema_state,
            meta={"step": global_step, "cfg": cfg},
            fake_score_net=fake_score_net,
        )
        logger.info("Finished %d steps in %.1fs -> %s", global_step, time.time() - t0, final)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0 and eval_after_train and eval_at_epochs:
        logger.info("Starting post-train eval for epochs %s", eval_at_epochs)
        _run_post_train_evals(cfg=cfg, repo_root=_REPO_ROOT, venv_python=venv_python)


if __name__ == "__main__":
    main()
