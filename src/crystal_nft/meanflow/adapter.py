"""Load MeanFlow-wrapped Clari models and run NFT / sampling."""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from crystal_nft.meanflow.chirality import (
    ChiralConditioning,
    chiral_bias_from_descriptor,
    chiral_descriptor_from_smiles,
    random_enantiomer_flip,
)
from crystal_nft.meanflow.interface import MeanFlowCrystalInterface
from crystal_nft.meanflow.net import (
    InstantaneousDiT,
    MeanFlowDiTWrapper,
    live_time_mix_meta,
    load_ema_preserving_time_mix,
    restore_time_mix_buffers,
    set_time_mix,
    wrap_dit_for_meanflow,
)
from crystal_nft.meanflow.sampler import MeanFlowCrystalSampler
from crystal_nft.meanflow.velocity import (
    clari_induced_velocity,
    flow_map_dudr,
    sample_nft_times,
)
from crystal_nft.nft.loss import (
    combine_nft_and_kl,
    kl_velocity_loss,
    nft_reconstruction_loss,
)

logger = logging.getLogger(__name__)


def attach_stereo_tokens(
    mf_net, *, device=None, pair_edges: bool = True, gain: float = 1.0,
    node_mod: bool = False,
    cond_path: bool = False,
    chiral_branch: bool = False,
    chiral_branch_scale: float = 1.0,
    chiral_branch_gate: float = 0.0,
    chiral_branch_endpoint_geom: bool = False,
    chiral_branch_bond_preserving: bool = False,
    chiral_branch_t_max: float = 1.0,
    chiral_branch_t_min: float = 0.0,
    head_t_min: float = 0.0,
    head_t_max: float = 1.0,
):
    """Attach the zero-init stereo conditioning heads to a wrapped student.

    Two heads, both zero-init so step 0 is bit-identical to the un-conditioned
    model: per-atom CIP tags on ``embed_feats`` and LoQI auxiliary edges on
    ``embed_bonds``.  Idempotent — re-binds hooks if already attached.
    """
    from crystal_nft.meanflow.net import (
        StereoCondEmbedding,
        StereoPairEmbedding,
        StereoTokenEmbedding,
    )
    from crystal_nft.meanflow.stereo import N_STEREO_EDGES, N_STEREO_TAGS

    dit = mf_net.dit
    existing = getattr(mf_net, "_stereo_tokens", None)
    if existing is not None:
        existing.bind(dit)
    else:
        dim = int(getattr(dit, "dim", 0) or 0) or int(
            getattr(dit.embed_feats, "out_features", 0) or 0
        )
        if dim <= 0:
            raise RuntimeError("Cannot infer DiT token dim for StereoTokenEmbedding")
        existing = StereoTokenEmbedding(dim, n_tags=N_STEREO_TAGS, gain=gain)
        if device is not None:
            existing = existing.to(device)
        mf_net.attach_stereo_tokens(existing)

    if pair_edges:
        pair_mod = getattr(mf_net, "_stereo_pairs", None)
        if pair_mod is not None:
            pair_mod.bind(dit)
        else:
            dim_pair = int(getattr(dit, "dim_pair", 0) or 0) or int(
                getattr(dit.embed_bonds, "embedding_dim", 0) or 0
            )
            if dim_pair <= 0:
                raise RuntimeError("Cannot infer DiT pair dim for StereoPairEmbedding")
            pair_mod = StereoPairEmbedding(dim_pair, n_types=N_STEREO_EDGES, gain=gain)
            if device is not None:
                pair_mod = pair_mod.to(device)
            mf_net.attach_stereo_pairs(pair_mod)

    if node_mod and getattr(mf_net, "_stereo_pair_mod", None) is None and pair_edges:
        from crystal_nft.meanflow.net import StereoPairMod

        dp = int(getattr(mf_net.dit, "dim_pair", 0) or 0)
        if dp > 0:
            mf_net.attach_stereo_pair_mod(
                StereoPairMod(dp, gain=gain, t_min=float(head_t_min),
                              t_max=float(head_t_max)).to(device)
            )
    if node_mod and getattr(mf_net, "_stereo_node_mod", None) is None:
        from crystal_nft.meanflow.net import StereoNodeMod

        dim = int(getattr(mf_net.dit, "dim", 0) or 0)
        if dim > 0:
            mf_net.attach_stereo_node_mod(
                StereoNodeMod(dim, gain=gain, t_min=float(head_t_min),
                              t_max=float(head_t_max)).to(device)
            )
    if chiral_branch and getattr(mf_net, "_stereo_chiral_branch", None) is None:
        from crystal_nft.meanflow.net import StereoChiralBranch

        br = StereoChiralBranch(
            scale=float(chiral_branch_scale), gate_margin=float(chiral_branch_gate),
            use_endpoint_geometry=bool(chiral_branch_endpoint_geom),
            bond_preserving=bool(chiral_branch_bond_preserving),
            t_max=float(chiral_branch_t_max),
            t_min=float(chiral_branch_t_min),
        )
        if device is not None:
            br = br.to(device)
        mf_net.attach_stereo_chiral_branch(br)
    if cond_path and getattr(mf_net, "_stereo_cond", None) is None:
        dim_cond = int(getattr(dit, "dim_cond", 0) or 0)
        if dim_cond <= 0:
            raise RuntimeError("Cannot infer dim_cond for StereoCondEmbedding")
        cond_mod = StereoCondEmbedding(dim_cond, gain=gain)
        if device is not None:
            cond_mod = cond_mod.to(device)
        mf_net.attach_stereo_cond(cond_mod)
    return existing


def load_frozen_clari_teacher(
    checkpoint: str,
    *,
    device: str | torch.device = "cuda",
    use_ema: bool = True,
) -> nn.Module:
    """Load an unwrapped Clari DiT and freeze it (AnyFlow teacher)."""
    from clari.inference.inputs import resolve_checkpoint
    from clari.inference.sample import load_lit, resolve_device

    device = resolve_device(device)
    ckpt = resolve_checkpoint(checkpoint)
    lit = load_lit(ckpt, device, use_ema=use_ema, n_steps=50, compile=False)
    teacher = lit.net
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def load_meanflow_bundle(
    checkpoint: str = "clari-h",
    *,
    device: str | torch.device = "cuda",
    use_ema: bool = True,
    meanflow_steps: int = 4,
    enable_chiral: bool = True,
    enable_stereo: bool = False,
    stereo_pair_edges: bool = True,
    stereo_gain: float = 1.0,
    stereo_cond_path: bool = False,
    stereo_node_mod: bool = False,
    stereo_chiral_branch: bool = False,
    stereo_chiral_branch_scale: float = 1.0,
    stereo_chiral_branch_gate: float = 0.0,
    stereo_chiral_branch_endpoint_geom: bool = False,
    stereo_chiral_branch_bond_preserving: bool = False,
    stereo_chiral_branch_t_max: float = 1.0,
    stereo_chiral_branch_t_min: float = 0.0,
    stereo_head_t_min: float = 0.0,
    stereo_head_t_max: float = 1.0,
    compile_model: bool = False,
    dit_preset: str | None = None,
    init_dit_from_checkpoint: bool = True,
    load_teacher: bool = False,
    load_fake_score: bool = False,
    gate_value: float = 0.25,
    conditioning_mode: str = "mix",
    dual_time_feature_mode: str = "both",
    dual_gate_value: float = 1.0,
    time_parameterization: str = "legacy",
    score_flowmap: bool = False,
    score_conditioning_mode: str = "mix",
    score_use_nft_init: bool = True,
    dmd_real_from: str = "teacher",
    fake_from: str = "teacher",
    align_from: str = "teacher",
    keep_nft_copies: bool = True,
    clari_nft_init: str | Path | None = None,
):
    """
    Load Clari LitDiT, wrap DiT for flow map u(x,r,t), and attach MeanFlow sampler.

    Returns trainable net, old/ref copies, interface, and optional chiral head.
    When ``load_teacher=True``, also returns a frozen unwrapped Clari DiT.
    """
    from clari.inference.inputs import resolve_checkpoint
    from clari.inference.sample import load_lit, resolve_device

    device = resolve_device(device)
    ckpt = resolve_checkpoint(checkpoint)
    lit = load_lit(ckpt, device, use_ema=use_ema, n_steps=50, compile=compile_model)
    lit.train()
    dmd_real_from = str(dmd_real_from or "teacher").lower()
    fake_from = str(fake_from or "teacher").lower()
    align_from = str(align_from or "teacher").lower()
    need_raw = (
        (load_teacher or load_fake_score)
        and (
            (not bool(score_use_nft_init))
            or dmd_real_from == "raw"
            or fake_from == "raw"
            or align_from == "raw"
        )
    )
    raw_score_dit: nn.Module | None = None
    if need_raw:
        raw_score_dit = copy.deepcopy(lit.net)
    if clari_nft_init:
        from crystal_nft.adapters.clari_adapter import load_clari_nft_state

        missing, unexpected = load_clari_nft_state(lit.net, clari_nft_init)
        logger.info(
            "Loaded Clari NFT init %s (missing=%d unexpected=%d)",
            clari_nft_init,
            len(missing),
            len(unexpected),
        )

    teacher_net: nn.Module | None = None
    fake_score_net: nn.Module | None = None
    real_score_net: nn.Module | None = None
    align_net: nn.Module | None = None
    nft_backbone = lit.net
    raw_backbone = raw_score_dit or lit.net
    teacher_src = nft_backbone if bool(score_use_nft_init) else raw_backbone
    fake_src = raw_backbone if fake_from == "raw" else teacher_src
    real_src = raw_backbone if dmd_real_from == "raw" else teacher_src
    align_src = raw_backbone if align_from == "raw" else teacher_src
    if align_from == "student":
        align_src = None

    def _make_score_net(src: nn.Module, *, train: bool) -> nn.Module:
        dit = copy.deepcopy(src)
        if score_flowmap:
            net = wrap_dit_for_meanflow(
                dit,
                gate_value=gate_value,
                conditioning_mode=score_conditioning_mode,
                time_parameterization=time_parameterization,
            )
        else:
            net = InstantaneousDiT(dit)
        if train:
            net.train()
            for p in net.parameters():
                p.requires_grad_(True)
        else:
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
        return net

    if load_teacher or load_fake_score:
        logger.info(
            "Score nets: anyflow_teacher=%s align=%s dmd_real=%s fake=%s score_flowmap=%s",
            "nft-init" if teacher_src is nft_backbone else "raw Clari-M",
            (
                "student MeanFlow (post-resume clone)"
                if align_from == "student"
                else (
                    "raw Clari-M" if align_from == "raw" else (
                        "nft-init" if teacher_src is nft_backbone else "raw Clari-M"
                    )
                )
            ),
            "raw Clari-M" if real_src is raw_backbone and dmd_real_from == "raw" else (
                "nft-init" if teacher_src is nft_backbone else "raw Clari-M"
            ),
            "raw Clari-M" if fake_src is raw_backbone and fake_from == "raw" else (
                "nft-init" if teacher_src is nft_backbone else "raw Clari-M"
            ),
            str(bool(score_flowmap)),
        )

    if load_teacher:
        teacher_net = _make_score_net(teacher_src, train=False)
        if align_from == "student":
            # Frozen dual-time student is cloned after resume in the trainer.
            align_net = None
        elif align_src is teacher_src:
            align_net = teacher_net
        else:
            # Heun/Euler targets must be instantaneous (r=t), even if the
            # AnyFlow teacher is a flow-map wrapper.
            align_net = InstantaneousDiT(copy.deepcopy(align_src))
            align_net.eval()
            for p in align_net.parameters():
                p.requires_grad_(False)
    if load_fake_score:
        fake_score_net = _make_score_net(fake_src, train=True)
        if dmd_real_from == "raw" and real_src is not teacher_src:
            real_score_net = _make_score_net(real_src, train=False)
        else:
            real_score_net = teacher_net
    raw_score_dit = None

    if dit_preset:
        from crystal_nft.meanflow.backbone import build_dit_from_preset, transfer_clari_dit_weights

        dit = build_dit_from_preset(dit_preset)
        if init_dit_from_checkpoint:
            transfer_clari_dit_weights(dit, lit.net)
        mf_net = wrap_dit_for_meanflow(
            dit,
            gate_value=gate_value,
            conditioning_mode=conditioning_mode,
            dual_time_feature_mode=dual_time_feature_mode,
            dual_gate_value=dual_gate_value,
            time_parameterization=time_parameterization,
        )
    else:
        mf_net = wrap_dit_for_meanflow(
            lit.net,
            gate_value=gate_value,
            conditioning_mode=conditioning_mode,
            dual_time_feature_mode=dual_time_feature_mode,
            dual_gate_value=dual_gate_value,
            time_parameterization=time_parameterization,
        )
    mf_net = mf_net.to(device)
    chiral: ChiralConditioning | None = None
    if enable_chiral:
        # dim_cond lives on the inner DiT after wrap (do not eagerly eval fallbacks).
        dim_cond = int(
            getattr(mf_net.dit, "dim_cond", None)
            or getattr(lit.net, "dim_cond", None)
            or 0
        )
        if dim_cond <= 0:
            raise RuntimeError("Cannot infer dim_cond for ChiralConditioning")
        chiral = ChiralConditioning(dim_cond).to(device)
        mf_net.attach_chiral_bias(chiral)

    if enable_stereo:
        attach_stereo_tokens(
            mf_net, device=device, pair_edges=stereo_pair_edges, gain=stereo_gain,
            node_mod=stereo_node_mod,
            cond_path=stereo_cond_path,
            chiral_branch=stereo_chiral_branch,
            chiral_branch_scale=stereo_chiral_branch_scale,
            chiral_branch_gate=stereo_chiral_branch_gate,
            chiral_branch_endpoint_geom=stereo_chiral_branch_endpoint_geom,
            chiral_branch_bond_preserving=stereo_chiral_branch_bond_preserving,
            chiral_branch_t_max=stereo_chiral_branch_t_max,
            chiral_branch_t_min=stereo_chiral_branch_t_min,
            head_t_min=stereo_head_t_min,
            head_t_max=stereo_head_t_max,
        )

    interface = MeanFlowCrystalInterface(prior=lit.interface.prior, train_tdist=lit.interface.train_tdist)
    sampler = MeanFlowCrystalSampler(
        num_steps=meanflow_steps,
        mode=str(os.environ.get("MEANFLOW_SAMPLER_MODE", "interval")).lower(),
        rho=float(os.environ.get("MEANFLOW_INTERVAL_RHO", "1")),
        schedule=str(os.environ.get("MEANFLOW_INTERVAL_SCHEDULE", "power")).lower(),
    )

    old_net = None
    ref_net = None
    if keep_nft_copies:
        old_net = copy.deepcopy(mf_net).eval()
        ref_net = copy.deepcopy(mf_net).eval()
        for p in old_net.parameters():
            p.requires_grad_(False)
        for p in ref_net.parameters():
            p.requires_grad_(False)

    return {
        "lit": lit,
        "net": mf_net,
        "old_net": old_net,
        "ref_net": ref_net,
        "teacher_net": teacher_net,
        "fake_score_net": fake_score_net,
        "real_score_net": real_score_net,
        "align_net": align_net,
        "interface": interface,
        "sampler": sampler,
        "chiral": chiral,
        "device": device,
        "checkpoint": ckpt,
    }


def meanflow_distill_train_step(
    interface,
    net,
    crystals,
    *,
    smiles: str | None = None,
    chiral=None,
    nfe_steps: int = 8,
    grid_rho: float = 0.75,
    diffusion_ratio: float = 0.0,
    consistency_ratio: float = 0.0,
    rank: int = 0,
    world_size: int = 1,
) -> dict:
    """One flow-map regression step onto UMA-relaxed targets (distillation).

    Supervised, not RL: ``crystals`` are already the *relaxed* endpoints, and
    the student is regressed onto them. No reward, no old/ref policy, no
    advantage -- which is what makes this distillation rather than NFT.

    Under Clari's linear path ``x_s = (1-s) x_0 + s x_1`` the exact flow map is
    constant in both times, ``U(x_r, r, t) = x_1 - x_0``, so the same target is
    valid at every ``(r, t)``. Times are drawn from the same grid the NFT arms
    use so the student sees the jump lengths it will be sampled on.
    """
    device = next(net.parameters()).device
    C0, C1 = interface.collate_fn(list(crystals))
    C0 = C0.to(device)
    C1 = C1.to(device)
    chiral_bias = _chiral_bias_for_batch(chiral, smiles, C1.batch_size, device)

    x0, x1 = C0.x, C1.x
    time_r, time_t, _ = sample_nft_times(
        C1.batch_size,
        device,
        diffusion_ratio=diffusion_ratio,
        consistency_ratio=consistency_ratio,
        nfe_steps=nfe_steps,
        rho=grid_rho,
        rank=rank,
        world_size=world_size,
    )
    xr = interface.sample_xt(x0, x1, time_r)

    xsc = None
    if net.self_cond:
        with torch.no_grad():
            nsc = max(1, C0.batch_size // 2)
            fsc = C0.subset(slice(0, nsc))
            out = interface.pred(
                net=net, xt=xr[:nsc], xsc=None, t=time_t[:nsc], r=time_r[:nsc],
                f=fsc, chiral_bias=chiral_bias[:nsc] if chiral_bias is not None else None,
            )
            xsc = torch.full_like(xr, torch.nan)
            xsc[:nsc] = interface.estimate_x1(xr[:nsc], time_r[:nsc], out)

    u = interface.pred(
        net=net, xt=xr, xsc=xsc, t=time_t, r=time_r, f=C0, chiral_bias=chiral_bias,
    )
    target = (x1 - x0).detach()

    # `_full_x_mask` returns None when nothing is padded, which is the usual
    # case here (one group is K candidates for the same family, so identical
    # atom counts). Only pay for the masked reduction when there is padding.
    mask = _full_x_mask(C1)
    err = (u - target) ** 2
    dims = tuple(range(1, err.ndim))
    if mask is None:
        per_sample = err.mean(dim=dims)
    else:
        m = mask
        while m.ndim < err.ndim:
            m = m.unsqueeze(-1)
        denom = m.mean(dim=dims).clamp(min=1e-6)
        per_sample = (err * m).mean(dim=dims) / denom
    loss = per_sample.mean()
    return {
        "loss": loss,
        "mse": loss.detach(),
        "target_norm": target.norm(dim=-1).mean().detach(),
        "mean_r": time_r.mean().detach(),
        "mean_t": time_t.mean().detach(),
    }


def save_meanflow_checkpoint(
    path: str | Path,
    *,
    net: MeanFlowDiTWrapper,
    chiral: ChiralConditioning | None,
    ema_state: dict | None,
    meta: dict,
    fake_score_net: nn.Module | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    meta.update(live_time_mix_meta(net))
    payload = {
        "net_state_dict": net.state_dict(),
        "chiral_state_dict": chiral.state_dict() if chiral is not None else None,
        "ema_state_dict": ema_state,
        "meta": meta,
    }
    if fake_score_net is not None:
        core = fake_score_net.module if hasattr(fake_score_net, "module") else fake_score_net
        payload["fake_score_state_dict"] = core.state_dict()
    torch.save(payload, path)


def load_meanflow_checkpoint(
    path: str | Path,
    bundle: dict,
    *,
    load_ema: bool = True,
) -> dict:
    """Restore MeanFlow weights into an existing ``load_meanflow_bundle`` result."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    bundle["net"].load_state_dict(payload["net_state_dict"], strict=False)
    chiral = bundle.get("chiral")
    if chiral is not None and payload.get("chiral_state_dict"):
        chiral.load_state_dict(payload["chiral_state_dict"], strict=True)
    bundle["meta"] = payload.get("meta", {})
    bundle["_mf_payload"] = payload
    if load_ema and payload.get("ema_state_dict"):
        bundle["ema_state_dict"] = payload["ema_state_dict"]
        load_ema_preserving_time_mix(bundle["net"], payload)
    else:
        restore_time_mix_buffers(
            bundle["net"], payload.get("net_state_dict"), bundle["meta"]
        )
    return bundle


def apply_meanflow_checkpoint_to_lit(
    lit,
    ckpt_path: str | Path,
    *,
    meanflow_steps: int = 4,
    load_ema: bool = True,
):
    """Inject trained MeanFlow weights into a Clari LitDiT for sampling / eval."""
    from crystal_nft.meanflow.interface import MeanFlowCrystalInterface
    from crystal_nft.meanflow.net import wrap_dit_for_meanflow
    from crystal_nft.meanflow.sampler import MeanFlowCrystalSampler

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = next(lit.net.parameters()).device
    meta = payload.get("meta") or {}
    saved_cfg = meta.get("cfg") if isinstance(meta.get("cfg"), dict) else {}
    gate_value = float(meta.get("gate_live", saved_cfg.get("gate_value", 0.25)))
    conditioning_mode = str(saved_cfg.get("conditioning_mode", "mix"))
    mf_net = wrap_dit_for_meanflow(
        lit.net,
        gate_value=gate_value,
        conditioning_mode=conditioning_mode,
        dual_time_feature_mode=str(saved_cfg.get("dual_time_feature_mode", "both")),
        dual_gate_value=float(
            meta.get("dual_gate_live", saved_cfg.get("dual_gate_value", 1.0))
        ),
        time_parameterization=str(saved_cfg.get("time_parameterization", "legacy")),
    )
    from crystal_nft.meanflow.tune import (
        apply_student_tune,
        resolve_eval_nfe8_lora_mix,
        set_nfe8_lora_mix,
    )

    apply_student_tune(mf_net, saved_cfg or {"student_tune": "full"})
    # Attach chiral head before load_state_dict: net_state_dict includes `_chiral_bias.*`
    # when training saved a MeanFlowDiTWrapper with attach_chiral_bias.
    net_sd = dict(payload["net_state_dict"])
    # Drop legacy output-correction head keys so old checkpoints load into
    # the new AnyFlow time-embedding architecture (strict would otherwise fail).
    for k in list(net_sd.keys()):
        if (
            k.startswith("_output_head.")
            or k.startswith("_delta_head.")
            or k.startswith("embed_deltat.")
        ):
            del net_sd[k]
    needs_chiral = payload.get("chiral_state_dict") or any(
        k.startswith("_chiral_bias.") for k in net_sd
    )
    if needs_chiral:
        dim = int(
            getattr(mf_net.dit, "dim_cond", None)
            or getattr(mf_net, "dim_cond", None)
            or 0
        )
        if dim <= 0:
            raise RuntimeError("Cannot infer dim_cond while restoring chiral head")
        chiral = ChiralConditioning(dim).to(device)
        if payload.get("chiral_state_dict"):
            chiral.load_state_dict(payload["chiral_state_dict"], strict=True)
        mf_net.attach_chiral_bias(chiral)
        payload["_chiral_module"] = chiral
    if any(k.startswith("_stereo_") for k in net_sd) or bool(
        saved_cfg.get("enable_stereo", False)
    ):
        attach_stereo_tokens(
            mf_net,
            device=device,
            pair_edges=any(k.startswith("_stereo_pairs.") for k in net_sd)
            or bool(saved_cfg.get("stereo_pair_edges", True)),
            gain=float(saved_cfg.get("stereo_gain", 1.0)),
            node_mod=any(k.startswith("_stereo_node_mod.") for k in net_sd)
            or any(k.startswith("_stereo_pair_mod.") for k in net_sd)
            or bool(saved_cfg.get("stereo_node_mod", False)),
            cond_path=any(k.startswith("_stereo_cond.") for k in net_sd)
            or bool(saved_cfg.get("stereo_cond_path", False)),
            chiral_branch=any(
                k.startswith("_stereo_chiral_branch.") for k in net_sd
            )
            or bool(saved_cfg.get("stereo_chiral_branch", False)),
            # scale/gate are buffers in the state_dict, but the module must be
            # constructed with them so eval matches training even when the
            # checkpoint predates the buffer.
            chiral_branch_scale=float(saved_cfg.get("stereo_chiral_branch_scale", 1.0)),
            chiral_branch_gate=float(saved_cfg.get("stereo_chiral_branch_gate", 0.0)),
            chiral_branch_endpoint_geom=bool(
                saved_cfg.get("stereo_chiral_branch_endpoint_geom", False)
            ),
            chiral_branch_bond_preserving=bool(
                saved_cfg.get("stereo_chiral_branch_bond_preserving", False)
            ),
            chiral_branch_t_max=float(saved_cfg.get("stereo_chiral_branch_t_max", 1.0)),
            chiral_branch_t_min=float(saved_cfg.get("stereo_chiral_branch_t_min", 0.0)),
            head_t_min=float(saved_cfg.get("stereo_head_t_min", 0.0)),
            head_t_max=float(saved_cfg.get("stereo_head_t_max", 1.0)),
        )
        payload["_stereo_enabled"] = True
    mf_net.load_state_dict(net_sd, strict=False)
    if load_ema and payload.get("ema_state_dict"):
        load_ema_preserving_time_mix(mf_net, payload)
    else:
        restore_time_mix_buffers(mf_net, net_sd, meta)
    # Interval residual stays off at NFE>=16. Report LoRA mix is 0 (16-head)
    # at every NFE; MEANFLOW_NFE8_LORA_MIX=1 is 8-LoRA ablation only.
    if int(meanflow_steps) >= 16:
        set_time_mix(mf_net, interval=0.0)
    nfe8_mix = resolve_eval_nfe8_lora_mix(
        meanflow_steps,
        float(saved_cfg.get("nfe8_lora_mix", 0.0) or 0.0),
    )
    set_nfe8_lora_mix(mf_net, nfe8_mix)
    logger.info(
        "Eval nfe8_lora_mix=%.3f steps=%s env=%r",
        nfe8_mix,
        meanflow_steps,
        os.environ.get("MEANFLOW_NFE8_LORA_MIX", ""),
    )
    lit.net = mf_net.to(device)
    lit.sampler = MeanFlowCrystalSampler(
        num_steps=meanflow_steps,
        mode=str(os.environ.get("MEANFLOW_SAMPLER_MODE", "interval")).lower(),
        rho=float(os.environ.get("MEANFLOW_INTERVAL_RHO", "1")),
        schedule=str(os.environ.get("MEANFLOW_INTERVAL_SCHEDULE", "power")).lower(),
    )
    lit.interface = MeanFlowCrystalInterface(
        prior=lit.interface.prior,
        train_tdist=lit.interface.train_tdist,
    )
    return payload


def _chiral_bias_for_batch(
    chiral: ChiralConditioning | None,
    smiles: str | None,
    batch_size: int,
    device: torch.device,
) -> Tensor | None:
    if chiral is None or not smiles:
        return None
    desc = chiral_descriptor_from_smiles(smiles, device)
    return chiral_bias_from_descriptor(chiral, desc, batch_size)


@torch.no_grad()
def sample_from_crystal_meanflow(
    bundle: dict,
    crystal,
    *,
    samples: int = 8,
    smiles: str | None = None,
    use_old_net: bool = True,
    batch_size: int | None = None,
):
    """Roll out ``samples`` candidates for one crystal template.

    Batched: the template is collated ``batch_size`` times so one interval
    rollout produces that many candidates, halving the batch on CUDA OOM
    (same shape as ``clari_adapter._LitProxy.sample_batch``). ``no_grad``
    rather than ``inference_mode`` -- the candidates come back as the "clean"
    target of the NFT loss, and inference tensors cannot be saved for backward.
    """
    interface = bundle["interface"]
    sampler = bundle["sampler"]
    device = bundle["device"]
    net = bundle["old_net"] if use_old_net else bundle["net"]
    chiral = bundle.get("chiral")
    crystal_cls = type(crystal)

    want_total = int(samples)
    chunk = int(batch_size) if batch_size else want_total
    chunk = max(1, min(chunk, want_total))
    out: list = []
    while len(out) < want_total:
        take = min(chunk, want_total - len(out))
        try:
            c_in = crystal_cls.collate([crystal] * take).to(device)
            bias = _chiral_bias_for_batch(chiral, smiles, take, device)
            c = sampler.sample(interface, net, c_in, chiral_bias=bias)
        except RuntimeError as exc:
            # Narrow on purpose: only OOM is retryable here, everything else
            # (shape / dtype / chirality bugs) must surface.
            if chunk <= 1 or "out of memory" not in str(exc).lower():
                raise
            chunk = max(1, chunk // 2)
            torch.cuda.empty_cache()
            continue
        got = c.cpu()
        out.extend(got.unbatch() if got.batched else [got])
    return out[:want_total], crystal


def meanflow_pretrain_step(
    interface: MeanFlowCrystalInterface,
    net: nn.Module,
    batch: tuple,
    *,
    chiral_bias: Tensor | None = None,
    use_fm_warmup: bool = False,
) -> dict[str, Tensor]:
    if use_fm_warmup:
        return interface.fm_supervision_loss(net, batch, chiral_bias=chiral_bias)
    return interface.loss(net, batch, chiral_bias=chiral_bias)


def _full_x_mask(crystal) -> Optional[Tensor]:
    """Validity mask over the rows of Clari's packed ``x`` (3 lattice + N atoms).

    ``Crystal.mask`` covers atoms only, while ``x`` is
    ``pack_to_x(lattice (B,3,3), coords (B,N,3))``. Returns ``(B, 3+N)`` with
    the three lattice rows always valid, or ``None`` when nothing is padded
    (the usual case here -- one NFT group is K candidates for the *same*
    family, so every candidate has the same atom count).
    """
    mask = getattr(crystal, "mask", None)
    if mask is None:
        return None
    if bool(mask.all()):
        return None
    lattice = torch.ones(
        mask.shape[0], 3, device=mask.device, dtype=mask.dtype
    )
    return torch.cat([lattice, mask], dim=1)


def meanflow_nft_train_step(
    interface: MeanFlowCrystalInterface,
    net: MeanFlowDiTWrapper,
    old_net: MeanFlowDiTWrapper,
    ref_net: MeanFlowDiTWrapper,
    crystals: Sequence,
    r_weights: Tensor,
    *,
    smiles: str | None = None,
    chiral: ChiralConditioning | None = None,
    beta: float = 0.1,
    kl_coef: float = 0.01,
    adv_clip_max: float = 5.0,
    nft_velocity_mode: str = "instantaneous",
    cd_velocity_source: str = "noise_minus_data",
    diffusion_ratio: float = 0.5,
    consistency_ratio: float = 0.0,
    nfe_steps: int = 16,
    grid_rho: float = 1.0,
    cd_eps: float = 5.0e-3,
    share_cd_with_old: bool = True,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Tensor]:
    """One MeanFlowNFT / DiffusionNFT update on the dual-time crystal student.

    Forward-process RL (DiffusionNFT, arXiv 2509.16117): the *generated*
    candidates are treated as the data, re-noised to a time on the flow-map
    grid, and an implicit positive / negative policy pair is regressed onto
    them with a rank weight ``r_weights`` in ``[0, 1]``.

    ``nft_velocity_mode``:

    - ``"instantaneous"`` (MeanFlowNFT as published): convert the flow map to
      the induced instantaneous velocity with the MeanFlow identity before
      applying NFT, so the update lives in the space where NFT's improvement
      guarantee holds. Under Clari time the state sits at the *start* time
      ``r``, which flips the identity's sign to
      ``V(x_r, r) = U(x_r, r, t) - (t - r) dU/dr`` (see
      ``clari_induced_velocity``). ``dU/dr`` is a stop-gradient central
      difference along the conditional velocity (``flow_map_dudr``).
    - ``"instantaneous_boundary"``: same conversion, but the correction is the
      *measured* average-vs-instantaneous gap of the old policy,
      ``U_old(x_r, r, t) - U_old(x_r, r, r)``, instead of a finite-difference
      derivative. Exact and noise-free for the same two extra forwards. On the
      CrystAF student the published estimator is not usable: the gap it has to
      reproduce is only ~3% of ``||U||`` on an adjacent 16-grid jump, while the
      central difference's spatial term ``(dU/dz).v`` is several times larger
      and per-sample uncorrelated with it (cos ~ -0.06 total vs +0.79 for the
      time partial alone). The gradient still flows through ``U(x_r, r, t)``,
      so this does not collapse to the ``r = t`` case the paper warns about.
    - ``"flow_map"`` / ``"direct_u"`` (DiffusionNFT): apply NFT straight to
      ``U`` with no identity conversion. This is the published DiffusionNFT
      update, which has no notion of the second time argument; with
      ``diffusion_ratio=1`` it degenerates to exactly plain DiffusionNFT on
      the ``r = t`` slice.

    ``share_cd_with_old`` runs the central difference once on ``old_net`` and
    reuses it for all three networks. That is what makes the positive /
    negative constructions exact mixtures in velocity space: the identity term
    then cancels out of ``V_theta - V_old`` instead of contributing its own
    finite-difference error.
    """
    device = next(net.parameters()).device
    C0, C1 = interface.collate_fn(list(crystals))
    C0 = C0.to(device)
    C1 = C1.to(device)
    r_w = r_weights.to(device=device, dtype=torch.float32)
    if r_w.shape[0] != C1.batch_size:
        raise ValueError(f"r size {r_w.shape[0]} != batch {C1.batch_size}")

    chiral_bias = _chiral_bias_for_batch(chiral, smiles, C1.batch_size, device)

    x0, x1 = C0.x, C1.x
    mode = str(nft_velocity_mode).lower().strip()
    if mode not in ("instantaneous", "instantaneous_boundary", "flow_map", "direct_u"):
        raise ValueError(f"Unknown nft_velocity_mode={nft_velocity_mode}")

    # Per-rank mode mix, not the reference's global prefix. These ranks never
    # all-reduce -- weights are averaged through the filesystem at epoch end --
    # so a global partition would hand rank 0 nothing but r = t and rank 1
    # nothing but interval jumps, and each would take its own optimiser steps on
    # half the objective. `official_mode_masks` made the same call for the same
    # reason. Pass rank / world_size only when the caller does synchronise.
    time_r, time_t, _is_diffusion = sample_nft_times(
        C1.batch_size,
        device,
        diffusion_ratio=diffusion_ratio,
        consistency_ratio=consistency_ratio,
        nfe_steps=nfe_steps,
        rho=grid_rho,
        rank=rank,
        world_size=world_size,
    )
    # The state lives at the flow map's start time r (Clari: 0=noise, 1=data),
    # matching `CrystalAnyFlowLoss` (`state_time = r` for the official variant)
    # and the interval sampler, which calls U(z_r, r, t).
    xr = interface.sample_xt(x0, x1, time_r)

    xsc = None
    if net.self_cond:
        with torch.no_grad():
            nsc = max(1, C0.batch_size // 2)
            fsc = C0.subset(slice(0, nsc))
            out = interface.pred(
                net=net,
                xt=xr[:nsc],
                xsc=None,
                t=time_t[:nsc],
                r=time_r[:nsc],
                f=fsc,
                chiral_bias=chiral_bias[:nsc] if chiral_bias is not None else None,
            )
            xsc = torch.full_like(xr, torch.nan)
            xsc[:nsc] = interface.estimate_x1(xr[:nsc], time_r[:nsc], out)

    def _forward(target_net, *, grad: bool) -> Tensor:
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            return interface.forward(
                net=target_net,
                xt=xr,
                xsc=xsc,
                t=time_t,
                r=time_r,
                f=C0,
                chiral_bias=chiral_bias,
            )

    u_fwd = _forward(net, grad=True)
    with torch.no_grad():
        u_old = _forward(old_net, grad=False)
        u_ref = _forward(ref_net, grad=False)

    correction = None
    if mode == "instantaneous_boundary":
        # Exact induced-velocity correction, measured on the old policy:
        #   V_old = U_old(x_r, r, t) - [U_old(x_r, r, t) - U_old(x_r, r, r)]
        #         = U_old(x_r, r, r)   <- the old policy's own instantaneous velocity
        # One extra forward (the r = t boundary); no eps, no differencing.
        with torch.no_grad():
            u_old_boundary = interface.forward(
                net=old_net,
                xt=xr,
                xsc=xsc,
                t=time_r,
                r=time_r,
                f=C0,
                chiral_bias=chiral_bias,
            )
        correction = (u_old - u_old_boundary).detach()
    elif mode == "instantaneous":
        gap = (time_t - time_r).abs()
        if bool((gap > 1e-8).any()):
            cd_src = str(cd_velocity_source).lower().strip()
            if cd_src in ("time_partial", "none", "zero"):
                # dU/dr|_x instead of the total derivative along the path.
                #
                # The identity wants the total derivative, so this is formally
                # biased -- it drops the spatial term (dU/dz).v. On this
                # student that term is *measured* to be several times larger
                # than the gap it must reproduce and per-sample uncorrelated
                # with it (cos ~ -0.06 for the total vs +0.79 for the time
                # partial alone; see the `instantaneous_boundary` note above).
                # A biased estimator that correlates at +0.79 beats an
                # unbiased one that correlates at -0.06.
                v_dir = torch.zeros_like(xr)
            elif cd_src == "u_self":
                with torch.no_grad():
                    v_dir = interface.forward(
                        net=old_net,
                        xt=xr,
                        xsc=xsc,
                        t=time_r,
                        r=time_r,
                        f=C0,
                        chiral_bias=chiral_bias,
                    ).detach()
            else:
                v_dir = (x1 - x0).detach()

            def _cd(cd_net) -> Tensor:
                return flow_map_dudr(
                    interface,
                    cd_net,
                    xr,
                    xsc,
                    time_r,
                    time_t,
                    C0,
                    chiral_bias=chiral_bias,
                    v_dir=v_dir,
                    cd_eps=cd_eps,
                )

            if share_cd_with_old:
                shared = _cd(old_net)
                correction = (time_t - time_r).view(-1, 1, 1) * shared
            else:
                # Per-network CD. `net` may be DDP-wrapped; the CD runs under
                # no_grad and must not consume the one grad-enabled forward the
                # reducer allows, so unwrap it here.
                dt = (time_t - time_r).view(-1, 1, 1)
                correction = (
                    dt * _cd(net.module if hasattr(net, "module") else net),
                    dt * _cd(old_net),
                    dt * _cd(ref_net),
                )

    if correction is None:
        forward_pred, old_pred, ref_pred = u_fwd, u_old, u_ref
    else:
        # `correction` already carries the (t - r) factor. A single tensor means
        # it is shared by all three networks (the exact-gap mode, and the
        # central-difference mode under `share_cd_with_old`), which is what
        # leaves V_theta - V_old == U_theta - U_old.
        c = correction if isinstance(correction, tuple) else (correction,) * 3
        forward_pred = clari_induced_velocity(u_fwd, c[0])
        old_pred = clari_induced_velocity(u_old, c[1])
        ref_pred = clari_induced_velocity(u_ref, c[2])

    losses = nft_reconstruction_loss(
        forward_pred,
        old_pred,
        xt=xr,
        clean=x1,
        t=time_r,
        r=r_w,
        beta=beta,
        time_convention="clari",
        mask=_full_x_mask(C1),
    )
    kl = kl_velocity_loss(forward_pred, ref_pred)

    total = combine_nft_and_kl(
        losses["policy_loss"],
        kl,
        kl_coef=kl_coef,
        adv_clip_max=adv_clip_max,
    )
    return {
        "loss": total,
        "policy_loss": losses["policy_loss"].detach(),
        "kl_loss": kl.detach(),
        "unweighted_policy_loss": losses["unweighted_policy_loss"],
        "positive_loss": losses["positive_loss"],
        "negative_loss": losses["negative_loss"],
        "old_deviate": ((forward_pred - old_pred) ** 2).mean().detach(),
        "mean_t": time_t.mean().detach(),
        "mean_r": time_r.mean().detach(),
        "frac_r_eq_t": (
            ((time_t - time_r).abs() <= 1e-8).float().mean().detach()
        ),
    }


def enantiomer_augment_x1(x1: Tensor, crystal, flip_p: float) -> Tensor:
    """Apply random reflection augment when chiral tags are known.

    Batched crystals must be handled per-sample: a single concatenated tag
    vector from ``crystal.to_rdmol()`` does not align with padded ``body_ids``.
    """
    if flip_p <= 0:
        return x1
    try:
        from crystal_nft.meanflow.chirality import chiral_asu_tags_from_crystal

        body = crystal.body_ids
        if body.ndim == 1:
            body = body.unsqueeze(0).expand(x1.shape[0], -1)

        # Prefer per-crystal tags when the input is a batched Crystal.
        crystals = None
        try:
            if getattr(crystal, "batched", False) and hasattr(crystal, "unbatch"):
                crystals = list(crystal.unbatch())
        except Exception:
            crystals = None

        if crystals is not None and len(crystals) == int(x1.shape[0]):
            out = x1.clone()
            for b, c in enumerate(crystals):
                tags = chiral_asu_tags_from_crystal(c)
                n_coord = int(x1.shape[1] - 3)
                if tags is None:
                    tags_b = torch.ones(n_coord, dtype=torch.long, device=x1.device)
                else:
                    tags_b = tags.to(device=x1.device)
                    if tags_b.numel() < n_coord:
                        pad = torch.zeros(
                            n_coord - tags_b.numel(), dtype=torch.long, device=x1.device
                        )
                        tags_b = torch.cat([tags_b, pad], dim=0)
                    tags_b = tags_b[:n_coord]
                # Flip decision is per crystal (not per body-group RNG only).
                flipped = random_enantiomer_flip(
                    out[b : b + 1], tags_b, body[b : b + 1], p=flip_p
                )
                out[b] = flipped[0]
            return out

        tags = chiral_asu_tags_from_crystal(crystal)
        if tags is None:
            # Fall back: treat all ASU atoms as potentially chiral for augment.
            n_coord = int(x1.shape[1] - 3)
            tags = torch.ones(n_coord, dtype=torch.long, device=x1.device)
        else:
            tags = tags.to(device=x1.device)
            if tags.numel() < x1.shape[1] - 3:
                pad = torch.zeros(
                    x1.shape[1] - 3 - tags.numel(), dtype=torch.long, device=x1.device
                )
                tags = torch.cat([tags, pad], dim=0)
            tags = tags[: x1.shape[1] - 3]
        return random_enantiomer_flip(x1, tags, body, p=flip_p)
    except Exception:
        return x1


_DESC_CACHE: dict[str, Tensor] = {}


def make_chiral_bias_for_batch(
    chiral: ChiralConditioning | None,
    crystal,
    batch_size: int,
    device: torch.device,
) -> Tensor | None:
    """Build (B, dim_cond) chiral bias from crystal RDKit stereo when available."""
    if chiral is None:
        return None
    from crystal_nft.meanflow.chirality import (
        ChiralInfo,
        batch_chiral_descriptors_from_crystals,
        chiral_asu_tags_from_crystal,
        global_chiral_descriptor,
    )

    try:
        if hasattr(crystal, "unbatch"):
            crystals = list(crystal.unbatch())
        else:
            crystals = [crystal]
    except Exception:
        crystals = [crystal]
    if len(crystals) == 1 and batch_size > 1:
        crystals = crystals * batch_size
    elif len(crystals) != batch_size:
        crystals = (crystals * batch_size)[:batch_size]

    descs = []
    for c in crystals:
        cid = getattr(c, "csd_id", None)
        if isinstance(cid, (list, tuple)):
            cid = cid[0] if cid else None
        if isinstance(cid, str) and cid in _DESC_CACHE:
            descs.append(_DESC_CACHE[cid].to(device))
            continue
        tags = chiral_asu_tags_from_crystal(c)
        if tags is None:
            d = torch.zeros(8, dtype=torch.float32, device=device)
        else:
            info = ChiralInfo(
                asu_chiral=tags,
                n_chiral_centers=int((tags > 0).sum().item()),
                n_defined=int(((tags == 1) | (tags == 2)).sum().item()),
            )
            d = global_chiral_descriptor(info).to(device)
        if isinstance(cid, str) and cid:
            _DESC_CACHE[cid] = d.detach().cpu()
        descs.append(d)
    desc = torch.stack(descs, dim=0)
    return chiral_bias_from_descriptor(chiral, desc, batch_size)

def build_meanflow_teacher_from_ckpt(ckpt_path, *, base_lit, device, cfg):
    """A frozen, STEREO-AWARE AnyFlow teacher loaded from a meanflow checkpoint.

    The default teacher is `InstantaneousDiT(nft_backbone)` with no stereo head,
    so `teacher_instantaneous_velocity` returns a chirality-blind velocity -- and
    since the AnyFlow target is ~94% that velocity, distillation trains the
    student to ignore the CIP tag no matter what the chirality objective asks
    (measured in v16: af 0.19 -> 1.20, sampled chirality stuck at 0.50/fr0.24).

    Wrapping the teacher as a MeanFlowDiTWrapper with the same stereo modules --
    including StereoChiralBranch, which fires inside `crystal_forward` -- makes
    the teacher's velocity carry handedness, so the distillation target and the
    chirality objective point the same way.
    """
    import copy

    from crystal_nft.meanflow.net import wrap_dit_for_meanflow

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = payload.get("meta") or {}
    saved = meta.get("cfg") if isinstance(meta.get("cfg"), dict) else {}
    dit = copy.deepcopy(base_lit.net)
    if isinstance(dit, MeanFlowDiTWrapper):
        dit = copy.deepcopy(dit.dit)
    teacher = wrap_dit_for_meanflow(
        dit,
        gate_value=float(meta.get("gate_live", saved.get("gate_value", 0.25))),
        conditioning_mode=str(saved.get("conditioning_mode", "mix")),
        dual_time_feature_mode=str(saved.get("dual_time_feature_mode", "both")),
        dual_gate_value=float(meta.get("dual_gate_live", saved.get("dual_gate_value", 1.0))),
        time_parameterization=str(saved.get("time_parameterization", "legacy")),
    )
    net_sd = payload.get("net_state_dict") or {}
    attach_stereo_tokens(
        teacher,
        device=device,
        pair_edges=any(k.startswith("_stereo_pairs.") for k in net_sd)
        or bool(saved.get("stereo_pair_edges", True)),
        gain=float(saved.get("stereo_gain", 1.0)),
        node_mod=any(k.startswith("_stereo_node_mod.") for k in net_sd)
        or bool(saved.get("stereo_node_mod", False)),
        cond_path=any(k.startswith("_stereo_cond.") for k in net_sd)
        or bool(saved.get("stereo_cond_path", False)),
        chiral_branch=any(k.startswith("_stereo_chiral_branch.") for k in net_sd)
        or bool(saved.get("stereo_chiral_branch", False)),
        chiral_branch_scale=float(saved.get("stereo_chiral_branch_scale", 1.0)),
        chiral_branch_gate=float(saved.get("stereo_chiral_branch_gate", 0.0)),
        chiral_branch_endpoint_geom=bool(
            saved.get("stereo_chiral_branch_endpoint_geom", False)
        ),
        chiral_branch_bond_preserving=bool(
            saved.get("stereo_chiral_branch_bond_preserving", False)
        ),
        chiral_branch_t_max=float(saved.get("stereo_chiral_branch_t_max", 1.0)),
        chiral_branch_t_min=float(saved.get("stereo_chiral_branch_t_min", 0.0)),
        head_t_min=float(saved.get("stereo_head_t_min", 0.0)),
        head_t_max=float(saved.get("stereo_head_t_max", 1.0)),
    )
    # LoRA must exist before loading or the adapter tensors land as unexpected
    # keys and the distilled behaviour is silently discarded.
    from crystal_nft.meanflow.tune import apply_student_tune

    apply_student_tune(teacher, {"student_tune": str(saved.get("student_tune", "delta_lora")),
                                 "lora_rank": int(saved.get("lora_rank", 32)),
                                 "lora_alpha": float(saved.get("lora_alpha", 64.0))})
    inc = teacher.load_state_dict(net_sd, strict=False)
    logger.info(
        "stereo teacher: missing=%d unexpected=%d stereo_keys=%d",
        len(inc.missing_keys), len(inc.unexpected_keys),
        sum(1 for k in net_sd if "_stereo_" in k),
    )
    teacher = teacher.to(device).eval()
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    return teacher
