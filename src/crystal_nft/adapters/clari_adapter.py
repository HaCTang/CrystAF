"""Clari adapter: sampling under old policy and NFT flow-matching updates."""

from __future__ import annotations

import copy
import logging
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from crystal_nft.nft.loss import (
    combine_nft_and_kl,
    copy_params,
    kl_velocity_loss,
    nft_reconstruction_loss,
)

logger = logging.getLogger(__name__)


class _LitProxy:
    """Thin wrapper around lit.sampler.sample with OOM-safe batching."""

    def __init__(self, lit_module, device, *, filter_clashing: bool = False):
        self.lit = lit_module
        self.device = device
        self.use_bf16 = device.type == "cuda"
        self.filter_clashing = filter_clashing

    def sample_batch(self, crystal, count, pbar=None):
        from clari.chem import Crystal

        current = count
        while True:
            batch_gpu = None
            try:
                batch_gpu = Crystal.collate([crystal] * current).to(self.device)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.use_bf16,
                ):
                    out = self.lit.sampler.sample(
                        self.lit.interface,
                        self.lit.net,
                        batch_gpu,
                        pbar=pbar,
                    )
                return out.cpu().unbatch()
            except RuntimeError as exc:
                if (
                    self.device.type != "cuda"
                    or "out of memory" not in str(exc).lower()
                    or current <= 1
                ):
                    raise
                if batch_gpu is not None:
                    del batch_gpu
                torch.cuda.empty_cache()
                current = max(1, current // 2)


def crystal_n_mols(crystal) -> int:
    """Return the number of molecules (unique bodies) in a Crystal template."""
    assert not crystal.batched
    return int(torch.unique(crystal.body_ids).numel())


def load_clari_bundle(
    checkpoint: str = "clari-h",
    *,
    device: str | torch.device = "cuda",
    use_ema: bool = True,
    n_steps: int = 50,
    compile_model: bool = False,
):
    """Load trainable / old / ref Clari nets sharing the same interface/sampler."""
    from clari.inference.inputs import resolve_checkpoint
    from clari.inference.sample import load_lit, resolve_device

    device = resolve_device(device)
    ckpt = resolve_checkpoint(checkpoint)
    lit = load_lit(ckpt, device, use_ema=use_ema, n_steps=n_steps, compile=compile_model)
    lit.train()

    net = lit.net
    old_net = copy.deepcopy(net).eval()
    ref_net = copy.deepcopy(net).eval()
    for p in old_net.parameters():
        p.requires_grad_(False)
    for p in ref_net.parameters():
        p.requires_grad_(False)

    return {
        "lit": lit,
        "net": net,
        "old_net": old_net,
        "ref_net": ref_net,
        "interface": lit.interface,
        "sampler": lit.sampler,
        "device": device,
        "checkpoint": ckpt,
    }


def apply_clari_student_tune(bundle: dict, cfg: dict) -> dict[str, int]:
    """Inject teacher-preserving LoRA and rebuild old/ref copies."""
    from crystal_nft.meanflow.tune import apply_student_tune

    mode = str(cfg.get("student_tune", "full") or "full").lower().strip()
    stats = apply_student_tune(bundle["net"], cfg)
    if mode in ("full", "", "none", "ft"):
        return stats
    bundle["old_net"] = copy.deepcopy(bundle["net"]).eval()
    bundle["ref_net"] = copy.deepcopy(bundle["net"]).eval()
    for frozen in (bundle["old_net"], bundle["ref_net"]):
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
    bundle["lit"].net = bundle["net"]
    return stats


def resume_clari_nft_weights(bundle: dict, path) -> tuple[list[str], list[str]]:
    """Load a LoRA NFT checkpoint into ``net``; keep ``ref_net`` as the teacher."""
    missing, unexpected = load_clari_nft_state(bundle["net"], path)
    bundle["old_net"] = copy.deepcopy(bundle["net"]).eval()
    for parameter in bundle["old_net"].parameters():
        parameter.requires_grad_(False)
    bundle["lit"].net = bundle["net"]
    return missing, unexpected


def load_clari_nft_state(net: nn.Module, payload_or_path) -> tuple[list[str], list[str]]:
    """Load an NFT checkpoint, injecting LoRA only when the payload contains it."""
    from pathlib import Path

    from crystal_nft.meanflow.tune import apply_student_tune

    if isinstance(payload_or_path, (str, Path)):
        payload = torch.load(payload_or_path, map_location="cpu", weights_only=False)
    else:
        payload = payload_or_path
    state = payload.get("net_state_dict", payload)
    cfg = payload.get("config") or {}
    has_lora = any(("lora_A" in key) or ("lora_B" in key) for key in state)
    already_lora = any(type(mod).__name__ == "LoRALinear" for mod in net.modules())
    if has_lora and not already_lora:
        apply_student_tune(
            net,
            {
                "student_tune": cfg.get("student_tune", "attn_lora"),
                "lora_rank": int(cfg.get("lora_rank", 32)),
                "lora_alpha": float(cfg.get("lora_alpha", 64.0)),
            },
        )
    return net.load_state_dict(state, strict=False)


def merged_clari_state_dict(net: nn.Module) -> dict:
    """Return a Clari-native state_dict with LoRA baked into Linear weights."""
    from crystal_nft.meanflow.tune import merge_lora_into_linear

    merged = copy.deepcopy(net).cpu()
    merge_lora_into_linear(merged)
    return {key: value.detach().cpu() for key, value in merged.state_dict().items()}


@torch.inference_mode()
def sample_from_crystal(
    lit,
    crystal,
    *,
    samples: int = 32,
    batch_size: Optional[int] = None,
    filter_clashing: bool = False,
    pbar: bool = False,
    use_old_net: bool = True,
    old_net: Optional[nn.Module] = None,
):
    """
    Sample K crystal candidates starting from a provided Crystal template.

    When use_old_net=True, temporarily swaps lit.net with old_net for sampling.
    """
    device = next(lit.net.parameters()).device

    if use_old_net and old_net is not None:
        train_net = lit.net
        lit.net = old_net
    else:
        train_net = None

    try:
        proxy = _LitProxy(lit, device, filter_clashing=filter_clashing)
        produced = []
        remaining = samples
        empty_rounds = 0
        while remaining > 0:
            want = remaining if batch_size is None else min(batch_size, remaining)
            got = proxy.sample_batch(crystal, want, pbar="Denoising" if pbar else None)
            if not got:
                break
            if filter_clashing:
                from clari.pipelines.utils.metrics import is_clash_free

                got = [c for c in got if is_clash_free(c)]
            produced.extend(got)
            remaining = samples - len(produced)
            if len(got) == 0:
                empty_rounds += 1
                if empty_rounds >= 3:
                    break
            else:
                empty_rounds = 0
        return produced[:samples], crystal
    finally:
        if train_net is not None:
            lit.net = train_net


@torch.inference_mode()
def sample_candidates(
    lit,
    smiles: str,
    *,
    copies: int = 4,
    samples: int = 32,
    mol_id: Optional[str] = None,
    batch_size: Optional[int] = None,
    filter_clashing: bool = False,
    pbar: bool = False,
    use_old_net: bool = True,
    old_net: Optional[nn.Module] = None,
):
    """Sample K crystal candidates for one molecule (SMILES -> Crystal template)."""
    from clari.inference.inputs import make_request
    from clari.inference.sample import request_to_crystal

    req = make_request(smiles, id=mol_id or smiles, copies=copies, samples=samples)
    crystal = request_to_crystal(req)
    return sample_from_crystal(
        lit,
        crystal,
        samples=samples,
        batch_size=batch_size,
        filter_clashing=filter_clashing,
        pbar=pbar,
        use_old_net=use_old_net,
        old_net=old_net,
    )


def crystals_to_ase(crystals: Sequence):
    return [c.to_ase() for c in crystals]


def nft_train_step(
    lit,
    net: nn.Module,
    old_net: nn.Module,
    ref_net: nn.Module,
    crystals: Sequence,
    r_weights: Tensor,
    *,
    beta: float = 0.1,
    kl_coef: float = 0.01,
    adv_clip_max: float = 5.0,
    disable_self_cond: bool = True,
) -> dict[str, Tensor]:
    """One NFT update on a batch of clean Crystal samples."""
    from clari.pipelines.utils.utils import bcast_right

    interface = lit.interface
    device = next(net.parameters()).device

    # collate_fn builds (C0 prior, C1 aligned clean)
    C0, C1 = interface.collate_fn(list(crystals))
    C0 = C0.to(device)
    C1 = C1.to(device)
    r = r_weights.to(device=device, dtype=torch.float32)
    if r.shape[0] != C1.batch_size:
        raise ValueError(f"r size {r.shape[0]} != batch {C1.batch_size}")

    x0, x1 = C0.x, C1.x
    t = interface.sample_t([C1.batch_size], device=device)
    xt = interface.sample_xt(x0, x1, t)

    xsc = None
    if getattr(net, "self_cond", False) and not disable_self_cond:
        with torch.no_grad():
            nsc = C0.batch_size // 2
            if nsc > 0:
                fsc = C0.subset(slice(0, nsc))
                out = interface.pred(net=net, xt=xt[:nsc], xsc=None, t=t[:nsc], f=fsc)
                xsc = torch.full_like(xt, torch.nan)
                xsc[:nsc] = interface.estimate_x1(xt[:nsc], t[:nsc], out).detach()

    forward_pred = interface.forward(net=net, xt=xt, xsc=xsc, t=t, f=C0)
    with torch.no_grad():
        old_pred = interface.forward(net=old_net, xt=xt, xsc=xsc, t=t, f=C0)
        ref_pred = interface.forward(net=ref_net, xt=xt, xsc=xsc, t=t, f=C0)

    # Lattice + coord NFT reconstruction (Clari convention)
    losses = nft_reconstruction_loss(
        forward_pred,
        old_pred,
        xt=xt,
        clean=x1,
        t=t,
        r=r,
        beta=beta,
        time_convention="clari",
        reduce=True,
    )

    # Emphasize coordinate region via optional masked term (matches training split)
    t_ = bcast_right(t, xt)
    positive = beta * forward_pred + (1.0 - beta) * old_pred
    pos_x1 = xt + (1.0 - t_) * positive
    coord_err = ((pos_x1[:, 3:] - x1[:, 3:]) ** 2).mean()
    lattice_err = ((pos_x1[:, :3] - x1[:, :3]) ** 2).mean()

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
        "coord_mse": coord_err.detach(),
        "lattice_mse": lattice_err.detach(),
        "unweighted_policy_loss": losses["unweighted_policy_loss"],
    }
