"""Student-side freeze / LoRA for AnyFlow-faithful Stage-1.

Official SD3.5 AnyFlow keeps the teacher backbone (LoRA on attention only)
and trains a copied ``delta_embedder`` at ``gate=0.25``. Full DiT FT plus a
mid-range gate mixes a frozen Clari ``base(t)`` into a moved trunk and
collapses 16-step PB.

Modes:
  ``full``         — current behaviour (train DiT + delta; freeze base only)
  ``delta_only``   — freeze DiT; train ``embed_timestep.delta``
  ``delta_stem``   — freeze trunk; train delta + ``stem_cond``
  ``delta_lora``        — freeze DiT; train delta + LoRA on attention projections
  ``delta_lora_base``   — same as delta_lora, but also train ``embed_timestep.base``
  ``dual_cond_lora``    — train dual-time residual + attention/condition LoRA
  ``attn_lora``         — freeze the whole teacher; train attention LoRA only
"""

from __future__ import annotations

import logging
import math
import os
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

LORA_LINEAR_NAMES = ("proj_q", "proj_k", "proj_v", "proj_o")


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` plus zero-init B so the forward is unchanged at step 0."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        *,
        condition_on_interval: bool = False,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)
        r = int(rank)
        dev, dt = linear.weight.device, linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=dt, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=dt, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Second adapter used only on 8-NFE. Zero B ⇒ identity until trained.
        # 16-NFE eval sets nfe8_mix=0 so the shared LoRA is unchanged.
        self.lora8_A = nn.Parameter(torch.empty(r, self.in_features, dtype=dt, device=dev))
        self.lora8_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=dt, device=dev))
        nn.init.kaiming_uniform_(self.lora8_A, a=math.sqrt(5))
        self.scale = float(alpha) / float(r)
        self.condition_on_interval = bool(condition_on_interval)
        self._interval_gate: torch.Tensor | None = None
        self.register_buffer("nfe8_mix", torch.tensor([0.0], dtype=torch.float32))

    def set_interval_gate(self, value: torch.Tensor | None) -> None:
        self._interval_gate = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B)
        lora8 = F.linear(F.linear(x, self.lora8_A), self.lora8_B)
        mix8 = getattr(self, "nfe8_mix", None)
        if mix8 is None:
            adapted = lora
        else:
            mix8 = mix8.to(device=lora.device, dtype=lora.dtype)
            # Swap, not add: mix=0 is the frozen 16-NFE LoRA; mix=1 is the
            # 8-NFE copy. Additive zero-init (cont34) could not move 8-NFE.
            adapted = (1.0 - mix8) * lora + mix8 * lora8
        if self.condition_on_interval and self._interval_gate is not None:
            gate = self._interval_gate.to(device=adapted.device, dtype=adapted.dtype)
            gate = gate.reshape(gate.shape[0], *([1] * (adapted.ndim - 1)))
            adapted = adapted * gate
        return base + adapted * self.scale


def _is_delta_name(name: str) -> bool:
    return ("embed_timestep.delta" in name) or ("_time_proxy.delta" in name)


def _is_base_embed_name(name: str) -> bool:
    return ("embed_timestep.base" in name) or ("_time_proxy.base" in name)


def _is_nfe8_lora_name(name: str) -> bool:
    return (
        ".lora8_A" in name
        or name.endswith("lora8_A")
        or ".lora8_B" in name
        or name.endswith("lora8_B")
    )


def _is_lora_name(name: str) -> bool:
    if _is_nfe8_lora_name(name):
        return False
    return (
        (".lora_A" in name)
        or name.endswith("lora_A")
        or (".lora_B" in name)
        or name.endswith("lora_B")
    )


def resolve_eval_nfe8_lora_mix(
    meanflow_steps: int,
    saved_mix: float,
    *,
    override: str | None = None,
) -> float:
    """LoRA mix used at eval. Report protocol is mix=0 (the 16-NFE LoRA) at every NFE.

    ``MEANFLOW_NFE8_LORA_MIX`` can pin the dedicated 8-LoRA (mix=1) for ablations.
    Do not swap heads by NFE when reporting dual-time numbers.
    """
    raw = os.environ.get("MEANFLOW_NFE8_LORA_MIX", "") if override is None else override
    raw = str(raw).strip()
    if raw != "":
        return float(raw)
    del meanflow_steps, saved_mix
    return 0.0


def set_nfe8_lora_mix(net: nn.Module, value: float) -> int:
    """Scale the 8-NFE LoRA adapter. 0 leaves the shared 16-NFE LoRA unchanged."""
    n = 0
    mix = float(value)
    with torch.no_grad():
        for mod in net.modules():
            buf = getattr(mod, "nfe8_mix", None)
            if torch.is_tensor(buf):
                buf.fill_(mix)
                n += 1
    return n


def init_nfe8_lora_from_shared(net: nn.Module) -> int:
    """Copy the 16-NFE LoRA into the 8-NFE adapter so mix=1 starts identical."""
    n = 0
    with torch.no_grad():
        for mod in net.modules():
            if not isinstance(mod, LoRALinear):
                continue
            if not hasattr(mod, "lora8_A") or not hasattr(mod, "lora8_B"):
                continue
            mod.lora8_A.copy_(mod.lora_A)
            mod.lora8_B.copy_(mod.lora_B)
            n += 1
    return n


def _is_stem_cond_name(name: str) -> bool:
    return "stem_cond" in name


def _is_dual_name(name: str) -> bool:
    return (
        "stem_cond.conditioner." in name
        or "_dual_time." in name
        or "dual_time_conditioner." in name
    )


def inject_attention_lora(
    net: nn.Module,
    *,
    rank: int,
    alpha: float,
    condition_on_interval: bool = False,
) -> int:
    """Replace Clari ``Attention`` q/k/v/o linears. Idempotent."""
    n = 0
    for mod in net.modules():
        for child_name, child in list(mod.named_children()):
            if child_name not in LORA_LINEAR_NAMES:
                continue
            if isinstance(child, LoRALinear):
                continue
            if not isinstance(child, nn.Linear):
                continue
            setattr(
                mod,
                child_name,
                LoRALinear(
                    child,
                    rank=rank,
                    alpha=alpha,
                    condition_on_interval=condition_on_interval,
                ),
            )
            n += 1
    return n


def inject_condition_lora(
    net: nn.Module,
    *,
    rank: int,
    alpha: float,
    condition_on_interval: bool = False,
) -> int:
    """Inject LoRA into Clari Modulate scale/shift projections."""
    n = 0
    for mod in net.modules():
        if mod.__class__.__name__ != "Modulate":
            continue
        for child_name in ("scale", "shift"):
            child = getattr(mod, child_name, None)
            if isinstance(child, LoRALinear):
                continue
            if not isinstance(child, nn.Linear):
                continue
            setattr(
                mod,
                child_name,
                LoRALinear(
                    child,
                    rank=rank,
                    alpha=alpha,
                    condition_on_interval=condition_on_interval,
                ),
            )
            n += 1
    return n


def apply_student_tune(net: nn.Module, cfg: dict) -> dict[str, int]:
    """Freeze / LoRA the wrapped student. Safe to call on already-tuned nets."""
    mode = str(cfg.get("student_tune", "full") or "full").lower().strip()
    if mode in ("full", "", "none", "ft"):
        # Actually RESTORE full trainability. This branch used to only count
        # parameters and return, which made "full" a silent no-op whenever the
        # net had already been frozen by an earlier call -- and that is exactly
        # the order train_meanflow_nft.py uses (apply the checkpoint's own
        # `delta_lora` first to load it, then apply the NFT tuning mode). So
        # `nft_student_tune: full` left 3.5M/93.8M (3.8%) trainable and the
        # full fine-tune that made nft-m work could never be reproduced.
        #
        # The 8-NFE LoRA stays frozen unless explicitly requested: it is
        # zero-init and training it is a recorded dead end.
        train_nfe8 = bool(cfg.get("train_nfe8_lora", False))
        for name, prm in net.named_parameters():
            prm.requires_grad_(train_nfe8 if _is_nfe8_lora_name(name) else True)
        n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in net.parameters())
        logger.info(
            "Student tune=full trainable=%d / %d (%.3f%%)",
            n_train, n_all, 100.0 * n_train / max(n_all, 1),
        )
        return {"mode": 0, "n_lora": 0, "n_trainable": n_train, "n_all": n_all}

    n_lora = 0
    n_cond_lora = 0
    # `stereo_only` still INJECTS the attention LoRA (then freezes it below).
    # The checkpoints it resumes from were trained with student_tune=delta_lora,
    # so without the modules present their LoRA tensors load as "unexpected"
    # keys under strict=False and are silently dropped -- which would throw away
    # the very distillation this mode exists to preserve.
    if mode in ("delta_lora", "delta_lora_base", "dual_cond_lora", "attn_lora", "stereo_only"):
        n_lora = inject_attention_lora(
            net,
            rank=int(cfg.get("lora_rank", 32)),
            alpha=float(cfg.get("lora_alpha", 64.0)),
            condition_on_interval=(mode == "dual_cond_lora"),
        )
        if mode == "attn_lora" and n_lora == 0:
            already = sum(1 for m in net.modules() if isinstance(m, LoRALinear))
            if already == 0:
                raise ValueError(
                    "student_tune=attn_lora found no Attention proj_q/k/v/o modules"
                )
            n_lora = already
    if mode == "dual_cond_lora":
        n_cond_lora = inject_condition_lora(
            net,
            rank=int(cfg.get("cond_lora_rank", cfg.get("lora_rank", 32))),
            alpha=float(cfg.get("cond_lora_alpha", cfg.get("lora_alpha", 64.0))),
            condition_on_interval=True,
        )

    def keep(name: str) -> bool:
        if _is_nfe8_lora_name(name):
            return bool(cfg.get("train_nfe8_lora", False))
        # Stereo conditioning is a new zero-init head: always train it when
        # present, whatever the backbone freeze mode is.
        if name.startswith("_stereo_") or "._stereo_" in name:
            return not bool(cfg.get("freeze_stereo_tokens", False))
        if mode == "stereo_only":
            # Nothing but the stereo heads trains. Two consequences that matter:
            # (1) with all-zero tags the student is bit-identical to the
            #     checkpoint it resumed from, so PB/clash/vol/PDD cannot drift on
            #     achiral molecules and the few-step distillation is preserved --
            #     teaching chirality with a strong FM objective is what
            #     un-distilled the student before (PB 85 -> 10, pb_valid 0).
            # (2) the model can no longer satisfy the chirality hinge by reading
            #     handedness out of x_r, because how it reads x_r is frozen. The
            #     only route to lower hinge loss is the CIP tag, which is exactly
            #     the shortcut the mismatch/override machinery existed to block.
            return False
        if mode == "attn_lora":
            return _is_lora_name(name)
        if _is_delta_name(name):
            return True
        if "interval_proj" in name or "interval_mlp" in name:
            return True
        if mode in ("delta_lora", "delta_lora_base", "dual_cond_lora") and _is_lora_name(name):
            return True
        if mode == "dual_cond_lora" and _is_dual_name(name):
            return True
        if mode == "delta_lora_base" and _is_base_embed_name(name):
            return True
        if _is_base_embed_name(name):
            return False
        if mode == "delta_stem" and _is_stem_cond_name(name):
            return True
        return False

    n_enabled = 0
    for name, p in net.named_parameters():
        on = keep(name)
        p.requires_grad_(on)
        if on:
            n_enabled += p.numel()
    if bool(cfg.get("freeze_attn_lora", False)):
        for name, p in net.named_parameters():
            if _is_lora_name(name):
                p.requires_grad_(False)
    if bool(cfg.get("freeze_delta_embed", False)):
        for name, p in net.named_parameters():
            if _is_delta_name(name):
                p.requires_grad_(False)
    if bool(cfg.get("freeze_interval_residual", False)):
        for name, p in net.named_parameters():
            if "interval_proj" in name or "interval_mlp" in name:
                p.requires_grad_(False)
    n_enabled = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in net.parameters())
    stats = {
        "mode": {
            "delta_only": 1,
            "delta_stem": 2,
            "delta_lora": 3,
            "delta_lora_base": 4,
            "dual_cond_lora": 5,
            "attn_lora": 6,
            "stereo_only": 7,
        }.get(mode, -1),
        "n_lora": n_lora + n_cond_lora,
        "n_condition_lora": n_cond_lora,
        "n_trainable": n_enabled,
        "n_all": n_all,
    }
    if stats["mode"] < 0:
        raise ValueError(
            f"Unknown student_tune={mode!r}; expected "
            "full|delta_only|delta_stem|delta_lora|delta_lora_base|dual_cond_lora|attn_lora|stereo_only"
        )
    logger.info(
        "Student tune=%s lora_modules=%d trainable=%d / %d (%.3f%%)",
        mode,
        n_lora,
        n_enabled,
        n_all,
        100.0 * n_enabled / max(n_all, 1),
    )
    return stats


def merge_lora_into_linear(net: nn.Module) -> int:
    """Bake LoRA deltas into the frozen Linear weights and restore nn.Linear.

    After this the module matches the original Clari architecture and can be
    loaded by ``eval_clari_table1`` without injecting LoRA.
    """
    n = 0
    for mod in net.modules():
        for child_name, child in list(mod.named_children()):
            if not isinstance(child, LoRALinear):
                continue
            with torch.no_grad():
                delta = (child.lora_B @ child.lora_A) * child.scale
                child.linear.weight.add_(delta.to(dtype=child.linear.weight.dtype))
            setattr(mod, child_name, child.linear)
            n += 1
    return n


def iter_trainable_names(net: nn.Module) -> Iterable[str]:
    for n, p in net.named_parameters():
        if p.requires_grad:
            yield n
