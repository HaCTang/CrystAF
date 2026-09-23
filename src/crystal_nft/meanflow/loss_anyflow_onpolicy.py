"""AnyFlow Stage-2 on-policy DMD for Clari crystals.

Maps the SD3.5 trainer in ``anyflow_onpolicy_trainer.py`` onto Clari time
(t=0 noise -> t=1 data):

- Student shortcut / full flow-map rollout (with grad).
- DMD on x1: noise the student sample, extrapolate fake/real instantaneous
  scores from t to 1 via ``estimate_x1`` at the r=t boundary, then
  ``mse(x1, (x1 - grad).detach())`` with
  ``grad = (x1_fake - x1_real) / |x1_student - x1_real|``.
- Frozen Clari teacher is the real score (FP32).
- Trainable fake score tracks the student distribution with FM at r=t.
- Optional Stage-1 AnyFlow cotrain on GT interpolants.

This is distribution matching, not paired same-noise MSE to a 50-step ODE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from clari.pipelines.utils import masked_mean
from clari.geometry import zero_com_suffix

from crystal_nft.meanflow.loss_anyflow import AnyFlowLossConfig, CrystalAnyFlowLoss
from crystal_nft.meanflow.loss_sample_align import (
    _synced_choice,
    _synced_randint,
    periodic_frac_coord_mse,
)
from crystal_nft.meanflow.sampler import flow_map_rollout


@dataclass
class AnyFlowOnPolicyConfig:
    student_steps_list: list[int] = field(default_factory=lambda: [2, 4, 8])
    rollout_mode: str = "shortcut"
    n_jumps: int = 8
    detach_between_jumps: bool = False
    checkpoint_jumps: bool = False
    student_self_cond: bool = False
    dmd_weight: float = 1.0
    dmd_t_min: float = 0.02
    dmd_t_max: float = 0.98
    gradient_normalization: bool = True
    anyflow_cotrain_weight: float = 1.0
    discriminator_update_ratio: int = 1
    real_score_fp32: bool = True
    periodic_coord: bool = True
    lattice_weight: float = 1.0
    coord_weight: float = 1.0


def _sample_prior_x(interface, C) -> Tensor:
    zeros = torch.zeros_like(C.x)
    return interface.sample_prior(C.replace(x=zeros)).x


def _masked_abs_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Per-sample mean |x| over lattice + masked coords, keepdim [B,1,1]."""
    b = x.shape[0]
    lat = x[:, :3].reshape(b, -1).abs().mean(dim=-1)
    coord = masked_mean(x[:, 3:].abs(), mask.unsqueeze(-1), dim=[1, 2])
    return (0.5 * (lat + coord)).view(b, 1, 1)


def _masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    err = pred - target
    b = err.shape[0]
    lat = err[:, :3].pow(2).reshape(b, -1).mean(dim=-1)
    coord = masked_mean(err[:, 3:].pow(2), mask.unsqueeze(-1), dim=[1, 2])
    return (lat + coord).mean()


def dmd_identity_loss(
    pred: Tensor,
    grad: Tensor,
    mask: Tensor,
    *,
    periodic_coord: bool = True,
    lattice_weight: float = 1.0,
    coord_weight: float = 1.0,
) -> Tensor:
    """AnyFlow DMD identity in packed x, with optional periodic coord wrap."""
    tgt = (pred - grad).detach()
    b = pred.shape[0]
    lat = (pred[:, :3] - tgt[:, :3]).pow(2).reshape(b, -1).mean(dim=-1)
    packed = masked_mean((pred[:, 3:] - tgt[:, 3:]).pow(2), mask.unsqueeze(-1), dim=[1, 2])
    if periodic_coord:
        coord = periodic_frac_coord_mse(pred, tgt, mask)
        if not torch.isfinite(coord).all():
            coord = packed
    else:
        coord = packed
    return (float(lattice_weight) * lat + float(coord_weight) * coord).mean()


def dmd_kl_gradient(
    pred_x1: Tensor,
    x1_fake: Tensor,
    x1_real: Tensor,
    mask: Tensor,
    *,
    normalize: bool = True,
) -> Tensor:
    """Official DMD gradient: fake score minus real score, optionally normalized."""
    grad = x1_fake - x1_real
    if normalize:
        normalizer = _masked_abs_mean(pred_x1 - x1_real, mask) + 1e-8
        grad = grad / normalizer
    return torch.nan_to_num(grad)


class CrystalAnyFlowOnPolicyLoss:
    """Generator DMD + discriminator FM, AnyFlow Stage-2 on crystals."""

    def __init__(
        self,
        cfg: AnyFlowOnPolicyConfig | None = None,
        *,
        teacher_net: nn.Module | None = None,
        fake_score_net: nn.Module | None = None,
        real_score_net: nn.Module | None = None,
        anyflow_cfg: AnyFlowLossConfig | None = None,
    ):
        self.cfg = cfg or AnyFlowOnPolicyConfig()
        self.teacher_net = teacher_net
        self.fake_score_net = fake_score_net
        self.real_score_net = real_score_net if real_score_net is not None else teacher_net
        self.anyflow_loss: CrystalAnyFlowLoss | None = None
        if float(self.cfg.anyflow_cotrain_weight) > 0:
            self.anyflow_loss = CrystalAnyFlowLoss(anyflow_cfg, teacher_net=teacher_net)

    def set_score_nets(self, teacher_net: nn.Module, fake_score_net: nn.Module) -> None:
        self.teacher_net = teacher_net
        self.fake_score_net = fake_score_net
        for p in self.teacher_net.parameters():
            p.requires_grad_(False)
        self.teacher_net.eval()
        if self.anyflow_loss is not None:
            self.anyflow_loss.set_teacher(teacher_net)

    def _score_to_x1(
        self,
        score_net: nn.Module,
        interface,
        xt: Tensor,
        t: Tensor,
        f,
        chiral_bias: Optional[Tensor],
        *,
        fp32: bool,
    ) -> Tensor:
        device_type = "cuda" if xt.is_cuda else xt.device.type
        orig_dtype = xt.dtype
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=not fp32 and device_type == "cuda",
        ):
            xt_in = xt.float() if fp32 else xt
            t_in = t.float() if fp32 else t
            v = interface.pred(
                net=score_net,
                xt=xt_in,
                xsc=None,
                t=t_in,
                r=t_in,
                f=f,
                chiral_bias=chiral_bias,
            )
            x1 = interface.estimate_x1(xt_in, t_in, v)
            x1 = zero_com_suffix(x1, w=f.mask)
        return x1.to(dtype=orig_dtype)

    def _student_rollout(
        self,
        net: nn.Module,
        interface,
        C0,
        x0: Tensor,
        chiral_bias: Optional[Tensor],
        *,
        with_grad: bool,
        ddp_sync_last: bool,
    ) -> tuple[Tensor, int, int]:
        cfg = self.cfg
        device = x0.device
        student_steps = _synced_choice(cfg.student_steps_list, device)
        rollout_mode = str(cfg.rollout_mode).lower()
        grad_timestep = 0
        if rollout_mode == "shortcut":
            grad_timestep = _synced_randint(0, student_steps, device)
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            x_student = flow_map_rollout(
                interface,
                net,
                C0,
                x_init=x0,
                num_steps=student_steps,
                chiral_bias=chiral_bias,
                detach_between_jumps=bool(cfg.detach_between_jumps),
                use_self_cond=bool(cfg.student_self_cond),
                checkpoint_jumps=bool(cfg.checkpoint_jumps),
                rollout_mode=rollout_mode,
                grad_timestep=grad_timestep,
                n_jumps=int(cfg.n_jumps),
                ddp_sync_last=ddp_sync_last,
            )
        return x_student, student_steps, grad_timestep

    def _sample_dmd_t(self, batch_size: int, device, dtype) -> Tensor:
        cfg = self.cfg
        t = torch.rand(batch_size, device=device, dtype=dtype)
        lo, hi = float(cfg.dmd_t_min), float(cfg.dmd_t_max)
        return t * (hi - lo) + lo

    def _sample_discriminator_t(self, batch_size: int, device, dtype) -> Tensor:
        """Official fake-score FM uses logit-normal rather than uniform time."""
        t = torch.sigmoid(torch.randn(batch_size, device=device, dtype=dtype))
        lo, hi = float(self.cfg.dmd_t_min), float(self.cfg.dmd_t_max)
        return t.clamp(min=lo, max=hi)

    @torch.no_grad()
    def _dmd_grad(
        self,
        interface,
        pred_x1: Tensor,
        noisy: Tensor,
        t: Tensor,
        f,
        chiral_bias: Optional[Tensor],
    ) -> Tensor:
        if self.real_score_net is None or self.fake_score_net is None:
            raise RuntimeError("On-policy DMD requires real_score_net and fake_score_net")
        cfg = self.cfg
        x1_fake = self._score_to_x1(
            self.fake_score_net, interface, noisy, t, f, chiral_bias, fp32=False
        )
        x1_real = self._score_to_x1(
            self.real_score_net,
            interface,
            noisy,
            t,
            f,
            chiral_bias,
            fp32=bool(cfg.real_score_fp32),
        )
        return dmd_kl_gradient(
            pred_x1,
            x1_fake,
            x1_real,
            f.mask,
            normalize=bool(cfg.gradient_normalization),
        )

    def generator_loss(
        self,
        net: nn.Module,
        interface,
        C0,
        C1,
        *,
        chiral_bias: Optional[Tensor] = None,
        chirality_fn: Optional[Callable] = None,
        chiral_consistency_fn: Optional[Callable] = None,
    ) -> dict[str, Tensor]:
        del chirality_fn, chiral_consistency_fn
        cfg = self.cfg
        device = C1.x.device
        x0 = C0.x
        cotrain = self.anyflow_loss is not None and cfg.anyflow_cotrain_weight > 0

        if self.fake_score_net is not None:
            self.fake_score_net.eval()
        if self.teacher_net is not None:
            self.teacher_net.eval()

        pred_x1, student_steps, grad_timestep = self._student_rollout(
            net,
            interface,
            C0,
            x0,
            chiral_bias,
            with_grad=True,
            ddp_sync_last=not cotrain,
        )

        noise = _sample_prior_x(interface, C0)
        t = self._sample_discriminator_t(int(C1.batch_size), device, pred_x1.dtype)
        noisy = interface.sample_xt(noise, pred_x1, t).detach()
        grad = self._dmd_grad(interface, pred_x1, noisy, t, C0, chiral_bias)

        dmd = cfg.dmd_weight * dmd_identity_loss(
            pred_x1.double(),
            grad.double(),
            C0.mask,
            periodic_coord=bool(cfg.periodic_coord),
            lattice_weight=float(cfg.lattice_weight),
            coord_weight=float(cfg.coord_weight),
        )
        loss = dmd.to(dtype=pred_x1.dtype)

        loss_anyflow = torch.zeros((), device=device, dtype=loss.dtype)
        if cotrain:
            af = self.anyflow_loss(net, interface, C0, C1, chiral_bias=chiral_bias)
            loss_anyflow = af["loss"]
            loss = loss + cfg.anyflow_cotrain_weight * loss_anyflow

        return {
            "loss": loss,
            "loss_dmd": dmd.detach().to(dtype=loss.dtype),
            "loss_anyflow": loss_anyflow.detach(),
            "loss_disc": torch.zeros((), device=device, dtype=loss.dtype),
            "student_steps": torch.tensor(float(student_steps), device=device),
            "grad_timestep": torch.tensor(float(grad_timestep), device=device),
            "dmd_t_mean": t.mean().detach(),
        }

    def discriminator_loss(
        self,
        net: nn.Module,
        interface,
        C0,
        C1,
        *,
        chiral_bias: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if self.fake_score_net is None:
            raise RuntimeError("discriminator_loss requires fake_score_net")
        cfg = self.cfg
        device = C1.x.device
        student = net.module if hasattr(net, "module") else net
        student.eval()
        self.fake_score_net.train()
        if self.teacher_net is not None:
            self.teacher_net.eval()

        with torch.no_grad():
            pred_x1, student_steps, grad_timestep = self._student_rollout(
                net,
                interface,
                C0,
                C0.x,
                chiral_bias,
                with_grad=False,
                ddp_sync_last=True,
            )
            pred_x1 = pred_x1.detach()

        noise = _sample_prior_x(interface, C0)
        t = self._sample_dmd_t(int(C1.batch_size), device, pred_x1.dtype)
        xt = interface.sample_xt(noise, pred_x1, t)
        target = pred_x1 - noise
        v_pred = interface.pred(
            net=self.fake_score_net,
            xt=xt,
            xsc=None,
            t=t,
            r=t,
            f=C0,
            chiral_bias=chiral_bias,
        )
        loss_disc = _masked_mse(v_pred.float(), target.float(), C0.mask)
        return {
            "loss": loss_disc,
            "loss_disc": loss_disc.detach(),
            "student_steps": torch.tensor(float(student_steps), device=device),
            "grad_timestep": torch.tensor(float(grad_timestep), device=device),
            "dmd_t_mean": t.mean().detach(),
        }

    def __call__(self, *args, **kwargs) -> dict[str, Tensor]:
        return self.generator_loss(*args, **kwargs)
