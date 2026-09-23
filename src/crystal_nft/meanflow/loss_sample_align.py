"""Few-step sample alignment: teacher ODE vs student flow-map.

Same molecule and the same prior noise: a frozen Clari teacher integrates
Euler steps; the MeanFlow student integrates a few-step flow map. The
student crystal is aligned to the teacher (or GT) crystal on lattice,
periodic fractional coordinates, volume, and bonded distances (LDD).

This follows AnyFlow Stage-2 *mechanics* (not the SD DMD objective):

- Default student rollout is the 3-jump shortcut (``training_rollout``):
  prev / current / post, always at most 3 transformer forwards.
- ``cotrain_forward_kl``: mix Stage-1 AnyFlow velocity loss so 4-step
  capability is not overwritten.
- Official shortcut default is ``rollout_detach_between_jumps=False``.
- DDP: student jumps except the last go through ``no_sync()``; if
  AnyFlow cotrain is on, the cotrain forward is the reducer-visible call.

Do **not** use adaptive p=1 on lattice/coord MSE. That maps large errors
to ``err/(err+eps) ≈ 1`` and kills the gradient (``eps/(err+eps)^2``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from clari.chem import Crystal
from clari.pipelines.utils import masked_mean

from crystal_nft.meanflow.loss_anyflow import AnyFlowLossConfig, CrystalAnyFlowLoss
from crystal_nft.meanflow.sampler import (
    euler_instantaneous_rollout,
    flow_map_rollout,
    heun_instantaneous_rollout,
    interval_flow_map_jump,
)


@dataclass
class SampleAlignLossConfig:
    teacher_steps: int = 50
    student_steps_list: list[int] = field(default_factory=lambda: [4, 8])
    detach_between_jumps: bool = False
    checkpoint_jumps: bool = False
    teacher_self_cond: bool = True
    student_self_cond: bool = False
    lattice_weight: float = 1.0
    coord_weight: float = 1.0
    vol_weight: float = 0.5
    ldd_weight: float = 1.0
    # Multiply the packed geometry terms so they are O(1) like AnyFlow loss.
    # Probe scale: mse_lat~0.02, mse_coord~0.06, vol~0.03, ldd~0.22 → ~0.3 raw.
    geom_scale: float = 8.0
    # teacher = same-noise teacher sample; data = ground-truth C1 (debug / ablation)
    align_to: str = "teacher"
    # euler/heun: instantaneous Clari ODE. interval: full dual-time Euler.
    # interval_compose: local 16-step jump vs two frozen 32-step jumps
    # along the teacher interval trajectory (eval-aligned, not endpoint MSE).
    teacher_sampler: str = "euler"
    anyflow_cotrain_weight: float = 1.0
    # Adaptive p=1 saturates and must stay off for geometry MSE.
    use_adaptive: bool = False
    norm_p: float = 1.0
    norm_eps: float = 0.01
    # AnyFlow Stage-2 default: 3-jump shortcut. "full" unrolls every step.
    rollout_mode: str = "shortcut"
    periodic_coord: bool = True
    interval_rho: float = 0.75
    interval_schedule: str = "power"


def _synced_choice(choices: list[int], device: torch.device) -> int:
    vals = [int(x) for x in choices]
    if not vals:
        raise ValueError("student_steps_list must be non-empty")
    if len(vals) == 1:
        return vals[0]
    t = torch.zeros(1, dtype=torch.long, device=device)
    use_dist = dist.is_initialized() and dist.get_world_size() > 1
    if (not use_dist) or dist.get_rank() == 0:
        t[0] = vals[int(torch.randint(len(vals), (1,)).item())]
    if use_dist:
        dist.broadcast(t, src=0)
    return int(t.item())


def _synced_randint(low: int, high: int, device: torch.device) -> int:
    """Inclusive-low, exclusive-high integer, broadcast from rank 0."""
    lo = int(low)
    hi = int(high)
    if hi <= lo:
        return lo
    t = torch.zeros(1, dtype=torch.long, device=device)
    use_dist = dist.is_initialized() and dist.get_world_size() > 1
    if (not use_dist) or dist.get_rank() == 0:
        t[0] = int(torch.randint(lo, hi, (1,)).item())
    if use_dist:
        dist.broadcast(t, src=0)
    return int(t.item())


def periodic_frac_coord_mse(x_pred: Tensor, x_tgt: Tensor, mask: Tensor) -> Tensor:
    """Per-sample MSE of wrapped fractional coords in the target unit cell.

    Packed Clari ``x`` stores cartesian coords / COORD_NORM, not fractional
    values, so a naive packed-coord MSE has no periodic wrap.
    """
    coord_norm = float(Crystal.COORD_NORM)
    lattice = 2.0 * coord_norm * x_tgt[:, :3]
    cart_pred = coord_norm * x_pred[:, 3:]
    cart_tgt = coord_norm * x_tgt[:, 3:]
    inv = torch.linalg.pinv(lattice)
    frac_pred = torch.einsum("bji,bnj->bni", inv, cart_pred)
    frac_tgt = torch.einsum("bji,bnj->bni", inv, cart_tgt)
    delta = frac_pred - frac_tgt
    delta = delta - torch.round(delta)
    return masked_mean(delta.pow(2), mask.unsqueeze(-1), dim=[1, 2])


class CrystalSampleAlignLoss:
    """Align student few-step crystals to teacher many-step crystals."""

    def __init__(
        self,
        cfg: SampleAlignLossConfig | None = None,
        *,
        teacher_net: nn.Module | None = None,
        align_net: nn.Module | None = None,
        anyflow_cfg: AnyFlowLossConfig | None = None,
    ):
        self.cfg = cfg or SampleAlignLossConfig()
        self.teacher_net = teacher_net
        # Frozen instantaneous net for Heun/Euler crystal targets. Separate
        # from ``teacher_net`` so AnyFlow cotrain can keep the rank800 field
        # while the geometry target is Clari-M Heun-50.
        self.align_net = align_net if align_net is not None else teacher_net
        self.anyflow_loss: CrystalAnyFlowLoss | None = None
        if float(self.cfg.anyflow_cotrain_weight) > 0:
            self.anyflow_loss = CrystalAnyFlowLoss(anyflow_cfg, teacher_net=teacher_net)

    def set_teacher(self, teacher_net: nn.Module) -> None:
        self.teacher_net = teacher_net
        for p in self.teacher_net.parameters():
            p.requires_grad_(False)
        self.teacher_net.eval()
        if self.anyflow_loss is not None:
            self.anyflow_loss.set_teacher(teacher_net)

    def set_align_net(self, align_net: nn.Module) -> None:
        self.align_net = align_net
        for p in self.align_net.parameters():
            p.requires_grad_(False)
        self.align_net.eval()

    def _maybe_adaptive(self, per_sample: Tensor) -> Tensor:
        if not self.cfg.use_adaptive:
            return per_sample
        cfg = self.cfg
        wt = (per_sample.detach() + cfg.norm_eps).pow(cfg.norm_p)
        return per_sample / wt

    def __call__(
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
        B = int(C1.batch_size)
        x0 = C0.x

        student_steps = _synced_choice(cfg.student_steps_list, device)
        rollout_mode = str(cfg.rollout_mode).lower()
        teacher_sampler = str(cfg.teacher_sampler).lower().strip()
        grad_timestep = 0
        if rollout_mode == "shortcut":
            grad_timestep = _synced_randint(0, student_steps, device)

        compose = teacher_sampler in ("interval_compose", "compose") or rollout_mode in (
            "compose",
            "local_jump",
        )
        cotrain = self.anyflow_loss is not None and cfg.anyflow_cotrain_weight > 0

        if compose:
            target_net = self.align_net if self.align_net is not None else self.teacher_net
            if target_net is None:
                raise RuntimeError(
                    "SampleAlignLoss compose mode requires align_net/teacher_net"
                )
            target_net.eval()
            fine = max(int(cfg.teacher_steps), 1)
            coarse = max(int(student_steps), 1)
            ratio = max(1, fine // coarse)
            k = _synced_randint(0, coarse, device)
            grad_timestep = k
            i0 = int(k) * int(ratio)
            z = x0
            with torch.no_grad():
                for i in range(i0):
                    z = interval_flow_map_jump(
                        interface,
                        target_net,
                        C0,
                        z,
                        t_from=float(i) / float(fine),
                        t_to=float(i + 1) / float(fine),
                        chiral_bias=chiral_bias,
                    )
                z_start = z.detach()
                z_tgt = z_start
                for j in range(ratio):
                    i = i0 + j
                    z_tgt = interval_flow_map_jump(
                        interface,
                        target_net,
                        C0,
                        z_tgt,
                        t_from=float(i) / float(fine),
                        t_to=float(i + 1) / float(fine),
                        chiral_bias=chiral_bias,
                    )
                x_target = z_tgt.detach()
            x_student = interval_flow_map_jump(
                interface,
                net,
                C0,
                z_start,
                t_from=float(i0) / float(fine),
                t_to=float(i0 + ratio) / float(fine),
                chiral_bias=chiral_bias,
            )
        else:
            if cfg.align_to == "data":
                x_target = C1.x.detach()
            else:
                target_net = self.align_net if self.align_net is not None else self.teacher_net
                if target_net is None:
                    raise RuntimeError(
                        "SampleAlignLoss requires align_net/teacher_net when align_to='teacher'"
                    )
                target_net.eval()
                if teacher_sampler == "heun":
                    x_target = heun_instantaneous_rollout(
                        interface,
                        target_net,
                        C0,
                        x_init=x0,
                        num_steps=int(cfg.teacher_steps),
                        chiral_bias=chiral_bias,
                    ).detach()
                elif teacher_sampler == "euler":
                    x_target = euler_instantaneous_rollout(
                        interface,
                        target_net,
                        C0,
                        x_init=x0,
                        num_steps=int(cfg.teacher_steps),
                        chiral_bias=chiral_bias,
                        use_self_cond=bool(cfg.teacher_self_cond),
                    ).detach()
                elif teacher_sampler in ("interval", "flowmap"):
                    with torch.no_grad():
                        x_target = flow_map_rollout(
                            interface,
                            target_net,
                            C0,
                            x_init=x0,
                            num_steps=int(cfg.teacher_steps),
                            chiral_bias=chiral_bias,
                            detach_between_jumps=False,
                            use_self_cond=bool(cfg.teacher_self_cond),
                            checkpoint_jumps=False,
                            rollout_mode="full",
                            ddp_sync_last=False,
                            rho=float(cfg.interval_rho),
                            schedule=str(cfg.interval_schedule),
                        ).detach()
                else:
                    raise ValueError(
                        "teacher_sampler must be euler|heun|interval|interval_compose, "
                        f"got {cfg.teacher_sampler!r}"
                    )

            # If AnyFlow cotrain will run a DDP-visible student forward after the
            # rollout, keep every shortcut jump under no_sync.
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
                ddp_sync_last=not cotrain,
                rho=float(cfg.interval_rho),
                schedule=str(cfg.interval_schedule),
            )

        err = x_student - x_target
        mse_lat_raw = err[:, :3].pow(2).reshape(B, -1).mean(dim=-1)
        mse_coord_packed = masked_mean(err[:, 3:].pow(2), C0.mask.unsqueeze(-1), dim=[1, 2])
        if cfg.periodic_coord:
            mse_coord_raw = periodic_frac_coord_mse(x_student, x_target, C0.mask)
            if not torch.isfinite(mse_coord_raw).all():
                mse_coord_raw = mse_coord_packed
        else:
            mse_coord_raw = mse_coord_packed

        loss_lattice = self._maybe_adaptive(mse_lat_raw).mean()
        loss_coord = self._maybe_adaptive(mse_coord_raw).mean()

        loss_vol = torch.zeros((), device=device, dtype=loss_lattice.dtype)
        loss_ldd = torch.zeros((), device=device, dtype=loss_lattice.dtype)
        if cfg.vol_weight > 0:
            loss_vol = interface._vol_losses(x_student, x_target).mean()
        if cfg.ldd_weight > 0:
            loss_ldd = interface._ldd_losses(x_student, x_target, f=C0).mean()

        geom = (
            cfg.lattice_weight * loss_lattice
            + cfg.coord_weight * loss_coord
            + cfg.vol_weight * loss_vol
            + cfg.ldd_weight * loss_ldd
        )
        loss = cfg.geom_scale * geom

        loss_anyflow = torch.zeros((), device=device, dtype=loss.dtype)
        if cotrain:
            af = self.anyflow_loss(net, interface, C0, C1, chiral_bias=chiral_bias)
            loss_anyflow = af["loss"]
            loss = loss + cfg.anyflow_cotrain_weight * loss_anyflow

        return {
            "loss": loss,
            "loss_lattice": loss_lattice.detach(),
            "loss_coord": loss_coord.detach(),
            "loss_vol": loss_vol.detach(),
            "loss_ldd": loss_ldd.detach(),
            "loss_anyflow": loss_anyflow.detach(),
            "mse_lat_raw": mse_lat_raw.mean().detach(),
            "mse_coord_raw": mse_coord_raw.mean().detach(),
            "mse_coord_packed": mse_coord_packed.mean().detach(),
            "loss_geom": geom.detach(),
            "student_steps": torch.tensor(float(student_steps), device=device),
            "teacher_steps": torch.tensor(float(cfg.teacher_steps), device=device),
            "grad_timestep": torch.tensor(float(grad_timestep), device=device),
        }
