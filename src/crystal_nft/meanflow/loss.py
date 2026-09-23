"""MeanFlow JVP loss for crystal flow maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from clari.pipelines.utils import masked_mean

from crystal_nft.meanflow.velocity import central_difference_dudt


def _unwrap_net(net: nn.Module) -> nn.Module:
    return net.module if hasattr(net, "module") else net


@dataclass
class MeanFlowLossConfig:
    time_sampler: str = "logit_normal"
    time_mu: float = -0.4
    time_sigma: float = 1.0
    ratio_r_not_equal_t: float = 0.75
    weighting: str = "uniform"
    adaptive_p: float = 1.0
    enantiomer_flip_p: float = 0.0
    aux_vol_weight: float = 1.0
    aux_ldd_weight: float = 5.0
    use_aux_losses: bool = True


class CrystalMeanFlowLoss:
    """Bootstrap target u* = v - (t-r) du/dt on Clari packed coordinates."""

    def __init__(self, cfg: MeanFlowLossConfig | None = None):
        self.cfg = cfg or MeanFlowLossConfig()

    def sample_time_steps(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        cfg = self.cfg
        if cfg.time_sampler == "uniform":
            samples = torch.rand(batch_size, 2, device=device)
        elif cfg.time_sampler == "logit_normal":
            normal = torch.randn(batch_size, 2, device=device) * cfg.time_sigma + cfg.time_mu
            samples = torch.sigmoid(normal)
        else:
            raise ValueError(f"Unknown time_sampler={cfg.time_sampler}")

        samples, _ = torch.sort(samples, dim=1)
        r, t = samples[:, 0], samples[:, 1]
        equal_mask = torch.rand(batch_size, device=device) < (1.0 - cfg.ratio_r_not_equal_t)
        r = torch.where(equal_mask, t, r)
        return r, t

    def __call__(
        self,
        net: nn.Module,
        interface,
        C0,
        C1,
        *,
        chiral_bias: Optional[Tensor] = None,
        chirality_fn: Optional[Callable[[Tensor, object], Tensor]] = None,
    ) -> dict[str, Tensor]:
        x0, x1 = C0.x, C1.x
        device = x1.device
        B = C1.batch_size

        if chirality_fn is not None and self.cfg.enantiomer_flip_p > 0:
            x1 = chirality_fn(x1, C1)

        r, t = self.sample_time_steps(B, device)
        xt = interface.sample_xt(x0, x1, t)
        v_t = x1 - x0

        xsc = None
        dit = getattr(_unwrap_net(net), "dit", _unwrap_net(net))
        if getattr(dit, "self_cond", False):
            # Avoid student no_grad forward under bf16 autocast before DDP (kills grads).
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                nsc = max(1, C0.batch_size // 2)
                fsc = C0.subset(slice(0, nsc))
                out = interface.pred(
                    net=_unwrap_net(net),
                    xt=xt[:nsc],
                    xsc=None,
                    t=t[:nsc],
                    r=r[:nsc],
                    f=fsc,
                    chiral_bias=chiral_bias[:nsc] if chiral_bias is not None else None,
                )
                xsc = torch.full_like(xt, torch.nan)
                xsc[:nsc] = interface.estimate_x1(xt[:nsc], t[:nsc], out)

        def fn(z, cur_r, cur_t):
            return interface.forward(
                net=net,
                xt=z,
                xsc=xsc,
                t=cur_t,
                r=cur_r,
                f=C0,
                chiral_bias=chiral_bias,
            )

        u = fn(xt, r, t)
        time_diff = (t - r).view(-1, 1, 1)

        dudt = central_difference_dudt(fn, xt, r, t, v_t, eps=1e-3)
        u_target = v_t - time_diff * dudt

        error = u - u_target.detach()
        loss_mid = torch.sum(error.reshape(B, -1) ** 2, dim=-1)

        if self.cfg.weighting == "adaptive":
            weights = 1.0 / (loss_mid.detach() + 1e-3).pow(self.cfg.adaptive_p)
            loss = (weights * loss_mid).mean()
        else:
            loss = loss_mid.mean()

        err_lattice = error[:, :3].pow(2).mean()
        err_coord = masked_mean(error[:, 3:].pow(2), C0.mask.unsqueeze(-1), dim=[1, 2]).mean()

        loss_vol = torch.tensor(0.0, device=device)
        loss_ldd = torch.tensor(0.0, device=device)
        if self.cfg.use_aux_losses and self.cfg.aux_vol_weight + self.cfg.aux_ldd_weight > 0:
            # estimate_x1 expects instantaneous v; convert dual-time u via MeanFlow identity.
            from crystal_nft.meanflow.velocity import flow_map_to_instantaneous_velocity

            v_inst = flow_map_to_instantaneous_velocity(u, xt, r, t, dudt.detach())
            pred_x1 = interface.estimate_x1(xt, t, v_inst)
            if self.cfg.aux_vol_weight > 0:
                loss_vol = interface._vol_losses(pred_x1, x1).mean()
            if self.cfg.aux_ldd_weight > 0:
                loss_ldd = interface._ldd_losses(pred_x1, x1, f=C0).mean()
            loss = loss + self.cfg.aux_vol_weight * loss_vol + self.cfg.aux_ldd_weight * loss_ldd

        return {
            "loss": loss,
            "loss_lattice": err_lattice.detach(),
            "loss_coord": err_coord.detach(),
            "loss_vol": loss_vol.detach() if torch.is_tensor(loss_vol) else loss_vol,
            "loss_ldd": loss_ldd.detach() if torch.is_tensor(loss_ldd) else loss_ldd,
            "loss_mean_ref": error.pow(2).mean().detach(),
        }
