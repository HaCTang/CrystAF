"""Improved MeanFlow (iMF) loss for crystal flow maps — arXiv:2512.02012."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from clari.pipelines.utils import masked_mean

from crystal_nft.meanflow.loss import MeanFlowLossConfig, _unwrap_net
from crystal_nft.meanflow.velocity import (
    central_difference_dudt,
    flow_map_to_instantaneous_velocity,
)


@dataclass
class IMFLossConfig(MeanFlowLossConfig):
    """Crystal iMF: V-loss + auxiliary instantaneous velocity loss (no CFG)."""

    data_proportion: float = 0.5
    cd_velocity_source: str = "u_self"
    cd_eps: float = 1e-3
    norm_p: float = 1.0
    norm_eps: float = 0.01
    loss_v_weight: float = 1.0


class CrystalIMFLoss:
    """
    iMF objective adapted to Clari crystals (single conditional path, no class CFG).

    - Compound velocity V = u + (t-r) * stop_grad(du/dt)
    - Target v_g = x1 - x0 (instantaneous OT velocity)
    - JVP tangent uses u(z,t,t) when cd_velocity_source=u_self (iMF Alg. 2)
    - Fraction ``data_proportion`` of samples use r=t (flow-matching / boundary)
    """

    def __init__(self, cfg: IMFLossConfig | None = None):
        self.cfg = cfg or IMFLossConfig()

    def sample_time_steps(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.cfg
        if cfg.time_sampler == "logit_normal":
            normal = torch.randn(batch_size, 2, device=device) * cfg.time_sigma + cfg.time_mu
            samples = torch.sigmoid(normal)
        elif cfg.time_sampler == "uniform":
            samples = torch.rand(batch_size, 2, device=device)
        else:
            raise ValueError(f"Unknown time_sampler={cfg.time_sampler}")
        samples, _ = torch.sort(samples, dim=1)
        r, t = samples[:, 0], samples[:, 1]
        fm_mask = torch.rand(batch_size, device=device) < cfg.data_proportion
        r = torch.where(fm_mask, t, r)
        return r, t, fm_mask

    def _adaptive_weight(self, per_sample: Tensor) -> Tensor:
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
        chirality_fn: Optional[Callable[[Tensor, object], Tensor]] = None,
    ) -> dict[str, Tensor]:
        cfg = self.cfg
        x0, x1 = C0.x, C1.x
        device = x1.device
        B = C1.batch_size

        if chirality_fn is not None and cfg.enantiomer_flip_p > 0:
            x1 = chirality_fn(x1, C1)

        r, t, fm_mask = self.sample_time_steps(B, device)
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

        if cfg.cd_velocity_source == "u_self":
            with torch.no_grad():
                v_c = fn(xt, t, t)
        elif cfg.cd_velocity_source == "noise_minus_data":
            v_c = v_t
        else:
            raise ValueError(f"Unknown cd_velocity_source={cfg.cd_velocity_source}")

        dudt = central_difference_dudt(fn, xt, r, t, v_c, eps=cfg.cd_eps)
        time_diff = (t - r).view(-1, 1, 1)
        v_compound = u + time_diff * dudt.detach()
        v_g = v_t.detach()

        err_u = (v_compound - v_g).reshape(B, -1)
        loss_u = self._adaptive_weight(torch.sum(err_u**2, dim=-1)).mean()

        v_aux = flow_map_to_instantaneous_velocity(u, xt, r, t, dudt)
        err_v = (v_aux - v_g).reshape(B, -1)
        loss_v = self._adaptive_weight(torch.sum(err_v**2, dim=-1)).mean()

        loss = loss_u + cfg.loss_v_weight * loss_v

        error = v_compound - v_g
        err_lattice = error[:, :3].pow(2).mean()
        err_coord = masked_mean(error[:, 3:].pow(2), C0.mask.unsqueeze(-1), dim=[1, 2]).mean()

        loss_vol = torch.tensor(0.0, device=device)
        loss_ldd = torch.tensor(0.0, device=device)
        if cfg.use_aux_losses and cfg.aux_vol_weight + cfg.aux_ldd_weight > 0:
            # estimate_x1 needs instantaneous v, not dual-time flow-map u.
            v_inst = flow_map_to_instantaneous_velocity(u, xt, r, t, dudt.detach())
            pred_x1 = interface.estimate_x1(xt, t, v_inst)
            if cfg.aux_vol_weight > 0:
                loss_vol = interface._vol_losses(pred_x1, x1).mean()
            if cfg.aux_ldd_weight > 0:
                loss_ldd = interface._ldd_losses(pred_x1, x1, f=C0).mean()
            loss = loss + cfg.aux_vol_weight * loss_vol + cfg.aux_ldd_weight * loss_ldd

        return {
            "loss": loss,
            "loss_u": loss_u.detach(),
            "loss_v": loss_v.detach(),
            "loss_lattice": err_lattice.detach(),
            "loss_coord": err_coord.detach(),
            "loss_vol": loss_vol.detach() if torch.is_tensor(loss_vol) else loss_vol,
            "loss_ldd": loss_ldd.detach() if torch.is_tensor(loss_ldd) else loss_ldd,
            "fm_frac": fm_mask.float().mean().detach(),
        }
