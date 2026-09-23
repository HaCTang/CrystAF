"""MeanFlow crystal interface: linear path + dual-time flow map on Clari state."""

from __future__ import annotations

import os

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from clari.geometry import zero_com_suffix
from clari.pipelines.base.interfaces import SiTInterface
from clari.pipelines.utils import bcast_right, masked_mean

from crystal_nft.meanflow.net import MeanFlowDiTWrapper


def _unwrap_net(net: nn.Module) -> nn.Module:
    return net.module if hasattr(net, "module") else net


class MeanFlowCrystalInterface(SiTInterface):
    """
    All-atom flexible crystal flow map (lattice + fractional coords in Clari packing).

    Clari time runs from noise (0) to data (1). Sampling uses
    ``z_t = z_r + (t-r) U(z_r, r, t)``.
    """

    def sample_prior(self, C):
        """Prior with the requested handedness already installed (if enabled).

        `chiral_prior_align` permutes two substituents' noise rows, which leaves
        the i.i.d. Gaussian prior exactly invariant (verified: the noise multiset
        is unchanged) while inverting the stereocentre's triple product. This is
        the only point in the trajectory where fixing handedness is free -- a
        flip is a permutation, not a rotation, so doing it once atoms are placed
        is a large rearrangement that distorts bonds and pins PB at ~69 (six
        configurations tested; the 82.26 baseline is never recovered).
        """
        out = super().sample_prior(C)
        if not bool(int(os.environ.get("CRYSTAF_CHIRAL_PRIOR", "0"))):
            return out
        from crystal_nft.meanflow.stereo import (
            chiral_prior_align,
            get_active_stereo_geometry,
        )

        centers, label_sign, valid = get_active_stereo_geometry()
        if centers is None or label_sign is None:
            return out
        spec = type("_S", (), {})()
        spec.centers, spec.label_sign, spec.valid = centers, label_sign, valid
        x = out.x
        coords = chiral_prior_align(x[:, 3:, :], spec)
        return out.replace(x=torch.cat([x[:, :3, :], coords], dim=1))

    def forward(
        self,
        net: nn.Module,
        xt: Tensor,
        xsc: Tensor | None,
        t: Tensor,
        f,
        *,
        r: Optional[Tensor] = None,
        chiral_bias: Optional[Tensor] = None,
    ) -> Tensor:
        if r is None:
            r = t
        if isinstance(t, float) or (isinstance(t, Tensor) and t.ndim == 0):
            t = torch.full([f.batch_size], float(t), device=xt.device, dtype=xt.dtype)
        if isinstance(r, float) or (isinstance(r, Tensor) and r.ndim == 0):
            r = torch.full([f.batch_size], float(r), device=xt.device, dtype=xt.dtype)
        if xsc is None:
            xsc = torch.full_like(xt, torch.nan)

        core = _unwrap_net(net)
        if isinstance(core, MeanFlowDiTWrapper):
            # FiLM stereo path: turn the per-crystal handedness descriptor into a
            # `cond` bias. Reuses the existing `_mf_chiral_bias` slot, which
            # `patch_dit_chiral_stem` adds to `stem_cond`'s input.
            stereo_cond = getattr(core, "_stereo_cond", None)
            if stereo_cond is not None and chiral_bias is None:
                from crystal_nft.meanflow.stereo import get_active_stereo_cell_desc

                desc = get_active_stereo_cell_desc()
                if desc is not None:
                    d = desc.to(
                        device=stereo_cond.net[0].weight.device,
                        dtype=stereo_cond.net[0].weight.dtype,
                    )
                    b = xt.shape[0]
                    if d.shape[0] == 1 and b > 1:
                        d = d.expand(b, -1)
                    elif d.shape[0] > b:
                        d = d[:b]
                    if d.shape[0] == b:
                        chiral_bias = stereo_cond(d)
            if chiral_bias is not None:
                core.dit._mf_chiral_bias = chiral_bias
            else:
                # Clear stale chiral bias from a previous batch (e.g. NFT train step
                # with B=8) so it does not broadcast-expand a B=1 sampling call.
                core.dit._mf_chiral_bias = None
            # DDP allows only one forward per backward. Self-cond / CD run under
            # no_grad and must unwrap; the grad-enabled student call goes through
            # the DDP container so embed_deltat stays in the reducer graph.
            if net is not core and torch.is_grad_enabled():
                pred = net(xt, xsc, r, t, f)
            else:
                pred = core.crystal_forward(x=xt, xsc=xsc, r=r, t=t, f=f)
        else:
            # Unwrapped Clari DiT / InstantaneousDiT. Grad-enabled DDP must
            # go through the container (crystal_forward on .module skips it).
            if net is not core and torch.is_grad_enabled():
                pred = net(xt, xsc, t, f)
            else:
                pred = core.crystal_forward(x=xt, xsc=xsc, t=t, f=f)

        # NOTE: the parity-sensitive residual is applied inside
        # MeanFlowDiTWrapper.crystal_forward, not here -- applying it outside
        # the DDP-wrapped module breaks the reducer ("mark a variable ready
        # only once"). zero_com_suffix still re-centres it below.
        pred = zero_com_suffix(pred, w=f.mask)
        return pred

    def pred(
        self,
        net: nn.Module,
        xt: Tensor,
        xsc: Tensor | None,
        t: Tensor,
        f,
        *,
        r: Optional[Tensor] = None,
        chiral_bias: Optional[Tensor] = None,
    ) -> Tensor:
        return self.forward(
            net=net, xt=xt, xsc=xsc, t=t, f=f, r=r, chiral_bias=chiral_bias
        )

    def flow_map_step(
        self,
        net: nn.Module,
        xt: Tensor,
        xsc: Tensor | None,
        t_from: Tensor,
        t_to: Tensor,
        f,
        *,
        chiral_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward Clari-time step: z_{t_to} = z_{t_from} + (t_to-t_from) u(z, r=t_from, t=t_to).

        Clari convention is t=0 noise → t=1 data (same as ``sample_xt``).
        """
        # Pred dual-time args: end time t = t_to, start time r = t_from (r <= t).
        r = torch.minimum(t_from, t_to)
        t = torch.maximum(t_from, t_to)
        u = self.pred(
            net=net,
            xt=xt,
            xsc=xsc,
            t=t,
            r=r,
            f=f,
            chiral_bias=chiral_bias,
        )
        dt = bcast_right(t_to - t_from, xt)
        return xt + dt * u

    def loss(
        self,
        net: nn.Module,
        batch: tuple,
        *,
        chiral_bias: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        from crystal_nft.meanflow.loss import CrystalMeanFlowLoss

        C0, C1 = batch
        return CrystalMeanFlowLoss()(net, self, C0, C1, chiral_bias=chiral_bias)

    def fm_supervision_loss(
        self,
        net: nn.Module,
        batch: tuple,
        *,
        chiral_bias: Optional[Tensor] = None,
        chiral_consistency_fn: Optional[Callable] = None,
        chiral_loss_weight: float = 0.0,
        low_t_frac: float = 0.0,
        low_t_max: float = 0.25,
        mismatch_frac: float = 0.0,
        mismatch_t_lo: float = 0.35,
        mismatch_t_hi: float = 0.85,
        chiral_on_flow_map: bool = False,
        self_state_frac: float = 0.0,
        self_state_steps: int = 4,
        self_state_denom_min: float = 0.2,
        self_state_hinge_only: bool = False,
        branch_l2_weight: float = 0.0,
    ) -> dict[str, Tensor]:
        """Standard Clari flow-matching loss via u(x,t,t) for warmup / fine-tune init.

        ``chiral_consistency_fn`` adds a chirality hinge on the endpoint estimate.
        Without it this path silently ignores ``stereo_loss_weight`` entirely --
        which made a full-scale "hinge" run a no-op (``stereo_n=0``). The hinge
        matters because at t~0 the handedness-dependent share of the FM target is
        small, so plain MSE is nearly minimised by the achiral compromise;
        measured on a controlled probe, adding it moves tag-following from 0.50
        to 0.83 while pure FM stays at chance.
        """
        C0, C1 = batch
        x0, x1 = C0.x, C1.x
        if bool(int(os.environ.get("CRYSTAF_CHIRAL_PRIOR", "0"))):
            # Train with the same prior the sampler will use: handedness set at
            # t = 0 by permuting noise rows, so the flow only has to PRESERVE it
            # instead of inventing it late (which is what costs geometry).
            from crystal_nft.meanflow.stereo import (
                chiral_prior_align,
                get_active_stereo_geometry,
            )

            _c, _s, _v = get_active_stereo_geometry()
            if _c is not None and _s is not None:
                _spec = type("_S", (), {})()
                _spec.centers, _spec.label_sign, _spec.valid = _c, _s, _v
                x0 = torch.cat(
                    [x0[:, :3, :], chiral_prior_align(x0[:, 3:, :], _spec)], dim=1
                )
                C0 = C0.replace(x=x0)
        t = self.sample_t([C1.batch_size], device=C1.x.device)
        if low_t_frac > 0:
            # Clari trains with t ~ Beta(1.8, 1): only 11.5% of mass below t=0.3,
            # 1.5% below t=0.1. But the CIP tag is the ONLY source of handedness
            # near t=0 -- at larger t, x_t is built from the true x1 and already
            # encodes it. So the chirality objective was almost never evaluated
            # where it is the sole signal. Flow matching is valid for any t
            # distribution, so mix low-t samples in.
            lo = torch.rand_like(t) * float(low_t_max)
            take = torch.rand_like(t) < float(low_t_frac)
            t = torch.where(take, lo, t)
        if mismatch_frac > 0:
            # Training always builds x_t from the SAME enantiomer the tag names,
            # so the model never sees a state whose handedness contradicts its
            # tag -- and never learns to override one. Measured on v5: the t=0
            # endpoint estimate follows the tag at 0.94 train / 0.97 val, but the
            # fully sampled trajectory follows it at 0.50, because along its own
            # trajectory the state's handedness sits near chance well past t=0.6
            # while the model has learned to read x_t instead of the tag:
            #   t         chirality(x_t)   chirality(x1_hat)
            #   0.0-0.5      ~0.50            0.97-1.00
            #   0.7           0.53             0.70
            #   0.9           0.63             0.57
            #
            # So build x_t from the OPPOSITE enantiomer for a fraction of
            # samples while the target and the tag keep asking for the correct
            # one -- but only inside a band where a correction is reachable.
            # Above `mismatch_t_hi` the model holds just (1 - t) * U of
            # authority over the endpoint, so demanding the opposite handedness
            # of an almost fully formed state is unsatisfiable: v7 applied it out
            # to t=1 and never learned even the in-sample tag-following v5 had by
            # the same step. Below `mismatch_t_lo` there is no handedness in the
            # state to override, which is why v6 stayed flat for 280 steps.
            mm = (
                (torch.rand(x1.shape[0], device=x1.device) < float(mismatch_frac))
                & (t > float(mismatch_t_lo))
                & (t < float(mismatch_t_hi))
            )
            # Inlined: a module import inside this hot loop under DDP coincided
            # with repeated SIGSEGVs (3 crashes in 10 min; 0 without this path).
            # Rows 0-2 are the lattice and must NOT flip -- negating all of x
            # re-describes the basis instead of mirroring the structure.
            x1_state = x1.clone()
            x1_state[:, 3:] = torch.where(mm.view(-1, 1, 1), -x1[:, 3:], x1[:, 3:])
            xt = self.sample_xt(x0, x1_state, t)
        else:
            mm = None
            xt = self.sample_xt(x0, x1, t)

        if self_state_frac > 0:
            # Every variant so far supervised states built as (1-t)x0 + t*x1
            # from the REAL crystal, which at large t is 80-90% finished crystal
            # and therefore leaks the handedness. The sampler's state does not.
            # Measured on a 50-step trajectory (200 val centres):
            #   t          0.00  0.20  0.50  0.70  0.80  0.90
            #   chir(x_t)  0.53  0.53  0.55  0.57  0.58  0.55   <- never chiral
            #   chir(x1^)  0.86  0.89  0.82  0.66  0.54  0.51   <- intent decays
            # and a step sweep puts the collapse between N=4 (0.63) and N=8
            # (0.54) while N=1/2 hold 0.87/0.85. So at large t the model faces an
            # achiral state with only the tag to go on, and it never learned to
            # use the tag there because training never put it in that position.
            #
            # Reproduce that state by ROLLING OUT the model's own Euler steps to
            # time t (stop-grad), rather than interpolating toward a finished
            # crystal. An earlier version of this used the one-shot endpoint
            # estimate x1_hat, but that is chirality-CORRECT (0.855), so the
            # synthetic state stayed chiral and taught nothing new -- it lifted
            # N=1/2 and left N>=8 at chance.
            with torch.no_grad(), torch.autocast(device_type=xt.device.type, enabled=False):
                core_ro = _unwrap_net(net)
                z_ro = x0
                k = max(1, int(self_state_steps))
                for j in range(k):
                    t_j = t * (float(j) / float(k))
                    u_j = self.pred(
                        net=core_ro, xt=z_ro, xsc=None, t=t_j, r=t_j, f=C0,
                        chiral_bias=chiral_bias,
                    )
                    z_ro = z_ro + (t / float(k)).view(
                        *([-1] + [1] * (x1.ndim - 1))
                    ) * u_j
                z_ro = z_ro.detach()
            take_self = (torch.rand(x1.shape[0], device=x1.device) < float(self_state_frac))
            sh_s = [-1] + [1] * (x1.ndim - 1)
            xt = torch.where(take_self.view(*sh_s), z_ro, xt)
        else:
            take_self = None

        xsc = None
        core = _unwrap_net(net)
        if getattr(core, "self_cond", False) or getattr(getattr(core, "dit", core), "self_cond", False):
            with torch.no_grad(), torch.autocast(device_type=xt.device.type, enabled=False):
                nsc = max(1, C0.batch_size // 2)
                fsc = C0.subset(slice(0, nsc))
                out = self.pred(
                    net=core,
                    xt=xt[:nsc],
                    xsc=None,
                    t=t[:nsc],
                    r=t[:nsc],
                    f=fsc,
                    chiral_bias=chiral_bias[:nsc] if chiral_bias is not None else None,
                )
                xsc = torch.full_like(xt, torch.nan)
                xsc[:nsc] = self.estimate_x1(xt[:nsc], t[:nsc], out)

        pred = self.pred(
            net=net,
            xt=xt,
            xsc=xsc,
            t=t,
            r=t,
            f=C0,
            chiral_bias=chiral_bias,
        )
        true = self.target(x0, x1, t)
        losses = F.mse_loss(pred, true, reduction="none")
        if take_self is not None and bool(take_self.any()) and self_state_hinge_only:
            # Self-state rows exist for ONE reason: to show the parity branch the
            # blob-like states the sampler actually visits, so it learns to fire
            # there. They do not need to teach geometry -- and regressing an
            # off-manifold rolled-out state toward x1 is a large, geometrically
            # meaningless target that corrupts the very MSE term structure
            # depends on. Measured cost of letting them into the MSE:
            #   frac 0.00 -> PB 82.26 clash 10.59 PDD 10.15 (chirality ~0.50)
            #   frac 0.60 -> PB 68.98 clash 23.03 PDD 13.28 (chirality ~0.80)
            #   frac 0.75 -> PB 64.64 clash 34.40 PDD 22.67 (chirality ~0.90)
            # So drop them from the MSE and keep only the chirality hinge, which
            # is the part doing the useful work. Geometry is then defined purely
            # by on-manifold states, exactly as in the clean v12 baseline.
            sh_t = [-1] + [1] * (losses.ndim - 1)
            losses = losses * (~take_self).to(losses.dtype).view(*sh_t)
        elif take_self is not None and bool(take_self.any()):
            # A rolled-out state is NOT on the straight line from x0 to x1, so
            # x1 - x0 is not the velocity that reaches x1 from it -- the same
            # trap the enantiomer override fell into. The velocity that does is
            # (x1 - x_t)/(1 - t), which explodes as t -> 1.
            #
            # Endpoint space (||x_t + (1-t)U - x1||^2) bounds it, but carries an
            # implicit (1-t)^2 weight -- at t=0.8 that is 25x LESS gradient than
            # a normal row, and large t is exactly where these states exist to
            # teach: the sampler's state there is achiral (chir(x_t) ~ 0.55 out
            # to t=0.9) so the CIP tag is the only signal left. Damping that
            # region defeats the purpose.
            #
            # So use the velocity target with a clamped denominator: full
            # gradient at large t, magnitude capped at 5x a normal target.
            sh_t = [-1] + [1] * (losses.ndim - 1)
            # The state-distribution benefit only needs the model to SEE blob-like
            # states; it does not need huge targets. denom_min caps the target at
            # 1/denom_min times a normal one (0.2 -> 5x, 0.4 -> 2.5x), and that
            # magnitude -- not the states themselves -- is the likely source of
            # the structural cost (self-state at frac 0.5-0.75 cost ~15 clash).
            denom = (1.0 - t).clamp_min(float(self_state_denom_min)).view(*sh_t)
            true_self = (x1 - xt) / denom
            losses = torch.where(
                take_self.view(*sh_t), (pred - true_self) ** 2, losses
            )
        if mm is not None and bool(mm.any()):
            # x1 - x0 is the correct velocity only for a state ON the path to
            # x1. A mismatched x_t lies on the path to mirror(x1), so the
            # velocity that actually reaches x1 from it is (x1 - x_t)/(1 - t) --
            # several times a normal target and swamping the geometry objective
            # (loss 5.4 vs 2-3, learning stalled). Supervise the *endpoint*
            # instead: ||x_t + (1 - t) U - x1||^2 has the same minimiser and
            # carries an automatic (1 - t)^2 weight, so it stays bounded and the
            # rows keep contributing to flow matching rather than being masked
            # out of it (v7 masked them and learned nothing at all).
            sh = [-1] + [1] * (losses.ndim - 1)
            losses = torch.where(
                mm.view(*sh), (self.estimate_x1(xt, t, pred) - x1) ** 2, losses
            )
        loss_lattice = losses[:, :3].mean()
        loss_coord = masked_mean(losses[:, 3:], C0.mask.unsqueeze(-1), dim=[1, 2]).mean()
        loss = loss_lattice + loss_coord
        loss_chiral = torch.zeros((), device=loss.device)
        if chiral_consistency_fn is not None and chiral_loss_weight > 0:
            pred_x1 = self.estimate_x1(xt, t, pred)
            # `t` doubles as `r` here: this path is r == t by construction.
            loss_chiral = chiral_consistency_fn(pred_x1, x1, C0, t)
            loss = loss + float(chiral_loss_weight) * loss_chiral
            if branch_l2_weight > 0:
                # Make the parity residual pay for its displacement, so it finds
                # the SMALLEST motion that flips the centre instead of shoving
                # atoms around and distorting bonds/angles.
                br = getattr(_unwrap_net(net), "_stereo_chiral_branch", None)
                sq = getattr(br, "_last_sqnorm", None) if br is not None else None
                if sq is not None:
                    loss = loss + float(branch_l2_weight) * sq
            if chiral_on_flow_map:
                # The hinge above only ever constrains the INSTANTANEOUS
                # velocity: this path predicts at r == t, and the AnyFlow path
                # deliberately strips the dual-time term (`v_r = u - (t-r)dudr`)
                # before estimate_x1. But sampling never calls that function --
                # every sampler mode uses the dual-time map U(z, r, t) with
                # r != t. Measured on v5 step1250 (24 val crystals, same weights
                # and conditioning): tag-following is 0.97 on the r == t endpoint
                # estimate and ~0.50 on every r != t jump, including NFE=1, which
                # *is* that endpoint via the map. So supervise the map itself:
                #   z_1 = z_r + (1 - r) U(z_r, r, 1)
                # is exactly the endpoint a 1-step sample produces, and multi-step
                # sampling composes it. Costs one extra forward per step.
                sh_r = [-1] + [1] * (xt.ndim - 1)
                one = torch.ones_like(t)
                u_map = self.pred(
                    net=net,
                    xt=xt,
                    xsc=xsc,
                    t=one,
                    r=t,
                    f=C0,
                    chiral_bias=chiral_bias,
                )
                pred_x1_map = xt + (1.0 - t).view(*sh_r) * u_map
                loss_map = chiral_consistency_fn(pred_x1_map, x1, C0, t)
                loss = loss + float(chiral_loss_weight) * loss_map
                loss_chiral = loss_chiral + loss_map
        return {
            "loss": loss,
            "loss_lattice": loss_lattice,
            "loss_coord": loss_coord,
            "loss_chiral": loss_chiral.detach(),
            "loss_vol": torch.tensor(0.0, device=loss.device),
            "loss_ldd": torch.tensor(0.0, device=loss.device),
        }
