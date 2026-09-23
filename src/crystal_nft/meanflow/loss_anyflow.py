"""AnyFlow-style teacher-distilled flow-map loss for Clari crystals."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from clari.pipelines.utils import masked_mean

from crystal_nft.meanflow.chirality import enantiomer_consistency_loss
from crystal_nft.meanflow.loss import MeanFlowLossConfig, _unwrap_net
from crystal_nft.meanflow.net import get_time_proxy
from crystal_nft.meanflow.sampler import (
    euler_instantaneous_segment,
    interval_compose_average_velocity,
    interval_state_at_times,
)
from crystal_nft.meanflow.velocity import (
    central_difference_dudr,
    central_difference_dudt,
    flow_map_to_instantaneous_velocity,
)


@dataclass
class AnyFlowLossConfig(MeanFlowLossConfig):
    """Distill frozen Clari teacher velocity into a dual-time flow map."""

    variant: str = "legacy"
    diffusion_ratio: float = 0.5
    consistency_ratio: float = 0.25
    weight_type: str = "beta08"
    weight_grid_size: int = 1000
    data_proportion: float = 0.5
    # Probability of forcing the full jump (r=0, t=1) — critical for few-step / 1-step.
    full_jump_prob: float = 0.2
    # teacher | noise_minus_data  (v_target)
    v_target_source: str = "teacher"
    # Below this r, regress the *data* velocity (x1 - x0) instead of the
    # teacher. The teacher has no stereo head, so its velocity provably cannot
    # depend on the CIP tag; regressing it therefore trains the student to be
    # tag-invariant, which is why cont4/5/6 all learned chance-level control.
    v_data_r_max: float = 0.0
    # teacher | noise_minus_data  (central-difference tangent)
    cd_velocity_source: str = "teacher"
    cd_eps: float = 1e-3
    # student: bootstrap dU/dr from the current flow map (can explode).
    # teacher: first-order map v + (t-r) dv_teacher/dr (stable with a strong teacher).
    jvp_source: str = "student"
    loss_clip: float = 0.0
    # If >0, interval samples are adjacent jumps on an NFE grid (1/nfe).
    nfe_steps: int = 0
    # If set, each sample draws one NFE from this list (overrides nfe_steps).
    nfe_steps_list: tuple[int, ...] = ()
    # One NFE for the whole batch so compose rollout can use a single grid.
    nfe_homogeneous: bool = False
    # Power grid t_i=(i/nfe)^rho. 1=uniform; 0.75 matches the report eval grid.
    nfe_grid_rho: float = 1.0
    # k ~ floor(U^p * nfe). p>1 oversamples early/large jumps (8-step bottleneck).
    nfe_k_power: float = 1.0
    # If >=0, every sample uses this grid index (0 = 8-step first jump).
    nfe_k_fixed: int = -1
    # If >0, interval samples with (t-r) above this use a teacher ODE average
    # velocity instead of the first-order JVP target (needed for 8-step Δt~0.21).
    large_jump_dt: float = 0.0
    large_jump_substeps: int = 6
    large_jump_method: str = "euler"
    # Extra loss weight on (t-r)>large_jump_dt interval samples.
    large_jump_loss_mult: float = 1.0
    # If <1, large-jump targets only apply for r at or below this (noise-side
    # first jump). 1.0 keeps the old "any r" behaviour.
    large_jump_r_max: float = 1.0
    # Fine grid for on-policy compose rollout to r (0 = max(nfe_steps_list) or 16).
    compose_fine_steps: int = 0
    # live: unwrap / DDP forward. frozen_student: snapshot with interval_mix=0.
    compose_from: str = "live"
    # Who rolls xt to r. Empty string follows compose_from. Set live while
    # compose_from=frozen_student to train 8-path states against a frozen 16-map.
    rollout_from: str = ""
    # If >0, traj_compose only runs when the homogeneous batch NFE is <= this
    # (8-path compose must not run on 16-keep batches).
    compose_nfe_max: int = 0
    norm_p: float = 1.0
    norm_eps: float = 0.01
    loss_v_weight: float = 1.0
    chiral_loss_weight: float = 0.1
    chiral_on_flow_map: bool = False


def large_jump_mask(
    fm_mask: Tensor,
    r: Tensor,
    t: Tensor,
    tau: float,
    r_max: float = 1.0,
) -> Tensor:
    """Interval samples with (t-r)>tau, optionally restricted to early r."""
    dt = t - r
    large = (~fm_mask) & (dt > float(tau))
    if float(r_max) < 1.0:
        large = large & (r <= float(r_max))
    return large


def freeze_student_snapshot(net: nn.Module) -> nn.Module:
    """Stop-grad copy with interval residual off. Do not refresh from live weights."""
    core = deepcopy(_unwrap_net(net))
    core.eval()
    for p in core.parameters():
        p.requires_grad_(False)
    targets = [core]
    proxy = get_time_proxy(core)
    if proxy is not None:
        targets.append(proxy)
    with torch.no_grad():
        for mod in core.modules():
            for buf_name in ("interval_mix", "nfe8_mix"):
                buf = getattr(mod, buf_name, None)
                if torch.is_tensor(buf):
                    buf.fill_(0.0)
        for mod in targets:
            mix = getattr(mod, "interval_mix", None)
            if torch.is_tensor(mix):
                mix.fill_(0.0)
    return core


def parse_nfe_steps_list(raw: object) -> tuple[int, ...]:
    """Parse yaml/cli NFE lists. Values <=1 are dropped."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        items: list[object] = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    elif isinstance(raw, (int, float)):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        try:
            items = list(raw)  # type: ignore[arg-type]
        except TypeError:
            items = [raw]
    out: list[int] = []
    for x in items:
        n = int(x)
        if n > 1:
            out.append(n)
    return tuple(out)


def nfe_grid_choices(cfg: AnyFlowLossConfig) -> tuple[tuple[int, ...], float]:
    """NFE grid mix and power rho used by ``sample_time_steps``."""
    rho = float(getattr(cfg, "nfe_grid_rho", 1.0) or 1.0)
    if rho <= 0.0:
        raise ValueError(f"nfe_grid_rho must be > 0, got {rho}")
    choices = parse_nfe_steps_list(getattr(cfg, "nfe_steps_list", None))
    if not choices:
        nfe = int(getattr(cfg, "nfe_steps", 0) or 0)
        if nfe > 1:
            choices = (nfe,)
    return choices, rho


def official_mode_masks(
    batch_size: int,
    *,
    rank: int = 0,
    world_size: int = 1,
    diffusion_ratio: float = 0.5,
    consistency_ratio: float = 0.25,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Return AnyFlow's deterministic global diffusion/consistency partition."""
    # Per-rank mix so every GPU sees diffusion + interval. The official
    # global prefix left rank 0 as all r=t (loss prints 0, weak dual-time).
    n_diffusion = round(float(diffusion_ratio) * int(batch_size))
    n_consistency = round(float(consistency_ratio) * int(batch_size))
    idx = torch.arange(int(batch_size), device=device)
    is_diffusion = idx < n_diffusion
    is_consistency = (idx >= n_diffusion) & (idx < n_diffusion + n_consistency)
    return is_diffusion, is_consistency


def official_beta08_weights(t: Tensor, grid_size: int = 1000) -> Tensor:
    """AnyFlow beta08 weights adapted to Clari time (0=noise, 1=data)."""
    n = max(2, int(grid_size))
    grid = torch.linspace(0.0, 1.0, n + 1, device=t.device, dtype=torch.float32)
    # Official uses sigma * sqrt(1-sigma); Clari t = 1-sigma.
    raw = (1.0 - grid) * torch.sqrt(grid.clamp_min(0.0))
    weights = raw * (float(n) / raw.sum().clamp_min(1e-12))
    index = torch.round(t.detach().float().clamp(0.0, 1.0) * n).long()
    return weights[index]


def official_scale_weight(
    weighted: Tensor,
    is_diffusion: Tensor,
) -> Tensor:
    """Rebalance non-diffusion samples exactly like AnyFlow forward training."""
    with torch.no_grad():
        local_weighted = weighted.detach()
        local_mask = is_diffusion.detach()
        if dist.is_available() and dist.is_initialized():
            gathered_weighted = [torch.empty_like(local_weighted) for _ in range(dist.get_world_size())]
            local_mask_u8 = local_mask.to(torch.uint8)
            gathered_mask = [torch.empty_like(local_mask_u8) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_weighted, local_weighted)
            dist.all_gather(gathered_mask, local_mask_u8)
            global_weighted = torch.cat(gathered_weighted, dim=0)
            global_mask = torch.cat(gathered_mask, dim=0).bool()
        else:
            global_weighted = local_weighted
            global_mask = local_mask

        scale = torch.ones_like(local_weighted)
        if global_mask.any() and (~global_mask).any() and (~local_mask).any():
            diff_mean = global_weighted[global_mask].mean()
            # Official code scales interval by diffusion_mean / interval.
            # With a matched teacher, diffusion_mean ~ 0 and that zeros the
            # flow-map residual. Never shrink interval terms.
            if float(diff_mean) > 1.0e-6:
                proposed = diff_mean / (local_weighted[~local_mask] + 1e-5)
                scale[~local_mask] = proposed.clamp(min=1.0)
    return scale


def teacher_instantaneous_velocity(
    teacher_net: nn.Module,
    interface,
    xt: Tensor,
    t: Tensor,
    f,
    *,
    xsc: Tensor | None = None,
) -> Tensor:
    """Frozen Clari DiT velocity at boundary r=t (no MeanFlow wrapper)."""
    with torch.no_grad():
        if xsc is None:
            xsc = torch.full_like(xt, torch.nan)
        return interface.pred(
            net=teacher_net,
            xt=xt,
            xsc=xsc,
            t=t,
            r=t,
            f=f,
            chiral_bias=None,
        ).detach()


class CrystalAnyFlowLoss:
    """
    AnyFlow Stage-1 style objective on crystals:

    - Student predicts flow map u(z, r, t)
    - Compound velocity V = u + (t-r) * stop_grad(du/dt)
    - Target v_g from frozen Clari-M teacher at (z,t,t) (distillation)
    - CD tangent from the same frozen teacher (low-variance, no student bootstrap)
    - Fraction ``data_proportion`` uses r=t (degenerates to FM distillation)
    """

    def __init__(
        self,
        cfg: AnyFlowLossConfig | None = None,
        *,
        teacher_net: nn.Module | None = None,
    ):
        self.cfg = cfg or AnyFlowLossConfig()
        self.teacher_net = teacher_net
        self._rollout_net: nn.Module | None = None
        self._frozen_student: nn.Module | None = None
        self._nfe_steps: Tensor | None = None

    def _compose_fine_steps(self) -> int:
        n_fine = int(getattr(self.cfg, "compose_fine_steps", 0) or 0)
        if n_fine > 1:
            return n_fine
        nfe_t = getattr(self, "_nfe_steps", None)
        if (
            bool(getattr(self.cfg, "nfe_homogeneous", False))
            and nfe_t is not None
            and torch.is_tensor(nfe_t)
            and nfe_t.numel() > 0
        ):
            n_fine = int(nfe_t.reshape(-1)[0].item())
        return n_fine if n_fine > 1 else 16

    def _compose_allowed_for_batch(self) -> bool:
        cap = int(getattr(self.cfg, "compose_nfe_max", 0) or 0)
        if cap <= 0:
            return True
        nfe_t = getattr(self, "_nfe_steps", None)
        if nfe_t is None or not torch.is_tensor(nfe_t) or nfe_t.numel() == 0:
            return True
        return int(nfe_t.reshape(-1)[0].item()) <= cap

    def _compose_src(self, field: str) -> str:
        raw = str(getattr(self.cfg, field, "") or "").lower().strip()
        if not raw:
            raw = str(getattr(self.cfg, "compose_from", "live") or "live").lower().strip()
        return raw or "live"

    def _frozen_student_net(self, net: nn.Module) -> nn.Module:
        if self._frozen_student is None:
            self._frozen_student = freeze_student_snapshot(net)
        return self._frozen_student

    def _live_rollout_net(self, net: nn.Module) -> nn.Module:
        core = _unwrap_net(net)
        if self._rollout_net is None:
            self._rollout_net = deepcopy(core)
            self._rollout_net.eval()
            for p in self._rollout_net.parameters():
                p.requires_grad_(False)
        else:
            self._rollout_net.load_state_dict(core.state_dict())
        return self._rollout_net

    def _student_rollout_net(self, net: nn.Module) -> nn.Module:
        """A separate frozen copy so rollout does not touch the DDP reducer."""
        src = self._compose_src("rollout_from")
        if src in ("frozen_student", "frozen"):
            return self._frozen_student_net(net)
        return self._live_rollout_net(net)

    def _compose_target_net(self, net: nn.Module) -> nn.Module:
        """Flow-map used to build the two-jump compose target."""
        src = self._compose_src("compose_from")
        if src in ("frozen_student", "frozen"):
            return self._frozen_student_net(net)
        return _unwrap_net(net)

    def set_teacher(self, teacher_net: nn.Module) -> None:
        self.teacher_net = teacher_net
        for p in self.teacher_net.parameters():
            p.requires_grad_(False)
        self.teacher_net.eval()

    def sample_time_steps(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.cfg
        if str(cfg.variant).lower() == "official":
            # Native AnyFlow runs from high-noise t to lower-noise r. Under
            # Clari's reversed time (0=noise, 1=data), diffusion keeps t=r,
            # consistency fixes the destination t=1, and generic keeps r<t.
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            fm_mask, consistency_mask = official_mode_masks(
                batch_size,
                rank=rank,
                world_size=world,
                diffusion_ratio=cfg.diffusion_ratio,
                consistency_ratio=cfg.consistency_ratio,
                device=device,
            )
            choices, rho = nfe_grid_choices(cfg)
            if choices:
                table = torch.tensor(list(choices), device=device, dtype=torch.long)
                if bool(getattr(cfg, "nfe_homogeneous", False)) and table.numel() > 0:
                    one = int(torch.randint(0, table.numel(), (1,), device=device).item())
                    pick = torch.full((batch_size,), one, device=device, dtype=torch.long)
                else:
                    pick = torch.randint(0, table.numel(), (batch_size,), device=device)
                nfe = table[pick]
                self._nfe_steps = nfe
                k_power = float(getattr(cfg, "nfe_k_power", 1.0) or 1.0)
                k_fixed = int(getattr(cfg, "nfe_k_fixed", -1))
                if k_fixed >= 0:
                    k = torch.full((batch_size,), k_fixed, device=device, dtype=torch.long)
                    k = torch.minimum(k.clamp(min=0), nfe - 1)
                else:
                    if k_power <= 0.0:
                        raise ValueError(f"nfe_k_power must be > 0, got {k_power}")
                    u = torch.rand(batch_size, device=device)
                    if abs(k_power - 1.0) > 1e-8:
                        u = u.pow(k_power)
                    k = torch.floor(u * nfe.to(torch.float32)).to(torch.long)
                    k = torch.minimum(k.clamp(min=0), nfe - 1)
                r_u = k.to(torch.float32) / nfe.to(torch.float32)
                t_u = (k + 1).to(torch.float32) / nfe.to(torch.float32)
                if abs(float(rho) - 1.0) > 1e-8:
                    r_u = torch.clamp(r_u, min=0.0, max=1.0).pow(rho)
                    t_u = torch.clamp(t_u, min=0.0, max=1.0).pow(rho)
                t = torch.where(fm_mask, r_u, t_u)
                self._dbg_nfe = nfe.float().mean().detach()
                self._dbg_k = k.float().mean().detach()
                return r_u, t, fm_mask
            samples = torch.rand(batch_size, 2, device=device)
            r, t = samples.min(dim=1).values, samples.max(dim=1).values
            t = torch.where(fm_mask, r, t)
            t = torch.where(consistency_mask, torch.ones_like(t), t)
            return r, t, fm_mask
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
        # Explicit one-step / full-interval supervision (overrides FM collapse).
        jump_p = float(getattr(cfg, "full_jump_prob", 0.0) or 0.0)
        if jump_p > 0:
            jump_mask = torch.rand(batch_size, device=device) < jump_p
            r = torch.where(jump_mask, torch.zeros_like(r), r)
            t = torch.where(jump_mask, torch.ones_like(t), t)
            fm_mask = fm_mask & ~jump_mask
        return r, t, fm_mask

    def _adaptive_weight(self, per_sample: Tensor) -> Tensor:
        cfg = self.cfg
        wt = (per_sample.detach() + cfg.norm_eps).pow(cfg.norm_p)
        return per_sample / wt

    def _resolve_velocity(
        self,
        source: str,
        *,
        interface,
        xt: Tensor,
        t: Tensor,
        f,
        xsc: Tensor | None,
        v_data: Tensor,
    ) -> Tensor:
        if source == "teacher":
            if self.teacher_net is None:
                raise RuntimeError("AnyFlowLoss requires teacher_net when source='teacher'")
            return teacher_instantaneous_velocity(
                self.teacher_net, interface, xt, t, f, xsc=xsc
            )
        if source in ("noise_minus_data", "data"):
            return v_data.detach()
        raise ValueError(f"Unknown velocity source={source}")

    def __call__(
        self,
        net: nn.Module,
        interface,
        C0,
        C1,
        *,
        chiral_bias: Optional[Tensor] = None,
        chirality_fn: Optional[Callable[[Tensor, object], Tensor]] = None,
        chiral_consistency_fn: Optional[Callable[[Tensor, object, object], Tensor]] = None,
    ) -> dict[str, Tensor]:
        cfg = self.cfg
        x0, x1 = C0.x, C1.x
        device = x1.device
        B = C1.batch_size

        if chirality_fn is not None and cfg.enantiomer_flip_p > 0:
            x1 = chirality_fn(x1, C1)

        r, t, fm_mask = self.sample_time_steps(B, device)
        state_time = r if str(cfg.variant).lower() == "official" else t
        xt = interface.sample_xt(x0, x1, state_time)
        v_data = x1 - x0
        tau = float(getattr(cfg, "large_jump_dt", 0.0) or 0.0)
        if not self._compose_allowed_for_batch():
            tau = 0.0
        r_max = float(getattr(cfg, "large_jump_r_max", 1.0) or 1.0)
        method = str(getattr(cfg, "large_jump_method", "euler") or "euler").lower()
        large = torch.zeros_like(fm_mask)
        onpol_end: Tensor | None = None
        if (
            str(cfg.variant).lower() == "official"
            and tau > 0.0
            and method in ("onpolicy_compose", "compose_onpolicy", "onpolicy_heun")
            and self.teacher_net is not None
        ):
            large = large_jump_mask(fm_mask, r, t, tau, r_max)
            if bool(large.any()):
                n_sub = int(getattr(cfg, "large_jump_substeps", 6) or 6)
                zeros_t = torch.zeros_like(r)
                z_r = euler_instantaneous_segment(
                    interface,
                    self.teacher_net,
                    C0,
                    x_init=x0,
                    t_from=zeros_t,
                    t_to=r,
                    num_steps=n_sub,
                    chiral_bias=chiral_bias,
                    use_self_cond=True,
                    method="heun",
                )
                z_t = euler_instantaneous_segment(
                    interface,
                    self.teacher_net,
                    C0,
                    x_init=z_r,
                    t_from=r,
                    t_to=t,
                    num_steps=max(1, n_sub // 2),
                    chiral_bias=chiral_bias,
                    use_self_cond=True,
                    method="heun",
                )
                onpol_end = z_t
                xt = torch.where(large.view(-1, 1, 1), z_r.to(dtype=xt.dtype), xt)

        if (
            str(cfg.variant).lower() == "official"
            and tau > 0.0
            and method in ("traj_compose", "rollout_compose")
        ):
            # Separate module copy (not DDP.module) so this rollout cannot
            # steal the one grad-enabled DDP forward.
            large = large_jump_mask(fm_mask, r, t, tau, r_max)
            if bool(large.any()):
                n_fine = self._compose_fine_steps()
                z_r = interval_state_at_times(
                    interface,
                    self._student_rollout_net(net),
                    C0,
                    x_init=x0,
                    t_at=r,
                    num_steps=n_fine,
                    rho=float(getattr(cfg, "nfe_grid_rho", 1.0) or 1.0),
                    chiral_bias=chiral_bias,
                )
                xt = torch.where(large.view(-1, 1, 1), z_r.to(dtype=xt.dtype), xt)

        xsc = None
        dit = getattr(_unwrap_net(net), "dit", _unwrap_net(net))
        if getattr(dit, "self_cond", False):
            # Student no_grad forward *inside* bf16 autocast before the DDP-tracked
            # student call drops grads on most params (incl. embed_deltat). Prefer
            # the frozen teacher, and always disable autocast for this prep.
            sc_net = self.teacher_net if self.teacher_net is not None else _unwrap_net(net)
            sc_t = state_time
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                nsc = max(1, C0.batch_size // 2)
                fsc = C0.subset(slice(0, nsc))
                out = interface.pred(
                    net=sc_net,
                    xt=xt[:nsc],
                    xsc=None,
                    t=sc_t[:nsc],
                    r=sc_t[:nsc],
                    f=fsc,
                    chiral_bias=chiral_bias[:nsc] if chiral_bias is not None else None,
                )
                xsc = torch.full_like(xt, torch.nan)
                xsc[:nsc] = interface.estimate_x1(xt[:nsc], sc_t[:nsc], out)

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

        if str(cfg.variant).lower() == "official":
            # Clari time (0=noise, 1=data): U = v(z,r) + (t-r) dU/dr.
            # v can be the frozen teacher (rank800) or the linear-path x1-x0.
            v = self._resolve_velocity(
                cfg.v_target_source,
                interface=interface,
                xt=xt,
                t=state_time,
                f=C0,
                xsc=xsc,
                v_data=v_data,
            )
            # Swap in the handed data velocity near the noise end. `x1` carries the
            # target's handedness, so matching it *requires* reading the tag --
            # unlike the teacher term, which is stereo-blind by construction.
            v_data_r_max = float(getattr(cfg, "v_data_r_max", 0.0) or 0.0)
            if v_data_r_max > 0.0:
                use_data = (r <= v_data_r_max).view(-1, 1, 1)
                v = torch.where(use_data, v_data.detach().to(v.dtype), v)
            jvp_src = str(getattr(cfg, "jvp_source", "student") or "student").lower()
            if jvp_src == "teacher":
                if self.teacher_net is None:
                    raise RuntimeError("jvp_source=teacher requires teacher_net")

                def teacher_fn(z, cur_r, cur_t):
                    return interface.forward(
                        net=self.teacher_net,
                        xt=z,
                        xsc=xsc,
                        t=cur_r,
                        r=cur_r,
                        f=C0,
                        chiral_bias=None,
                    )

                dudr = central_difference_dudr(
                    teacher_fn, xt, r, t, v, eps=cfg.cd_eps
                )
            elif jvp_src == "student":
                dudr = central_difference_dudr(fn, xt, r, t, v, eps=cfg.cd_eps)
            else:
                raise ValueError(
                    f"Official AnyFlow jvp_source must be student|teacher, got {jvp_src!r}"
                )
            time_diff = (t - r).view(-1, 1, 1)
            target = (v + time_diff * dudr).detach()
            if tau > 0.0:
                dt_1d = t - r
                if not bool(large.any()):
                    large = large_jump_mask(fm_mask, r, t, tau, r_max)
                if bool(large.any()):
                    if method in ("onpolicy_compose", "compose_onpolicy", "onpolicy_heun"):
                        if onpol_end is None:
                            u_ode = target
                        else:
                            u_ode = (onpol_end.float() - xt.float()) / dt_1d.view(
                                -1, 1, 1
                            ).clamp(min=1e-4)
                    elif method in ("compose", "student_compose", "traj_compose", "rollout_compose"):
                        u_ode = interval_compose_average_velocity(
                            interface,
                            self._compose_target_net(net),
                            C0,
                            x_init=xt,
                            t_from=r,
                            t_to=t,
                            rho=float(getattr(cfg, "nfe_grid_rho", 1.0) or 1.0),
                            chiral_bias=chiral_bias,
                        )
                    elif self.teacher_net is not None:
                        n_sub = int(getattr(cfg, "large_jump_substeps", 6) or 6)
                        z1 = euler_instantaneous_segment(
                            interface,
                            self.teacher_net,
                            C0,
                            x_init=xt,
                            t_from=r,
                            t_to=t,
                            num_steps=n_sub,
                            chiral_bias=chiral_bias,
                            use_self_cond=True,
                            method=method,
                        )
                        u_ode = (z1.float() - xt.float()) / dt_1d.view(-1, 1, 1).clamp(
                            min=1e-4
                        )
                    else:
                        u_ode = target
                    target = torch.where(
                        large.view(-1, 1, 1),
                        u_ode.to(dtype=target.dtype),
                        target,
                    )
            error = u.float() - target.float()
            err_lattice = error[:, :3].pow(2).mean(dim=(1, 2))
            err_coord = masked_mean(
                error[:, 3:].pow(2), C0.mask.unsqueeze(-1), dim=[1, 2]
            )
            per_sample = err_lattice + err_coord
            jump_mult = float(getattr(cfg, "large_jump_loss_mult", 1.0) or 1.0)
            if jump_mult != 1.0:
                per_sample = per_sample * torch.where(
                    large, torch.full_like(per_sample, jump_mult), torch.ones_like(per_sample)
                )
            clip = float(getattr(cfg, "loss_clip", 0.0) or 0.0)
            if clip > 0:
                per_sample = per_sample.clamp(max=clip)
            if str(cfg.weight_type).lower() == "beta08":
                weights = official_beta08_weights(r, cfg.weight_grid_size)
            elif str(cfg.weight_type).lower() == "uniform":
                weights = torch.ones_like(per_sample)
            else:
                raise ValueError(
                    f"Official AnyFlow weight_type must be beta08|uniform, got {cfg.weight_type!r}"
                )
            weighted = per_sample * weights
            scale = official_scale_weight(weighted, fm_mask)
            loss = (weighted * scale).mean()
            loss_vol = torch.zeros((), device=device)
            loss_ldd = torch.zeros((), device=device)
            loss_chiral = torch.zeros((), device=device)
            need_x1 = (
                cfg.use_aux_losses and cfg.aux_vol_weight + cfg.aux_ldd_weight > 0
            ) or (cfg.chiral_loss_weight > 0 and chiral_consistency_fn is not None)
            if need_x1:
                # Instantaneous v at r: U = v(z,r) + (t-r) dU/dr.
                # estimate_x1 must not see dual-time U when r≠t.
                v_r = u - time_diff * dudr.detach()
                pred_x1 = interface.estimate_x1(xt, r, v_r)
            if cfg.use_aux_losses and cfg.aux_vol_weight + cfg.aux_ldd_weight > 0:
                if cfg.aux_vol_weight > 0:
                    loss_vol = interface._vol_losses(pred_x1, x1).mean()
                if cfg.aux_ldd_weight > 0:
                    loss_ldd = interface._ldd_losses(pred_x1, x1, f=C0).mean()
                loss = loss + cfg.aux_vol_weight * loss_vol + cfg.aux_ldd_weight * loss_ldd
            # Stereochemistry: the AnyFlow target is built from a chirality-blind
            # teacher, so this hinge is the only term that supervises handedness.
            if cfg.chiral_loss_weight > 0 and chiral_consistency_fn is not None:
                # `r` goes through so the hinge can be concentrated near the noise
                # end, where the state does not already give away the handedness.
                loss_chiral = chiral_consistency_fn(pred_x1, x1, C0, r)
                loss = loss + cfg.chiral_loss_weight * loss_chiral
                if cfg.chiral_on_flow_map:
                    # pred_x1 above is built from v_r, the INSTANTANEOUS velocity
                    # (the line above strips the dual-time term on purpose). No
                    # sampler evaluates that function. Measured on one checkpoint,
                    # 308 val centres: tag-following 0.94 on the r == t endpoint
                    # vs 0.51 on the r=0 -> t=1 flow map that NFE=1 actually runs.
                    # So hinge the map too: z_1 = z_r + (1 - r) U(z_r, r, 1).
                    one = torch.ones_like(r)
                    u_map = interface.pred(
                        net=net, xt=xt, xsc=None, t=one, r=r, f=C0,
                        chiral_bias=chiral_bias,
                    )
                    sh_r = [-1] + [1] * (xt.ndim - 1)
                    pred_x1_map = xt + (1.0 - r).view(*sh_r) * u_map
                    loss_map = chiral_consistency_fn(pred_x1_map, x1, C0, r)
                    loss = loss + cfg.chiral_loss_weight * loss_map
                    loss_chiral = loss_chiral + loss_map
            return {
                "loss": loss,
                "loss_u": loss.detach(),
                "loss_v": torch.zeros((), device=device),
                "loss_anyflow": loss.detach(),
                "loss_lattice": err_lattice.mean().detach(),
                "loss_coord": err_coord.mean().detach(),
                "loss_vol": loss_vol.detach() if torch.is_tensor(loss_vol) else loss_vol,
                "loss_ldd": loss_ldd.detach() if torch.is_tensor(loss_ldd) else loss_ldd,
                "loss_chiral": loss_chiral.detach() if torch.is_tensor(loss_chiral) else loss_chiral,
                "fm_frac": fm_mask.float().mean().detach(),
                "weight_mean": weights.mean().detach(),
                "scale_mean": scale.mean().detach(),
                "student_steps": getattr(self, "_dbg_nfe", torch.zeros((), device=device)),
                "grad_timestep": getattr(self, "_dbg_k", torch.full((), -1.0, device=device)),
            }

        v_c = self._resolve_velocity(
            cfg.cd_velocity_source,
            interface=interface,
            xt=xt,
            t=t,
            f=C0,
            xsc=xsc,
            v_data=v_data,
        )
        dudt = central_difference_dudt(fn, xt, r, t, v_c, eps=cfg.cd_eps)
        time_diff = (t - r).view(-1, 1, 1)

        v_g = self._resolve_velocity(
            cfg.v_target_source,
            interface=interface,
            xt=xt,
            t=t,
            f=C0,
            xsc=xsc,
            v_data=v_data,
        )

        # Classic MeanFlow target (stop-grad RHS): clearer grads into dual-time u.
        u_target = (v_g - time_diff * dudt).detach()
        err_u = (u - u_target).reshape(B, -1)
        loss_u = self._adaptive_weight(torch.sum(err_u**2, dim=-1)).mean()

        v_compound = u + time_diff * dudt.detach()
        err_v = (v_compound - v_g).reshape(B, -1)
        loss_v = self._adaptive_weight(torch.sum(err_v**2, dim=-1)).mean()

        loss = loss_u + cfg.loss_v_weight * loss_v

        error = u - u_target
        err_lattice = error[:, :3].pow(2).mean()
        err_coord = masked_mean(error[:, 3:].pow(2), C0.mask.unsqueeze(-1), dim=[1, 2]).mean()

        # estimate_x1 assumes instantaneous velocity: x1 ≈ xt + (1-t)*v.
        # Never pass dual-time flow-map u(r≠t) here — that fights MeanFlow identity
        # and collapses the model back to single-time FM (kills few-step sampling).
        v_inst = flow_map_to_instantaneous_velocity(u, xt, r, t, dudt.detach())
        pred_x1 = interface.estimate_x1(xt, t, v_inst)

        loss_vol = torch.tensor(0.0, device=device)
        loss_ldd = torch.tensor(0.0, device=device)
        if cfg.use_aux_losses and cfg.aux_vol_weight + cfg.aux_ldd_weight > 0:
            if cfg.aux_vol_weight > 0:
                loss_vol = interface._vol_losses(pred_x1, x1).mean()
            if cfg.aux_ldd_weight > 0:
                loss_ldd = interface._ldd_losses(pred_x1, x1, f=C0).mean()
            loss = loss + cfg.aux_vol_weight * loss_vol + cfg.aux_ldd_weight * loss_ldd

        loss_chiral = torch.tensor(0.0, device=device)
        if cfg.chiral_loss_weight > 0:
            # Prefer post-augment x1 (not raw C1.x) so flipped enantiomers stay consistent.
            if chiral_consistency_fn is not None:
                loss_chiral = chiral_consistency_fn(pred_x1, x1, C0, r)
            else:
                loss_chiral = enantiomer_consistency_loss(
                    pred_x1, x1, getattr(C0, "mask", None)
                )
            loss = loss + cfg.chiral_loss_weight * loss_chiral

        return {
            "loss": loss,
            "loss_u": loss_u.detach(),
            "loss_v": loss_v.detach(),
            "loss_lattice": err_lattice.detach(),
            "loss_coord": err_coord.detach(),
            "loss_vol": loss_vol.detach() if torch.is_tensor(loss_vol) else loss_vol,
            "loss_ldd": loss_ldd.detach() if torch.is_tensor(loss_ldd) else loss_ldd,
            "loss_chiral": loss_chiral.detach() if torch.is_tensor(loss_chiral) else loss_chiral,
            "fm_frac": fm_mask.float().mean().detach(),
        }
