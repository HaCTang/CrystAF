"""MeanFlow sampling for crystal generators."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from torch import Tensor

from clari.chem import Crystal
from clari.geometry import zero_com_suffix
from clari.pipelines.base.samplers import HeunSampler
from clari.pipelines.utils import bcast_right

from crystal_nft.meanflow.interface import MeanFlowCrystalInterface


def _batch_time(value: float, batch_size: int, *, device, dtype) -> Tensor:
    return torch.full((batch_size,), float(value), device=device, dtype=dtype)


def interval_time_grid(
    steps: int,
    *,
    device,
    dtype,
    rho: float = 1.0,
    schedule: str = "power",
) -> Tensor:
    """Monotone knots on ``[0, 1]`` (0=noise, 1=data). ``rho=1`` is uniform.

    ``power``: ``rho>1`` uses smaller jumps near noise; ``rho<1`` near data.
    ``cosine``: denser near both endpoints (rho ignored).
    """
    n = max(1, int(steps))
    u = torch.linspace(0.0, 1.0, n + 1, device=device, dtype=dtype)
    sched = str(schedule).lower().strip() or "power"
    if sched in ("cosine", "cos"):
        return (0.5 * (1.0 - torch.cos(torch.pi * u))).clamp(min=0.0, max=1.0)
    if sched not in ("power", "uniform", ""):
        raise ValueError(f"interval schedule must be power|cosine, got {schedule!r}")
    rho_f = float(rho)
    if abs(rho_f - 1.0) < 1e-8:
        return u
    if rho_f <= 0.0:
        raise ValueError(f"interval rho must be > 0, got {rho_f}")
    return torch.clamp(u.pow(rho_f), min=0.0, max=1.0)


def interval_split_first_grid(
    steps: int,
    *,
    device,
    dtype,
    rho: float = 0.75,
    fine_steps: int | None = None,
) -> Tensor:
    """``steps`` jumps: split the first coarse jump into two fine-grid jumps.

    For 8-NFE / rho=0.75 the first coarse knot equals the second 16-grid knot.
    Remaining jumps remap a (steps-2) power grid from that knot to data, so
    NFE stays ``steps`` (not 9).
    """
    n = max(3, int(steps))
    fine = int(fine_steps) if fine_steps is not None else 2 * n
    g_fine = interval_time_grid(
        fine, device=device, dtype=dtype, rho=rho, schedule="power"
    )
    head = g_fine[:3]
    tail_n = n - 2
    g_tail = interval_time_grid(
        tail_n, device=device, dtype=dtype, rho=rho, schedule="power"
    )
    lo = head[-1]
    tail = lo + (1.0 - lo) * g_tail[1:]
    return torch.cat([head, tail], dim=0)


def interval_split_first_keep_coarse_grid(
    coarse_steps: int,
    *,
    device,
    dtype,
    rho: float = 0.75,
) -> Tensor:
    """Split the first coarse jump into two fine jumps; keep the rest of the coarse grid.

    For 8-step rho=0.75 this is 9 jumps (NFE=9): two 16-grid jumps to the first
    8-grid knot, then the original remaining seven 8-grid jumps to data.
    """
    n = max(2, int(coarse_steps))
    g_coarse = interval_time_grid(
        n, device=device, dtype=dtype, rho=rho, schedule="power"
    )
    g_fine = interval_time_grid(
        2 * n, device=device, dtype=dtype, rho=rho, schedule="power"
    )
    return torch.cat([g_fine[:3], g_coarse[2:]], dim=0)


def interval_flow_map_jump(
    interface: MeanFlowCrystalInterface,
    net: nn.Module,
    C: Crystal,
    z: Tensor,
    *,
    t_from: float,
    t_to: float,
    chiral_bias: Optional[Tensor] = None,
) -> Tensor:
    """One dual-time Euler jump: ``z += (t-r) u(z, r, t)``."""
    B = int(C.batch_size) if C.batched else 1
    device = z.device
    dtype = z.dtype
    r = _batch_time(t_from, B, device=device, dtype=dtype)
    t = _batch_time(t_to, B, device=device, dtype=dtype)
    u = interface.pred(
        net=net,
        xt=z,
        xsc=None,
        t=t,
        r=r,
        f=C,
        chiral_bias=chiral_bias,
    )
    dt = (t - r).view(-1, 1, 1)
    z = z + dt * u
    return zero_com_suffix(z, w=C.mask)


def power_grid_index(t: Tensor, nfe: int, rho: float) -> Tensor:
    """Round ``t`` to the nearest ``(i/nfe)^rho`` knot index in ``[0, nfe]``."""
    steps = max(1, int(nfe))
    rho_f = float(rho)
    t_c = t.reshape(-1).float().clamp(min=0.0, max=1.0)
    if abs(rho_f - 1.0) < 1e-8:
        u = t_c
    else:
        if rho_f <= 0.0:
            raise ValueError(f"interval rho must be > 0, got {rho}")
        u = t_c.pow(1.0 / rho_f)
    return (u * float(steps)).round().to(dtype=torch.long).clamp(min=0, max=steps)


@torch.no_grad()
def interval_grid_rollout_states(
    interface: MeanFlowCrystalInterface,
    net: nn.Module,
    C: Crystal,
    *,
    x_init: Tensor,
    num_steps: int,
    rho: float = 0.75,
    chiral_bias: Optional[Tensor] = None,
) -> Tensor:
    """Stop-grad interval states on the eval power grid, shape ``[steps+1, B, ...]``."""
    steps = max(1, int(num_steps))
    orig_dtype = x_init.dtype
    grid = interval_time_grid(
        steps,
        device=x_init.device,
        dtype=torch.float32,
        rho=rho,
        schedule="power",
    )
    z = x_init.float()
    states = [z]
    for i in range(steps):
        z = interval_flow_map_jump(
            interface,
            net,
            C,
            z,
            t_from=float(grid[i]),
            t_to=float(grid[i + 1]),
            chiral_bias=chiral_bias,
        )
        states.append(z)
    return torch.stack(states, dim=0).to(dtype=orig_dtype)


@torch.no_grad()
def interval_state_at_times(
    interface: MeanFlowCrystalInterface,
    net: nn.Module,
    C: Crystal,
    *,
    x_init: Tensor,
    t_at: Tensor,
    num_steps: int,
    rho: float = 0.75,
    chiral_bias: Optional[Tensor] = None,
) -> Tensor:
    """Stop-grad power-grid state at each sample's ``t_at`` (full batch, no subset)."""
    states = interval_grid_rollout_states(
        interface,
        net,
        C,
        x_init=x_init,
        num_steps=num_steps,
        rho=rho,
        chiral_bias=chiral_bias,
    )
    idx = power_grid_index(t_at, num_steps, rho)
    batch_idx = torch.arange(idx.numel(), device=idx.device)
    return states[idx, batch_idx]


def _self_cond_enabled(net: nn.Module) -> bool:
    core = net.module if hasattr(net, "module") else net
    return bool(getattr(core, "self_cond", False) or getattr(getattr(core, "dit", core), "self_cond", False))


@torch.no_grad()
def euler_instantaneous_rollout(
    interface: MeanFlowCrystalInterface,
    net: nn.Module,
    C: Crystal,
    *,
    x_init: Tensor,
    num_steps: int,
    chiral_bias: Optional[Tensor] = None,
    use_self_cond: bool = True,
) -> Tensor:
    """Clari-style Euler on an instantaneous velocity net (r = t). t=0 noise → t=1 data.

    Always FP32: Clari eval disables autocast. A 50-step Euler in bf16
    accumulates integration error and is a bad distillation target.
    """
    steps = max(1, int(num_steps))
    orig_dtype = x_init.dtype
    device_type = "cuda" if x_init.is_cuda else x_init.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        z = x_init.float()
        xsc = None
        B = int(C.batch_size) if C.batched else 1
        device = z.device
        dtype = z.dtype
        grid = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)
        for i in range(steps):
            t_cur = _batch_time(float(grid[i]), B, device=device, dtype=dtype)
            t_next = _batch_time(float(grid[i + 1]), B, device=device, dtype=dtype)
            v = interface.pred(
                net=net,
                xt=z,
                xsc=xsc,
                t=t_cur,
                r=t_cur,
                f=C,
                chiral_bias=chiral_bias,
            )
            dt = (t_next - t_cur).view(-1, 1, 1)
            if use_self_cond:
                xsc = interface.estimate_x1(z, t_cur, v)
            z = z + dt * v
            z = zero_com_suffix(z, w=C.mask)
        z = zero_com_suffix(z, w=C.mask)
        return z.to(dtype=orig_dtype)


@torch.no_grad()
def euler_instantaneous_segment(
    interface: MeanFlowCrystalInterface,
    net: nn.Module,
    C: Crystal,
    *,
    x_init: Tensor,
    t_from: Tensor,
    t_to: Tensor,
    num_steps: int,
    chiral_bias: Optional[Tensor] = None,
    use_self_cond: bool = True,
    method: str = "euler",
) -> Tensor:
    """Integrate instantaneous ``v`` from ``t_from`` to ``t_to`` (per sample), FP32.

    ``method=euler`` or ``heun``. Used as a large-Δt flow-map target so the
    student is not trained on a first-order JVP linearization.
    """
    steps = max(1, int(num_steps))
    method_l = str(method).lower().strip() or "euler"
    if method_l not in ("euler", "heun"):
        raise ValueError(f"segment method must be euler|heun, got {method!r}")
    orig_dtype = x_init.dtype
    device_type = "cuda" if x_init.is_cuda else x_init.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        z = x_init.float()
        r0 = t_from.reshape(-1).float()
        t1 = t_to.reshape(-1).float()
        xsc = None
        device = z.device
        dtype = z.dtype
        r0 = r0.to(device=device, dtype=dtype)
        t1 = t1.to(device=device, dtype=dtype)
        for i in range(steps):
            t_cur = r0 + (t1 - r0) * (float(i) / float(steps))
            t_next = r0 + (t1 - r0) * (float(i + 1) / float(steps))
            v0 = interface.pred(
                net=net,
                xt=z,
                xsc=xsc,
                t=t_cur,
                r=t_cur,
                f=C,
                chiral_bias=chiral_bias,
            )
            dt = (t_next - t_cur).view(-1, 1, 1)
            if use_self_cond:
                xsc = interface.estimate_x1(z, t_cur, v0)
            if method_l == "euler":
                z = z + dt * v0
            else:
                z_e = z + dt * v0
                v1 = interface.pred(
                    net=net,
                    xt=z_e,
                    xsc=xsc,
                    t=t_next,
                    r=t_next,
                    f=C,
                    chiral_bias=chiral_bias,
                )
                z = z + 0.5 * dt * (v0 + v1)
            z = zero_com_suffix(z, w=C.mask)
        z = zero_com_suffix(z, w=C.mask)
    return z.to(dtype=orig_dtype)


def power_index_midpoint(r: Tensor, t: Tensor, rho: float) -> Tensor:
    """Midpoint of ``[r, t]`` in power-grid index space ``u=t^{1/rho}``.

    Adjacent 8-step knots with ``t_i=(i/8)^rho`` map to the skipped 16-step knot.
    """
    rho_f = float(rho)
    r_c = r.reshape(-1).float().clamp(min=0.0, max=1.0)
    t_c = t.reshape(-1).float().clamp(min=0.0, max=1.0)
    if abs(rho_f - 1.0) < 1e-8:
        return (0.5 * (r_c + t_c)).to(dtype=r.dtype)
    if rho_f <= 0.0:
        raise ValueError(f"interval rho must be > 0, got {rho}")
    inv = 1.0 / rho_f
    mid_u = 0.5 * (r_c.pow(inv) + t_c.pow(inv))
    return mid_u.clamp(min=0.0, max=1.0).pow(rho_f).to(dtype=r.dtype)


@torch.no_grad()
def interval_compose_average_velocity(
    interface: MeanFlowCrystalInterface,
    net: nn.Module,
    C: Crystal,
    *,
    x_init: Tensor,
    t_from: Tensor,
    t_to: Tensor,
    rho: float = 0.75,
    chiral_bias: Optional[Tensor] = None,
) -> Tensor:
    """Average velocity of two stop-grad interval jumps that skip a fine knot.

    Used so an 8-step jump matches two 16-step jumps of the same flow map
    instead of a first-order JVP or a coarse teacher ODE.
    """
    orig_dtype = x_init.dtype
    device_type = "cuda" if x_init.is_cuda else x_init.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        z = x_init.float()
        r0 = t_from.reshape(-1).float().to(device=z.device)
        t1 = t_to.reshape(-1).float().to(device=z.device)
        t_mid = power_index_midpoint(r0, t1, rho).to(device=z.device, dtype=z.dtype)
        u1 = interface.pred(
            net=net,
            xt=z,
            xsc=None,
            t=t_mid,
            r=r0,
            f=C,
            chiral_bias=chiral_bias,
        )
        z1 = zero_com_suffix(z + (t_mid - r0).view(-1, 1, 1) * u1.float(), w=C.mask)
        u2 = interface.pred(
            net=net,
            xt=z1,
            xsc=None,
            t=t1,
            r=t_mid,
            f=C,
            chiral_bias=chiral_bias,
        )
        z2 = zero_com_suffix(z1 + (t1 - t_mid).view(-1, 1, 1) * u2.float(), w=C.mask)
        dt = (t1 - r0).view(-1, 1, 1).clamp(min=1e-4)
        u_star = (z2 - z) / dt
    return u_star.to(dtype=orig_dtype)


@torch.no_grad()
def heun_instantaneous_rollout(
    interface: MeanFlowCrystalInterface,
    net: nn.Module,
    C: Crystal,
    *,
    x_init: Tensor,
    num_steps: int,
    chiral_bias: Optional[Tensor] = None,
) -> Tensor:
    """Clari HeunSampler protocol (stochasticity=none), FP32.

    The last grid step is Euler, matching teacher Table1 NFE=16.
    ``chiral_bias`` is attached on the DiT the same way eval Heun sampling does.
    """
    steps = max(1, int(num_steps))
    orig_dtype = x_init.dtype
    device_type = "cuda" if x_init.is_cuda else x_init.device.type
    core = net.module if hasattr(net, "module") else net
    dit = getattr(core, "dit", None)
    with torch.autocast(device_type=device_type, enabled=False):
        if chiral_bias is not None and dit is not None:
            dit._mf_chiral_bias = chiral_bias
        try:
            C_in = C.replace(x=x_init.float())
            sampler = HeunSampler(num_steps=steps, stochasticity="none")
            out = sampler.sample(
                interface,
                net,
                C_in,
                sample_prior=False,
                pbar=None,
            )
            z = out.x if hasattr(out, "x") else out
        finally:
            if dit is not None:
                dit._mf_chiral_bias = None
    return z.to(dtype=orig_dtype)


def _uniform_jump_index_pairs(num_steps: int, n_jumps: int) -> list[tuple[int, int]]:
    """Split ``[0, num_steps]`` into ``n_jumps`` index intervals (eval grid aligned)."""
    steps = max(1, int(num_steps))
    nj = max(1, min(int(n_jumps), steps))
    edges = [int(round(i * steps / nj)) for i in range(nj + 1)]
    edges[0] = 0
    edges[-1] = steps
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = min(steps, edges[i - 1] + 1)
    for i in range(len(edges) - 2, -1, -1):
        if edges[i] >= edges[i + 1]:
            edges[i] = max(0, edges[i + 1] - 1)
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if a < b]


def _flow_map_jump_index_pairs(
    num_steps: int,
    *,
    mode: str = "full",
    grad_timestep: int | None = None,
    n_jumps: int | None = None,
) -> list[tuple[int, int]]:
    """Grid-index pairs for a Clari-time (t=0 noise → t=1 data) flow-map rollout.

    ``full``: every uniform step ``i -> i+1``.
    ``shortcut``: AnyFlow Stage-2 3-jump Backward Simulation
    (prev / current / post), always at most 3 transformer forwards::

        prev:    0 -> k
        current: k -> k+1
        post:    k+1 -> num_steps

    ``jumps``: ``n_jumps`` uniform index intervals on the same grid (e.g. 8
    jumps on a 16-step lattice). Degenerate zero-width segments are dropped.
    """
    steps = max(1, int(num_steps))
    mode_l = str(mode).lower()
    if mode_l == "full":
        return [(i, i + 1) for i in range(steps)]
    if mode_l in ("jumps", "uniform_jumps"):
        nj = 8 if n_jumps is None else int(n_jumps)
        return _uniform_jump_index_pairs(steps, nj)
    if mode_l != "shortcut":
        raise ValueError(f"Unknown rollout mode={mode}")
    k = 0 if grad_timestep is None else int(grad_timestep)
    k = min(max(k, 0), steps - 1)
    pairs = [(0, k), (k, k + 1), (k + 1, steps)]
    return [(a, b) for a, b in pairs if a != b]


def _ddp_no_sync(net: nn.Module, enabled: bool):
    if enabled and hasattr(net, "no_sync"):
        return net.no_sync()
    return nullcontext()


def allreduce_grads(module: nn.Module) -> None:
    """Average parameter grads across DDP ranks.

    Flow-map rollouts run extra DiT forwards under ``no_sync`` / checkpoint
    on the unwrapped module. Autograd walks the graph in reverse, so the
    DDP-visible forward (last jump or AnyFlow cotrain) allreduces *before*
    earlier jump grads exist. Those geom/DMD grads stay rank-local and
    the live weights diverge unless we allreduce after the full backward.
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    world = float(dist.get_world_size())
    for p in module.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world)


def flow_map_rollout(
    interface: MeanFlowCrystalInterface,
    net: nn.Module,
    C: Crystal,
    *,
    x_init: Tensor,
    num_steps: int,
    chiral_bias: Optional[Tensor] = None,
    detach_between_jumps: bool = False,
    use_self_cond: bool = False,
    checkpoint_jumps: bool = False,
    rollout_mode: str = "full",
    grad_timestep: int | None = None,
    n_jumps: int | None = None,
    ddp_sync_last: bool = True,
    rho: float = 0.75,
    schedule: str = "power",
) -> Tensor:
    """Flow-map Euler on the eval power grid: z' = z + (t-r) * U(z, r, t).

    ``rho=0.75`` matches Table1 interval eval. ``rollout_mode='shortcut'`` is
    AnyFlow Stage-2 training_rollout (3 jumps).

    ``detach_between_jumps`` matches AnyFlow Feature 1: u is evaluated on a
    detached state so grads do not flow through earlier jump *inputs*, but
    each ``dt * u`` term still trains the student. Official default is False.

    DDP: every grad-enabled DiT call goes through the DDP container. All but
    the last jump run under ``no_sync()`` so the reducer fires once per
    backward. Checkpointed jumps use the unwrapped module. Callers must
    ``allreduce_grads`` after the full backward: autograd reverses jump
    order, so DDP allreduces before earlier geom/DMD grads exist.
    Do not checkpoint through DDP (checkpoint is skipped on DDP).
    """
    steps = max(1, int(num_steps))
    z = x_init
    xsc = None
    B = int(C.batch_size) if C.batched else 1
    device = z.device
    dtype = z.dtype
    grid = interval_time_grid(
        steps, device=device, dtype=dtype, rho=rho, schedule=schedule
    )
    self_cond = bool(use_self_cond) and _self_cond_enabled(net)
    pairs = _flow_map_jump_index_pairs(
        steps, mode=rollout_mode, grad_timestep=grad_timestep, n_jumps=n_jumps
    )
    n_pairs = len(pairs)
    is_ddp = hasattr(net, "no_sync")
    core = net.module if is_ddp else net
    # Checkpoint the unwrapped DiT. Last jump still goes through the DDP
    # container so the reducer fires. Calling checkpoint(DDP) is unsafe.
    use_ckpt = bool(checkpoint_jumps)

    def _jump_with(net_obj: nn.Module, z_in: Tensor, t_from: Tensor, t_to: Tensor, dummy: Tensor) -> Tensor:
        u = interface.pred(
            net=net_obj,
            xt=z_in,
            xsc=xsc,
            t=t_to,
            r=t_from,
            f=C,
            chiral_bias=chiral_bias,
        )
        return u + dummy.to(dtype=u.dtype) * 0

    def _jump_inner(z_in: Tensor, t_from: Tensor, t_to: Tensor, dummy: Tensor) -> Tensor:
        return _jump_with(core, z_in, t_from, t_to, dummy)

    dummy = torch.zeros((), device=device, dtype=dtype, requires_grad=True)
    for j, (i0, i1) in enumerate(pairs):
        t_from = _batch_time(float(grid[i0]), B, device=device, dtype=dtype)
        t_to = _batch_time(float(grid[i1]), B, device=device, dtype=dtype)
        sync_this = bool(ddp_sync_last) and (j == n_pairs - 1)
        z_in = z.detach() if detach_between_jumps else z
        with _ddp_no_sync(net, enabled=not sync_this):
            if use_ckpt and not sync_this:
                u = torch.utils.checkpoint.checkpoint(
                    _jump_inner,
                    z_in,
                    t_from,
                    t_to,
                    dummy,
                    use_reentrant=False,
                )
            else:
                u = _jump_with(net if is_ddp else core, z_in, t_from, t_to, dummy)
        dt = (t_to - t_from).view(-1, 1, 1)
        z = z + dt * u
        z = zero_com_suffix(z, w=C.mask)
        if self_cond:
            with torch.no_grad():
                u_inst = interface.pred(
                    net=net,
                    xt=z.detach(),
                    xsc=None,
                    t=t_to,
                    r=t_to,
                    f=C,
                    chiral_bias=chiral_bias,
                )
                xsc = interface.estimate_x1(z.detach(), t_to, u_inst)
    return zero_com_suffix(z, w=C.mask)


def _pcfm_env(name: str, default: str) -> str:
    return str(os.environ.get(name, default)).strip()


def _pcfm_correct_step(
    *,
    z_prev: Tensor,
    z: Tensor,
    u: Tensor,
    t_from: Tensor,
    t_to: Tensor,
    step: int,
) -> Tensor:
    """PCFM endpoint look-ahead + projection, re-aimed onto the current step.

    Follows Cai et al. (ICLR AI4Mat 2026): estimate the endpoint as
    ``x̂(1) = x(t) + (1-t) u``, project it towards the constraint set, then take
    the step towards the *corrected* endpoint.  The re-aiming factor
    ``dt / (1 - t_from)`` is < 1 early in the trajectory and exactly 1 on the
    final jump, so the correction is applied gradually and is only enforced
    outright at ``t = 1``.

    No-op unless a constraint has been published via ``pcfm.active_constraint``.
    """
    parity = _pcfm_env("CRYSTAF_PCFM_PARITY", "0").lower() in ("body", "1", "true")
    project = _pcfm_env("CRYSTAF_PCFM", "").lower() in ("rs", "1", "true", "chirality")
    if not (parity or project):
        return z
    from crystal_nft.meanflow import pcfm as _pcfm

    cst = _pcfm.get_active_constraint()
    if cst is None or cst.n_active == 0:
        return z
    every = max(1, int(_pcfm_env("CRYSTAF_PCFM_EVERY", "1")))
    if step % every != 0:
        return z
    # Stop projecting near t=1: on the final jumps the re-aiming factor is ~1, so
    # the sample *is* the raw projected endpoint and there is no model step left
    # to repair the bond lengths / angles PoseBusters checks.  Steering early and
    # letting the flow finish keeps the geometry and most of the chirality gain.
    t_max = float(_pcfm_env("CRYSTAF_PCFM_T_MAX", "1.0"))
    if t_max < 1.0 and float(t_from.min()) >= t_max:
        return z

    denom = (1.0 - t_from).view(-1, *([1] * (z.ndim - 1)))
    if float(denom.min()) <= 1e-6:
        return z
    x1_hat = z_prev + denom * u
    coords = x1_hat[:, 3:].float()

    skip_tol = float(_pcfm_env("CRYSTAF_PCFM_SKIP_TOL", "0") or 0.0)
    if skip_tol > 0 and float(_pcfm.chirality_residual(coords, cst).max()) <= skip_tol:
        return z

    fixed = coords
    if parity:
        # Discrete parity operator on the *endpoint estimate*, in the same
        # look-ahead/re-aim slot the Gauss-Newton projection uses.  Flipping a
        # body mid-trajectory (rather than after the last step) leaves the
        # remaining jumps free to relax the packing around it, which is where
        # the clash cost of a post-hoc flip comes from.
        fixed = _pcfm.resolve_body_handedness(
            fixed, cst,
            realign=_pcfm_env("CRYSTAF_MIRROR_REALIGN", "1") != "0",
            minimize_flips=_pcfm_env("CRYSTAF_MIRROR_MIN_FLIPS", "1") != "0",
        )
    if project:
        fixed = _pcfm.project_chirality(
            fixed,
            cst,
            n_iter=int(_pcfm_env("CRYSTAF_PCFM_ITERS", "2")),
            ridge=float(_pcfm_env("CRYSTAF_PCFM_RIDGE", "1e-4")),
            max_shift=float(_pcfm_env("CRYSTAF_PCFM_MAX_SHIFT", "0.5")),
        )
    if torch.equal(fixed, coords):
        return z
    x1_corr = x1_hat.clone()
    x1_corr[:, 3:] = fixed.to(x1_hat.dtype)
    dt = (t_to - t_from).view(-1, *([1] * (z.ndim - 1)))
    return z_prev + (dt / denom) * (x1_corr - z_prev)


def _rigid_project(z: Tensor) -> Tensor:
    """MolCrystalFlow-style rigid-body projection (``CRYSTAF_RIGID``).

    Exact no-op unless `crystal_nft.rigid` has published a conformer, so the
    trunk behaviour is bit-identical when the feature is off. See
    `crystal_nft/rigid/__init__.py` for why SO(3) makes chirality free.
    """
    if _pcfm_env("CRYSTAF_RIGID", "0").lower() in ("0", "", "false", "no"):
        return z
    from crystal_nft.rigid.projector import project_z

    return project_z(z)


def _pcfm_global_parity(z: Tensor) -> Tensor:
    """Final enantiomorph pick. ``CRYSTAF_MIRROR_FIX=cell|body|1``.

    ``cell`` inverts the whole unit cell (an isometry: every metric is exactly
    preserved).  ``body`` inverts individual molecules, which keeps every
    intramolecular distance — so PoseBusters is invariant — but moves the
    packing.  ``body`` is the useful one: a chirality-blind model picks each
    molecule's handedness independently, so cells come out mixed.
    """
    mode = _pcfm_env("CRYSTAF_MIRROR_FIX", "0").lower()
    if mode in ("0", "", "false", "no"):
        return z
    from crystal_nft.meanflow import pcfm as _pcfm

    cst = _pcfm.get_active_constraint()
    if cst is None or cst.n_active == 0:
        return z
    coords = z[:, 3:].float()
    if mode in ("body", "per_body", "molecule"):
        fixed = _pcfm.resolve_body_handedness(
            coords, cst,
            realign=_pcfm_env("CRYSTAF_MIRROR_REALIGN", "1") != "0",
            minimize_flips=_pcfm_env("CRYSTAF_MIRROR_MIN_FLIPS", "1") != "0",
        )
    else:
        fixed = _pcfm.resolve_global_handedness(coords, cst)
    out = z.clone()
    out[:, 3:] = fixed.to(z.dtype)
    return out



def _pcfm_bond_project(z: Tensor) -> Tensor:
    """PCFM bond-length projection (``CRYSTAF_PCFM_BOND=1``).

    Unlike the chirality residual, this targets a constraint PoseBusters actually
    checks (``bond_lengths_within_bounds``), so it can move ``pb_score``.
    """
    if _pcfm_env("CRYSTAF_PCFM_BOND", "0").lower() in ("0", "", "false", "no"):
        return z
    from crystal_nft.meanflow import pcfm as _pcfm

    cst = _pcfm.get_active_bond_constraint()
    if cst is None or cst.n_active == 0:
        return z
    # `z[:, 3:]` holds Cartesian coordinates divided by `Crystal.COORD_NORM`,
    # while the DG bounds are in Angstrom. Projecting without converting treats
    # every 1.5 A bond as 0.19 A -- "far too short" -- and blows the molecule
    # apart (measured: pb 2.1%, clash 91.7%). Work in Angstrom, convert back.
    scale = float(Crystal.COORD_NORM)
    fixed = _pcfm.project_bonds(
        z[:, 3:].float() * scale, cst,
        n_iter=int(_pcfm_env("CRYSTAF_PCFM_BOND_ITERS", "6")),
        max_shift=float(_pcfm_env("CRYSTAF_PCFM_BOND_MAX_SHIFT", "0.25")),
    )
    out = z.clone()
    out[:, 3:] = (fixed / scale).to(z.dtype)
    return out



_RELAX_WARNED: set[str] = set()


def _relax_warn_once(kind: str, exc: BaseException) -> None:
    """Report each distinct relaxation failure once, never silently."""
    if kind in _RELAX_WARNED:
        return
    _RELAX_WARNED.add(kind)
    import warnings

    warnings.warn(f"clash relaxation failed ({kind}): {exc}", RuntimeWarning, stacklevel=2)


def _pcfm_reflect_stereo(z: Tensor, C: Crystal) -> Tensor:
    """Branch-reflection fix for the post-flip residual (``CRYSTAF_STEREO_REFLECT=1``).

    Generalises the whole-molecule parity flip: reflecting the branch hanging off
    an acyclic bond, through a plane containing that bond, inverts only the
    stereocentres inside it and preserves every bond length and bond angle.
    """
    if _pcfm_env("CRYSTAF_STEREO_REFLECT", "0").lower() in ("0", "", "false", "no"):
        return z
    from crystal_nft.meanflow import pcfm as _pcfm

    cst = _pcfm.get_active_constraint()
    bonds = getattr(C, "bonds", None)
    if cst is None or cst.n_active == 0 or bonds is None:
        return z
    coords = z[:, 3:].float()
    fixed = _pcfm.fix_stereocentres_exact(
        coords, cst, bonds,
        rounds=int(_pcfm_env("CRYSTAF_STEREO_REFLECT_ROUNDS", "4")),
        atom_nums=getattr(C, "atom_nums", None),
        n_theta=int(_pcfm_env("CRYSTAF_STEREO_THETA", "24")),
    )
    if torch.equal(fixed, coords):
        return z
    out = z.clone()
    out[:, 3:] = fixed.to(z.dtype)
    return out


def _pcfm_swap_stereo(z: Tensor, C: Crystal) -> Tensor:
    """Exact per-centre inversion for the post-flip residual (``CRYSTAF_STEREO_SWAP=1``).

    The whole-molecule parity flip can only rescue molecules whose centres are
    uniformly wrong; measured on generated crystals, single-stereocentre
    molecules reach 100% while every remaining error sits in multi-centre ones.
    This swaps two substituent branches of each still-wrong centre with a proper
    rotation, inverting exactly that centre while keeping every bond length and
    all branch-internal geometry exact.
    """
    if _pcfm_env("CRYSTAF_STEREO_SWAP", "0").lower() in ("0", "", "false", "no"):
        return z
    from crystal_nft.meanflow import pcfm as _pcfm

    cst = _pcfm.get_active_constraint()
    bonds = getattr(C, "bonds", None)
    if cst is None or cst.n_active == 0 or bonds is None:
        return z
    coords = z[:, 3:].float()
    fixed = _pcfm.swap_fix_stereocentres(
        coords, cst, bonds,
        max_branch=int(_pcfm_env("CRYSTAF_STEREO_SWAP_MAX_BRANCH", "24")),
    )
    if torch.equal(fixed, coords):
        return z
    out = z.clone()
    out[:, 3:] = fixed.to(z.dtype)
    return out


def _pcfm_final_chirality(z: Tensor) -> Tensor:
    """Project the stereocentres the parity flip could not fix (``CRYSTAF_PCFM_FINAL=1``).

    Inverting a whole molecule flips every one of its stereocentres at once, so
    it can only rescue a molecule whose centres are *uniformly* wrong; a molecule
    with 3 centres and 2 wrong gets majority-flipped and keeps one error. The
    residual therefore lives in multi-stereocentre molecules.

    ``project_chirality`` masks itself to centres that still violate the margin,
    so running it *after* the flip touches only that residual instead of the
    ~50% a full-trajectory projection sees -- which is what made the standalone
    ``CRYSTAF_PCFM=rs`` cost 10 points of PB.
    """
    if _pcfm_env("CRYSTAF_PCFM_FINAL", "0").lower() in ("0", "", "false", "no"):
        return z
    from crystal_nft.meanflow import pcfm as _pcfm

    cst = _pcfm.get_active_constraint()
    if cst is None or cst.n_active == 0:
        return z
    coords = z[:, 3:].float()
    # `max_shift` here is in ANGSTROM. `project_chirality` sees the normalized
    # tensor (cartesian / COORD_NORM), where 0.35 would silently mean 2.8 A --
    # the same units trap that made the bond projection explode. Convert.
    scale = float(Crystal.COORD_NORM)
    ang = float(_pcfm_env("CRYSTAF_PCFM_FINAL_MAX_SHIFT", "0.30"))
    fixed = _pcfm.project_chirality(
        coords, cst,
        n_iter=int(_pcfm_env("CRYSTAF_PCFM_FINAL_ITERS", "12")),
        ridge=float(_pcfm_env("CRYSTAF_PCFM_RIDGE", "1e-4")),
        max_shift=ang / scale,
    )
    if torch.equal(fixed, coords):
        return z
    out = z.clone()
    out[:, 3:] = fixed.to(z.dtype)
    return out


def _pcfm_mmff_relax(z: Tensor, C: Crystal) -> Tensor:
    """Restrained MMFF relaxation of each molecule (``CRYSTAF_MMFF=1``).

    Repairs the local strain the stereochemistry operations leave behind
    (internal steric clash and bond angles) without moving any atom more than
    ``CRYSTAF_MMFF_DISPL`` from where the generator placed it, so the packing --
    and hence clash and PDD -- is preserved.
    """
    if _pcfm_env("CRYSTAF_MMFF", "0").lower() in ("0", "", "false", "no"):
        return z
    from crystal_nft.meanflow.relax import mmff_relax_bodies

    scale = float(Crystal.COORD_NORM)
    displ = float(_pcfm_env("CRYSTAF_MMFF_DISPL", "0.15"))
    k = float(_pcfm_env("CRYSTAF_MMFF_K", "50.0"))
    its = int(_pcfm_env("CRYSTAF_MMFF_ITS", "400"))
    out = z.clone()
    try:
        items = list(C.unbatch()) if getattr(C, "batched", False) else [C]
    except Exception as exc:  # noqa: BLE001
        _relax_warn_once("mmff:unbatch", exc)
        return z
    mask = getattr(C, "mask", None)
    for b, item in enumerate(items):
        try:
            new = mmff_relax_bodies(
                item, max_displ=displ, force_constant=k, max_its=its
            )
        except Exception as exc:  # noqa: BLE001
            _relax_warn_once(f"mmff:{type(exc).__name__}", exc)
            continue
        tgt = out[b, 3:]
        val = (new / scale).to(tgt.dtype).to(tgt.device)
        if mask is None:
            tgt.copy_(val)
        else:
            tgt[mask[b]] = val
    return out


def _pcfm_relax_torsions(z: Tensor, C: Crystal) -> Tensor:
    """Torsion-space repair of internal steric clash (``CRYSTAF_RELAX_TORSION=1``).

    The stereochemistry operations preserve bond lengths and angles exactly, but
    they swing branches into new positions, and the measured cost is almost
    entirely PoseBusters' ``internal_steric_clash`` (+2.96 points of fragments).
    Rotating about rotatable bonds changes only 1-4-and-longer distances, so it
    repairs that without disturbing bond lengths, bond angles or the chirality
    just fixed. Optimised against the metric's own DG lower bounds.
    """
    if _pcfm_env("CRYSTAF_RELAX_TORSION", "0").lower() in ("0", "", "false", "no"):
        return z
    from crystal_nft.meanflow.relax import relax_torsions
    from crystal_nft.meanflow.stereo import dg_clash_bounds

    scale = float(Crystal.COORD_NORM)
    rounds = int(_pcfm_env("CRYSTAF_RELAX_TORSION_ROUNDS", "3"))
    n_theta = int(_pcfm_env("CRYSTAF_RELAX_TORSION_THETA", "24"))
    out = z.clone()
    try:
        items = list(C.unbatch()) if getattr(C, "batched", False) else [C]
    except Exception as exc:  # noqa: BLE001
        _relax_warn_once("unbatch", exc)
        return z
    mask = getattr(C, "mask", None)
    for b, item in enumerate(items):
        try:
            dg = dg_clash_bounds(item)
            if not dg:
                continue
            keep = slice(None) if mask is None else mask[b]
            cart = (z[b, 3:].float() * scale)[keep]
            bonds = item.bonds
            fixed = relax_torsions(cart, bonds, dg, rounds=rounds, n_theta=n_theta)
        except Exception as exc:  # noqa: BLE001
            _relax_warn_once(f"torsion:{type(exc).__name__}", exc)
            continue
        tgt = out[b, 3:]
        if mask is None:
            tgt.copy_((fixed / scale).to(tgt.dtype))
        else:
            tgt[mask[b]] = (fixed / scale).to(tgt.dtype)
    return out


def _pcfm_relax_clashes(z: Tensor, C: Crystal) -> Tensor:
    """Rigid-body clash relaxation (``CRYSTAF_RELAX_CLASH=1``).

    ``clash_rate`` is a binary per-crystal flag (any intermolecular pair inside
    the sum of covalent radii), so clearing the worst contacts flips a crystal's
    score outright.  Moving whole molecules rigidly leaves intramolecular
    geometry and the lattice untouched, so ``pb_score`` and ``volume_error`` are
    invariant by construction -- only ``clash_rate`` and ``dist_pdd`` respond.
    """
    if _pcfm_env("CRYSTAF_RELAX_CLASH", "0").lower() in ("0", "", "false", "no"):
        return z
    from crystal_nft.meanflow.relax import relax_clashes

    body_ids = getattr(C, "body_ids", None)
    atom_nums = getattr(C, "atom_nums", None)
    if body_ids is None or atom_nums is None:
        return z
    n_iter = int(_pcfm_env("CRYSTAF_RELAX_ITERS", "60"))
    lr = float(_pcfm_env("CRYSTAF_RELAX_LR", "0.05"))
    slack = float(_pcfm_env("CRYSTAF_RELAX_SLACK", "0.05"))
    spring = float(_pcfm_env("CRYSTAF_RELAX_SPRING", "0.02"))
    max_shift = float(_pcfm_env("CRYSTAF_RELAX_MAX_SHIFT", "1.5"))

    scale = float(Crystal.COORD_NORM)
    out = z.clone()
    lattice = C.lattice  # (B, 3, 3) in Angstrom
    mask = getattr(C, "mask", None)
    for b in range(z.shape[0]):
        keep = slice(None) if mask is None else mask[b]
        cart = (z[b, 3:].float() * scale)[keep]
        bid = body_ids[b][keep] if body_ids.dim() > 1 else body_ids[keep]
        znum = atom_nums[b][keep] if atom_nums.dim() > 1 else atom_nums[keep]
        if cart.shape[0] == 0 or int(torch.unique(bid).numel()) < 2:
            continue  # single-molecule cell: no intermolecular pair to fix
        try:
            fixed = relax_clashes(
                cart, lattice[b].float(), bid, znum,
                n_iter=n_iter, lr=lr, slack=slack, spring=spring, max_shift=max_shift,
            )
        except Exception as exc:  # noqa: BLE001
            # A bare `continue` here turns the whole feature into a silent no-op;
            # two real bugs in this repo already hid behind a broad except.
            _relax_warn_once(type(exc).__name__, exc)
            continue
        tgt = out[b, 3:]
        if mask is None:
            tgt.copy_((fixed / scale).to(tgt.dtype))
        else:
            tgt[mask[b]] = (fixed / scale).to(tgt.dtype)
    return out



def _vol_scale_lattice(z: Tensor) -> Tensor:
    """Rescale the cell by ``CRYSTAF_VOL_SCALE`` (a *volume* factor).

    CrystAF systematically over-predicts cell volume: on 60 val families x 5
    samples at NFE=16 rho=0.75 the SIGNED relative volume error is
    **+3.21% (SE 0.33, bias/SE 9.8)**, with 74.7% of cells too large. Removing
    that offset cuts mean |rel err| from 4.68% to 3.96%.

    Clari stores the lattice and **Cartesian** coordinates independently
    (``to_ase`` uses ``positions=coords, cell=lattice``), so scaling only the
    lattice rows changes the cell without moving any atom. Intramolecular
    geometry -- and therefore PoseBusters -- is bit-identical; only the
    periodic quantities (clash, PDD) shift.

    ``CRYSTAF_VOL_SCALE`` is the factor applied to the volume, so each lattice
    row is scaled by its cube root. It MUST be calibrated on the train split
    (see scripts/calibrate_vol_scale.py); reading it off the evaluation targets
    would be fitting the metric.
    """
    raw = os.environ.get("CRYSTAF_VOL_SCALE", "")
    if not raw:
        return z
    vol_factor = float(raw)
    if vol_factor <= 0:
        raise ValueError(f"CRYSTAF_VOL_SCALE must be > 0, got {vol_factor}")
    if abs(vol_factor - 1.0) < 1e-12:
        return z
    row_scale = vol_factor ** (1.0 / 3.0)
    out = z.clone()
    out[..., :3, :] = out[..., :3, :] * row_scale
    return out


class MeanFlowCrystalSampler:
    """16-step (or few-step) sampler for a MeanFlow / AnyFlow student.

    ``mode="interval"`` is the dual-time flow map: ``z += (t-r) u(z,r,t)``.
    ``mode="interval_heun"`` is the same map with a predictor-corrector
    (2 U evals per jump; 8 jumps = 16 NFE).
    ``mode="heun"`` / ``mode="euler"`` are single-time slices (r=t) used
    only to compare against the Clari teacher protocol.
    """

    def __init__(
        self,
        num_steps: int = 1,
        mode: str = "interval",
        rho: float = 1.0,
        schedule: str = "power",
    ):
        self.num_steps = int(num_steps)
        mode_norm = str(mode).lower().strip()
        if mode_norm not in ("heun", "euler", "interval", "interval_heun"):
            raise ValueError(
                f"sampler mode must be heun|euler|interval|interval_heun, got {mode!r}"
            )
        self.mode = mode_norm
        self.rho = float(rho)
        self.schedule = str(schedule).lower().strip() or "power"
        self._heun = HeunSampler(num_steps=int(num_steps))

    def sample(
        self,
        interface: MeanFlowCrystalInterface,
        net: nn.Module,
        C: Crystal,
        *,
        sample_prior: bool = True,
        pbar: str | None = None,
        chiral_bias: Optional[Tensor] = None,
        return_trajectory: bool = False,
    ) -> Crystal | Tensor:
        if self.mode == "heun":
            self._heun.num_steps = int(self.num_steps)
            core = net.module if hasattr(net, "module") else net
            dit = getattr(core, "dit", None)
            if chiral_bias is not None and dit is not None:
                dit._mf_chiral_bias = chiral_bias
            try:
                return self._heun.sample(
                    interface,
                    net,
                    C,
                    sample_prior=sample_prior,
                    pbar=pbar,
                    return_trajectory=return_trajectory,
                )
            finally:
                if dit is not None:
                    dit._mf_chiral_bias = None

        if sample_prior:
            C0 = C.replace(x=torch.zeros_like(C.x))
            C0 = interface.sample_prior(C0)
        else:
            C0 = C

        z = C0.x
        xsc = None
        traj = [z] if return_trajectory else None
        B = C.batch_size if C.batched else 1
        device = C.device

        if self.num_steps <= 1 and self.mode != "interval_heun":
            t = torch.ones(B, device=device, dtype=z.dtype)
            r = torch.zeros(B, device=device, dtype=z.dtype)
            if self.mode == "euler":
                r = t
            u = interface.pred(
                net=net,
                xt=z,
                xsc=xsc,
                t=t,
                r=r,
                f=C,
                chiral_bias=chiral_bias,
            )
            z = z + u
        elif self.mode == "euler":
            # Match Clari EulerSampler: r=t, same-forward self-cond, no extra NFE.
            grid = torch.linspace(
                0.0, 1.0, self.num_steps + 1, device=device, dtype=z.dtype
            ).unsqueeze(-1)
            for i in tqdm.trange(
                self.num_steps,
                desc=pbar if isinstance(pbar, str) else "mf",
                leave=False,
                disable=not bool(pbar),
            ):
                t_curr = grid[i]
                t_next = grid[i + 1]
                dt = bcast_right(t_next - t_curr, z)
                v = interface.pred(
                    net=net,
                    xt=z,
                    xsc=xsc,
                    t=t_curr,
                    r=t_curr,
                    f=C,
                    chiral_bias=chiral_bias,
                )
                z_prev = z
                z = z + v * dt
                xsc = interface.estimate_x1(xt=z_prev, t=t_curr, pred=v)
                z = zero_com_suffix(z, w=C.mask)
                xsc = zero_com_suffix(xsc, w=C.mask)
                if return_trajectory:
                    traj.append(z)
        else:
            n_jump = max(1, int(self.num_steps))
            split_raw = str(os.environ.get("MEANFLOW_INTERVAL_SPLIT_FIRST", "0")).lower()
            if split_raw in ("keep8", "keep_coarse"):
                grid = interval_split_first_keep_coarse_grid(
                    8 if n_jump in (8, 9) else n_jump,
                    device=device,
                    dtype=z.dtype,
                    rho=self.rho,
                )
                n_jump = int(grid.numel()) - 1
            elif split_raw in ("1", "true", "yes") and n_jump >= 3:
                grid = interval_split_first_grid(
                    n_jump,
                    device=device,
                    dtype=z.dtype,
                    rho=self.rho,
                )
            else:
                grid = interval_time_grid(
                    n_jump,
                    device=device,
                    dtype=z.dtype,
                    rho=self.rho,
                    schedule=self.schedule,
                )
            use_heun = self.mode == "interval_heun"
            for i in tqdm.trange(
                n_jump,
                desc=pbar if isinstance(pbar, str) else "mf",
                leave=False,
                disable=not bool(pbar),
            ):
                t_from = torch.full((B,), float(grid[i]), device=device, dtype=z.dtype)
                t_to = torch.full((B,), float(grid[i + 1]), device=device, dtype=z.dtype)
                u = interface.pred(
                    net=net,
                    xt=z,
                    xsc=xsc,
                    t=t_to,
                    r=t_from,
                    f=C,
                    chiral_bias=chiral_bias,
                )
                dt = bcast_right(t_to - t_from, z)
                z_prev = z
                u_sc = u
                if use_heun:
                    z_pred = zero_com_suffix(z + dt * u, w=C.mask)
                    u2 = interface.pred(
                        net=net,
                        xt=z_pred,
                        xsc=xsc,
                        t=t_to,
                        r=t_from,
                        f=C,
                        chiral_bias=chiral_bias,
                    )
                    z = z + dt * 0.5 * (u + u2)
                    u_sc = 0.5 * (u + u2)
                else:
                    z = z + dt * u
                z = _pcfm_correct_step(
                    z_prev=z_prev, z=z, u=u_sc, t_from=t_from, t_to=t_to, step=i
                )
                if _self_cond_enabled(net):
                    xsc = interface.estimate_x1(xt=z_prev, t=t_from, pred=u_sc)
                    xsc = zero_com_suffix(xsc, w=C.mask)
                z = zero_com_suffix(z, w=C.mask)
                if return_trajectory:
                    traj.append(z)

        # Correct the known cell-volume bias BEFORE the repair steps, so the
        # rigid clash relaxation below can resolve any contacts the shrink
        # creates. (Applying it last would leave those contacts in place.)
        z = _vol_scale_lattice(z)
        z = _rigid_project(z)
        z = _pcfm_global_parity(z)
        z = _pcfm_reflect_stereo(z, C.replace(x=z))
        z = _pcfm_swap_stereo(z, C.replace(x=z))
        z = _pcfm_final_chirality(z)
        z = _pcfm_relax_torsions(z, C.replace(x=z))
        z = _pcfm_mmff_relax(z, C.replace(x=z))
        z = _pcfm_bond_project(z)
        z = _pcfm_relax_clashes(z, C.replace(x=z))
        z = zero_com_suffix(z, w=C.mask)
        if return_trajectory:
            return torch.stack(traj)
        return C.replace(x=z)
