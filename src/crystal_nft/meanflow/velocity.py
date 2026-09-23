"""Convert flow-map predictions to instantaneous velocity (MeanFlowNFT-compatible)."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor


def central_difference_dudt(
    fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    z: Tensor,
    r: Tensor,
    t: Tensor,
    v_dir: Tensor,
    eps: float = 1e-3,
) -> Tensor:
    """
    Stop-gradient central difference of total d u / d t along tangent (v_dir, 0, 1).

    MeanFlow / AnyFlow identity needs the *trajectory* derivative
    ``du/dt = (du/dz)·v + du/dt_explicit``, not a pure time partial.
    Matches ``AnyFlowPretrainTrainer._compute_central_difference`` with
    continuous time in ``[0, 1]`` (equivalent to ``T=1`` discrete scaling):

        z± = z ± v_dir * eps
        t± = t ± eps
        dudt ≈ (u(z+, r, t+) - u(z-, r, t-)) / (2 * eps)
    """
    eps_f = float(eps)
    # Keep t in [0, 1]; scale spatial steps by the *actual* half-step used.
    t_plus = (t + eps_f).clamp(max=1.0)
    t_minus = (t - eps_f).clamp(min=0.0)
    dt_plus = (t_plus - t).view(-1, 1, 1)
    dt_minus = (t - t_minus).view(-1, 1, 1)
    z_plus = z + v_dir * dt_plus
    z_minus = z - v_dir * dt_minus
    denom = (t_plus - t_minus).view(-1, 1, 1).clamp(min=1e-6)
    with torch.no_grad():
        u_plus = fn(z_plus, r, t_plus)
        u_minus = fn(z_minus, r, t_minus)
    return (u_plus - u_minus) / denom


def central_difference_dudr(
    fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    z: Tensor,
    r: Tensor,
    t: Tensor,
    v_dir: Tensor,
    eps: float = 5e-3,
) -> Tensor:
    """Total derivative along the Clari forward trajectory at start time ``r``.

    Native AnyFlow uses a noise-to-data reverse step and differentiates its
    current (noise) time. Clari time is reversed (0=noise, 1=data), so the
    equivalent derivative is with respect to the start time ``r``, holding
    endpoint ``t`` fixed.
    """
    eps_f = float(eps)
    r_plus = r + eps_f
    r_minus = r - eps_f
    z_plus = z + v_dir * eps_f
    z_minus = z - v_dir * eps_f
    with torch.no_grad():
        u_plus = fn(z_plus, r_plus, t)
        u_minus = fn(z_minus, r_minus, t)
    return (u_plus - u_minus) / (2.0 * eps_f)


def flow_map_to_instantaneous_velocity(
    u: Tensor,
    z: Tensor,
    r: Tensor,
    t: Tensor,
    dudt: Tensor,
) -> Tensor:
    """V(z, t) = u(z, r, t) + (t - r) * du/dt -- the SD3-time form.

    Only valid when the differentiated time is the one the *state* sits at and
    it is the integral's **upper** limit, i.e. MeanFlowNFT's own convention
    (``t=0`` data, ``t=T`` noise, state at ``t``, jump down to ``s``). Clari
    reverses time, which flips the sign -- use ``clari_induced_velocity``.
    """
    return u + (t - r).view(-1, 1, 1) * dudt


def clari_induced_velocity(u: Tensor, correction: Tensor) -> Tensor:
    """Induced instantaneous velocity under Clari time: ``V = U - correction``.

    ``correction`` is the already-scaled term ``(t - r) dU/dr`` -- passed
    pre-multiplied so the exact-gap variant (which measures
    ``U_old(x_r,r,t) - U_old(x_r,r,r)`` directly and has no ``dU/dr`` to
    scale) can share this path without dividing by ``t - r``, which is zero on
    the ``r = t`` slice.

    The sign differs from the paper (Eq. 3, ``v = u + (t-s) du/dt``) because
    Clari runs ``0=noise -> 1=data`` and the flow map jumps *up* from ``r`` to
    ``t``, so the time the state sits at is the integral's **lower** limit::

        (t - r) U(x_r, r, t) = int_r^t v(x_s, s) ds
        d/dr LHS = -U + (t - r) dU/dr
        d/dr RHS = -v(x_r, r)                    # lower limit -> minus
        => v(x_r, r) = U(x_r, r, t) - (t - r) dU/dr

    Equivalent to the ``U = v_teacher(z, r) + (t-r) dU/dr`` form the AnyFlow
    distillation regresses against, solved for ``v`` instead of ``U``.

    Verified numerically on the cont3 student: the time partial alone has
    ``cos((t-r) dU/dr, U - U(x_r,r,r)) = +0.79``, so subtracting it is what
    moves ``U`` toward the model's own boundary velocity.
    """
    return u - correction


def central_difference_dudr_clamped(
    fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    z: Tensor,
    r: Tensor,
    t: Tensor,
    v_dir: Tensor,
    eps: float = 5e-3,
) -> Tensor:
    """``central_difference_dudr`` with the time steps clamped to ``[0, t]``.

    ``central_difference_dudr`` walks ``r`` by a fixed ``±eps``, which leaves
    the time axis whenever ``r < eps`` (the consistency mode puts ``r`` exactly
    at 0) or ``r > t - eps``. Here each side is clamped and the spatial step
    and denominator use the half-step that was actually taken, so the estimate
    stays a valid one-sided/two-sided difference at the ends of the interval.
    """
    eps_f = float(eps)
    r_plus = torch.minimum(r + eps_f, t)
    r_minus = (r - eps_f).clamp(min=0.0)
    dr_plus = (r_plus - r).view(-1, 1, 1)
    dr_minus = (r - r_minus).view(-1, 1, 1)
    denom = (r_plus - r_minus).view(-1, 1, 1)
    z_plus = z + v_dir * dr_plus
    z_minus = z - v_dir * dr_minus
    with torch.no_grad():
        u_plus = fn(z_plus, r_plus, t)
        u_minus = fn(z_minus, r_minus, t)
    out = (u_plus - u_minus) / denom.clamp(min=1e-6)
    # Degenerate interval (r == t == 0 or r == t == 1): no direction to walk.
    return torch.where(denom > 1e-6, out, torch.zeros_like(out))


def flow_map_dudr(
    interface,
    net,
    z: Tensor,
    xsc: Optional[Tensor],
    r: Tensor,
    t: Tensor,
    f,
    *,
    chiral_bias: Optional[Tensor] = None,
    v_dir: Tensor,
    cd_eps: float = 5e-3,
) -> Tensor:
    """Stop-gradient ``dU/dr`` of one flow-map network along the Clari path.

    This is the derivative the MeanFlow / AnyFlow identity needs under Clari
    time (``0=noise``, ``1=data``): the state ``z`` lives at the *start* time
    ``r``, so the "current time" being differentiated is ``r``, not ``t``
    (MeanFlowNFT's SD3 convention has the state at the noisier time ``t`` and
    differentiates that one instead).
    """

    def fn(cur_z, cur_r, cur_t):
        return interface.forward(
            net=net,
            xt=cur_z,
            xsc=xsc,
            t=cur_t,
            r=cur_r,
            f=f,
            chiral_bias=chiral_bias,
        )

    return central_difference_dudr_clamped(fn, z, r, t, v_dir, eps=cd_eps)


def sample_nft_times(
    batch_size: int,
    device: torch.device | str,
    *,
    diffusion_ratio: float = 0.5,
    consistency_ratio: float = 0.0,
    nfe_steps: int = 16,
    rho: float = 1.0,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[Tensor, Tensor, Tensor]:
    """MeanFlowNFT three-mode ``(r, t)`` draw in Clari time (``r <= t``).

    Mirrors ``MeanFlowNFTTrainer._sample_training_tr_three_mode`` with the two
    changes Clari's reversed time forces:

    - the state lives at ``r`` (the noisier end), so ``r`` is the "current"
      time and the diffusion mode pins ``r = t``;
    - the consistency mode jumps to the data end (``t = 1``) instead of to 0.

    With ``nfe_steps > 1`` the generic mode draws *adjacent* jumps on the
    ``N``-step power grid ``t_i = (i/N)**rho`` -- the same restriction that took
    the CrystAF student from 58% to 77% during distillation, and the grid the
    interval sampler actually walks at eval. ``nfe_steps <= 1`` falls back to
    continuous ``U(0, 1)`` pairs (the reference's own recipe).

    The mode partition is a deterministic function of the *global* index, so
    every rank sees the same mix without a collective.
    """
    total = float(diffusion_ratio) + float(consistency_ratio)
    if not 0.0 <= total <= 1.0 + 1e-6:
        raise ValueError(
            f"diffusion_ratio ({diffusion_ratio}) + consistency_ratio "
            f"({consistency_ratio}) must lie in [0, 1], got {total}"
        )
    if rho <= 0.0:
        raise ValueError(f"rho must be > 0, got {rho}")

    b = int(batch_size)
    ws = max(1, int(world_size))
    global_bsz = ws * b
    start = int(rank) * b
    idx = torch.arange(start, start + b, device=device)
    n_diffusion = round(float(diffusion_ratio) * global_bsz)
    n_consistency = round(float(consistency_ratio) * global_bsz)
    is_diffusion = idx < n_diffusion
    is_consistency = (idx >= n_diffusion) & (idx < n_diffusion + n_consistency)

    n = int(nfe_steps)
    if n > 1:
        k = torch.randint(0, n, (b,), device=device)
        r_u = k.to(torch.float32) / float(n)
        t_u = (k + 1).to(torch.float32) / float(n)
    else:
        pair = torch.rand(b, 2, device=device)
        r_u = pair.min(dim=1).values
        t_u = pair.max(dim=1).values

    if abs(float(rho) - 1.0) > 1e-8:
        r_u = r_u.clamp(0.0, 1.0).pow(float(rho))
        t_u = t_u.clamp(0.0, 1.0).pow(float(rho))

    r = r_u
    t = torch.where(is_diffusion, r_u, t_u)
    t = torch.where(is_consistency, torch.ones_like(t), t)
    return r, t, is_diffusion
