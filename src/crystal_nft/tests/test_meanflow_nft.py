"""CPU tests for the MeanFlowNFT / DiffusionNFT update on the dual-time student.

No GPU, no Clari checkpoint: the flow-map network is a tiny analytic stand-in
with the same ``interface.forward(net, xt, xsc, t, f, r=..)`` contract.
"""

from __future__ import annotations

import torch

from crystal_nft.meanflow.velocity import (
    central_difference_dudr_clamped,
    clari_induced_velocity,
    flow_map_to_instantaneous_velocity,
    sample_nft_times,
)
from crystal_nft.nft.loss import nft_mixture_predictions


def test_nft_times_ordering_and_grid():
    """r <= t always, and off-diffusion pairs are adjacent grid jumps."""
    r, t, is_diff = sample_nft_times(
        64, "cpu", diffusion_ratio=0.5, consistency_ratio=0.0, nfe_steps=16, rho=1.0
    )
    assert torch.all(r <= t + 1e-6)
    assert torch.all(r >= 0.0) and torch.all(t <= 1.0)
    # Diffusion slice is pinned to r == t.
    assert torch.allclose(r[is_diff], t[is_diff])
    # The rest jump exactly one 16-step cell.
    gap = (t[~is_diff] - r[~is_diff])
    assert torch.allclose(gap, torch.full_like(gap, 1.0 / 16.0), atol=1e-6)
    # r sits on the grid.
    on_grid = (r * 16.0).round() / 16.0
    assert torch.allclose(r, on_grid, atol=1e-6)


def test_nft_times_power_grid_matches_eval_schedule():
    """rho reshapes both endpoints exactly like the interval sampler's grid."""
    rho = 0.75
    r, t, is_diff = sample_nft_times(
        128, "cpu", diffusion_ratio=0.0, consistency_ratio=0.0, nfe_steps=16, rho=rho
    )
    grid = (torch.arange(17, dtype=torch.float32) / 16.0).pow(rho)
    for r_i, t_i in zip(r.tolist(), t.tolist()):
        assert torch.isclose(grid, torch.tensor(r_i)).any()
        assert torch.isclose(grid, torch.tensor(t_i)).any()
        assert t_i > r_i
    assert not bool(is_diff.any())


def test_nft_times_consistency_mode_targets_data_end():
    """Clari time runs 0=noise -> 1=data, so 'consistency' means t = 1."""
    r, t, is_diff = sample_nft_times(
        32, "cpu", diffusion_ratio=0.0, consistency_ratio=1.0, nfe_steps=16, rho=1.0
    )
    assert torch.allclose(t, torch.ones_like(t))
    assert not bool(is_diff.any())
    assert torch.all(r <= t)


def test_nft_times_partition_is_global():
    """Every rank computes the same mode assignment from the global index."""
    kwargs = dict(diffusion_ratio=0.5, consistency_ratio=0.25, nfe_steps=16, rho=1.0)
    masks = []
    for rank in range(4):
        _, _, is_diff = sample_nft_times(8, "cpu", rank=rank, world_size=4, **kwargs)
        masks.append(is_diff)
    flat = torch.cat(masks)
    # 0.5 * 32 = 16 diffusion samples globally, all in the low global indices.
    assert int(flat.sum()) == 16
    assert bool(flat[:16].all()) and not bool(flat[16:].any())


def test_uniform_grid_is_default_when_rho_one():
    r, t, _ = sample_nft_times(
        16, "cpu", diffusion_ratio=0.0, nfe_steps=4, rho=1.0
    )
    assert torch.allclose((t - r), torch.full_like(t, 0.25))


def test_continuous_fallback_when_nfe_steps_disabled():
    r, t, _ = sample_nft_times(
        256, "cpu", diffusion_ratio=0.0, nfe_steps=0, rho=1.0
    )
    assert torch.all(r <= t)
    # Continuous draws are not on any coarse grid.
    assert (t - r).unique().numel() > 100


def _linear_flow_map(dudr_true: float):
    """Flow map whose dU/dr along the path is exactly ``dudr_true``.

    ``U(z, r, t) = base(z) + dudr_true * r`` with ``base`` linear in ``z`` and
    zero slope, so the central difference must recover ``dudr_true`` regardless
    of the tangent direction.
    """

    def fn(z, r, t):
        return torch.zeros_like(z) + dudr_true * r.view(-1, 1, 1)

    return fn


def test_central_difference_recovers_known_dudr():
    z = torch.randn(5, 6, 3)
    r = torch.full((5,), 0.4)
    t = torch.full((5,), 0.9)
    v_dir = torch.randn_like(z)
    got = central_difference_dudr_clamped(_linear_flow_map(2.5), z, r, t, v_dir, eps=5e-3)
    assert torch.allclose(got, torch.full_like(got, 2.5), atol=1e-4)


def test_central_difference_is_one_sided_at_r_zero():
    """r = 0 (consistency mode) must not step off the time axis."""
    z = torch.randn(3, 4, 3)
    r = torch.zeros(3)
    t = torch.ones(3)
    v_dir = torch.randn_like(z)
    got = central_difference_dudr_clamped(_linear_flow_map(1.0), z, r, t, v_dir, eps=0.01)
    # Half-step forward only: (U(eps) - U(0)) / eps = 1.0, still exact here.
    assert torch.allclose(got, torch.ones_like(got), atol=1e-4)


def test_central_difference_zero_on_degenerate_interval():
    z = torch.randn(2, 4, 3)
    zeros = torch.zeros(2)
    got = central_difference_dudr_clamped(
        _linear_flow_map(3.0), z, zeros, zeros, torch.randn_like(z), eps=0.01
    )
    assert torch.count_nonzero(got) == 0


def test_identity_is_a_no_op_at_the_boundary():
    """At r = t the flow map already *is* the instantaneous velocity."""
    u = torch.randn(4, 5, 3)
    dudr = torch.randn_like(u)
    t = torch.rand(4)
    v = flow_map_to_instantaneous_velocity(u, None, t, t, dudr)
    assert torch.allclose(v, u)
    zero = (t - t).view(-1, 1, 1) * dudr
    assert torch.allclose(clari_induced_velocity(u, zero), u)


def test_clari_identity_sign_is_minus_not_plus():
    """Clari time flips the MeanFlow identity's sign.

    The paper differentiates the time the state sits at *and* which is the
    integral's upper limit (``v = u + (t-s) du/dt``). Clari's state sits at the
    jump's start ``r``, the integral's **lower** limit, so the Leibniz term
    comes in negative and ``v = U - (t-r) dU/dr``. Getting this backwards does
    not fail loudly -- it silently doubles the discrepancy it should cancel,
    which cost ~12 PB points in the first Stage-2 run.
    """
    u = torch.randn(6, 5, 3)
    dudr = torch.randn_like(u)
    r = torch.rand(6) * 0.5
    t = r + 0.1
    scaled = (t - r).view(-1, 1, 1) * dudr
    got = clari_induced_velocity(u, scaled)
    assert torch.allclose(got, u - scaled)
    # Explicitly NOT the SD3-time form.
    assert not torch.allclose(got, flow_map_to_instantaneous_velocity(u, None, r, t, dudr))


def test_exact_gap_correction_recovers_the_boundary_velocity():
    """The exact-gap mode's correction turns U_old into U_old at r = t.

    ``correction = U_old(x_r,r,t) - U_old(x_r,r,r)`` so
    ``V_old = U_old(x_r,r,t) - correction == U_old(x_r,r,r)`` exactly -- the old
    policy's own instantaneous velocity, with no finite differencing. This is
    the property the central-difference estimator failed to deliver on the
    cont3 student (its correction was ~6x too large and per-sample
    uncorrelated with this gap).
    """
    u_old_interval = torch.randn(4, 5, 3)
    u_old_boundary = torch.randn(4, 5, 3)
    correction = u_old_interval - u_old_boundary
    assert torch.allclose(
        clari_induced_velocity(u_old_interval, correction), u_old_boundary
    )


def test_shared_correction_keeps_policy_gap_on_the_average_velocity():
    """Any shared correction leaves V_theta - V_old == U_theta - U_old.

    True for both the central-difference and the exact-gap mode, and it is why
    neither collapses to the ``r = t`` case: the gradient still reaches
    ``U_theta(x_r, r, t)``, the average-velocity head at the real interval.
    """
    u_new, u_old = torch.randn(4, 5, 3), torch.randn(4, 5, 3)
    shared = torch.randn(4, 5, 3)
    v_new = clari_induced_velocity(u_new, shared)
    v_old = clari_induced_velocity(u_old, shared)
    assert torch.allclose(v_new - v_old, u_new - u_old, atol=1e-6)


def test_shared_cd_cancels_from_the_policy_difference():
    """MeanFlowNFT's shared dU/dr leaves V_theta - V_old equal to U_theta - U_old.

    That cancellation is the reason the positive / negative constructions stay
    exact mixtures in velocity space; a per-network CD would inject its own
    finite-difference error into the difference the loss is built from.
    """
    u_new = torch.randn(4, 5, 3)
    u_old = torch.randn(4, 5, 3)
    shared = torch.randn(4, 5, 3)
    r = torch.rand(4) * 0.5
    t = r + 0.25
    v_new = flow_map_to_instantaneous_velocity(u_new, None, r, t, shared)
    v_old = flow_map_to_instantaneous_velocity(u_old, None, r, t, shared)
    assert torch.allclose(v_new - v_old, u_new - u_old, atol=1e-6)

    beta = 0.1
    pos_v, neg_v = nft_mixture_predictions(v_new, v_old, beta=beta)
    pos_u, neg_u = nft_mixture_predictions(u_new, u_old, beta=beta)
    shift = (t - r).view(-1, 1, 1) * shared
    # Both implicit policies are shifted by the same identity term, so the
    # MeanFlowNFT and DiffusionNFT arms differ by a common offset -- not by a
    # rescaling of the positive/negative gap.
    assert torch.allclose(pos_v, pos_u + shift, atol=1e-6)
    assert torch.allclose(neg_v, neg_u + shift, atol=1e-6)


def test_identity_term_vanishes_only_on_the_diffusion_slice():
    """The two arms are identical at r = t and differ everywhere else."""
    u = torch.randn(8, 5, 3)
    dudr = torch.ones_like(u)
    r, t, is_diff = sample_nft_times(
        8, "cpu", diffusion_ratio=0.5, nfe_steps=16, rho=1.0
    )
    v = flow_map_to_instantaneous_velocity(u, None, r, t, dudr)
    delta = (v - u).abs().flatten(1).max(dim=1).values
    assert torch.all(delta[is_diff] == 0)
    assert torch.all(delta[~is_diff] > 0)


def test_student_tune_full_restores_trainability_after_a_freeze():
    """`student_tune: full` must unfreeze, not just count.

    `train_meanflow_nft.py` applies the checkpoint's own tuning mode first (to
    load it correctly) and then the NFT tuning mode. The "full" branch used to
    early-return after only counting parameters, so the earlier `delta_lora`
    freeze survived and `nft_student_tune: full` silently trained 3.8% of the
    network -- which is why the full fine-tune that made the Clari `nft-m` run
    work (+3.2 PB) could never be reproduced on the student.
    """
    import torch.nn as nn

    from crystal_nft.meanflow.tune import apply_student_tune

    net = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    total = sum(p.numel() for p in net.parameters())
    for p in net.parameters():
        p.requires_grad_(False)
    assert sum(p.numel() for p in net.parameters() if p.requires_grad) == 0

    stats = apply_student_tune(net, {"student_tune": "full"})
    assert sum(p.numel() for p in net.parameters() if p.requires_grad) == total
    assert stats["n_trainable"] == total


def test_student_tune_full_keeps_the_8nfe_lora_frozen():
    """Training the zero-init 8-NFE LoRA is a recorded dead end; "full" must
    not silently switch it on."""
    import torch
    import torch.nn as nn

    from crystal_nft.meanflow.tune import apply_student_tune

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.lora8_A = nn.Parameter(torch.zeros(4, 4))

    net = Net()
    for p in net.parameters():
        p.requires_grad_(False)
    apply_student_tune(net, {"student_tune": "full"})
    got = {n: p.requires_grad for n, p in net.named_parameters()}
    assert got["lora8_A"] is False, got
    assert got["proj.weight"] is True, got

    apply_student_tune(net, {"student_tune": "full", "train_nfe8_lora": True})
    assert dict(net.named_parameters())["lora8_A"].requires_grad is True


def test_param_ema_lags_the_live_weights_and_only_covers_trainables():
    """Generator EMA: both reference implementations keep one and report it.

    This trainer saved `ema_state=None`, so evaluation always read live
    weights. That matters because every NFT arm peaks at its first checkpoint
    and then degrades -- an EMA stays near the peak instead of following the
    drift.
    """
    import torch
    import torch.nn as nn

    from crystal_nft.train.train_meanflow_nft import _ParamEMA

    net = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    net[1].weight.requires_grad_(False)          # simulate a frozen slice
    ema = _ParamEMA(net, 0.9)
    assert "1.weight" not in ema.shadow, "frozen params must not be tracked"
    assert "0.weight" in ema.shadow

    start = net[0].weight.detach().clone()
    with torch.no_grad():
        net[0].weight.add_(1.0)
    ema.update(net)
    got = ema.state_dict(net)["0.weight"]
    # decay 0.9 => EMA moves 10% of the way toward the new value
    assert torch.allclose(got, start + 0.1, atol=1e-6), got
    # untracked tensors pass through unchanged
    assert torch.allclose(ema.state_dict(net)["1.weight"], net[1].weight.detach())


def test_param_ema_converges_toward_a_stable_target():
    import torch
    import torch.nn as nn

    from crystal_nft.train.train_meanflow_nft import _ParamEMA

    net = nn.Linear(3, 3)
    with torch.no_grad():
        net.weight.zero_()
    ema = _ParamEMA(net, 0.5)
    with torch.no_grad():
        net.weight.fill_(1.0)
    for _ in range(20):
        ema.update(net)
    assert torch.allclose(ema.state_dict(net)["weight"], torch.ones(3, 3), atol=1e-4)


def test_flat_pb_skip_respects_clash_rank_weight():
    """A clash-weighted rank key must not be short-circuited by flat PB.

    `clash_rank_weight > 0` puts clash into the ranking key, so a family whose
    PoseBusters scores are all equal is still rankable -- and those are the
    families that carry clash signal. Skipping them silently drops the exact
    gradient the clash target needs.
    """
    from crystal_nft.train.train_meanflow_nft import pb_is_whole_rank_key

    assert pb_is_whole_rank_key({}) is True
    assert pb_is_whole_rank_key({"vol_rank_weight": 0.0, "clash_rank_weight": 0.0})
    assert not pb_is_whole_rank_key({"clash_rank_weight": 0.5})
    assert not pb_is_whole_rank_key({"vol_rank_weight": 1.0})
    assert not pb_is_whole_rank_key({"vol_rank_weight": 0.0, "clash_rank_weight": 1.0})


def test_instantaneous_degenerates_to_flow_map_when_r_equals_t():
    """The `r = t` slice makes MeanFlowNFT bitwise identical to DiffusionNFT.

    `nft_velocity_mode: instantaneous` differs from `flow_map` only through the
    identity term `(t - r) * dU/dr`. On the diffusion slice `(t - r) = 0`, so
    those samples carry *no* MeanFlowNFT content at all -- with
    `nft_diffusion_ratio: 0.5` that is half the batch. This is the property that
    made the two modes nearly indistinguishable in practice; it must stay
    visible rather than be rediscovered.
    """
    import torch

    from crystal_nft.meanflow.velocity import clari_induced_velocity
    from crystal_nft.nft.loss import nft_reconstruction_loss

    torch.manual_seed(0)
    B, N, D, beta = 4, 16, 3, 0.1
    x1, xr = torch.randn(B, N, D), torch.randn(B, N, D)
    u_old = torch.randn(B, N, D)
    time_r, r_w = torch.rand(B) * 0.9, torch.randint(0, 2, (B,)).float()
    zero_corr = torch.zeros(B, N, D)  # (t - r) = 0 on the whole batch

    def grad(mode, corr):
        u = torch.randn(B, N, D, generator=torch.Generator().manual_seed(1))
        u.requires_grad_(True)
        fwd = clari_induced_velocity(u, corr) if mode == "inst" else u
        old = clari_induced_velocity(u_old, corr) if mode == "inst" else u_old
        nft_reconstruction_loss(
            fwd, old, xt=xr, clean=x1, t=time_r, r=r_w, beta=beta,
            time_convention="clari",
        )["policy_loss"].backward()
        return u.grad

    assert torch.equal(grad("inst", zero_corr), grad("flow_map", zero_corr))

    # With a real jump the modes must actually diverge, else the identity is
    # doing nothing anywhere and `instantaneous` is mislabelled.
    corr = (1.0 / 32.0) * torch.randn(B, N, D)
    gi, gf = grad("inst", corr), grad("flow_map", corr)
    assert not torch.equal(gi, gf)
    # ...but only slightly, at adjacent-grid jump length. Guarding the order of
    # magnitude documents *why* the two arms behaved alike.
    rel = ((gi - gf).norm() / gf.norm()).item()
    assert 1e-4 < rel < 0.10, rel


def test_kl_term_is_identical_in_both_velocity_modes():
    """`share_cd_with_old` makes the KL anchor blind to the velocity mode.

    forward - ref = (U_fwd - c) - (U_ref - c) = U_fwd - U_ref, so the KL cannot
    distinguish MeanFlowNFT from DiffusionNFT at all. Only the policy term can.
    """
    import torch

    from crystal_nft.meanflow.velocity import clari_induced_velocity
    from crystal_nft.nft.loss import kl_velocity_loss

    torch.manual_seed(0)
    u_fwd, u_ref = torch.randn(3, 8, 3), torch.randn(3, 8, 3)
    corr = 0.3 * torch.randn(3, 8, 3)
    assert torch.allclose(
        kl_velocity_loss(u_fwd, u_ref),
        kl_velocity_loss(clari_induced_velocity(u_fwd, corr),
                         clari_induced_velocity(u_ref, corr)),
    )


def test_induced_velocity_sign_matches_the_sampler_state_time():
    """The identity must return v at `r` -- the time the sampler's state sits at.

    Both `interval_flow_map_jump` (`r=t_from`, `z += (t-r)*u(z,r,t)`) and the NFT
    step (`xr = sample_xt(x0, x1, time_r)`) put the state at the *start* of the
    jump. So `clari_induced_velocity` has to recover v(x_r, **r**), not v(x_r, t).

    Checked against an analytic case rather than the algebra it was derived
    from. For v(x, s) = s * a the exact flow map is

        x_t = x_r + a (t^2 - r^2) / 2   =>   U(x_r, r, t) = a (t + r) / 2
        dU/dr = a / 2                   =>   (t - r) dU/dr = a (t - r) / 2

    so  U - (t-r) dU/dr = a*r  (start, correct) and
        U + (t-r) dU/dr = a*t  (end, the sign error that cost 12 PB).
    """
    import torch

    from crystal_nft.meanflow.velocity import clari_induced_velocity

    a = torch.tensor([2.0, -1.0, 0.5])
    r, t = 0.25, 0.75  # r < t, Clari time: 0=noise -> 1=data

    u = a * (t + r) / 2.0
    correction = (t - r) * (a / 2.0)

    v = clari_induced_velocity(u, correction)
    assert torch.allclose(v, a * r), v          # velocity at the jump START
    assert not torch.allclose(v, a * t)         # ...not at its end

    # The opposite sign would silently return the end-time velocity.
    assert torch.allclose(u + correction, a * t)


def test_sampler_and_nft_step_agree_on_which_time_holds_the_state():
    """Guard the shared convention: r is the jump start and r <= t, in both paths."""
    import inspect
    import torch

    from crystal_nft.meanflow import sampler
    from crystal_nft.meanflow.velocity import sample_nft_times

    # Sampler: r comes from t_from, t from t_to, and it steps by (t - r).
    src = inspect.getsource(sampler.interval_flow_map_jump)
    assert "r = _batch_time(t_from" in src
    assert "t = _batch_time(t_to" in src
    assert "dt = (t - r)" in src

    # NFT (r, t) draw: r <= t always, so the state time is the smaller one.
    r, t, is_diff = sample_nft_times(
        64, torch.device("cpu"), diffusion_ratio=0.5, consistency_ratio=0.25,
        nfe_steps=32, rho=1.0,
    )
    assert bool((r <= t + 1e-9).all())
    assert bool((t[is_diff] == r[is_diff]).all())      # diffusion slice: r == t
    consistency = (~is_diff) & (t >= 1.0 - 1e-9)
    assert int(consistency.sum()) > 0                   # consistency slice: t == 1


def test_zero_v_dir_gives_the_pure_time_partial():
    """`cd_velocity_source: time_partial` must drop the spatial term exactly.

    `central_difference_dudr_clamped` walks the state by `v_dir * dr` while
    stepping `r`, so it returns the *total* derivative along the path. With
    `v_dir = 0` the state is held fixed and it reduces to dU/dr|_x -- which is
    the whole point of the option: the repo measured the spatial term
    (dU/dz).v to be per-sample uncorrelated with the gap it must reproduce.
    """
    import torch

    from crystal_nft.meanflow.velocity import central_difference_dudr_clamped

    # U depends on the state as well as on r, so a total derivative and a time
    # partial genuinely differ here.
    def fn(z, r, t):
        return 3.0 * z + r.view(-1, 1, 1) * 2.0

    z = torch.randn(4, 6, 3)
    r = torch.full((4,), 0.4)
    t = torch.full((4,), 0.9)
    v = torch.randn(4, 6, 3)

    total = central_difference_dudr_clamped(fn, z, r, t, v, eps=1e-3)
    partial = central_difference_dudr_clamped(fn, z, r, t, torch.zeros_like(v), eps=1e-3)

    # d/dr [3z(r) + 2r] with z moving at v = 3v + 2 ; with z fixed = 2.
    assert torch.allclose(partial, torch.full_like(partial, 2.0), atol=1e-3)
    assert torch.allclose(total, 3.0 * v + 2.0, atol=1e-2)
    assert not torch.allclose(total, partial)


def test_atoms_to_state_x_round_trips_the_crystal_normalisation():
    """`atoms_to_state_x` must invert Clari's state normalisation exactly.

    `Crystal` stores `x = cat([0.5 * lattice, coords]) / COORD_NORM` and reads
    back `lattice = 2*COORD_NORM*x[:3]`, `coords = COORD_NORM*x[3:]`. A UMA
    distillation target is built by relaxing the ASE view and writing it back
    through this function, so any factor-of-two slip here would silently train
    the student toward a half-size cell.
    """
    import numpy as np
    import torch
    from ase import Atoms

    from crystal_nft.meanflow.uma_relax import atoms_to_state_x

    cell = np.array([[9.0, 0.0, 0.0], [0.5, 10.0, 0.0], [0.3, 0.2, 11.0]])
    pos = np.array([[0.1, 0.2, 0.3], [3.0, 4.0, 5.0], [8.0, 9.0, 10.0]])
    atoms = Atoms(numbers=[6, 7, 8], positions=pos, cell=cell, pbc=True)

    x = atoms_to_state_x(atoms, coord_norm=8.0)
    assert x.shape == (6, 3) and x.dtype == torch.float32

    # Read back exactly as Crystal does.
    assert np.allclose((2 * 8.0 * x[:3]).numpy(), cell, atol=1e-5)
    assert np.allclose((8.0 * x[3:]).numpy(), pos, atol=1e-5)


def test_relax_atoms_uma_is_a_noop_for_zero_steps_and_preserves_indexing():
    """None entries must survive relaxation so caller indexing is preserved."""
    import numpy as np
    from ase import Atoms

    from crystal_nft.meanflow.uma_relax import relax_atoms_uma

    a = Atoms(numbers=[6], positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 8, pbc=True)
    out, stats = relax_atoms_uma(object(), [a, None, a], steps=0)
    assert len(out) == 3
    assert out[1] is None
    assert out[0] is not a           # copies, never aliases the input
    assert stats.n_in == 2 and stats.steps == 0


def test_lattice_scale_moves_the_cell_but_never_the_atoms():
    """The volume correction must leave Cartesian coords -- and so PB -- alone.

    Measured on 36 generated structures: UMA's own cell relaxation makes signed
    volume error *worse* (+4.16% -> +6.50%), because UMA's equilibrium cell is
    larger than the experimental CSD reference. A lattice-only rescale instead
    gives -0.46% while keeping clash at 0.000. That is only safe if atoms do not
    move with the cell, which is what this pins.
    """
    import numpy as np
    from ase import Atoms

    from crystal_nft.meanflow.uma_relax import relax_atoms_uma

    pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    cell = np.diag([10.0, 11.0, 12.0])
    atoms = Atoms(numbers=[6, 8], positions=pos, cell=cell, pbc=True)

    out, _ = relax_atoms_uma(object(), [atoms], steps=0, lattice_scale=0.985)
    assert np.allclose(out[0].get_cell()[:], cell * 0.985)
    assert np.allclose(out[0].get_positions(), pos)   # NOT scaled with the cell
    assert np.allclose(atoms.get_cell()[:], cell)     # input untouched
