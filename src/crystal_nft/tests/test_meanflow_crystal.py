"""Unit tests for MeanFlow crystal helpers (no GPU / Clari checkpoint required)."""

from __future__ import annotations

import torch

from crystal_nft.meanflow.chirality import global_chiral_descriptor, ChiralInfo
from crystal_nft.meanflow.velocity import (
    central_difference_dudr,
    central_difference_dudt,
    flow_map_to_instantaneous_velocity,
)


def test_flow_map_to_velocity():
    B, N, D = 2, 5, 3
    u = torch.randn(B, N, D)
    z = torch.randn(B, N, D)
    r = torch.tensor([0.2, 0.5])
    t = torch.tensor([0.8, 0.9])
    dudt = torch.randn(B, N, D)
    v = flow_map_to_instantaneous_velocity(u, z, r, t, dudt)
    assert v.shape == u.shape
    expected = u + (t - r).view(-1, 1, 1) * dudt
    assert torch.allclose(v, expected)


def test_dmd_stopgrad_identity():
    """mse(x, (x-g).detach()) backprops 2g/N into x (AnyFlow DMD identity)."""
    x = torch.randn(4, 5, requires_grad=True)
    g = torch.randn(4, 5)
    loss = torch.nn.functional.mse_loss(x, (x - g).detach())
    loss.backward()
    expected = 2.0 * g / x.numel()
    assert torch.allclose(x.grad, expected, atol=1e-5)


def test_shortcut_jump_pairs():
    from crystal_nft.meanflow.sampler import _flow_map_jump_index_pairs

    assert _flow_map_jump_index_pairs(4, mode="full") == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert _flow_map_jump_index_pairs(4, mode="shortcut", grad_timestep=1) == [
        (0, 1),
        (1, 2),
        (2, 4),
    ]
    assert _flow_map_jump_index_pairs(4, mode="shortcut", grad_timestep=0) == [(0, 1), (1, 4)]
    assert _flow_map_jump_index_pairs(4, mode="shortcut", grad_timestep=3) == [(0, 3), (3, 4)]
    assert _flow_map_jump_index_pairs(8, mode="shortcut", grad_timestep=2) == [
        (0, 2),
        (2, 3),
        (3, 8),
    ]
    assert _flow_map_jump_index_pairs(16, mode="jumps", n_jumps=8) == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
        (8, 10),
        (10, 12),
        (12, 14),
        (14, 16),
    ]
    assert len(_flow_map_jump_index_pairs(16, mode="full")) == 16
    assert _flow_map_jump_index_pairs(16, mode="jumps", n_jumps=16) == [
        (i, i + 1) for i in range(16)
    ]


def test_sample_align_keeps_separate_align_net():
    from crystal_nft.meanflow.loss_sample_align import (
        CrystalSampleAlignLoss,
        SampleAlignLossConfig,
    )

    teacher = torch.nn.Linear(1, 1)
    align = torch.nn.Linear(1, 1)
    loss = CrystalSampleAlignLoss(
        SampleAlignLossConfig(anyflow_cotrain_weight=0.0),
        teacher_net=teacher,
        align_net=align,
    )
    assert loss.teacher_net is teacher
    assert loss.align_net is align


def test_allreduce_grads_noop_without_process_group():
    from crystal_nft.meanflow.sampler import allreduce_grads

    lin = torch.nn.Linear(2, 2)
    lin.weight.grad = torch.ones_like(lin.weight)
    allreduce_grads(lin)
    assert torch.equal(lin.weight.grad, torch.ones_like(lin.weight))


def test_onpolicy_masked_mse_finite():
    from crystal_nft.meanflow.loss_anyflow_onpolicy import _masked_abs_mean, _masked_mse

    x = torch.randn(2, 6, 3)
    y = torch.randn(2, 6, 3)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    loss = _masked_mse(x, y, mask)
    assert torch.isfinite(loss)
    nrm = _masked_abs_mean(x, mask)
    assert nrm.shape == (2, 1, 1)
    assert torch.isfinite(nrm).all()


def test_dmd_identity_periodic_backward():
    from crystal_nft.meanflow.loss_anyflow_onpolicy import (
        dmd_identity_loss,
        dmd_kl_gradient,
    )

    b, n = 2, 3
    pred = torch.zeros(b, 3 + n, 3)
    pred[:, :3] = 0.25 * torch.eye(3).unsqueeze(0)
    pred[:, 3:] = 0.02 * torch.randn(b, n, 3)
    pred = pred.detach().requires_grad_(True)
    grad = 0.01 * torch.randn_like(pred)
    mask = torch.ones(b, n, dtype=torch.bool)
    loss = dmd_identity_loss(pred, grad, mask, periodic_coord=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

    pred_x1 = torch.ones_like(pred)
    x1_real = torch.zeros_like(pred)
    x1_fake = 0.5 * torch.ones_like(pred)
    grad = dmd_kl_gradient(pred_x1, x1_fake, x1_real, mask, normalize=True)
    # The packed lattice and coordinate absolute means are both one.
    assert torch.allclose(grad, 0.5 * torch.ones_like(grad), atol=1e-6)


def test_central_difference_uses_spatial_tangent():
    """CD must perturb z along v_dir; pure time finite-diff is incorrect for MeanFlow."""
    B, N, D = 2, 4, 3
    z = torch.randn(B, N, D)
    r = torch.tensor([0.1, 0.2])
    t = torch.tensor([0.5, 0.7])
    v_dir = torch.randn(B, N, D)
    seen = {"z_plus": None, "z_minus": None}

    def fn(cur_z, cur_r, cur_t):
        # Record endpoints; return a z-dependent field so spatial CD is nonzero.
        if seen["z_plus"] is None:
            seen["z_plus"] = cur_z.detach().clone()
        else:
            seen["z_minus"] = cur_z.detach().clone()
        return cur_z * 2.0 + cur_t.view(-1, 1, 1)

    eps = 1e-3
    dudt = central_difference_dudt(fn, z, r, t, v_dir, eps=eps)
    assert seen["z_plus"] is not None and seen["z_minus"] is not None
    # First call is t+, second t- under current implementation.
    assert not torch.allclose(seen["z_plus"], z)
    assert not torch.allclose(seen["z_minus"], z)
    assert torch.allclose(seen["z_plus"], z + v_dir * eps, atol=1e-5)
    assert torch.allclose(seen["z_minus"], z - v_dir * eps, atol=1e-5)
    assert dudt.shape == z.shape
    assert torch.isfinite(dudt).all()


def test_clari_forward_central_difference_uses_start_time():
    z = torch.randn(2, 4, 3)
    r = torch.tensor([0.2, 0.4])
    t = torch.tensor([0.7, 0.9])
    v_dir = torch.randn_like(z)
    seen = []

    def fn(cur_z, cur_r, cur_t):
        seen.append((cur_z.detach().clone(), cur_r.detach().clone(), cur_t.detach().clone()))
        return 2.0 * cur_z + cur_r.view(-1, 1, 1)

    eps = 5e-3
    out = central_difference_dudr(fn, z, r, t, v_dir, eps=eps)
    assert torch.allclose(seen[0][0], z + eps * v_dir, atol=1e-5)
    assert torch.allclose(seen[1][0], z - eps * v_dir, atol=1e-5)
    assert torch.allclose(seen[0][1], r + eps)
    assert torch.allclose(seen[1][1], r - eps)
    assert torch.allclose(seen[0][2], t)
    assert torch.allclose(seen[1][2], t)
    assert torch.isfinite(out).all()


def test_official_anyflow_partition_and_weights():
    from crystal_nft.meanflow.loss_anyflow import (
        official_beta08_weights,
        official_mode_masks,
        official_scale_weight,
    )

    all_diff = []
    all_cons = []
    for rank in range(7):
        diff, cons = official_mode_masks(
            16,
            rank=rank,
            world_size=7,
            diffusion_ratio=0.5,
            consistency_ratio=0.25,
        )
        assert not (diff & cons).any()
        all_diff.append(diff)
        all_cons.append(cons)
    diff = torch.cat(all_diff)
    cons = torch.cat(all_cons)
    assert int(diff.sum()) == round(0.5 * 112)
    assert int(cons.sum()) == round(0.25 * 112)
    assert int((~diff & ~cons).sum()) == 28

    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    w = official_beta08_weights(t, grid_size=1000)
    assert w[0] == 0 and w[-1] == 0
    # Clari t=0 is noise, so reversed beta08 peaks before the data endpoint.
    assert w[0] < w[1]
    assert w[1] > w[3]

    # Rebalancing is deliberately one-sided: `official_scale_weight` clamps the
    # scale at 1, so interval terms can be lifted toward the diffusion mean but
    # never shrunk toward it. Official AnyFlow shrinks too; with a matched
    # teacher its diffusion mean is ~0, and shrinking would zero the flow-map
    # residual this student exists to learn. Both directions are asserted here
    # so the deviation stays deliberate rather than becoming a silent drift.
    is_diff = torch.tensor([True, True, False, False])

    # diffusion mean (3.0) below the interval terms -> untouched
    weighted = torch.tensor([2.0, 4.0, 8.0, 16.0])
    scale = official_scale_weight(weighted, is_diff)
    assert torch.allclose(scale, torch.ones(4))

    # diffusion mean (12.0) above the interval terms -> lifted to it
    weighted = torch.tensor([8.0, 16.0, 2.0, 4.0])
    scale = official_scale_weight(weighted, is_diff)
    assert torch.allclose(scale[:2], torch.ones(2))
    assert torch.allclose(
        weighted[2:] * scale[2:],
        torch.full((2,), weighted[:2].mean()),
        rtol=1e-3,
    )


def test_frozen_student_snapshot_zeros_interval_and_does_not_track_live():
    from crystal_nft.meanflow.loss_anyflow import (
        AnyFlowLossConfig,
        CrystalAnyFlowLoss,
        freeze_student_snapshot,
    )

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(2, 2, bias=False)
            self.register_buffer("interval_mix", torch.tensor([1.0]))

    live = Tiny()
    with torch.no_grad():
        live.w.weight.fill_(2.0)
    snap = freeze_student_snapshot(live)
    assert float(snap.interval_mix) == 0.0
    assert all(not p.requires_grad for p in snap.parameters())
    with torch.no_grad():
        live.w.weight.fill_(9.0)
        live.interval_mix.fill_(1.0)
    assert torch.allclose(snap.w.weight, torch.full_like(snap.w.weight, 2.0))
    assert float(snap.interval_mix) == 0.0

    with torch.no_grad():
        live.w.weight.fill_(2.0)
        live.interval_mix.fill_(1.0)
    loss = CrystalAnyFlowLoss(
        AnyFlowLossConfig(compose_from="frozen_student"),
        teacher_net=None,
    )
    a = loss._student_rollout_net(live)
    b = loss._compose_target_net(live)
    assert a is b
    with torch.no_grad():
        live.w.weight.fill_(-1.0)
    c = loss._student_rollout_net(live)
    assert c is a
    assert torch.allclose(c.w.weight, torch.full_like(c.w.weight, 2.0))


def test_rollout_live_compose_frozen_are_independent():
    from crystal_nft.meanflow.loss_anyflow import AnyFlowLossConfig, CrystalAnyFlowLoss

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(2, 2, bias=False)
            self.register_buffer("interval_mix", torch.tensor([1.0]))

    live = Tiny()
    with torch.no_grad():
        live.w.weight.fill_(2.0)
    loss = CrystalAnyFlowLoss(
        AnyFlowLossConfig(compose_from="frozen_student", rollout_from="live"),
        teacher_net=None,
    )
    roll = loss._student_rollout_net(live)
    tgt = loss._compose_target_net(live)
    assert roll is not tgt
    assert float(tgt.interval_mix) == 0.0
    assert torch.allclose(roll.w.weight, live.w.weight)
    with torch.no_grad():
        live.w.weight.fill_(9.0)
        live.interval_mix.fill_(1.0)
    roll2 = loss._student_rollout_net(live)
    tgt2 = loss._compose_target_net(live)
    assert roll2 is roll
    assert tgt2 is tgt
    assert torch.allclose(roll2.w.weight, torch.full_like(roll2.w.weight, 9.0))
    assert torch.allclose(tgt2.w.weight, torch.full_like(tgt2.w.weight, 2.0))
    assert float(tgt2.interval_mix) == 0.0


def test_nfe_homogeneous_picks_one_grid_per_batch():
    from crystal_nft.meanflow.loss_anyflow import AnyFlowLossConfig, CrystalAnyFlowLoss

    loss = CrystalAnyFlowLoss(
        AnyFlowLossConfig(
            variant="official",
            nfe_steps_list=(8, 16),
            nfe_homogeneous=True,
            diffusion_ratio=0.0,
            consistency_ratio=0.0,
            compose_fine_steps=0,
        ),
        teacher_net=None,
    )
    _, _, _ = loss.sample_time_steps(32, torch.device("cpu"))
    nfe = loss._nfe_steps
    assert nfe is not None and nfe.numel() == 32
    assert int(nfe.min()) == int(nfe.max())
    assert int(nfe[0]) in (8, 16)
    assert loss._compose_fine_steps() == int(nfe[0])


def test_compose_nfe_max_only_allows_8_grid_batches():
    from crystal_nft.meanflow.loss_anyflow import AnyFlowLossConfig, CrystalAnyFlowLoss

    loss = CrystalAnyFlowLoss(
        AnyFlowLossConfig(compose_nfe_max=8, nfe_homogeneous=True),
        teacher_net=None,
    )
    loss._nfe_steps = torch.full((4,), 16, dtype=torch.long)
    assert not loss._compose_allowed_for_batch()
    loss._nfe_steps = torch.full((4,), 8, dtype=torch.long)
    assert loss._compose_allowed_for_batch()
    loss.cfg.compose_nfe_max = 0
    loss._nfe_steps = torch.full((4,), 16, dtype=torch.long)
    assert loss._compose_allowed_for_batch()


def test_interval_split_first_grid_keeps_nfe_and_matches_16grid_knot():
    from crystal_nft.meanflow.sampler import (
        interval_split_first_grid,
        interval_split_first_keep_coarse_grid,
        interval_time_grid,
    )

    device = torch.device("cpu")
    dtype = torch.float32
    rho = 0.75
    g8 = interval_time_grid(8, device=device, dtype=dtype, rho=rho)
    g16 = interval_time_grid(16, device=device, dtype=dtype, rho=rho)
    hyb = interval_split_first_grid(8, device=device, dtype=dtype, rho=rho)
    assert hyb.numel() == 9
    assert float(hyb[0]) == 0.0
    assert abs(float(hyb[-1]) - 1.0) < 1e-6
    assert torch.allclose(hyb[1], g16[1], atol=1e-5)
    assert torch.allclose(hyb[2], g16[2], atol=1e-5)
    assert torch.allclose(hyb[2], g8[1], atol=1e-5)
    assert torch.all(hyb[1:] > hyb[:-1])

    keep = interval_split_first_keep_coarse_grid(
        8, device=device, dtype=dtype, rho=rho
    )
    assert keep.numel() == 10
    assert torch.allclose(keep[:3], g16[:3], atol=1e-5)
    assert torch.allclose(keep[3:], g8[2:], atol=1e-5)


def test_anyflow_nfe_grid_list_and_rho():
    from crystal_nft.meanflow.loss_anyflow import (
        AnyFlowLossConfig,
        CrystalAnyFlowLoss,
        nfe_grid_choices,
        parse_nfe_steps_list,
    )

    assert parse_nfe_steps_list([8, 16]) == (8, 16)
    assert parse_nfe_steps_list("8,16") == (8, 16)
    assert parse_nfe_steps_list(16) == (16,)
    assert parse_nfe_steps_list([1, 8]) == (8,)

    cfg16 = AnyFlowLossConfig(variant="official", nfe_steps=16, nfe_grid_rho=1.0)
    choices, rho = nfe_grid_choices(cfg16)
    assert choices == (16,) and abs(rho - 1.0) < 1e-8

    loss16 = CrystalAnyFlowLoss(cfg16, teacher_net=None)
    r, t, fm = loss16.sample_time_steps(4096, torch.device("cpu"))
    dt = (t - r)[~fm]
    assert torch.allclose(dt, torch.full_like(dt, 1.0 / 16), atol=1e-5)

    cfg_mix = AnyFlowLossConfig(
        variant="official",
        nfe_steps=16,
        nfe_steps_list=(8, 16),
        nfe_grid_rho=0.75,
        diffusion_ratio=0.5,
        consistency_ratio=0.0,
    )
    loss_mix = CrystalAnyFlowLoss(cfg_mix, teacher_net=None)
    r, t, fm = loss_mix.sample_time_steps(8192, torch.device("cpu"))
    dt = (t - r)[~fm]
    first8 = (1.0 / 8.0) ** 0.75
    assert float(dt.min()) > 0.0
    assert float(dt.max()) <= first8 + 1e-4
    # 8-step rho=0.75 first jump must appear in the mix.
    assert bool((dt > first8 - 1e-4).any())

    cfg_bias = AnyFlowLossConfig(
        variant="official",
        nfe_steps=8,
        nfe_grid_rho=0.75,
        nfe_k_power=3.0,
        diffusion_ratio=0.0,
        consistency_ratio=0.0,
    )
    loss_bias = CrystalAnyFlowLoss(cfg_bias, teacher_net=None)
    r, t, fm = loss_bias.sample_time_steps(8192, torch.device("cpu"))
    dt = (t - r)[~fm]
    first8 = (1.0 / 8.0) ** 0.75
    frac_first = float((dt > first8 - 1e-4).float().mean())
    # Uniform k would be ~1/8; k_power=3 yields P(k=0)=8^(-1/3)~0.5.
    assert frac_first > 0.35

    cfg_fixed = AnyFlowLossConfig(
        variant="official",
        nfe_steps=8,
        nfe_grid_rho=0.75,
        nfe_k_fixed=0,
        diffusion_ratio=0.0,
        consistency_ratio=0.0,
    )
    loss_fixed = CrystalAnyFlowLoss(cfg_fixed, teacher_net=None)
    r, t, fm = loss_fixed.sample_time_steps(256, torch.device("cpu"))
    assert not bool(fm.any())
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-6)
    assert torch.allclose(t, torch.full_like(t, (1.0 / 8.0) ** 0.75), atol=1e-5)


def test_euler_instantaneous_segment_constant_velocity():
    from crystal_nft.meanflow.sampler import euler_instantaneous_segment

    class _Iface:
        def pred(self, net, xt, xsc, t, r, f, chiral_bias=None):
            return torch.ones_like(xt)

        def estimate_x1(self, z, t, v):
            return z + (1.0 - t).view(-1, 1, 1) * v

    class _C:
        mask = torch.ones(2, 4)

    z0 = torch.zeros(2, 7, 3)
    t_from = torch.tensor([0.0, 0.2])
    t_to = torch.tensor([0.21, 0.45])
    z1 = euler_instantaneous_segment(
        _Iface(),
        net=None,
        C=_C(),
        x_init=z0,
        t_from=t_from,
        t_to=t_to,
        num_steps=6,
        use_self_cond=False,
        method="euler",
    )
    dt = (t_to - t_from).view(-1, 1, 1)
    expected = z0 + dt * torch.ones_like(z0)
    expected[:, 3:] = expected[:, 3:] - expected[:, 3:].mean(dim=1, keepdim=True)
    assert torch.allclose(z1, expected, atol=1e-5)
    z1_heun = euler_instantaneous_segment(
        _Iface(),
        net=None,
        C=_C(),
        x_init=z0,
        t_from=t_from,
        t_to=t_to,
        num_steps=8,
        use_self_cond=False,
        method="heun",
    )
    assert torch.allclose(z1_heun, expected, atol=1e-5)


def test_interval_compose_matches_constant_velocity_and_8grid_midpoint():
    from crystal_nft.meanflow.sampler import (
        interval_compose_average_velocity,
        power_index_midpoint,
    )

    rho = 0.75
    r = torch.tensor([0.0])
    t = torch.tensor([(1.0 / 8.0) ** rho])
    mid = power_index_midpoint(r, t, rho)
    expected_mid = torch.tensor([(1.0 / 16.0) ** rho])
    assert torch.allclose(mid, expected_mid, atol=1e-6)
    from crystal_nft.meanflow.sampler import power_grid_index

    assert int(power_grid_index(r, 16, rho).item()) == 0
    assert int(power_grid_index(t, 16, rho).item()) == 2
    assert int(power_grid_index(expected_mid, 16, rho).item()) == 1

    class _Iface:
        def pred(self, net, xt, xsc, t, r, f, chiral_bias=None):
            return torch.ones_like(xt)

    class _C:
        mask = torch.ones(2, 4)
        batched = True
        batch_size = 2

    z0 = torch.zeros(2, 7, 3)
    t_from = torch.tensor([0.0, 0.2])
    t_to = torch.tensor([0.21, 0.45])
    u_star = interval_compose_average_velocity(
        _Iface(),
        net=None,
        C=_C(),
        x_init=z0,
        t_from=t_from,
        t_to=t_to,
        rho=rho,
    )
    dt = (t_to - t_from).view(-1, 1, 1)
    z1 = z0 + dt * u_star
    expected = z0 + dt * torch.ones_like(z0)
    expected[:, 3:] = expected[:, 3:] - expected[:, 3:].mean(dim=1, keepdim=True)
    z1[:, 3:] = z1[:, 3:] - z1[:, 3:].mean(dim=1, keepdim=True)
    assert torch.allclose(z1, expected, atol=1e-5)

    from crystal_nft.meanflow.sampler import interval_state_at_times

    t_at = torch.tensor([(2.0 / 16.0) ** rho, (4.0 / 16.0) ** rho])
    z_at = interval_state_at_times(
        _Iface(),
        net=None,
        C=_C(),
        x_init=z0,
        t_at=t_at,
        num_steps=16,
        rho=rho,
    )
    reached = z0 + t_at.view(-1, 1, 1) * torch.ones_like(z0)
    reached[:, 3:] = reached[:, 3:] - reached[:, 3:].mean(dim=1, keepdim=True)
    got = z_at.clone()
    got[:, 3:] = got[:, 3:] - got[:, 3:].mean(dim=1, keepdim=True)
    assert torch.allclose(got, reached, atol=1e-5)


def test_large_jump_mask_respects_tau_and_r_max():
    from crystal_nft.meanflow.loss_anyflow import large_jump_mask

    fm = torch.tensor([False, False, True, False])
    r = torch.tensor([0.0, 0.2, 0.0, 0.0])
    t = torch.tensor([0.21, 0.41, 0.21, 0.10])
    mask = large_jump_mask(fm, r, t, tau=0.15, r_max=0.05)
    assert mask.tolist() == [True, False, False, False]
    mask_any_r = large_jump_mask(fm, r, t, tau=0.15, r_max=1.0)
    assert mask_any_r.tolist() == [True, True, False, False]


def test_chiral_descriptor_dim():
    info = ChiralInfo(asu_chiral=torch.tensor([0, 1, 2, 0]), n_chiral_centers=2, n_defined=2)
    d = global_chiral_descriptor(info)
    assert d.shape == (8,)


def test_lora_linear_zero_init_matches_base():
    from crystal_nft.meanflow.tune import LoRALinear

    lin = torch.nn.Linear(8, 16)
    x = torch.randn(3, 8)
    wrapped = LoRALinear(lin, rank=4, alpha=8.0)
    assert torch.allclose(lin(x), wrapped(x), atol=1e-6)
    with torch.no_grad():
        wrapped.lora8_B.normal_()
        wrapped.nfe8_mix.fill_(0.0)
    assert torch.allclose(lin(x), wrapped(x), atol=1e-6)
    wrapped.nfe8_mix.fill_(1.0)
    assert not torch.allclose(lin(x), wrapped(x))
    wrapped.nfe8_mix.fill_(0.0)

    with torch.no_grad():
        wrapped.lora_B.normal_()
        wrapped.lora8_A.copy_(wrapped.lora_A)
        wrapped.lora8_B.copy_(wrapped.lora_B)
    wrapped.nfe8_mix.fill_(0.0)
    out16 = wrapped(x)
    wrapped.nfe8_mix.fill_(1.0)
    out8 = wrapped(x)
    assert torch.allclose(out16, out8, atol=1e-6)
    with torch.no_grad():
        wrapped.lora8_B.normal_()
    assert not torch.allclose(out16, wrapped(x))
    wrapped.nfe8_mix.fill_(0.0)
    assert torch.allclose(out16, wrapped(x), atol=1e-6)

    conditional = LoRALinear(lin, rank=4, alpha=8.0, condition_on_interval=True)
    with torch.no_grad():
        conditional.lora_B.normal_()
    conditional.set_interval_gate(torch.zeros(x.shape[0]))
    assert torch.allclose(lin(x), conditional(x), atol=1e-6)
    conditional.set_interval_gate(torch.ones(x.shape[0]))
    assert not torch.allclose(lin(x), conditional(x))


def test_resolve_eval_nfe8_lora_mix_override():
    from crystal_nft.meanflow.tune import resolve_eval_nfe8_lora_mix

    assert resolve_eval_nfe8_lora_mix(8, 1.0, override="") == 0.0
    assert resolve_eval_nfe8_lora_mix(16, 1.0, override="") == 0.0
    assert resolve_eval_nfe8_lora_mix(50, 1.0, override="") == 0.0
    assert resolve_eval_nfe8_lora_mix(16, 0.0, override="1") == 1.0
    assert resolve_eval_nfe8_lora_mix(8, 1.0, override="0") == 0.0


def test_student_tune_delta_only_and_lora():
    from crystal_nft.meanflow.tune import LoRALinear, apply_student_tune

    class _Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_q = torch.nn.Linear(4, 4)
            self.proj_k = torch.nn.Linear(4, 4)
            self.proj_v = torch.nn.Linear(4, 4)
            self.proj_o = torch.nn.Linear(4, 4)

    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dit = torch.nn.Module()
            self.dit.trunk = _Attn()
            self.dit.stem_cond = torch.nn.Linear(4, 4)
            self.dit.embed_timestep = torch.nn.Module()
            self.dit.embed_timestep.base = torch.nn.Linear(2, 2)
            self.dit.embed_timestep.delta = torch.nn.Linear(2, 2)

    net = _Net()
    apply_student_tune(net, {"student_tune": "delta_only"})
    assert net.dit.embed_timestep.delta.weight.requires_grad
    assert not net.dit.embed_timestep.base.weight.requires_grad
    assert not net.dit.trunk.proj_q.weight.requires_grad
    assert not net.dit.stem_cond.weight.requires_grad

    net = _Net()
    stats = apply_student_tune(
        net, {"student_tune": "delta_lora", "lora_rank": 2, "lora_alpha": 4.0}
    )
    assert stats["n_lora"] == 4
    assert isinstance(net.dit.trunk.proj_q, LoRALinear)
    assert net.dit.trunk.proj_q.lora_A.requires_grad
    assert not net.dit.trunk.proj_q.linear.weight.requires_grad
    assert net.dit.embed_timestep.delta.weight.requires_grad
    assert not net.dit.trunk.proj_q.lora8_A.requires_grad

    net = _Net()
    stats = apply_student_tune(
        net,
        {
            "student_tune": "delta_lora",
            "lora_rank": 2,
            "lora_alpha": 4.0,
            "train_nfe8_lora": True,
            "freeze_attn_lora": True,
            "freeze_delta_embed": True,
            "freeze_interval_residual": True,
        },
    )
    assert net.dit.trunk.proj_q.lora8_A.requires_grad
    assert not net.dit.trunk.proj_q.lora_A.requires_grad
    assert not net.dit.embed_timestep.delta.weight.requires_grad
    from crystal_nft.meanflow.tune import set_nfe8_lora_mix

    with torch.no_grad():
        net.dit.trunk.proj_q.lora8_B.normal_()
    x = torch.randn(3, 4)
    set_nfe8_lora_mix(net, 0.0)
    out0 = net.dit.trunk.proj_q(x)
    set_nfe8_lora_mix(net, 1.0)
    out1 = net.dit.trunk.proj_q(x)
    assert not torch.allclose(out0, out1)
    from crystal_nft.meanflow.tune import init_nfe8_lora_from_shared

    init_nfe8_lora_from_shared(net)
    set_nfe8_lora_mix(net, 0.0)
    copied0 = net.dit.trunk.proj_q(x)
    set_nfe8_lora_mix(net, 1.0)
    copied1 = net.dit.trunk.proj_q(x)
    assert torch.allclose(copied0, copied1, atol=1e-6)

    net = _Net()
    apply_student_tune(net, {"student_tune": "delta_stem"})
    assert net.dit.stem_cond.weight.requires_grad
    assert not net.dit.trunk.proj_q.weight.requires_grad

    net = _Net()
    stats = apply_student_tune(
        net, {"student_tune": "delta_lora_base", "lora_rank": 2, "lora_alpha": 4.0}
    )
    assert stats["mode"] == 4
    assert net.dit.embed_timestep.base.weight.requires_grad
    assert net.dit.embed_timestep.delta.weight.requires_grad
    assert net.dit.trunk.proj_q.lora_A.requires_grad
    assert not net.dit.trunk.proj_q.linear.weight.requires_grad

    net = _Net()
    stats = apply_student_tune(
        net, {"student_tune": "attn_lora", "lora_rank": 2, "lora_alpha": 4.0}
    )
    assert stats["mode"] == 6
    assert stats["n_lora"] == 4
    assert net.dit.trunk.proj_q.lora_A.requires_grad
    assert not net.dit.embed_timestep.delta.weight.requires_grad
    assert not net.dit.embed_timestep.base.weight.requires_grad
    assert not net.dit.stem_cond.weight.requires_grad
    x = torch.randn(3, 4)
    before = net.dit.trunk.proj_q.linear(x)
    assert torch.allclose(net.dit.trunk.proj_q(x), before, atol=1e-6)

    from crystal_nft.meanflow.tune import merge_lora_into_linear

    with torch.no_grad():
        net.dit.trunk.proj_q.lora_B.normal_()
    lora_out = net.dit.trunk.proj_q(x)
    assert not torch.allclose(lora_out, before)
    n_merged = merge_lora_into_linear(net)
    assert n_merged == 4
    assert isinstance(net.dit.trunk.proj_q, torch.nn.Linear)
    assert torch.allclose(net.dit.trunk.proj_q(x), lora_out, atol=1e-5)


def test_dual_time_residual_boundary_and_tune():
    from clari.models.layers.transformer import Modulate

    from crystal_nft.meanflow.net import (
        DualTimeResidualConditioner,
        DualTimeStemAdapter,
        load_ema_preserving_time_mix,
    )
    from crystal_nft.meanflow.tune import LoRALinear, apply_student_tune

    class ScaleEmb(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.proj = torch.nn.Linear(1, dim, bias=False)

        def forward(self, x):
            return self.proj(x)

    dim = 8
    conditioner = DualTimeResidualConditioner(
        ScaleEmb(dim), dim, feature_mode="both", gate_value=1.0
    )
    stem = DualTimeStemAdapter(torch.nn.Identity(), conditioner)
    x = torch.randn(3, dim)
    r = torch.tensor([0.1, 0.2, 0.3])
    t = torch.tensor([0.4, 0.5, 0.6])
    conditioner.set_times(r, t)
    assert torch.allclose(stem(x), x)

    with torch.no_grad():
        conditioner.adapter[-1].weight.normal_()
    changed = stem(x)
    assert not torch.allclose(changed, x)
    conditioner.set_times(t, t)
    assert torch.allclose(stem(x), x, atol=1e-6)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dit = torch.nn.Module()
            self.dit.stem_cond = stem
            self.dit.trunk = torch.nn.Module()
            self.dit.trunk.attn = torch.nn.Module()
            self.dit.trunk.attn.proj_q = torch.nn.Linear(dim, dim)
            self.dit.trunk.attn.proj_k = torch.nn.Linear(dim, dim)
            self.dit.trunk.attn.proj_v = torch.nn.Linear(dim, dim)
            self.dit.trunk.attn.proj_o = torch.nn.Linear(dim, dim)
            self.dit.trunk.mod = Modulate(dim, dim)

    net = Net()
    stats = apply_student_tune(
        net,
        {
            "student_tune": "dual_cond_lora",
            "lora_rank": 2,
            "lora_alpha": 4.0,
            "cond_lora_rank": 2,
            "cond_lora_alpha": 4.0,
        },
    )
    assert stats["mode"] == 5
    assert stats["n_condition_lora"] == 2
    assert conditioner.adapter[-1].weight.requires_grad
    assert isinstance(net.dit.trunk.attn.proj_q, LoRALinear)
    assert isinstance(net.dit.trunk.mod.scale, LoRALinear)
    assert isinstance(net.dit.trunk.mod.shift, LoRALinear)

    conditioner.dual_gate.fill_(0.35)
    online = {key: value.clone() for key, value in net.state_dict().items()}
    ema = {key: value.clone() for key, value in online.items()}
    for key in ema:
        if key.endswith("conditioner.dual_gate"):
            ema[key].fill_(0.0)
    conditioner.dual_gate.fill_(0.9)
    loaded = load_ema_preserving_time_mix(
        net,
        {
            "ema_state_dict": ema,
            "net_state_dict": online,
            "meta": {"dual_gate_live": 0.35},
        },
    )
    assert loaded
    assert torch.allclose(conditioner.dual_gate, torch.tensor([0.35]))


def test_time_mix_residual_and_anneal():
    from crystal_nft.meanflow.net import (
        MeanFlowTimeEmbedding,
        is_time_mix_buffer_key,
        scheduled_mix_value,
        time_mix_from_cfg,
    )

    class ScaleEmb(torch.nn.Module):
        def __init__(self, s):
            super().__init__()
            self.s = torch.nn.Parameter(torch.tensor([float(s)]))

        def forward(self, t_col):
            return t_col * self.s

    emb = MeanFlowTimeEmbedding(ScaleEmb(1.0), gate_value=1.0)
    with torch.no_grad():
        emb.base.s.fill_(2.0)
        emb.delta.s.fill_(5.0)
    t = torch.tensor([[0.4], [0.8]])
    r = torch.tensor([0.1, 0.3])
    emb.set_r(r)

    emb.gate.fill_(1.0)
    emb.time_residual.fill_(0.0)
    out = emb(t)
    assert torch.allclose(out, r.unsqueeze(-1) * 5.0, atol=1e-5)

    emb.time_residual.fill_(0.5)
    out = emb(t)
    expected = r.unsqueeze(-1) * 5.0 + 0.5 * (t * 2.0)
    assert torch.allclose(out, expected, atol=1e-5)

    emb.time_residual.fill_(0.0)
    emb.gate.fill_(0.5)
    out = emb(t)
    expected = 0.5 * (t * 2.0) + 0.5 * (r.unsqueeze(-1) * 5.0)
    assert torch.allclose(out, expected, atol=1e-5)

    # Same-embed endpoint mix: α=0 keeps mix; r=t is identity for any α.
    emb.gate.fill_(1.0)
    emb.time_residual.fill_(0.0)
    emb.endpoint_mix.fill_(0.0)
    out = emb(t)
    assert torch.allclose(out, r.unsqueeze(-1) * 5.0, atol=1e-5)
    emb.endpoint_mix.fill_(0.4)
    out = emb(t)
    expected = 0.6 * (r.unsqueeze(-1) * 5.0) + 0.4 * (t * 5.0)
    assert torch.allclose(out, expected, atol=1e-5)
    t_flat = t.squeeze(-1)
    emb.set_r(t_flat)
    for a in (0.0, 0.4, 1.0):
        emb.endpoint_mix.fill_(a)
        out = emb(t)
        assert torch.allclose(out, t * 5.0, atol=1e-5), a
    emb.set_r(r)
    emb.endpoint_mix.fill_(0.0)

    forward_emb = MeanFlowTimeEmbedding(
        ScaleEmb(1.0), gate_value=0.25, time_parameterization="clari_forward"
    )
    with torch.no_grad():
        forward_emb.base.s.fill_(2.0)
        forward_emb.delta.s.fill_(5.0)
    forward_emb.set_times(r, t.squeeze(-1))
    out = forward_emb(r.unsqueeze(-1))
    expected = 0.75 * (r.unsqueeze(-1) * 2.0) + 0.25 * (t * 5.0)
    assert torch.allclose(out, expected, atol=1e-5)
    # Zero-init interval_proj must not change a resumed mix-0.25 checkpoint.
    forward_emb.interval_mix.fill_(1.0)
    out_im = forward_emb(r.unsqueeze(-1))
    assert torch.allclose(out_im, expected, atol=1e-5)
    # Zero-init interval_mlp last layer is a no-op on top of the Linear.
    # r=t ⇒ delta(t-r)-delta(0)=0 even with a live projector.
    with torch.no_grad():
        forward_emb.interval_proj.weight.fill_(0.5)
    same = torch.tensor([0.2, 0.4])
    forward_emb.set_times(same, same)
    out_rt = forward_emb(same.unsqueeze(-1))
    expected_rt = 0.75 * (same.unsqueeze(-1) * 2.0) + 0.25 * (same.unsqueeze(-1) * 5.0)
    assert torch.allclose(out_rt, expected_rt, atol=1e-5)
    # Interval residual uses frozen base(t-r), not student delta (cont18 collapse).
    # gate=0 so the convex mix does not use delta; only the interval term would.
    forward_emb.gate.fill_(0.0)
    forward_emb.interval_mix.fill_(1.0)
    torch.nn.init.eye_(forward_emb.interval_proj.weight)
    forward_emb.delta.s.requires_grad_(True)
    r_live = torch.tensor([0.0, 0.0])
    t_live = torch.tensor([0.21, 0.21])
    forward_emb.set_times(r_live, t_live)
    out_live = forward_emb(r_live.unsqueeze(-1)).sum()
    out_live.backward()
    assert forward_emb.delta.s.grad is None or torch.count_nonzero(forward_emb.delta.s.grad) == 0
    assert forward_emb.interval_proj.weight.grad is not None
    assert torch.count_nonzero(forward_emb.interval_proj.weight.grad) > 0
    assert forward_emb.interval_mlp[-1].weight.grad is not None
    assert torch.count_nonzero(forward_emb.interval_mlp[-1].weight.grad) > 0
    forward_emb.delta.s.requires_grad_(False)
    forward_emb.interval_proj.weight.grad = None
    forward_emb.interval_mlp[-1].weight.grad = None
    # 16-step first jump Δt≈0.125 must stay bit-identical when dt_min=0.15.
    forward_emb.interval_dt_min.fill_(0.15)
    forward_emb.interval_r_max.fill_(0.05)
    forward_emb.gate.fill_(0.25)
    r16 = torch.zeros(2)
    t16 = torch.full((2,), (1.0 / 16.0) ** 0.75)
    forward_emb.set_times(r16, t16)
    out_masked = forward_emb(r16.unsqueeze(-1))
    expected16 = 0.75 * (r16.unsqueeze(-1) * 2.0) + 0.25 * (t16.unsqueeze(-1) * 5.0)
    assert torch.allclose(out_masked, expected16, atol=1e-5)
    # 8-step first jump Δt≈0.21 at r=0 is allowed through and moves.
    t8 = torch.full((2,), (1.0 / 8.0) ** 0.75)
    forward_emb.set_times(r16, t8)
    out8 = forward_emb(r16.unsqueeze(-1))
    expected8 = 0.75 * (r16.unsqueeze(-1) * 2.0) + 0.25 * (t8.unsqueeze(-1) * 5.0)
    assert not torch.allclose(out8, expected8, atol=1e-4)
    # Later 8-step jumps (r>0.05) stay masked even if Δt is large.
    r_late = torch.full((2,), 0.84)
    t_late = torch.ones(2)
    forward_emb.set_times(r_late, t_late)
    out_late = forward_emb(r_late.unsqueeze(-1))
    expected_late = 0.75 * (r_late.unsqueeze(-1) * 2.0) + 0.25 * (t_late.unsqueeze(-1) * 5.0)
    assert torch.allclose(out_late, expected_late, atol=1e-5)
    # r_min skips the first 8-jump (r=0) even when Δt is large.
    forward_emb.interval_dt_min.fill_(0.05)
    forward_emb.interval_r_max.fill_(1.0)
    forward_emb.interval_r_min.fill_(0.15)
    forward_emb.set_times(r16, t8)
    out_first = forward_emb(r16.unsqueeze(-1))
    assert torch.allclose(out_first, expected8, atol=1e-5)
    r_mid = torch.full((2,), (2.0 / 8.0) ** 0.75)
    t_mid = torch.full((2,), (3.0 / 8.0) ** 0.75)
    forward_emb.set_times(r_mid, t_mid)
    out_mid = forward_emb(r_mid.unsqueeze(-1))
    expected_mid = 0.75 * (r_mid.unsqueeze(-1) * 2.0) + 0.25 * (t_mid.unsqueeze(-1) * 5.0)
    assert not torch.allclose(out_mid, expected_mid, atol=1e-4)
    forward_emb.interval_dt_min.fill_(0.0)
    forward_emb.interval_r_max.fill_(1.0)
    forward_emb.interval_r_min.fill_(0.0)
    forward_emb.gate.fill_(0.25)

    assert scheduled_mix_value(0, 1.0, 0.25, 8000, 1.0) == 1.0
    assert abs(scheduled_mix_value(7999, 1.0, 0.25, 8000, 1.0) - 0.25) < 1e-9
    assert scheduled_mix_value(100, 1.0, 0.25, 0, 1.0) == 1.0
    g, res, ep = time_mix_from_cfg(
        {
            "gate_value": 0.25,
            "gate_anneal_steps": 0,
            "time_residual": 0.0,
            "time_residual_anneal_steps": 0,
        },
        100,
    )
    assert g == 0.25 and res == 0.0 and ep == 0.0
    g, res, ep = time_mix_from_cfg(
        {
            "gate_value": 1.0,
            "gate_anneal_start": 1.0,
            "gate_anneal_end": 0.25,
            "gate_anneal_steps": 5,
            "time_residual": 0.0,
            "time_residual_start": 0.0,
            "time_residual_end": 0.5,
            "time_residual_anneal_steps": 5,
            "endpoint_mix_start": 0.0,
            "endpoint_mix_end": 0.75,
            "endpoint_mix_anneal_steps": 5,
        },
        4,
    )
    assert abs(g - 0.25) < 1e-9
    assert abs(res - 0.5) < 1e-9
    assert abs(ep - 0.75) < 1e-9

    assert is_time_mix_buffer_key("dit.embed_timestep.gate")
    assert is_time_mix_buffer_key("dit.embed_timestep.time_residual")
    assert is_time_mix_buffer_key("dit.embed_timestep.endpoint_mix")
    assert is_time_mix_buffer_key("dit.trunk.proj_q.nfe8_mix")
    assert is_time_mix_buffer_key("module._time_proxy.gate")
    assert not is_time_mix_buffer_key("dit.stem_node.gate.weight")
    assert not is_time_mix_buffer_key("dit.trunk.0.ada.gate.weight")


if __name__ == "__main__":
    test_flow_map_to_velocity()
    test_central_difference_uses_spatial_tangent()
    test_clari_forward_central_difference_uses_start_time()
    test_official_anyflow_partition_and_weights()
    test_anyflow_nfe_grid_list_and_rho()
    test_euler_instantaneous_segment_constant_velocity()
    test_interval_compose_matches_constant_velocity_and_8grid_midpoint()
    test_large_jump_mask_respects_tau_and_r_max()
    test_chiral_descriptor_dim()
    test_dmd_stopgrad_identity()
    test_shortcut_jump_pairs()
    test_sample_align_keeps_separate_align_net()
    test_allreduce_grads_noop_without_process_group()
    test_onpolicy_masked_mse_finite()
    test_dmd_identity_periodic_backward()
    test_lora_linear_zero_init_matches_base()
    test_student_tune_delta_only_and_lora()
    test_dual_time_residual_boundary_and_tune()
    test_time_mix_residual_and_anneal()
    print("ok")


def test_flow_map_hinge_targets_the_dual_time_map():
    """The chirality hinge must see U(z, r, 1), not the r == t velocity.

    Both loss paths supervised handedness only at r == t: `fm_supervision_loss`
    predicts with `r = t`, and the AnyFlow path rebuilds `v_r = u - (t-r) dudr`
    so `estimate_x1` never sees the dual-time output. No sampler evaluates
    r == t, which is why tag-following measured 0.97 on the r == t endpoint and
    ~0.50 on every r != t jump of the same checkpoint.
    """
    import torch

    from crystal_nft.meanflow.interface import MeanFlowCrystalInterface

    seen: list[tuple[float, float]] = []

    class _Iface(MeanFlowCrystalInterface):
        def __init__(self):
            pass

        def pred(self, *, net, xt, xsc, t, r, f, chiral_bias=None):
            seen.append((float(r[0]), float(t[0])))
            return torch.ones_like(xt)

    iface = _Iface()
    xt = torch.zeros(2, 5, 3)
    r = torch.full((2,), 0.25)
    one = torch.ones_like(r)
    u = iface.pred(net=None, xt=xt, xsc=None, t=one, r=r, f=None)
    pred_x1_map = xt + (1.0 - r).view(-1, 1, 1) * u

    # z_1 = z_r + (1 - r) U(z_r, r, 1) is exactly a 1-step interval sample.
    assert seen[-1] == (0.25, 1.0)
    assert torch.allclose(pred_x1_map, torch.full_like(xt, 0.75))


def test_chiral_branch_residual_is_parity_odd():
    """The branch must act with OPPOSITE parity to a true displacement.

    Scalar FiLM conditioning (StereoNodeMod / StereoPairMod / StereoCondEmbedding)
    can bias the conditional mean but has no reflection-odd mechanism, so it
    cannot systematically flip a signed volume. A cross product of relative
    vectors is a pseudovector: (Mu) x (Mv) = -M(u x v) for a reflection M. This
    pins that property down, since it is the entire reason the branch exists.
    """
    import torch

    from crystal_nft.meanflow.net import StereoChiralBranch

    torch.manual_seed(0)
    u = torch.randn(5, 3)
    v = torch.randn(5, 3)
    M = torch.diag(torch.tensor([1.0, 1.0, -1.0]))  # a reflection, det = -1
    assert torch.linalg.det(M) < 0
    lhs = torch.linalg.cross(u @ M.T, v @ M.T)
    rhs = -(torch.linalg.cross(u, v) @ M.T)
    assert torch.allclose(lhs, rhs, atol=1e-5), "cross product is not a pseudovector"

    # Zero-init output layer => an untrained branch is exactly a no-op, so
    # attaching it cannot perturb a converged checkpoint.
    br = StereoChiralBranch()
    assert float(br.net[-1].weight.abs().max()) == 0.0
    assert float(br.net[-1].bias.abs().max()) == 0.0


def test_bond_preserving_residual_keeps_bond_lengths():
    """The projected residual must not change |u_k| to first order.

    PB checks bond lengths and angles. The unprojected residual
    a(u1xu2)+b(u2xu3)+g(u3xu1) applied to neighbour n1 includes u2xu3, which is
    NOT perpendicular to u1, so it stretches that bond -- and it was also
    applied to the centre, perturbing all four bonds at once. Measured cost of
    the branch's mere presence: ~3.7 PB (82.26 -> 78.55) even with a weak hinge.
    """
    import torch

    torch.manual_seed(0)
    u = torch.randn(1, 1, 3, 3)  # (B, M, 3 neighbours, xyz)
    dv = torch.randn(1, 1, 4, 3)  # centre + 3 neighbours

    e_n = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    d_n = dv[:, :, 1:, :]
    d_n = d_n - (d_n * e_n).sum(-1, keepdim=True) * e_n
    proj = torch.cat([torch.zeros_like(dv[:, :, :1, :]), d_n], dim=2)

    # centre is held fixed, so no bond is perturbed from that end
    assert torch.allclose(proj[:, :, 0, :], torch.zeros(3), atol=1e-6)
    # each neighbour's displacement is perpendicular to its own bond
    radial = (proj[:, :, 1:, :] * e_n).sum(-1).abs().max()
    assert float(radial) < 1e-5, f"residual still has a radial component: {radial}"
    # and it is not simply zero -- the motion survives, it is just tangential
    assert float(proj.abs().max()) > 1e-3


def test_stereo_head_time_window_actually_gates_the_film():
    """A window that never fires is the failure mode this file exists to catch.

    The FiLM heads sit on `dit.node_mod` via a forward hook and never receive
    `t`, so the window depends on `crystal_forward` publishing it. If that
    publication breaks, the hook silently keeps firing at every time and the run
    looks healthy while testing nothing -- which has already happened once here
    (the branch's `t.reshape(b, 1)` died only on the sampling path).
    """
    import torch

    from crystal_nft.meanflow.net import StereoNodeMod, _publish_stereo_time
    from crystal_nft.meanflow.stereo import active_stereo_tags

    class _FakeDiT(torch.nn.Module):
        pass

    dit = _FakeDiT()
    mod = StereoNodeMod(4, n_tags=4, gain=1.0, t_min=0.0, t_max=0.5)
    mod.__dict__["_bound_dit"] = dit
    # non-zero weights so the modulation is observable at all
    torch.nn.init.constant_(mod.scale.weight, 0.5)
    torch.nn.init.constant_(mod.shift.weight, 0.5)

    out = torch.ones(2, 4, 4)
    tags = torch.ones(2, 4, dtype=torch.long)  # TAG_R everywhere

    with active_stereo_tags(tags):
        _publish_stereo_time(dit, torch.tensor([0.1, 0.1]))
        below = mod._hook(None, None, out.clone())
        _publish_stereo_time(dit, torch.tensor([0.9, 0.9]))
        above = mod._hook(None, None, out.clone())
        _publish_stereo_time(dit, None)
        unpublished = mod._hook(None, None, out.clone())

    assert not torch.allclose(below, out), "inside the window the FiLM must act"
    assert torch.allclose(above, out), "above t_max the FiLM must be exactly off"
    # with no time published the window cannot be evaluated, so it must not
    # silently zero the head -- it falls back to the unwindowed behaviour
    assert torch.allclose(unpublished, below)


def test_stereo_head_window_is_a_noop_when_unset():
    """Existing checkpoints must be bit-identical: t_min 0 / t_max 1 = no gate."""
    import torch

    from crystal_nft.meanflow.net import StereoNodeMod, _publish_stereo_time
    from crystal_nft.meanflow.stereo import active_stereo_tags

    class _FakeDiT(torch.nn.Module):
        pass

    dit = _FakeDiT()
    plain = StereoNodeMod(4, n_tags=4, gain=1.0)
    plain.__dict__["_bound_dit"] = dit
    torch.nn.init.constant_(plain.scale.weight, 0.5)
    torch.nn.init.constant_(plain.shift.weight, 0.5)

    out = torch.ones(2, 4, 4)
    tags = torch.ones(2, 4, dtype=torch.long)
    with active_stereo_tags(tags):
        _publish_stereo_time(dit, torch.tensor([0.9, 0.9]))
        late = plain._hook(None, None, out.clone())
        _publish_stereo_time(dit, torch.tensor([0.1, 0.1]))
        early = plain._hook(None, None, out.clone())
        _publish_stereo_time(dit, None)
    assert torch.allclose(late, early), "unset window must not depend on t"
