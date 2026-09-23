"""CPU-only unit tests for reward / NFT helpers."""

from __future__ import annotations

import numpy as np
import torch

from crystal_nft.nft.loss import nft_reconstruction_loss, return_decay
from crystal_nft.rewards.advantages import (
    advantage_to_nft_weight,
    compute_group_advantages,
    compute_pb_rank_advantages,
    pb_has_signal,
)
from crystal_nft.rewards.uma_scorer import ScoreResult


def _scores():
    return [
        ScoreResult(10.0, 2.5, 0.1, 0.2, 0.0, 0.0, 1.2, True, 4),
        ScoreResult(12.0, 3.0, 0.3, 0.5, 0.1, 0.0, 1.1, True, 4),
        ScoreResult(8.0, 2.0, 0.05, 0.1, 0.0, 0.0, 1.3, True, 4),
        ScoreResult(20.0, 5.0, 1.0, 2.0, 1.0, 1.0, 0.2, False, 4),
    ]


def test_group_advantages_shape():
    adv = compute_group_advantages(_scores(), ["a"] * 4, ef_lambda=0.5)
    assert adv.shape == (4,)
    assert np.isfinite(adv[:3]).all()


def test_used_pb_score_invalid_is_zero():
    from crystal_nft.rewards.posebusters import used_pb_score

    assert used_pb_score(1.0, 0.0) == 0.0
    assert used_pb_score(1.0, False) == 0.0
    assert used_pb_score(0.75, 1.0) == 0.75
    assert used_pb_score(float("nan"), 1.0) == 0.0


def test_pb_advantage_ranks_higher_score_first():
    """Higher PB ranks higher -- among candidates that are physically usable.

    Candidate 3 of `_scores()` has clash 1.0 and density 0.2 g/cm^3, so it
    fails the geometry filter and arrives invalid. It is now floored at <= 0 no
    matter how high its PB reads, because reinforcing a structure with a hard
    atomic clash is what let the UMA arms run away. The original version of
    this test asserted `adv[3] > adv[0]`, i.e. that this candidate should
    outrank a valid one on PB alone; that assertion is what the floor removes.
    """
    dummy = _scores()
    adv = compute_group_advantages(
        dummy,
        ["a"] * 4,
        ef_lambda=0.0,
        w_e=0.0,
        w_f=0.0,
        w_clash=0.0,
        w_pb=1.0,
        pb_rewards=[0.2, 0.9, 0.0, 1.0],
    )
    assert adv[1] > adv[0], adv          # PB 0.9 over PB 0.2
    assert adv[0] > adv[2], adv          # PB 0.2 over PB 0.0
    assert adv[3] <= 0.0, adv            # clashing candidate never reinforced


def test_pb_rank_skips_flat_and_vetoes_clash():
    assert not pb_has_signal([0.5, 0.5, 0.5, 0.5])
    assert pb_has_signal([0.0, 0.0, 1.0, 1.0])
    flat = compute_pb_rank_advantages([1.0, 1.0, 1.0, 1.0])
    assert np.allclose(flat, 0.0)
    pb = [1.0, 0.8, 0.2, 0.0]
    adv = compute_pb_rank_advantages(pb, top_frac=0.25, bottom_frac=0.25)
    assert adv[0] == 1.0
    assert adv[3] == -1.0
    assert adv[1] == 0.0 and adv[2] == 0.0
    vetoed = compute_pb_rank_advantages(
        pb, clash=[1.0, 0.0, 0.0, 0.0], clash_veto_positive=True
    )
    assert vetoed[0] == 0.0
    assert vetoed[3] == -1.0


def test_target_alignment_advantage_and_exact_match():
    from ase import Atoms

    from crystal_nft.rewards.target_alignment import target_alignment_rewards

    target = Atoms(
        "CO",
        positions=[[0.0, 0.0, 0.0], [1.2, 1.2, 1.2]],
        cell=np.diag([5.0, 5.0, 5.0]),
        pbc=True,
    )
    distorted = target.copy()
    distorted.set_cell(np.diag([6.0, 5.0, 4.5]), scale_atoms=True)
    alignment = target_alignment_rewards([target, distorted], target)
    assert np.isclose(alignment[0], 0.0)
    assert alignment[1] < alignment[0]

    adv = compute_group_advantages(
        _scores()[:2],
        ["a", "a"],
        ef_lambda=0.0,
        w_e=0.0,
        w_f=0.0,
        w_alignment=1.0,
        alignment_rewards=alignment,
    )
    assert adv[0] > adv[1]


def test_nft_weight_range():
    r = advantage_to_nft_weight(np.array([-5.0, 0.0, 5.0]), adv_clip_max=5.0)
    assert np.allclose(r, [0.0, 0.5, 1.0])


def test_nft_reconstruction_clari():
    B, N, D = 4, 6, 3
    xt = torch.randn(B, N, D)
    clean = torch.randn(B, N, D)
    fwd = torch.randn(B, N, D, requires_grad=True)
    old = torch.randn(B, N, D)
    t = torch.rand(B)
    r = torch.tensor([0.2, 0.8, 0.5, 0.1])
    out = nft_reconstruction_loss(
        fwd, old, xt=xt, clean=clean, t=t, r=r, beta=0.1, time_convention="clari"
    )
    out["policy_loss"].backward()
    assert fwd.grad is not None
    assert torch.isfinite(out["policy_loss"])


def test_return_decay():
    assert return_decay(0, 1) == 0.0
    assert return_decay(1000, 1) == 0.5


if __name__ == "__main__":
    test_group_advantages_shape()
    test_used_pb_score_invalid_is_zero()
    test_pb_advantage_ranks_higher_score_first()
    test_pb_rank_skips_flat_and_vetoes_clash()
    test_target_alignment_advantage_and_exact_match()
    test_nft_weight_range()
    test_nft_reconstruction_clari()
    test_return_decay()
    print("ok")


def test_pb_rank_volume_weight_is_a_tiebreak_not_an_override():
    """Volume error breaks ties among equal-PB candidates, but PB still leads.

    PB-only ranking leaves cell volume unconstrained, and a Stage-2 NFT run on
    CrystAF held PB/clash at baseline while volume error went 2.09% -> 2.58%
    (2.9 sigma). PB is coarse (~0.25 granularity) and volume error is a few
    percent, so at volume_weight ~ 1 the volume term must not reorder
    different-PB candidates.
    """
    import numpy as np

    from crystal_nft.rewards.advantages import compute_pb_rank_advantages

    # Same PB, different volume: the better cell must win.
    adv = compute_pb_rank_advantages(
        [1.0, 1.0, 1.0, 1.0],
        volume_error=[0.01, 0.09, 0.02, 0.08],
        volume_weight=1.0,
        top_frac=0.25,
        bottom_frac=0.25,
        clash_veto_positive=False,
    )
    assert adv[0] == 1.0, adv       # lowest volume error -> positive
    assert adv[1] == -1.0, adv      # highest volume error -> negative

    # Different PB: a 0.25 PB gap must outrank a few percent of volume error.
    adv = compute_pb_rank_advantages(
        [1.0, 0.75],
        volume_error=[0.09, 0.01],
        volume_weight=1.0,
        top_frac=0.5,
        bottom_frac=0.5,
        clash_veto_positive=False,
    )
    assert adv[0] == 1.0 and adv[1] == -1.0, adv


def test_pb_rank_volume_weight_zero_is_the_old_behaviour():
    import numpy as np

    from crystal_nft.rewards.advantages import compute_pb_rank_advantages

    pb = [1.0, 0.5, 0.75, 0.25]
    vol = [0.9, 0.0, 0.5, 0.1]
    a = compute_pb_rank_advantages(pb, volume_weight=0.0, volume_error=vol)
    b = compute_pb_rank_advantages(pb)
    assert np.array_equal(a, b)


def test_pb_rank_flat_pb_becomes_usable_with_volume():
    """A completely flat-PB family carries no signal alone, but does with volume.

    ~50% of CSD families come back all-PB-1.0, and those were being skipped
    outright. With the volume term they become "all PB-valid, prefer the
    better cell" -- which is exactly a Table-1 objective.
    """
    import numpy as np

    from crystal_nft.rewards.advantages import compute_pb_rank_advantages

    flat = [1.0] * 8
    vol = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.20]
    assert np.count_nonzero(compute_pb_rank_advantages(flat)) == 0
    adv = compute_pb_rank_advantages(
        flat, volume_error=vol, volume_weight=1.0, clash_veto_positive=False
    )
    assert np.count_nonzero(adv) > 0
    assert adv[0] == 1.0 and adv[-1] == -1.0


def test_pb_rank_nan_volume_ranks_worst():
    import numpy as np

    from crystal_nft.rewards.advantages import compute_pb_rank_advantages

    adv = compute_pb_rank_advantages(
        [1.0, 1.0, 1.0, 1.0],
        volume_error=[0.01, float("nan"), 0.02, 0.03],
        volume_weight=1.0,
        top_frac=0.25,
        bottom_frac=0.25,
        clash_veto_positive=False,
    )
    assert adv[1] == -1.0, adv
    assert np.isfinite(adv).all()


def _fake_scores(n, valid=True):
    from crystal_nft.rewards.uma_scorer import ScoreResult
    import inspect
    sig = inspect.signature(ScoreResult)
    out = []
    for i in range(n):
        kw = {}
        for name, prm in sig.parameters.items():
            if prm.default is not inspect._empty:
                kw[name] = prm.default
            elif name == "valid":
                kw[name] = valid
            else:
                kw[name] = 0.0
        kw["valid"] = valid
        out.append(ScoreResult(**kw))
    return out


def test_w_volume_creates_advantage_from_volume_alone():
    """The dedicated volume advantage must rank on |V - V_true| by itself.

    Volume error is a Table-1 column and this is the lever the MolCrystalFlow
    line used for RMAD 3.88% -> 3.16%, so the term has to rank a group on cell
    volume even when energy and force are uninformative (identical here).

    This test used to build the group from `valid=False` scores. That is not a
    group with no energy signal -- `valid=False` means the structure failed the
    geometry filter, i.e. it has an atomic clash or an impossible density -- so
    the old assertion demanded that a clashing structure be reinforced for
    having a good cell. See
    `test_all_invalid_group_carries_no_signal_even_with_a_volume_target`.
    """
    import numpy as np

    from crystal_nft.rewards.advantages import compute_group_advantages

    n = 6
    scores = _fake_scores(n, valid=True)  # valid, but energy/force uninformative
    target = 1000.0
    volumes = [1000.0, 1010.0, 1050.0, 1100.0, 1200.0, 1400.0]
    adv = compute_group_advantages(
        scores, ["fam"] * n, w_volume=1.0,
        volumes=volumes, volume_targets=[target] * n,
        adv_mode="continuous", advantage_clip=5.0,
    )
    assert np.any(adv != 0.0), adv
    # Closest volume must score highest, furthest lowest.
    assert adv[0] == max(adv), adv
    assert adv[-1] == min(adv), adv


def test_w_volume_zero_leaves_advantages_untouched():
    import numpy as np

    from crystal_nft.rewards.advantages import compute_group_advantages

    n = 5
    scores = _fake_scores(n, valid=False)
    adv = compute_group_advantages(
        scores, ["fam"] * n, w_volume=0.0,
        volumes=[1.0] * n, volume_targets=[2.0] * n,
    )
    assert np.count_nonzero(adv) == 0


def test_relative_volume_error_can_be_fed_as_a_negated_reward():
    """The PB path passes an already-relative volume error, not a raw volume.

    `compute_group_advantages` computes rel_dev = |vol - target| / |target|, so
    an already-relative error is fed as volumes=-err with volume_targets=0.
    Check the ranking direction, since getting this backwards would reinforce
    the WORST cells while looking perfectly healthy in the logs.
    """
    import inspect

    import numpy as np

    from crystal_nft.rewards.advantages import compute_group_advantages
    from crystal_nft.rewards.uma_scorer import ScoreResult

    sig = inspect.signature(ScoreResult)

    def blanks(n):
        # valid=True: `valid=False` means "failed the geometry filter", and such
        # a candidate is floored at advantage 0 so it can never be reinforced.
        out = []
        for _ in range(n):
            kw = {
                k: (v.default if v.default is not inspect._empty else 0.0)
                for k, v in sig.parameters.items()
            }
            kw["valid"] = True
            out.append(ScoreResult(**kw))
        return out

    vol_rel = [0.01, 0.02, 0.05, 0.10]
    adv = compute_group_advantages(
        blanks(4), ["fam"] * 4, w_volume=1.0,
        volumes=[-v for v in vol_rel], volume_targets=[0.0] * 4,
        adv_mode="continuous", advantage_clip=5.0,
    )
    assert np.any(adv != 0.0), adv
    assert adv[0] == max(adv), adv      # smallest volume error -> best
    assert adv[-1] == min(adv), adv     # largest -> worst


def test_clash_enters_the_pb_rank_key_not_just_the_veto():
    """Clash must be able to REORDER candidates, not only veto positives.

    `clash_veto_positive` zeroes a clashing top-ranked crystal but cannot push
    it down the order, so clash was the weakest signal in the group and the
    first thing sacrificed when PB and volume competed (measured:
    vol_rank_weight 1 -> 6 moved clash 12.08 -> 15.46).
    """
    import numpy as np

    from crystal_nft.rewards.advantages import compute_pb_rank_advantages

    adv = compute_pb_rank_advantages(
        [1.0] * 4, clash=[0, 0, 1, 1], clash_weight=0.5,
        top_frac=0.25, bottom_frac=0.25, clash_veto_positive=False,
    )
    assert adv[0] == 1.0 and adv[-1] == -1.0, adv

    # PB still leads: a 0.25 PB gap outranks a 0.1 clash penalty.
    adv = compute_pb_rank_advantages(
        [1.0, 0.75], clash=[1, 0], clash_weight=0.1,
        top_frac=0.5, bottom_frac=0.5, clash_veto_positive=False,
    )
    assert adv[0] == 1.0, adv

    # Off by default.
    a = compute_pb_rank_advantages([1.0, 0.5, 0.75], clash=[1, 0, 1], clash_weight=0.0)
    b = compute_pb_rank_advantages([1.0, 0.5, 0.75], clash=[1, 0, 1])
    assert np.array_equal(a, b)


def test_clash_and_volume_rank_weights_compose():
    """Both extra rank weights must apply; the volume term must not reset the key.

    With PB flat, clash alone decides the order. Adding a volume weight must
    refine that order, not replace it -- rebuilding `key` from `pb` inside the
    volume branch silently dropped clash whenever both weights were on.
    """
    import numpy as np

    from crystal_nft.rewards.advantages import compute_pb_rank_advantages

    # 8 candidates so the 25% quartiles are two each (with n=4 every rank but
    # the single best and single worst is 0, which proves nothing).
    pb = [1.0] * 8
    clash = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    # Volume only separates within each clash pair, so clash must still lead.
    vol = [0.04, 0.03, 0.02, 0.01, 0.04, 0.03, 0.02, 0.01]

    adv = compute_pb_rank_advantages(
        pb,
        clash=clash,
        clash_weight=1.0,
        volume_error=vol,
        volume_weight=0.1,
        top_frac=0.25,
        bottom_frac=0.25,
        clash_veto_positive=False,
    )
    # Clash leads: every positive is clash-free, every negative clashes.
    assert set(np.flatnonzero(adv > 0)) == {2, 3}
    assert set(np.flatnonzero(adv < 0)) == {4, 5}
    # Volume refines within the clash-free block: 3 (vol .01) and 2 (.02) beat
    # 1 (.03) and 0 (.04). Dropping the clash term would order purely by volume
    # and put 7 (clashing, vol .01) at the top instead.
    assert adv[7] <= 0.0


def _uma_score(valid: bool, energy: float = -100.0, fmax: float = 1.0, clash: float = 0.0):
    """One UMA score; an invalid one has no usable energy, as in the real path."""
    return ScoreResult(
        energy * 4, (energy if valid else float("nan")), 0.5, fmax, 0.01,
        clash, 1.2, valid, 4,
    )


_UMA_KW = dict(
    adv_mode="continuous", advantage_clip=1.0, w_clash=1.0, w_fmax=0.25,
    w_stress=0.1, w_volume=1.0, w_alignment=0.0,
)


def test_all_invalid_group_carries_no_signal_even_with_a_volume_target():
    """A structure the potential cannot score must never be a positive.

    With a target-driven term on, the volume reward alone used to rank an
    all-invalid group, so a broken molecule whose cell happened to match the
    target scored +clip. That is self-reinforcing -- it took two UMA arms from
    PB 89.3 to 41.5 in three epochs while KL stayed at 0.017 -- so the group
    must contribute nothing instead.
    """
    adv = compute_group_advantages(
        [_uma_score(False) for _ in range(4)], ["f"] * 4,
        volumes=[1000.0, 1400.0, 1500.0, 1600.0], volume_targets=[1000.0] * 4,
        **_UMA_KW,
    )
    assert np.allclose(adv, 0.0), adv


def test_invalid_candidate_is_never_reinforced_by_a_good_cell_volume():
    adv = compute_group_advantages(
        [_uma_score(True, -100.0), _uma_score(True, -99.0),
         _uma_score(True, -98.0), _uma_score(False)], ["f"] * 4,
        # the invalid candidate has the *best* cell of the four
        volumes=[1100.0, 1150.0, 1200.0, 1000.0], volume_targets=[1000.0] * 4,
        **_UMA_KW,
    )
    assert adv[3] <= 0.0, f"invalid candidate reinforced: {adv}"
    assert adv[0] > adv[2], f"valid candidates no longer ranked: {adv}"


def test_fmax_channel_survives_an_invalid_candidate_in_the_group():
    """`r_fmax`/`r_stress` must get the valid mean substituted like `r_e`.

    Left raw they still hold the -1e6 invalid sentinel, and standardising a
    vector containing it collapses every valid entry onto one value -- the
    channel then carries no information whenever any candidate is invalid.
    """
    scores = [
        _uma_score(True, fmax=0.5), _uma_score(True, fmax=5.0),
        _uma_score(True, fmax=50.0), _uma_score(False),
    ]
    adv = compute_group_advantages(
        scores, ["f"] * 4, volumes=[1000.0] * 4, volume_targets=[1000.0] * 4,
        **_UMA_KW,
    )
    assert adv[0] > adv[1] > adv[2], f"fmax ordering lost: {adv}"
    assert adv[0] - adv[2] > 0.1, f"fmax channel is flat: {adv}"


def test_valid_only_group_is_unaffected_by_the_invalid_handling():
    adv = compute_group_advantages(
        [_uma_score(True, -102.0), _uma_score(True, -100.0), _uma_score(True, -98.0)],
        ["f"] * 3, volumes=[1000.0] * 3, volume_targets=[1000.0] * 3, **_UMA_KW,
    )
    assert adv[0] > adv[1] > adv[2], f"lower energy must rank higher: {adv}"


def test_continuous_advantages_are_balanced_within_the_group():
    """The continuous path must be zero-centred, like the PB-rank path.

    `advantage_to_nft_weight` maps adv -> r = adv/2 + 0.5 and DiffusionNFT
    reads r as a relative within-group preference. `compute_pb_rank_advantages`
    is balanced by construction (+1 top quartile, -1 bottom, 0 otherwise); this
    path adds a raw clash penalty that is always <= 0 plus a floor on invalid
    candidates, so without re-centring the group mean drifts negative and the
    update becomes "push away from every sample" with no positive anchor. That
    drift tracked the divergence of the UMA arms (mean advantage -0.02 -> -0.43
    over four epochs) while mfpure held ~0 and was stable.
    """
    scores = [_uma_score(True, energy=-100.0 - i) for i in range(8)]
    adv = compute_group_advantages(
        scores, ["f"] * 8,
        volumes=[1000.0 + 30 * i for i in range(8)], volume_targets=[1000.0] * 8,
        **_UMA_KW,
    )
    assert abs(float(np.mean(adv))) < 1e-9, f"group is not zero-centred: {adv}"
    assert (adv > 0).sum() >= 3 and (adv < 0).sum() >= 3, f"unbalanced split: {adv}"
    assert adv[0] > adv[-1], f"lower energy must still rank higher: {adv}"


def test_a_uniform_clash_penalty_does_not_make_every_candidate_negative():
    """An absolute penalty shared by the whole group carries no ranking signal.

    Clash is still vetoed absolutely: a clashing crystal fails the geometry
    filter, so it arrives as invalid and is floored at <= 0 by
    `test_invalid_candidate_is_never_reinforced_by_a_good_cell_volume`.
    """
    scores = [_uma_score(True, energy=-100.0 - i, clash=1.0) for i in range(4)]
    adv = compute_group_advantages(
        scores, ["f"] * 4, volumes=[1000.0] * 4, volume_targets=[1000.0] * 4,
        **_UMA_KW,
    )
    assert (adv > 0).any(), f"whole group pushed away from: {adv}"


def test_clash_margin_ranks_clearance_before_the_filter_rejects():
    """The geometry filter is a cliff; `w_margin` gives it a slope.

    Every other channel treats validity as binary, so a candidate drifting
    toward the clash cutoff pays nothing until it crosses and is excluded from
    reinforcement. Four UMA arms drifted over exactly that edge (PB ~93 at
    epoch 2-6, then UMA valid fraction 0.8 -> 0.02, learning rate setting only
    when). With `w_margin` on, structures that are all valid but differently
    close to the cutoff must be ranked by clearance.
    """
    def s(min_d: float):
        return ScoreResult(-400.0, -100.0, 0.5, 1.0, 0.01,
                           1.0 if min_d < 0.8 else 0.0, 1.2, True, 4, min_d)

    scores = [s(1.6), s(1.0), s(0.85), s(0.81)]
    vols = [1000.0] * 4
    flat = compute_group_advantages(
        scores, ["f"] * 4, volumes=vols, volume_targets=vols, **_UMA_KW)
    graded = compute_group_advantages(
        scores, ["f"] * 4, volumes=vols, volume_targets=vols, w_margin=1.0, **_UMA_KW)
    assert np.allclose(flat, 0.0), f"expected no signal without w_margin: {flat}"
    assert graded[0] > graded[1] > graded[2] > graded[3], f"not ranked: {graded}"
    assert graded[0] - graded[3] > 0.5, f"margin signal too weak: {graded}"


def test_clash_margin_saturates_so_comfortable_structures_are_not_pushed_apart():
    """Past `margin_headroom` of clearance the term must stop caring."""
    def s(min_d: float):
        return ScoreResult(-400.0, -100.0, 0.5, 1.0, 0.01, 0.0, 1.2, True, 4, min_d)

    vols = [1000.0] * 3
    adv = compute_group_advantages(
        [s(1.4), s(2.5), s(4.0)], ["f"] * 3, volumes=vols, volume_targets=vols,
        w_margin=1.0, **_UMA_KW)
    assert np.allclose(adv, 0.0, atol=1e-9), f"saturation broken: {adv}"
