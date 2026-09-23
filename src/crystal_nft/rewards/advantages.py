"""PackFlow-style group-relative advantages and project.md reward extras."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from crystal_nft.rewards.uma_scorer import FAILED_ENERGY, ScoreResult


def _group_standardize(values: np.ndarray, eps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    return (values - mean) / (std + eps)


def _mad_normalize(values: np.ndarray, eps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    return -(values - med) / (mad + eps)


def score_to_rewards(
    scores: Sequence[ScoreResult],
    *,
    w_e: float = 1.0,
    w_f: float = 1.0,
    w_fmax: float = 0.0,
    w_stress: float = 0.0,
    w_clash: float = 1.0,
    w_margin: float = 0.0,
    clash_cutoff: float = 0.8,
    margin_headroom: float = 0.4,
    invalid_reward: float = -1e6,
) -> dict[str, np.ndarray]:
    """Convert ScoreResult list into raw reward channels (higher is better).

    `r_margin` is the one channel defined for invalid structures as well. Every
    other channel treats the geometry filter as a cliff: a candidate that
    crosses it is simply excluded from reinforcement, so nothing opposes the
    policy drifting toward it. Four UMA-reward arms did exactly that -- PB
    climbed to ~93 by epoch 2-6 and then fell as the UMA valid fraction went
    0.8 -> 0.02, with the learning rate setting only *when*. This channel
    rewards clearance above `clash_cutoff` and saturates once there is
    `margin_headroom` of it, so approaching the cliff costs something while a
    comfortable structure is not pushed to spread out forever.
    """
    n = len(scores)
    r_e = np.full(n, invalid_reward, dtype=np.float64)
    r_f = np.full(n, invalid_reward, dtype=np.float64)
    r_fmax = np.full(n, invalid_reward, dtype=np.float64)
    r_stress = np.full(n, invalid_reward, dtype=np.float64)
    r_clash = np.zeros(n, dtype=np.float64)
    r_margin = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    for i, s in enumerate(scores):
        r_clash[i] = -float(s.clash) * w_clash
        if w_margin != 0.0:
            min_d = float(getattr(s, "min_pair_distance", float("nan")))
            if np.isfinite(min_d):
                headroom = min(max(min_d - clash_cutoff, 0.0), margin_headroom)
                r_margin[i] = (headroom / margin_headroom - 1.0) * w_margin
        if not s.valid or not np.isfinite(s.energy_per_mol) or s.energy_per_mol == FAILED_ENERGY:
            continue
        valid[i] = True
        r_e[i] = -float(s.energy_per_mol) * w_e
        if np.isfinite(s.force_mean_norm) and s.force_mean_norm != FAILED_ENERGY:
            r_f[i] = -float(s.force_mean_norm) * w_f
        if np.isfinite(s.force_max) and s.force_max != FAILED_ENERGY:
            r_fmax[i] = -np.log1p(float(s.force_max)) * w_fmax
        if np.isfinite(s.stress_norm):
            r_stress[i] = -float(s.stress_norm) * w_stress

    return {
        "r_e": r_e,
        "r_f": r_f,
        "r_fmax": r_fmax,
        "r_stress": r_stress,
        "r_clash": r_clash,
        "r_margin": r_margin,
        "valid": valid,
    }


def compute_group_advantages(
    scores: Sequence[ScoreResult],
    group_ids: Sequence[str],
    *,
    ef_lambda: float = 0.5,
    advantage_eps: float = 1e-8,
    advantage_clip: float = 5.0,
    use_mad_energy: bool = False,
    w_e: float = 1.0,
    w_f: float = 1.0,
    w_fmax: float = 0.0,
    w_stress: float = 0.0,
    w_clash: float = 1.0,
    w_margin: float = 0.0,
    w_volume: float = 0.0,
    volumes: Sequence[float] | None = None,
    volume_targets: Sequence[float] | None = None,
    w_alignment: float = 0.0,
    alignment_rewards: Sequence[float] | None = None,
    w_pb: float = 0.0,
    pb_rewards: Sequence[float] | None = None,
    adv_mode: str = "continuous",
    top_frac: float = 0.2,
    bottom_frac: float = 0.3,
) -> np.ndarray:
    """
    PackFlow AdvantageMixing within each molecule/group.

    A = clip(lambda * A_E + (1-lambda) * A_F + extras, -c, c)

    adv_mode:
      - continuous: return clipped mixed advantages
      - binary: +1 for top_frac, -1 for bottom_frac, 0 otherwise (by advantage rank)
    """
    rewards = score_to_rewards(
        scores,
        w_e=w_e,
        w_f=w_f,
        w_fmax=w_fmax,
        w_stress=w_stress,
        w_clash=w_clash,
        w_margin=w_margin,
    )
    group_ids = np.asarray(list(group_ids))
    advantages = np.zeros(len(scores), dtype=np.float64)

    vol_arr = None
    vt_arr = None
    if w_volume != 0.0 and volumes is not None and volume_targets is not None:
        vol_arr = np.asarray(volumes, dtype=np.float64)
        vt_arr = np.asarray(volume_targets, dtype=np.float64)
    align_arr = None
    if w_alignment != 0.0 and alignment_rewards is not None:
        align_arr = np.asarray(alignment_rewards, dtype=np.float64)
    pb_arr = None
    if w_pb != 0.0 and pb_rewards is not None:
        pb_arr = np.asarray(pb_rewards, dtype=np.float64)

    # When a lattice-volume target is supplied, volume is well-defined for every
    # candidate regardless of UMA validity (clash/density). Since volume RMAD is
    # exactly the target metric, do NOT gate the volume signal on UMA validity.
    # PB is the Table-1 packing metric: also well-defined after invalid->0.
    volume_driven = vol_arr is not None and vt_arr is not None
    target_driven = volume_driven or align_arr is not None or pb_arr is not None

    for gid in np.unique(group_ids):
        mask = group_ids == gid
        idx = np.where(mask)[0]
        valid_mask = rewards["valid"][idx]
        n = len(idx)

        # Energy/force advantages require >=2 valid structures; otherwise they
        # contribute nothing (but volume can still drive learning).
        ef = np.zeros(n, dtype=np.float64)
        if valid_mask.sum() >= 2:
            r_e = rewards["r_e"][idx].copy()
            r_f = rewards["r_f"][idx].copy()
            if (~valid_mask).any():
                r_e[~valid_mask] = np.nanmean(r_e[valid_mask])
                r_f[~valid_mask] = np.nanmean(r_f[valid_mask])

            if use_mad_energy:
                epm = np.array(
                    [scores[i].energy_per_mol if scores[i].valid else np.nan for i in idx],
                    dtype=np.float64,
                )
                a_e = _mad_normalize(epm, advantage_eps)
                a_e = np.nan_to_num(a_e, nan=0.0)
            else:
                a_e = _group_standardize(r_e, advantage_eps)
            a_f = _group_standardize(r_f, advantage_eps)
            ef = ef_lambda * a_e + (1.0 - ef_lambda) * a_f
        elif valid_mask.sum() == 0 or not target_driven:
            # Nothing usable in this group -> leave advantages at 0.
            #
            # The `valid_mask.sum() == 0` half matters even when a target-driven
            # term is on. Without it the volume/PB/alignment term alone decided
            # the ranking inside an all-invalid group, so a structure the
            # potential could not even score would be reinforced as a positive
            # whenever its cell happened to match the target. That is a
            # positive-feedback channel, and it destroyed two UMA-reward arms:
            # the policy learns broken geometry with a plausible cell, validity
            # collapses, and a larger share of the gradient then comes through
            # this path. KL gives no warning because the direction is
            # consistent (PB 89.3 -> 41.5 by epoch 3 at epoch-mean KL 0.017).
            advantages[idx] = 0.0
            continue

        extras = np.zeros(n, dtype=np.float64)
        # Substitute the valid mean for invalid entries before standardising,
        # exactly as r_e/r_f above. Left raw, those entries still hold the
        # -1e6 `invalid_reward` sentinel, and standardising a vector that
        # contains it collapses every valid entry onto the same value -- so
        # these two channels carried no information at all whenever a single
        # candidate in the group was invalid.
        def _fill_invalid(key: str) -> np.ndarray:
            arr = rewards[key][idx].copy()
            if (~valid_mask).any() and valid_mask.any():
                arr[~valid_mask] = np.nanmean(arr[valid_mask])
            return arr

        if w_fmax != 0.0 and valid_mask.sum() >= 2:
            extras = extras + _group_standardize(_fill_invalid("r_fmax"), advantage_eps)
        if w_stress != 0.0 and valid_mask.sum() >= 2:
            extras = extras + _group_standardize(_fill_invalid("r_stress"), advantage_eps)
        # Lattice-volume matching reward: prefer candidates whose cell volume is
        # closest to the target (GT) volume. Directly reduces volume RMAD.
        if volume_driven:
            rel_dev = np.abs(vol_arr[idx] - vt_arr[idx]) / np.maximum(
                np.abs(vt_arr[idx]), advantage_eps
            )
            a_vol = _group_standardize(-rel_dev, advantage_eps)
            a_vol = np.nan_to_num(a_vol, nan=0.0)
            extras = extras + w_volume * a_vol
        if align_arr is not None:
            a_align = _group_standardize(align_arr[idx], advantage_eps)
            a_align = np.nan_to_num(a_align, nan=0.0)
            extras = extras + w_alignment * a_align
        if pb_arr is not None:
            a_pb = _group_standardize(pb_arr[idx], advantage_eps)
            a_pb = np.nan_to_num(a_pb, nan=0.0)
            extras = extras + w_pb * a_pb
        # Clash penalty (raw, per-candidate). Keeps volume-good but clashy
        # packings from being reinforced too strongly.
        extras = extras + rewards["r_clash"][idx]
        # Clash headroom, raw and defined for invalid candidates too: this is
        # the term meant to bite *before* the geometry filter rejects a
        # structure, so unlike fmax/stress it must not be gated on validity.
        if w_margin != 0.0:
            extras = extras + rewards["r_margin"][idx]

        mixed = ef + extras
        # Re-centre within the group before clipping.
        #
        # `advantage_to_nft_weight` maps an advantage to r = adv/2 + 0.5, which
        # DiffusionNFT reads as a *relative* within-group preference: r > 0.5
        # reinforces, r < 0.5 pushes away. The PB path is balanced by
        # construction (+1 top quartile, -1 bottom, 0 otherwise, so mean ~ 0),
        # but this path adds a raw clash penalty that is always <= 0 and floors
        # invalid candidates, so the group mean drifts negative and the update
        # becomes "move away from everything you just sampled" with no positive
        # anchor. Measured: mfpure held mean advantage ~0 and KL <= 0.047 for 12
        # epochs, while the UMA arms drifted to -0.43 and diverged. Use the
        # median of the valid candidates as the reference, since the clash term
        # makes the mean sensitive to a single bad candidate.
        if valid_mask.any():
            mixed = mixed - np.median(mixed[valid_mask])
        mixed = np.clip(mixed, -advantage_clip, advantage_clip)
        if not target_driven:
            mixed[~valid_mask] = -advantage_clip
        else:
            # Keep the target signal for ranking the valid candidates, but an
            # invalid one may never be reinforced -- floor it at zero rather
            # than letting a good cell volume outvote the fact that the
            # structure is broken.
            mixed[~valid_mask] = np.minimum(mixed[~valid_mask], 0.0)
        advantages[idx] = mixed

        if adv_mode == "binary":
            order = np.argsort(-advantages[idx])  # high advantage first
            n = len(idx)
            n_pos = max(1, int(np.ceil(top_frac * n)))
            n_neg = max(1, int(np.ceil(bottom_frac * n)))
            binary = np.zeros(n, dtype=np.float64)
            binary[order[:n_pos]] = 1.0
            binary[order[-n_neg:]] = -1.0
            # keep invalid as negative
            binary[~valid_mask] = -1.0
            advantages[idx] = binary

    return advantages


def pb_has_signal(pb_rewards: Sequence[float], *, min_range: float = 1e-6) -> bool:
    """True when a group has at least two distinct PoseBusters values."""
    values = np.asarray(pb_rewards, dtype=np.float64)
    if values.size < 2:
        return False
    return float(np.nanmax(values) - np.nanmin(values)) >= float(min_range)


def compute_pb_rank_advantages(
    pb_rewards: Sequence[float],
    *,
    clash: Sequence[float] | None = None,
    clash_weight: float = 0.0,
    volume_error: Sequence[float] | None = None,
    volume_weight: float = 0.0,
    top_frac: float = 0.25,
    bottom_frac: float = 0.25,
    clash_veto_positive: bool = True,
    min_range: float = 1e-6,
) -> np.ndarray:
    """Rank by PB (optionally minus clash / volume-error terms): +1 top, -1 bottom.

    With both extra weights at 0 the ranking is PB alone and clash only vetoes:
    a clashing top-ranked crystal is zeroed (not reinforced) when
    ``clash_veto_positive`` is set. The two weights below put those signals into
    the key itself; they compose, so setting both subtracts both.

    ``clash_weight > 0`` subtracts ``clash_weight * clash`` from the ranking
    key, so clash can actually reorder candidates instead of only vetoing
    positives. ``volume_weight > 0`` subtracts ``volume_weight * volume_error``
    from the ranking key. This exists because PB-only ranking leaves cell volume
    completely unconstrained, and volume error is itself a Table-1 column: a
    Stage-2 NFT run on CrystAF held PB and clash at baseline while volume error
    went 2.09% -> 2.58% (2.9 sigma worse), because nothing in the reward
    opposed it. ``volume_error`` is the relative error Clari's ``volume_error``
    already returns in every ``score_crystal_pb`` row, so this costs nothing
    extra to compute.

    PB is coarse (fragment pass fractions, granularity ~0.25) while volume
    error is a few percent, so at ``volume_weight`` ~ 1 this acts mainly as a
    **tie-break among equal-PB candidates** rather than overriding PB. That
    also makes the ~50% of families whose PB is completely flat usable instead
    of skipped: "these are all PB-valid, prefer the one with the better cell".
    NaN volume (no reference cell, or a failed computation) ranks worst.
    """
    pb = np.asarray(pb_rewards, dtype=np.float64)
    n = int(pb.size)
    advantages = np.zeros(n, dtype=np.float64)
    key = pb
    if clash_weight and clash is not None:
        # Clash in the RANKING KEY, not just as a veto. `clash_veto_positive`
        # can only decline to reinforce a clashing crystal; it cannot push one
        # down the order, which makes clash the weakest signal in the group and
        # the first thing sacrificed when PB and volume compete (measured:
        # vol_rank_weight 1->6 moved clash 12.08 -> 15.46). PB is a fragment
        # pass fraction with ~0.25 granularity and clash is 0/1 per crystal, so
        # clash_weight ~0.25-0.5 costs a clashing candidate one-to-two PB steps.
        key = key - float(clash_weight) * np.asarray(clash, dtype=np.float64)
    if volume_weight and volume_error is not None:
        vol = np.asarray(volume_error, dtype=np.float64)
        if vol.size == n:
            worst = np.nanmax(vol) if np.any(np.isfinite(vol)) else 0.0
            vol = np.where(np.isfinite(vol), vol, worst + 1.0)
            # `key`, not `pb`: rebuilding from `pb` here silently discarded
            # the clash term above whenever both weights were on.
            key = key - float(volume_weight) * vol
    if not pb_has_signal(key, min_range=min_range):
        return advantages
    order = np.argsort(-key, kind="stable")
    n_pos = max(1, int(np.ceil(float(top_frac) * n)))
    n_neg = max(1, int(np.ceil(float(bottom_frac) * n)))
    if n_pos + n_neg > n:
        n_neg = max(1, n - n_pos)
    advantages[order[:n_pos]] = 1.0
    advantages[order[-n_neg:]] = -1.0
    if clash_veto_positive and clash is not None:
        clash_arr = np.asarray(clash, dtype=np.float64)
        advantages[(advantages > 0.0) & (clash_arr > 0.0)] = 0.0
    return advantages


def advantage_to_nft_weight(
    advantages: np.ndarray,
    *,
    adv_clip_max: float = 5.0,
) -> np.ndarray:
    """Map advantages in [-c, c] to DiffusionNFT r in [0, 1]."""
    advantages = np.asarray(advantages, dtype=np.float64)
    clipped = np.clip(advantages, -adv_clip_max, adv_clip_max)
    r = (clipped / adv_clip_max) / 2.0 + 0.5
    return np.clip(r, 0.0, 1.0)
