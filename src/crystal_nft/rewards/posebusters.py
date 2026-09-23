"""Table-1 PoseBusters scoring for NFT (invalid crystals contribute 0).

Clari's ``posebusters_score`` returns ``pb_score=1.0`` when no fragment is
eligible. That naive default must never be used as a training reward.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from crystal_nft.rewards.uma_scorer import ScoreResult

logger = logging.getLogger(__name__)


def used_pb_score(pb_score: float, pb_valid: float | bool) -> float:
    """Map a raw PoseBusters row onto the value used by paper_bootstrap.

    Invalid crystals (``pb_valid=0``) contribute 0, not the naive 1.0.
    """
    if not pb_valid:
        return 0.0
    value = float(pb_score)
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, value))


def score_crystal_pb(pred, true=None) -> dict[str, float]:
    """Score one Crystal with the Table-1 PB + clash (+ optional volume) metrics."""
    from clari.pipelines.utils.metrics import check_clashes_eval, posebusters_score, volume_error

    try:
        raw = posebusters_score(pred)
        pb_score = float(raw.get("pb_score", 1.0))
        pb_valid = float(raw.get("pb_valid", 0.0))
    except Exception as exc:
        logger.warning("PoseBusters failed (%s); recording invalid PB=0", exc)
        pb_score, pb_valid = 1.0, 0.0

    try:
        clash_rate = float(check_clashes_eval(pred))
    except Exception as exc:
        logger.warning("Clash check failed (%s); recording clash=1", exc)
        clash_rate = 1.0

    volume = float("nan")
    if true is not None:
        try:
            volume = float(volume_error(pred, true))
        except Exception as exc:
            logger.warning("Volume error failed (%s)", exc)
            volume = float("nan")

    return {
        "pb_score": pb_score,
        "pb_valid": pb_valid,
        "pb_used": used_pb_score(pb_score, pb_valid),
        "clash_rate": clash_rate,
        "volume_error": volume,
    }


def score_crystals_pb(preds: Sequence[Any], true=None) -> list[dict[str, float]]:
    return [score_crystal_pb(pred, true) for pred in preds]


def scores_from_pb_rows(rows: Sequence[dict[str, float]]) -> list[ScoreResult]:
    """Build ScoreResult shells so clash still flows through advantage extras."""
    out: list[ScoreResult] = []
    for row in rows:
        out.append(
            ScoreResult(
                energy=0.0,
                energy_per_mol=0.0,
                force_mean_norm=0.0,
                force_max=0.0,
                stress_norm=0.0,
                clash=float(row.get("clash_rate", 1.0)),
                density=0.0,
                valid=True,
                n_mol=1,
            )
        )
    return out
