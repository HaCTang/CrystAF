"""Proper-rotation Kabsch fit.

The `det = +1` clamp is the entire chirality argument, so it is enforced here
rather than left to the caller. Plain Kabsch minimises RMSD over O(3), which
includes reflections -- and a reflection is exactly the enantiomer flip we are
trying to make impossible. Restricting to SO(3) means a mirrored target is
fitted *badly* (large RMSD) instead of being matched by flipping the conformer,
which is the behaviour we want: the conformer's handedness always wins.
"""

from __future__ import annotations

import torch
from torch import Tensor


def kabsch_proper(src: Tensor, dst: Tensor, weights: Tensor | None = None) -> Tensor:
    """Rotation in SO(3) taking centred `src` onto centred `dst`.

    `src`, `dst`: (..., n, 3), already centred. Returns (..., 3, 3) with
    determinant +1 (never -1), so the map is orientation-preserving.
    """
    if weights is not None:
        w = weights.unsqueeze(-1)
        h = (src * w).transpose(-1, -2) @ dst
    else:
        h = src.transpose(-1, -2) @ dst
    u, _, vh = torch.linalg.svd(h.double())
    v = vh.transpose(-1, -2)
    # Flip the least-significant singular direction if the naive product is a
    # reflection. This is the standard Kabsch correction and is what keeps the
    # result inside SO(3).
    d = torch.linalg.det(v @ u.transpose(-1, -2))
    sign = torch.ones_like(v[..., 0, :])
    sign[..., -1] = torch.sign(d)
    r = (v * sign.unsqueeze(-2)) @ u.transpose(-1, -2)
    return r.to(src.dtype)


def rigid_fit_bodies(
    sampled: Tensor,
    conformer: Tensor,
    body_ids: Tensor,
    mask: Tensor | None = None,
    conformer_ok: Tensor | None = None,
) -> Tensor:
    """Replace each body's geometry by the best proper-rotation fit of `conformer`.

    `sampled`, `conformer`: (B, N, 3) in the SAME atom order (a CrystAF
    prediction is the reference crystal carrying new coordinates, so this holds
    by construction). `body_ids`: (B, N) or (N,).

    Each body keeps the *sampled* centroid -- only its internal geometry is
    replaced -- so intermolecular packing is left to the generative model and
    only intramolecular structure comes from the conformer. That split is
    exactly MolCrystalFlow's factorisation.
    """
    out = sampled.clone()
    b, n, _ = sampled.shape
    for i in range(b):
        bid = body_ids[i] if body_ids.dim() > 1 else body_ids
        valid = mask[i] if mask is not None else torch.ones(n, dtype=torch.bool, device=sampled.device)
        for uid in torch.unique(bid[valid]):
            sel = (bid == uid) & valid
            if int(sel.sum()) < 3:      # 1-2 atoms have no handedness to fix
                continue
            if conformer_ok is not None and not bool(conformer_ok[i][sel].all()):
                continue        # no conformer for this body: leave the sample as-is
            y = sampled[i][sel].double()
            x = conformer[i][sel].double()
            cy, cx = y.mean(0, keepdim=True), x.mean(0, keepdim=True)
            r = kabsch_proper((x - cx).unsqueeze(0), (y - cy).unsqueeze(0))[0]
            out[i][sel] = (((x - cx) @ r.transpose(-1, -2)) + cy).to(out.dtype)
    return out
