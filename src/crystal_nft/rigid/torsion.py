"""Torsion-space refinement: give back flexibility without giving back chirality.

The rigid route buys exact stereocontrol and pays ~1.5 EMD PDD, because the
model's own predicted conformation is thrown away and replaced by a gas-phase
embedding. This module recovers that conformation while keeping the guarantee.

Why torsions specifically:

* Rotating about a single acyclic bond leaves every 1-2 and 1-3 distance fixed,
  so bond lengths and angles stay at ETKDG's ideal values.
* The four substituents at a tetrahedral centre keep their relative arrangement
  under any such rotation, so the sign of the chiral volume is INVARIANT. No
  torsion, at any angle, can turn R into S.
* Ring bonds are excluded, so ring closure is never violated -- rings move as
  rigid units. This is the standard ETKDG/OMEGA/docking factorisation.

The fit target is the model's OWN sampled molecule -- no reference geometry.
The generator predicted a conformation; we keep its torsions and take only the
ideal internal geometry from the embedding.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor

Torsion = Tuple[int, int, Tensor]      # (anchor, axis-end, moving-atom mask)


def rotatable_torsions(fm) -> List[Torsion]:
    """Acyclic single bonds with something to move on both sides."""
    from rdkit import Chem

    n = fm.GetNumAtoms()
    adj: list[list[int]] = [[] for _ in range(n)]
    for b in fm.GetBonds():
        adj[b.GetBeginAtomIdx()].append(b.GetEndAtomIdx())
        adj[b.GetEndAtomIdx()].append(b.GetBeginAtomIdx())

    out: List[Torsion] = []
    for b in fm.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if len(adj[i]) < 2 or len(adj[j]) < 2:
            continue                      # terminal bond: rotation is a no-op
        # atoms reachable from j without crossing back through i
        seen = {i, j}
        stack, moving = [j], []
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in seen:
                    continue
                seen.add(v)
                moving.append(v)
                stack.append(v)
        if not moving or len(moving) >= n - 1:
            continue
        m = torch.zeros(n, dtype=torch.bool)
        m[moving] = True
        out.append((i, j, m))
    return out


def intramolecular_bounds(fm):
    """(mask, lb) over atom pairs at least 4 bonds apart, with covalent bounds.

    Torsion fitting against a distorted target will happily fold a molecule onto
    itself: RMSD does not care, but PoseBusters' internal-clash check does
    (measured: PB 0.99 -> 0.87 without this term). Only 1-4-and-beyond pairs are
    included -- 1-2 and 1-3 distances are invariant under torsion anyway.
    """
    from rdkit import Chem
    from clari.chem import element_radii

    n = fm.GetNumAtoms()
    d = torch.from_numpy(Chem.GetDistanceMatrix(fm)).float()
    far = d >= 4.0
    rad = torch.tensor([element_radii(a.GetAtomicNum(), "cov") for a in fm.GetAtoms()],
                       dtype=torch.float32)
    return far, rad.unsqueeze(-1) + rad


def _rodrigues(axis: Tensor, theta: Tensor) -> Tensor:
    k = axis / axis.norm().clamp_min(1e-12)
    K = torch.zeros(3, 3, dtype=axis.dtype, device=axis.device)
    K[0, 1], K[0, 2] = -k[2], k[1]
    K[1, 0], K[1, 2] = k[2], -k[0]
    K[2, 0], K[2, 1] = -k[1], k[0]
    return (torch.eye(3, dtype=axis.dtype, device=axis.device)
            + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K))


def apply_torsions(coords: Tensor, tors: List[Torsion], angles: Tensor) -> Tensor:
    """Rotate about each acyclic single bond in turn. Chirality-invariant."""
    out = coords
    for (i, j, m), th in zip(tors, angles, strict=False):
        p_i = out[i]
        R = _rodrigues(out[j] - p_i, th)
        moved = (out[m] - p_i) @ R.transpose(0, 1) + p_i
        out = out.masked_scatter(m.unsqueeze(-1).to(out.device), moved.reshape(-1))
    return out


def fit_torsions_to(
    conformer: Tensor,
    target: Tensor,
    tors: List[Torsion],
    *,
    steps: int = 120,
    lr: float = 0.15,
    steric: tuple | None = None,
    steric_weight: float = 4.0,
) -> Tensor:
    """Torsion angles of `conformer` that best reproduce `target`'s shape.

    Returns coordinates with ETKDG's bond lengths/angles and the target's
    torsions, rigidly superposed onto `target`.
    """
    from crystal_nft.rigid.kabsch import kabsch_proper

    if not tors:
        return conformer
    with torch.inference_mode(False):
        x = conformer.detach().clone().double()
        y = target.detach().clone().double()
        y = y - y.mean(0, keepdim=True)
        th = torch.zeros(len(tors), dtype=x.dtype, device=x.device, requires_grad=True)
        opt = torch.optim.Adam([th], lr=lr)
        best, best_v = x, float("inf")
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            p = apply_torsions(x, tors, th)
            p = p - p.mean(0, keepdim=True)
            R = kabsch_proper(p.unsqueeze(0), y.unsqueeze(0))[0]
            aligned = p @ R.transpose(0, 1)
            loss = ((aligned - y) ** 2).sum(-1).mean()
            if steric is not None:
                far, lb = steric
                dd = torch.cdist(p, p)
                pen = torch.relu(lb.to(dd) - dd) * far.to(dd)
                loss = loss + steric_weight * (pen ** 2).sum()
            v = float(loss.detach())
            if v < best_v:
                best_v, best = v, aligned.detach().clone()
            loss.backward()
            opt.step()
        return (best + target.mean(0, keepdim=True).double()).to(conformer.dtype)
