"""Apply the rigid-body representation to a finished CrystAF sample.

Conformer sources, and what each one is allowed to claim:

* ``oracle`` -- the reference crystal's own molecular geometry. This is a
  CEILING measurement, not a method: it answers "if the conformer were perfect,
  what do the four packing metrics become?" It uses reference *internal*
  geometry, never reference packing (lattice, centroids and orientations all
  still come from the model), but it is still information the Table1 baseline
  did not get. Numbers from this source are NOT comparable to the CrystAF
  8/16/50 table and must be labelled as a ceiling.
* ``etkdg``  -- RDKit distance-geometry conformer built from the molecular graph
  plus the REQUESTED stereo tags. This is MolCrystalFlow's actual setting (they
  take conformers from OMEGA/ETKDGv2) and uses no reference geometry, so it is
  a fair method. Not implemented until the ceiling justifies it.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
from torch import Tensor

from crystal_nft.rigid.kabsch import rigid_fit_bodies


class RigidReference:
    """The conformer source published for the duration of one sampling call."""

    def __init__(self, conformer: Tensor, body_ids: Tensor, mask: Optional[Tensor] = None,
                 source: str = "oracle", atom_nums: Optional[Tensor] = None,
                 relax: bool = False, relax_steps: int = 250, relax_margin: float = 0.05,
                 conformer_ok: Optional[Tensor] = None, torsions=None):
        self.conformer = conformer          # (B, N, 3), coords / COORD_NORM
        self.body_ids = body_ids            # (B, N) or (N,)
        self.mask = mask                    # (B, N) bool or None
        self.source = source
        self.atom_nums = atom_nums          # (B, N) long, needed for covalent radii
        self.relax = relax
        self.relax_steps = relax_steps
        self.relax_margin = relax_margin
        self.conformer_ok = conformer_ok   # (B, N) bool; False = no conformer, leave alone
        self.torsions = torsions           # list[B] of (atom_idx, torsion list) per body
        self.torsion_steps = 120


_ACTIVE: Optional[RigidReference] = None


def get_active_rigid() -> Optional[RigidReference]:
    return _ACTIVE


@contextmanager
def active_rigid(ref: Optional[RigidReference]):
    global _ACTIVE
    prev = _ACTIVE
    _ACTIVE = ref
    try:
        yield ref
    finally:
        _ACTIVE = prev


class RigidProjector:
    """`EnantiomerSelector`-shaped: projects a sample onto the rigid manifold.

    `select_prior` is inert -- the whole point is that nothing has to be decided
    up front, because handedness is carried by the conformer rather than chosen
    by the flow.
    """

    def __init__(self, ref: Optional[RigidReference] = None):
        self._ref = ref

    def select_prior(self, x0: Tensor, spec, *, generator=None) -> Tensor:
        return x0

    def select_output(self, x: Tensor, spec) -> Tensor:
        ref = self._ref or get_active_rigid()
        if ref is None:
            return x
        out = x.clone()
        if ref.torsions is not None:
            out[:, 3:, :] = _fit_with_torsions(x[:, 3:, :], ref)
            if ref.relax:
                out = _relax_pack(out, ref)
            return out
        out[:, 3:, :] = rigid_fit_bodies(
            x[:, 3:, :].float(), ref.conformer.float().to(x.device),
            ref.body_ids.to(x.device),
            None if ref.mask is None else ref.mask.to(x.device),
            None if ref.conformer_ok is None else ref.conformer_ok.to(x.device),
        ).to(x.dtype)
        if ref.relax:
            out = _relax_pack(out, ref)
        return out


def _relax_pack(x: Tensor, ref: RigidReference) -> Tensor:
    """Clear inter-body overlaps with 6 DOF per molecule (see rigid/relax.py)."""
    from clari.chem import Crystal

    from crystal_nft.rigid.relax import relax_bodies

    if ref.atom_nums is None:
        raise ValueError("relax needs atom_nums for covalent radii")
    from clari.chem import element_radii

    scale = float(Crystal.COORD_NORM)
    out = x.clone()
    for i in range(x.shape[0]):
        m = (ref.mask[i] if ref.mask is not None else
             torch.ones(x.shape[1] - 3, dtype=torch.bool, device=x.device))
        if int(m.sum()) < 2:
            continue
        z = ref.atom_nums[i][m].tolist()
        bid = (ref.body_ids[i] if ref.body_ids.dim() > 1 else ref.body_ids)[m]
        if int(torch.unique(bid).numel()) < 2:      # one molecule: nothing to clear
            continue
        rad = torch.tensor([element_radii(int(a), "cov") for a in z],
                           dtype=torch.float64, device=x.device)
        lb = rad.unsqueeze(-1) + rad
        coords = x[i, 3:, :][m].double() * scale
        lat = x[i, :3, :].double() * 2.0 * scale
        fixed = relax_bodies(coords, lat, bid, lb,
                             margin=ref.relax_margin, steps=ref.relax_steps)
        buf = out[i, 3:, :].clone()
        buf[m] = (fixed / scale).to(out.dtype)
        out[i, 3:, :] = buf
    return out


def project_z(z: Tensor) -> Tensor:
    """Trunk entry point. Exact no-op unless a reference is active."""
    ref = get_active_rigid()
    if ref is None:
        return z
    return RigidProjector(ref).select_output(z, None)


def _fit_with_torsions(sampled: Tensor, ref: RigidReference) -> Tensor:
    """ETKDG's ideal bond lengths/angles, the MODEL's torsions, locked chirality.

    Runs on CPU: the molecules are tiny, and per-body kernel-launch overhead on
    GPU dominates the actual arithmetic.
    """
    from crystal_nft.rigid.torsion import fit_torsions_to

    dev, dt = sampled.device, sampled.dtype
    # Eval runs under torch.inference_mode. `.clone()` alone is not enough here:
    # the tensors are still inference tensors and blow up as "cannot be saved
    # for backward" inside the optimiser. Round-trip through numpy INSIDE
    # inference_mode(False) to get genuinely normal tensors.
    # `with torch.inference_mode(), ..., _active_rigid(batch)` ENTERS inference
    # mode before the later context managers are evaluated, so the conformer,
    # the body indices and the torsion masks are all inference tensors. Every
    # one of them has to be materialised before it touches autograd -- the mask
    # is the one that bites, via masked_scatter.
    with torch.inference_mode(False):
        out = torch.from_numpy(sampled.detach().cpu().float().numpy()).clone()
        conf = torch.from_numpy(ref.conformer.detach().cpu().float().numpy()).clone()
        for i, bodies in enumerate(ref.torsions or []):
            if i >= out.shape[0]:
                break
            for idxs, tors, steric in bodies:
                idxs = torch.from_numpy(idxs.cpu().numpy()).clone()
                if idxs.numel() < 3 or int(idxs.max()) >= out.shape[1]:
                    continue
                tors = [(a, b, torch.from_numpy(m.cpu().numpy()).clone())
                        for a, b, m in tors]
                far, lb = steric
                steric = (torch.from_numpy(far.cpu().numpy()).clone(),
                          torch.from_numpy(lb.cpu().numpy()).clone())
                fitted = fit_torsions_to(conf[i][idxs], out[i][idxs], tors,
                                         steps=int(ref.torsion_steps), steric=steric)
                out[i][idxs] = fitted.float()
    return out.to(device=dev, dtype=dt)
