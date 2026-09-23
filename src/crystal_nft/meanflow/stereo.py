"""Tetrahedral stereochemistry for CrystAF: CIP conditioning + chiral-volume loss.

Clari conditions the DiT on the molecular *graph* only (`atom_feats` is radii /
charge / degree / adjacent bond types, `bonds` is bond order + topological
distance).  Every one of those is isomorphic between two enantiomers, so the
generator has no way to know which one it is supposed to pack.  This module adds
the missing bit, following two papers:

* :func:`find_stereo_centers_full` — a per-atom CIP tag (``R`` / ``S``) at every
  tetrahedral stereocentre, plus the CIP-ordered substituent frame.  Clari's DiT
  has no positional encoding and is permutation-equivariant over atoms, so the
  label has to be *canonical*: an atom-order parity would be unreadable, while
  CIP priority is a function of the graph the model already sees.
* :func:`build_stereo_pair_edges` — LoQI's graph augmentation (Nikitin et al.):
  the lowest-CIP substituent is linked to the other three, and those three form
  a *directed* cycle oriented by the configuration.
* :func:`chiral_volumes` — the scale-normalised signed triple product, the same
  ``tau`` PCFM (Cai et al.) uses.  Its sign is the handedness.  The loss and the
  metric compare a prediction against a reference *in the same frame*, so the
  frame cancels and they need no canonical order; only the targets used by
  inference-time correction do, and those read the sign in the canonical CIP
  frame — which is the R/S label (99.2% agreement with RDKit's own label on the
  CSD val split; the rest are RDKit CIP edge cases).

Conditioning on CIP tags is the same class of information Clari already gets for
free (bond orders, formal charges): it describes the *molecule*, not the target
crystal.  A CSP input is a molecule with defined stereochemistry.  The one
caveat is racemic cells: tags are attached per molecule, so for those the
conditioning also fixes which site holds which enantiomer.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Per-atom conditioning tags.
TAG_NONE = 0
TAG_R = 1
TAG_S = 2
TAG_UNDEF = 3  # a stereocentre RDKit could not label
N_STEREO_TAGS = 4

_EPS = 1e-8


# --------------------------------------------------------------------------
# Geometry: signed chiral volume
# --------------------------------------------------------------------------
def chiral_volumes(coords: Tensor, centers: Tensor, *, eps: float = _EPS) -> Tensor:
    """Scale-normalised signed triple product at each stereocentre.

    ``coords``  (..., N, 3) cartesian atom positions.
    ``centers`` (..., M, 4) long ``[centre, n1, n2, n3]`` atom indices.

    Returns (..., M) in [-1, 1]: the triple product of the three unit vectors
    from the centre to its three lowest-index neighbours.  Sign = handedness,
    magnitude = how far the centre is from planar.
    """
    if centers.numel() == 0:
        return coords.new_zeros(centers.shape[:-1])
    if coords.ndim == 2:
        p = coords[centers]  # (M, 4, 3)
    else:
        b, m, _ = centers.shape
        idx = centers.reshape(b, m * 4, 1).expand(b, m * 4, 3)
        p = torch.gather(coords, 1, idx).reshape(b, m, 4, 3)
    e = p[..., 1:, :] - p[..., :1, :]  # (..., M, 3, 3)
    n = e.norm(dim=-1).clamp_min(eps)  # (..., M, 3)
    e = e / n.unsqueeze(-1)
    # Explicit triple product rather than ``linalg.det``: the determinant
    # backward is undefined for a singular matrix and PyTorch returns a zero
    # gradient there, which would silently kill the hinge exactly at the planar
    # (maximally ambiguous) centres the margin exists to push off the plane.
    return (torch.linalg.cross(e[..., 0, :], e[..., 1, :]) * e[..., 2, :]).sum(-1)


def chiral_signs(coords: Tensor, centers: Tensor) -> Tensor:
    """``sign(chiral_volumes)`` with 0 mapped to +1 so it is always ±1."""
    v = chiral_volumes(coords, centers)
    return torch.where(v < 0, -torch.ones_like(v), torch.ones_like(v))


# --------------------------------------------------------------------------
# RDKit: locate stereocentres and read their CIP labels
# --------------------------------------------------------------------------
_CENTER_CACHE: dict[str, tuple] = {}


def _silence_rdkit() -> None:
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:  # noqa: BLE001 — RDKit is optional
        pass


def _use_accurate_cip() -> bool:
    return os.environ.get("CRYSTAF_ACCURATE_CIP", "0").strip() not in ("", "0", "false")


def _neighbors_from_bonds(bonds: Tensor, idx: int) -> list[int]:
    row = bonds[idx]
    nbr = torch.nonzero(row > 0, as_tuple=False).flatten().tolist()
    return sorted(n for n in nbr if n != idx)


def _cip_rank(Chem, mol, idx: int) -> Optional[int]:
    atom = mol.GetAtomWithIdx(int(idx))
    if not atom.HasProp("_CIPRank"):
        return None
    return int(atom.GetProp("_CIPRank"))


class _CpuCell:
    """CPU snapshot of the fields RDKit perception needs.

    Taken once per crystal: the training batch lives on CUDA, and touching it
    element-by-element from Python both fails (``Tensor.numpy()`` raises on a
    CUDA tensor) and would sync on every atom.
    """

    __slots__ = ("atom_nums", "atom_charges", "bonds", "coords", "body_ids", "num_atoms")

    def __init__(self, crystal):
        self.atom_nums = crystal.atom_nums.detach().cpu()
        self.atom_charges = crystal.atom_charges.detach().cpu()
        self.bonds = crystal.bonds.detach().cpu()
        self.coords = crystal.coords.detach().float().cpu()
        self.body_ids = crystal.body_ids.detach().cpu()
        self.num_atoms = int(crystal.num_atoms)


def _body_spans(cell: "_CpuCell") -> list[tuple[int, int]]:
    """``(offset, size)`` of every molecule; bodies are contiguous in atom order."""
    _, counts = torch.unique_consecutive(cell.body_ids, return_counts=True)
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    return [(int(o), int(c)) for o, c in zip(offsets, counts, strict=False)]


def _body_signature(cell: "_CpuCell", start: int, size: int) -> bytes:
    """Identity of a molecule: element sequence + its bond submatrix."""
    nums = cell.atom_nums[start : start + size].to(torch.int32)
    sub = cell.bonds[start : start + size, start : start + size].to(torch.int32)
    return nums.numpy().tobytes() + b"|" + sub.numpy().tobytes()


def _body_rdmol(Chem, cell: "_CpuCell", start: int, size: int):
    """Build an RDKit mol for a single molecule of the cell (with its conformer)."""
    from clari.chem import INDEX_TO_BOND

    mol = Chem.RWMol()
    nums = cell.atom_nums[start : start + size].tolist()
    charges = cell.atom_charges[start : start + size].tolist()
    for i in range(size):
        atom = Chem.Atom(int(nums[i]))
        atom.SetFormalCharge(int(charges[i]))
        mol.AddAtom(atom)
    sub = cell.bonds[start : start + size, start : start + size]
    for u, v in zip(*torch.nonzero(sub > 0, as_tuple=True), strict=False):
        u, v = int(u), int(v)
        if u < v:
            mol.AddBond(u, v, INDEX_TO_BOND[int(sub[u, v])])
    coords = cell.coords[start : start + size].tolist()
    conf = Chem.Conformer(size)
    for i in range(size):
        conf.SetAtomPosition(i, (coords[i][0], coords[i][1], coords[i][2]))
    mol.AddConformer(conf)
    return mol.GetMol()


def _max_body_atoms() -> int:
    return int(os.environ.get("CRYSTAF_STEREO_MAX_ATOMS", "160"))


def _tetrahedral_atoms(Chem, mol) -> list[int]:
    """Atom indices of potential tetrahedral stereocentres.

    Uses ``FindPotentialStereo`` (stereo *perception*) rather than
    ``FindMolChiralCenters(useLegacyImplementation=False)``, which internally
    runs the accurate CIP labeller: that is both far slower and raises
    ``Non integer-order bonds are not allowed`` on any aromatic ring.
    """
    try:
        info = Chem.FindPotentialStereo(mol)
        want = getattr(getattr(Chem, "StereoType", None), "Atom_Tetrahedral", None)
        return [
            int(e.centeredOn)
            for e in info
            if (e.type == want if want is not None else str(e.type) == "Atom_Tetrahedral")
        ]
    except Exception:  # noqa: BLE001 — fall back to the legacy perception
        found = Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=True
        )
        return [int(idx) for idx, _ in found]


def _centers_for_one_body(Chem, cell: "_CpuCell", start: int, size: int):
    """``[(idx, p1, p2, p3, low, has_frame), ...]`` in *body-local* indices."""
    mol = _body_rdmol(Chem, cell, start, size)
    Chem.SanitizeMol(mol, catchErrors=True)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    if _use_accurate_cip():
        from rdkit.Chem import rdCIPLabeler

        rdCIPLabeler.AssignCIPLabels(mol)
    else:
        # Legacy CIP ranking: sets `_CIPRank`, is fast, and tolerates aromatics.
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    found = _tetrahedral_atoms(Chem, mol)
    if not found:
        return []
    sub = cell.bonds[start : start + size, start : start + size]
    out = []
    for idx in found:
        if idx >= size:
            continue
        nbr = _neighbors_from_bonds(sub, idx)
        if len(nbr) < 3:
            continue
        frame, low, has_frame = nbr[:3], -1, False
        if len(nbr) >= 4:
            ranks = [(_cip_rank(Chem, mol, n), n) for n in nbr]
            if all(r is not None for r, _ in ranks) and len({r for r, _ in ranks}) == len(ranks):
                ranks.sort(reverse=True)  # highest CIP priority first
                frame = [n for _, n in ranks[:3]]
                low = int(ranks[3][1])
                has_frame = True
        out.append((idx, frame[0], frame[1], frame[2], low, has_frame))
    return out


_WARNED_ERRORS: set[str] = set()


def _warn_once(exc: Exception, crystal) -> None:
    """Report a swallowed perception failure once per error kind.

    Returning ``None`` on any exception silently means "this crystal has no
    stereochemistry", which has already hidden two real bugs (the CIP labeller
    rejecting aromatic bonds, and ``Tensor.numpy()`` on a CUDA batch).  Never
    fail silently again.
    """
    key = f"{type(exc).__name__}: {str(exc)[:120]}"
    if key in _WARNED_ERRORS:
        return
    _WARNED_ERRORS.add(key)
    logger.warning(
        "stereo perception failed (reported once per kind) on %s: %s — "
        "these crystals are being scored as achiral",
        getattr(crystal, "csd_id", "?"),
        key,
    )


def _find_centers_uncached(crystal):
    """``(centers (M,4), low_prio (M,), has_frame (M,))`` — all graph-derived.

    RDKit runs on **one representative molecule per distinct species**, not on
    the whole cell.  A unit cell is Z copies of the same molecule, and stereo
    perception has to break that symmetry across all Z fragments — on a
    350-atom, Z=16 cell that can run for *minutes*, which stalls every other DDP
    rank at the next collective.  Per molecule the graph is ~20-80 atoms and the
    perception is trivial; results are replicated to sibling bodies by offset.

    Everything returned is a function of the constitutional graph only, so it is
    safe to cache per ``csd_id``: Clari's training collate permutes equivalent
    bodies (``Crystal.aligned_perm``) without touching ``bonds``, and a cached
    *label* would go stale for a racemic cell under that permutation.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return None, None, None

    _silence_rdkit()
    max_atoms = _max_body_atoms()
    per_species: dict[bytes, list] = {}
    rows: list[list[int]] = []
    lows: list[int] = []
    frames: list[bool] = []
    t_start = time.perf_counter()
    try:
        cell = _CpuCell(crystal)
        for start, size in _body_spans(cell):
            if size > max_atoms:
                continue  # pathological molecule: drop rather than stall training
            sig = _body_signature(cell, start, size)
            if sig not in per_species:
                per_species[sig] = _centers_for_one_body(Chem, cell, start, size)
            for idx, p1, p2, p3, low, has_frame in per_species[sig]:
                rows.append([start + idx, start + p1, start + p2, start + p3])
                lows.append(-1 if low < 0 else start + low)
                frames.append(has_frame)
    except Exception as exc:  # noqa: BLE001 — malformed graphs happen; log them
        _warn_once(exc, crystal)
        return None, None, None

    # This runs on the critical path before the loss: a slow crystal stalls every
    # other DDP rank at the next collective, which looks like an idle GPU rather
    # than an error.  Make it loud instead of silent.
    elapsed = time.perf_counter() - t_start
    if elapsed > float(os.environ.get("CRYSTAF_STEREO_WARN_SEC", "2.0")):
        logger.warning(
            "slow stereo perception: %.1fs for %s (%d atoms, %d bodies, %d species) "
            "— this blocks the whole DDP step; consider CRYSTAF_STEREO_MAX_ATOMS",
            elapsed,
            getattr(crystal, "csd_id", "?"),
            int(crystal.num_atoms),
            len(per_species),
            len(per_species),
        )
    if not rows:
        return None, None, None
    return (
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(lows, dtype=torch.long),
        torch.tensor(frames, dtype=torch.bool),
    )


def find_stereo_centers_full(crystal):
    """``(centers, atom_tags, label_sign, low_prio)`` for one unbatched Crystal.

    The graph part is cached; ``atom_tags`` and ``label_sign`` are re-derived
    from *this* crystal's coordinates every call (a 3x3 determinant per centre),
    so they always describe the copy in front of us.
    """
    key = getattr(crystal, "csd_id", None)
    if isinstance(key, (list, tuple)):
        key = key[0] if key else None
    if isinstance(key, str) and key in _CENTER_CACHE:
        cached = _CENTER_CACHE[key]
    else:
        cached = _find_centers_uncached(crystal)
        if isinstance(key, str) and key:
            _CENTER_CACHE[key] = tuple(
                None if v is None else v.detach().clone() for v in cached
            )
    centers, low_prio, has_frame = cached
    if centers is None:
        return None, None, None, None
    centers = centers.clone()
    low_prio = low_prio.clone()
    has_frame = has_frame.clone()

    coords = crystal.coords.detach().float().cpu()
    sign = chiral_signs(coords, centers)  # (M,), +1 / -1
    label_sign = torch.where(has_frame, sign, torch.zeros_like(sign))

    n_atoms = int(crystal.num_atoms)
    tags = torch.zeros(n_atoms, dtype=torch.long)
    tag_vals = torch.where(
        has_frame,
        torch.where(
            sign > 0,
            torch.full_like(sign, float(TAG_S)),
            torch.full_like(sign, float(TAG_R)),
        ),
        torch.full_like(sign, float(TAG_UNDEF)),
    ).long()
    tags[centers[:, 0]] = tag_vals
    return centers, tags, label_sign, low_prio


def find_stereo_centers(crystal) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Cached ``(centers, atom_tags)``; see :func:`find_stereo_centers_full`."""
    centers, tags, _, _ = find_stereo_centers_full(crystal)
    return centers, tags


def clear_stereo_cache() -> None:
    _CENTER_CACHE.clear()


# --------------------------------------------------------------------------
# Batched spec
# --------------------------------------------------------------------------
@dataclass
class BatchStereo:
    """Padded per-batch stereocentre bookkeeping."""

    centers: Tensor  # (B, M, 4) long, [centre, p1, p2, p3]
    valid: Tensor  # (B, M) bool
    atom_tags: Tensor  # (B, N) long, 0 for padding / achiral
    n_centers: int  # total valid centres in the batch
    label_sign: Optional[Tensor] = None  # (B, M) float, +1 = S, -1 = R, 0 = unknown
    low_prio: Optional[Tensor] = None  # (B, M) long, lowest-CIP substituent, -1 = unknown

    def to(self, device) -> "BatchStereo":
        return BatchStereo(
            centers=self.centers.to(device),
            valid=self.valid.to(device),
            atom_tags=self.atom_tags.to(device),
            n_centers=self.n_centers,
            label_sign=None if self.label_sign is None else self.label_sign.to(device),
            low_prio=None if self.low_prio is None else self.low_prio.to(device),
        )


def build_batch_stereo(crystal, *, device=None) -> Optional[BatchStereo]:
    """Build a :class:`BatchStereo` from a (possibly batched) conditioning Crystal.

    Returns ``None`` when no stereocentre is found anywhere in the batch, so
    callers can skip the whole stereo path with one check.
    """
    try:
        items = list(crystal.unbatch()) if getattr(crystal, "batched", False) else [crystal]
    except Exception:  # noqa: BLE001
        return None

    per_centers: list[Optional[Tensor]] = []
    per_tags: list[Optional[Tensor]] = []
    per_signs: list[Optional[Tensor]] = []
    per_low: list[Optional[Tensor]] = []
    n_atoms: list[int] = []
    for c in items:
        centers, tags, label_sign, low = find_stereo_centers_full(c)
        per_centers.append(centers)
        per_tags.append(tags)
        per_signs.append(label_sign)
        per_low.append(low)
        n_atoms.append(int(c.num_atoms))

    m_max = max((0 if c is None else int(c.shape[0])) for c in per_centers)
    if m_max == 0:
        return None
    n_max = max(n_atoms) if n_atoms else 0
    b = len(items)

    centers = torch.zeros(b, m_max, 4, dtype=torch.long)
    valid = torch.zeros(b, m_max, dtype=torch.bool)
    atom_tags = torch.zeros(b, n_max, dtype=torch.long)
    label_sign = torch.zeros(b, m_max, dtype=torch.float32)
    low_prio = torch.full((b, m_max), -1, dtype=torch.long)
    total = 0
    for i, (c, t, sg, lo) in enumerate(
        zip(per_centers, per_tags, per_signs, per_low, strict=False)
    ):
        if c is not None and c.shape[0]:
            m = int(c.shape[0])
            centers[i, :m] = c
            valid[i, :m] = True
            total += m
            if sg is not None and sg.numel() == m:
                label_sign[i, :m] = sg
            if lo is not None and lo.numel() == m:
                low_prio[i, :m] = lo
        if t is not None and t.numel():
            atom_tags[i, : t.numel()] = t
    out = BatchStereo(
        centers=centers,
        valid=valid,
        atom_tags=atom_tags,
        n_centers=total,
        label_sign=label_sign,
        low_prio=low_prio,
    )
    return out.to(device) if device is not None else out


# --------------------------------------------------------------------------
# RDKit distance-geometry bond bounds (for PCFM's bond-length constraint)
# --------------------------------------------------------------------------
_BOUNDS_CACHE: dict[str, tuple] = {}


def _bond_bounds_uncached(crystal):
    """``(pairs (P,2), lo (P,), hi (P,))`` for every bonded pair in the cell.

    Bounds come from RDKit's DistanceGeometry on one representative molecule per
    species (same reason as the stereo perception: doing it per cell is slow),
    widened by PoseBusters' ``threshold_bad_bond_length`` so that satisfying this
    is exactly what its ``distance_geometry`` check tests.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDistGeom
    except ImportError:
        return None, None, None

    _silence_rdkit()
    thr = float(os.environ.get("CRYSTAF_PCFM_BOND_SLACK", "0.25"))
    cell = _CpuCell(crystal)
    pairs, lo, hi = [], [], []
    per_species: dict[bytes, tuple] = {}
    try:
        for start, size in _body_spans(cell):
            if size > _max_body_atoms():
                continue
            sig = _body_signature(cell, start, size)
            if sig not in per_species:
                mol = _body_rdmol(Chem, cell, start, size)
                Chem.SanitizeMol(mol, catchErrors=True)
                # GetMoleculeBoundsMatrix needs ring info; a partially-failed
                # sanitize leaves it uninitialised ("RingInfo not initialized").
                mol.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(mol)
                bm = rdDistGeom.GetMoleculeBoundsMatrix(
                    mol, set15bounds=True, scaleVDW=True, doTriangleSmoothing=True
                )
                sub = cell.bonds[start : start + size, start : start + size]
                loc = []
                for u, v in zip(*torch.nonzero(sub > 0, as_tuple=True), strict=False):
                    u, v = int(u), int(v)
                    if u >= v:
                        continue
                    # RDKit stores upper bounds above the diagonal, lower below.
                    hi_uv = float(bm[min(u, v)][max(u, v)])
                    lo_uv = float(bm[max(u, v)][min(u, v)])
                    if not (lo_uv > 0 and hi_uv >= lo_uv):
                        continue
                    loc.append((u, v, lo_uv * (1.0 - thr), hi_uv * (1.0 + thr)))
                per_species[sig] = tuple(loc)
            for u, v, l, h in per_species[sig]:
                pairs.append([start + u, start + v]); lo.append(l); hi.append(h)
    except Exception as exc:  # noqa: BLE001
        _warn_once(exc, crystal)
        return None, None, None
    if not pairs:
        return None, None, None
    return (torch.tensor(pairs, dtype=torch.long),
            torch.tensor(lo, dtype=torch.float32),
            torch.tensor(hi, dtype=torch.float32))


def bond_bounds(crystal):
    """Cached ``(pairs, lo, hi)``; graph-derived, so safe to key on ``csd_id``."""
    key = getattr(crystal, "csd_id", None)
    if isinstance(key, (list, tuple)):
        key = key[0] if key else None
    if isinstance(key, str) and key in _BOUNDS_CACHE:
        return tuple(None if v is None else v.clone() for v in _BOUNDS_CACHE[key])
    out = _bond_bounds_uncached(crystal)
    if isinstance(key, str) and key:
        _BOUNDS_CACHE[key] = tuple(None if v is None else v.detach().clone() for v in out)
    return out


def build_bond_constraint(crystal, *, device=None):
    """Padded :class:`~crystal_nft.meanflow.pcfm.BondConstraint` for a batch."""
    from crystal_nft.meanflow.pcfm import BondConstraint

    try:
        items = list(crystal.unbatch()) if getattr(crystal, "batched", False) else [crystal]
    except Exception:  # noqa: BLE001
        return None
    per = [bond_bounds(c) for c in items]
    p_max = max((0 if p[0] is None else int(p[0].shape[0])) for p in per)
    if p_max == 0:
        return None
    b = len(items)
    pairs = torch.zeros(b, p_max, 2, dtype=torch.long)
    lo = torch.ones(b, p_max, dtype=torch.float32)
    hi = torch.full((b, p_max), 1e6, dtype=torch.float32)
    valid = torch.zeros(b, p_max, dtype=torch.bool)
    for i, (pr, l, h) in enumerate(per):
        if pr is None:
            continue
        n = int(pr.shape[0])
        pairs[i, :n] = pr; lo[i, :n] = l; hi[i, :n] = h; valid[i, :n] = True
    cst = BondConstraint(pairs=pairs, lo=lo, hi=hi, valid=valid)
    return cst.to(device) if device is not None else cst


_DG_CACHE: dict[bytes, tuple] = {}


def dg_clash_bounds(crystal):
    """Per body: ``(atom_index, lower_bound, pair_mask)`` for the internal-clash check.

    PoseBusters' ``no_internal_clash`` compares atom pairs beyond 1-4 against the
    RDKit distance-geometry *lower* bound, widened by ``threshold_clash`` (0.3).
    Returning those bounds lets a repair optimise exactly the quantity the metric
    tests, rather than a hand-rolled radii proxy.

    Cached per molecular species, like the other perception products -- the
    bounds are graph-derived, so they survive ``aligned_perm``.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDistGeom, rdmolops
    except ImportError:
        return []

    _silence_rdkit()
    cell = _CpuCell(crystal)
    out = []
    for start, size in _body_spans(cell):
        if size > _max_body_atoms():
            continue
        sig = _body_signature(cell, start, size)
        hit = _DG_CACHE.get(sig)
        if hit is None:
            try:
                mol = _body_rdmol(Chem, cell, start, size)
                Chem.SanitizeMol(mol, catchErrors=True)
                mol.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(mol)
                bm = rdDistGeom.GetMoleculeBoundsMatrix(
                    mol, set15bounds=True, scaleVDW=True, doTriangleSmoothing=True
                )
                gd = rdmolops.GetDistanceMatrix(mol)
            except Exception as exc:  # noqa: BLE001
                _warn_once("dg_bounds", exc)
                _DG_CACHE[sig] = (None, None)
                continue
            lo = torch.zeros(size, size, dtype=torch.float32)
            for i in range(size):
                for j in range(i + 1, size):
                    lo[i, j] = lo[j, i] = float(bm[j][i])  # lower bounds below diagonal
            mask = torch.from_numpy(gd >= 4.0)
            _DG_CACHE[sig] = (lo, mask)
            hit = _DG_CACHE[sig]
        lo, mask = hit
        if lo is None:
            continue
        out.append((start, size, lo.clone(), mask.clone()))
    return out


# --------------------------------------------------------------------------
# LoQI-style graph augmentation: R/S as directed auxiliary edges
# --------------------------------------------------------------------------
EDGE_NONE = 0
EDGE_LOW = 1  # undirected: lowest-CIP substituent <-> each of the other three
EDGE_CYCLE_FWD = 2  # directed cycle over p1,p2,p3, oriented by the R/S label
EDGE_CYCLE_BWD = 3
N_STEREO_EDGES = 4


def build_stereo_pair_edges(spec: "BatchStereo", n_atoms: int) -> Optional[Tensor]:
    """(B, N, N) auxiliary-edge types encoding R/S, following LoQI.

    Nikitin et al. encode a stereocentre by connecting the lowest-CIP-priority
    substituent to the other three with undirected edges, and orienting a
    *directed* cycle over those three according to the configuration.  That is
    permutation-equivariant and index-free — unlike an atom-order parity, which
    Clari's DiT could not read, since it has no positional encoding.

    Only centres with a canonical CIP frame (``label_sign != 0``) get edges.
    """
    if spec is None or spec.label_sign is None or spec.low_prio is None:
        return None
    device = spec.centers.device
    b, m, _ = spec.centers.shape
    usable = spec.valid & (spec.label_sign != 0) & (spec.low_prio >= 0)
    if not bool(usable.any()):
        return None
    edges = torch.zeros(b, n_atoms, n_atoms, dtype=torch.long, device=device)
    bi, mi = torch.nonzero(usable, as_tuple=True)
    p1 = spec.centers[bi, mi, 1]
    p2 = spec.centers[bi, mi, 2]
    p3 = spec.centers[bi, mi, 3]
    low = spec.low_prio[bi, mi]
    # label_sign > 0 (S) keeps CIP order; < 0 (R) reverses the cycle.
    flip = spec.label_sign[bi, mi] < 0
    a, c_ = torch.where(flip, p3, p2), torch.where(flip, p2, p3)
    for u, v in ((p1, a), (a, c_), (c_, p1)):
        edges[bi, u, v] = EDGE_CYCLE_FWD
        edges[bi, v, u] = EDGE_CYCLE_BWD
    for other in (p1, p2, p3):
        edges[bi, low, other] = EDGE_LOW
        edges[bi, other, low] = EDGE_LOW
    return edges


# --------------------------------------------------------------------------
# Enantiomer (mirror) augmentation
# --------------------------------------------------------------------------
def mirror_coords_in_place_x(x: Tensor, flip: Tensor) -> Tensor:
    """Return the enantiomorph of the selected crystals in Clari packing.

    ``x`` is ``(B, 3+N, 3)``: rows 0-2 are the lattice, the rest are cartesian
    coordinates.  Inverting the coordinates through the origin while keeping the
    lattice gives the enantiomorphic crystal — a lattice is centrosymmetric, so
    the same basis still describes it, and every interatomic distance (hence
    every pair feature the DiT sees) is unchanged.

    Negating the *whole* of ``x`` would instead flip the basis as well and leave
    the fractional structure untouched — a re-description, not a mirror.
    """
    if not bool(flip.any()):
        return x
    out = x.clone()
    sel = flip.view(-1, *([1] * (x.ndim - 1)))
    out[:, 3:] = torch.where(sel, -x[:, 3:], x[:, 3:])
    return out


def swap_rs_tags(tags: Tensor, flip: Tensor) -> Tensor:
    """R <-> S on the selected rows; ``TAG_NONE`` / ``TAG_UNDEF`` are unchanged."""
    if not bool(flip.any()):
        return tags
    swapped = tags.clone()
    swapped[tags == TAG_R] = TAG_S
    swapped[tags == TAG_S] = TAG_R
    return torch.where(flip.view(-1, 1), swapped, tags)


# --------------------------------------------------------------------------
# Loss + metric
# --------------------------------------------------------------------------
def chiral_hinge_loss(
    pred_coords: Tensor,
    ref_coords: Tensor,
    spec: BatchStereo,
    *,
    margin: float = 0.62,
    sample_weight: Optional[Tensor] = None,
) -> Tensor:
    """Hinge on the handedness of every stereocentre.

    ``relu(margin - s_ref * v_pred)`` where ``s_ref`` is the reference sign and
    ``v_pred`` the normalised chiral volume of the prediction.  Zero once the
    centre is correctly handed *and* clearly non-planar; the reference frame
    cancels, so no canonical neighbour order is needed.

    ``sample_weight`` (B,) re-weights each crystal.  It exists to kill a
    shortcut: at moderate ``r`` the noised state ``x_r`` already reveals the
    handedness, so the model can satisfy this hinge by copying it out of the
    state and the gradient into the CIP-tag embedding collapses to ~0.  Only
    near ``r = 0`` is the tag the *only* available source, so concentrating the
    weight there is what makes the conditioning causal.
    """
    if spec is None or spec.n_centers == 0:
        return pred_coords.new_zeros(())
    s_ref = chiral_signs(ref_coords.detach(), spec.centers)
    v_pred = chiral_volumes(pred_coords, spec.centers)
    pen = torch.relu(float(margin) - s_ref * v_pred)
    w = spec.valid.to(pen.dtype)
    if sample_weight is not None:
        w = w * sample_weight.reshape(-1, 1).to(dtype=w.dtype, device=w.device)
    return (pen * w).sum() / w.sum().clamp_min(1e-6)


@torch.no_grad()
def stereo_agreement(
    pred_coords: Tensor,
    ref_coords: Tensor,
    spec: BatchStereo,
) -> tuple[float, int]:
    """Fraction of stereocentres whose predicted handedness matches the reference."""
    if spec is None or spec.n_centers == 0:
        return float("nan"), 0
    s_ref = chiral_signs(ref_coords, spec.centers)
    s_pred = chiral_signs(pred_coords, spec.centers)
    ok = ((s_ref == s_pred) & spec.valid).sum().item()
    n = int(spec.valid.sum().item())
    return (ok / n if n else float("nan")), n


# --------------------------------------------------------------------------
# Active-tag context (avoids threading a new kwarg through every call site)
# --------------------------------------------------------------------------
def cell_stereo_descriptor(spec) -> Optional[Tensor]:
    """(B, 4) per-crystal handedness summary for the FiLM/``cond`` path.

    ``[mean_sign, frac_R, frac_S, has_centres]`` -- ``mean_sign`` is +-1 for an
    enantiopure cell and ~0 for a racemate, which is exactly the information a
    per-crystal channel can carry.
    """
    if spec is None or spec.label_sign is None:
        return None
    v = spec.valid & (spec.label_sign != 0)
    n = v.sum(dim=1).clamp_min(1).float()
    sgn = (spec.label_sign * v).sum(dim=1) / n
    frac_s = ((spec.label_sign > 0) & v).sum(dim=1).float() / n
    frac_r = ((spec.label_sign < 0) & v).sum(dim=1).float() / n
    has = (v.any(dim=1)).float()
    return torch.stack([sgn, frac_r, frac_s, has], dim=-1)


@dataclass
class StereoConditioning:
    """What the DiT is told about stereochemistry for the current batch."""

    atom_tags: Optional[Tensor] = None  # (B, N) long CIP tags
    pair_edges: Optional[Tensor] = None  # (B, N, N) long LoQI auxiliary edges
    cell_desc: Optional[Tensor] = None  # (B, 4) per-crystal handedness summary
    centers: Optional[Tensor] = None  # (B, M, 4) long [centre, n1, n2, n3]
    label_sign: Optional[Tensor] = None  # (B, M) target handedness in {-1,0,+1}
    valid: Optional[Tensor] = None  # (B, M) bool


_ACTIVE_TAGS: Optional[object] = None


def set_active_stereo_tags(tags) -> None:
    global _ACTIVE_TAGS
    _ACTIVE_TAGS = tags


def get_active_stereo_tags() -> Optional[Tensor]:
    if isinstance(_ACTIVE_TAGS, StereoConditioning):
        return _ACTIVE_TAGS.atom_tags
    return _ACTIVE_TAGS


def get_active_stereo_pair_edges() -> Optional[Tensor]:
    if isinstance(_ACTIVE_TAGS, StereoConditioning):
        return _ACTIVE_TAGS.pair_edges
    return None


def get_active_stereo_cell_desc() -> Optional[Tensor]:
    if isinstance(_ACTIVE_TAGS, StereoConditioning):
        return _ACTIVE_TAGS.cell_desc
    return None


def get_active_stereo_geometry():
    """(centers, label_sign, valid) for the parity-sensitive velocity branch."""
    if isinstance(_ACTIVE_TAGS, StereoConditioning):
        return _ACTIVE_TAGS.centers, _ACTIVE_TAGS.label_sign, _ACTIVE_TAGS.valid
    return None, None, None


@contextmanager
def active_stereo_tags(tags):
    """Publish stereo conditioning to every DiT forward inside the block.

    Accepts a plain ``(B, N)`` tag tensor or a :class:`StereoConditioning`.
    Threading a new kwarg instead would touch every sampler / loss / interface
    signature in the package.
    """
    global _ACTIVE_TAGS
    prev = _ACTIVE_TAGS
    _ACTIVE_TAGS = tags
    try:
        yield
    finally:
        _ACTIVE_TAGS = prev


def build_stereo_conditioning(spec, n_atoms: int, *, pair_edges: bool = True):
    """Node tags + (optionally) LoQI auxiliary edges for a batch."""
    if spec is None:
        return None
    return StereoConditioning(
        atom_tags=spec.atom_tags,
        pair_edges=build_stereo_pair_edges(spec, n_atoms) if pair_edges else None,
        cell_desc=cell_stereo_descriptor(spec),
        centers=spec.centers,
        label_sign=spec.label_sign,
        valid=spec.valid,
    )

@torch.no_grad()
def chiral_prior_align(coords: Tensor, spec, *, eps: float = _EPS) -> Tensor:
    """Give the PRIOR the requested handedness, for free, by permuting noise.

    Flipping a stereocentre is a permutation, not a rotation -- no rotation
    changes chirality -- so correcting handedness once atoms are nearly placed
    is inherently a large rearrangement, and it leaves bond/angle distortion.
    Measured across six configurations: any hinge strong enough to make
    chirality survive integration pins PB at ~69 vs an 82.26 baseline, while
    clash (intermolecular) moves freely, and every softening trades chirality
    away one-for-one.

    But at t = 0 the state is i.i.d. Gaussian noise, which is EXCHANGEABLE:
    swapping two atoms' noise rows leaves the prior distribution exactly
    invariant while flipping the sign of the stereocentre's triple product.
    So the requested handedness can be installed at the one point in the
    trajectory where rearrangement costs nothing, and the flow then only has to
    preserve it rather than invent it.
    """
    if spec is None or spec.centers is None or spec.label_sign is None:
        return coords
    if spec.centers.numel() == 0:
        return coords
    out = coords.clone()
    centers, sign = spec.centers, spec.label_sign
    if centers.ndim == 2:  # (M, 4) -> (1, M, 4)
        centers = centers.unsqueeze(0)
        sign = sign.unsqueeze(0)
    b = out.shape[0]
    if centers.shape[0] == 1 and b > 1:
        centers = centers.expand(b, -1, -1)
        sign = sign.expand(b, -1)
    v = chiral_volumes(out.float(), centers)
    active = (sign != 0)
    if spec.valid is not None:
        val = spec.valid
        if val.ndim == 1:
            val = val.unsqueeze(0)
        if val.shape[0] == 1 and b > 1:
            val = val.expand(b, -1)
        active = active & val
    wrong = active & (torch.sign(v) != sign) & (v.abs() > eps)
    if not bool(wrong.any()):
        return out
    bi, mi = torch.nonzero(wrong, as_tuple=True)
    # Swap the first two substituents: exchanges two rows of i.i.d. noise, so the
    # prior is unchanged in distribution, and inverts the triple product sign.
    a_idx = centers[bi, mi, 1]
    c_idx = centers[bi, mi, 2]
    tmp = out[bi, a_idx].clone()
    out[bi, a_idx] = out[bi, c_idx]
    out[bi, c_idx] = tmp
    return out
