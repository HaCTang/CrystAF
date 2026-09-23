"""Stereochemistry featurization and enantiomer-safe augmentations for flexible molecules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ChiralInfo:
    """Per-ASU chiral tags from RDKit (when available)."""

    asu_chiral: Tensor  # (N,) int: 0=achiral, 1=R-ish, 2=S-ish (CIP when possible)
    n_chiral_centers: int
    n_defined: int


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDistGeom
    except ImportError as exc:
        raise ImportError("RDKit is required for chirality featurization") from exc
    return Chem, rdDistGeom


def chiral_tags_from_smiles(smiles: str) -> ChiralInfo:
    Chem, _ = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    n_asu = mol.GetNumAtoms()
    tags = torch.zeros(n_asu, dtype=torch.long)
    n_chiral = 0
    n_defined = 0
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        chiral = atom.GetChiralTag()
        if chiral == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            continue
        n_chiral += 1
        if chiral in (Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,):
            tags[idx] = 1
            n_defined += 1
        elif chiral in (Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,):
            tags[idx] = 2
            n_defined += 1
        else:
            tags[idx] = 3
            n_defined += 1
    return ChiralInfo(asu_chiral=tags, n_chiral_centers=n_chiral, n_defined=n_defined)


def global_chiral_descriptor(info: ChiralInfo) -> Tensor:
    """Fixed 8-d summary for conditioning (counts + normalized tag histogram)."""
    tags = info.asu_chiral.float()
    n = max(int(tags.numel()), 1)
    hist = torch.zeros(4)
    for k in range(4):
        hist[k] = (info.asu_chiral == k).float().sum() / n
    vec = torch.tensor(
        [
            info.n_chiral_centers,
            info.n_defined,
            hist[1],
            hist[2],
            hist[3],
            float(info.n_chiral_centers > 0),
            float(info.n_defined == info.n_chiral_centers and info.n_chiral_centers > 0),
            hist[1] - hist[2],
        ],
        dtype=torch.float32,
    )
    return vec


class ChiralConditioning(nn.Module):
    """Maps global stereochemistry vector to dim_cond bias for DiT."""

    def __init__(self, dim_cond: int, in_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, dim_cond),
            nn.SiLU(),
            nn.Linear(dim_cond, dim_cond),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, desc: Tensor) -> Tensor:
        return self.net(desc)


def random_enantiomer_flip(x: Tensor, chiral_asu: Tensor, body_ids: Tensor, p: float = 0.5) -> Tensor:
    """
    Mirror coordinates within each chiral body (flexible torsions preserved aside from reflection).

    x: (B, 3+N, 3) packed crystal state; only coord rows (:, 3:, :) are modified.
    chiral_asu: (N_asu,) tags on asymmetric unit atoms mapped to coord rows via body_ids.
    """
    if p <= 0.0 or chiral_asu.numel() == 0:
        return x
    out = x.clone()
    B = x.shape[0]
    device = x.device
    for b in range(B):
        if torch.rand(1, device=device).item() >= p:
            continue
        bodies = torch.unique(body_ids[b])
        for body in bodies.tolist():
            mask = body_ids[b] == body
            if not mask.any():
                continue
            if (chiral_asu[mask] > 0).any():
                coords = out[b, 3:][mask]
                center = coords.mean(dim=0, keepdim=True)
                out[b, 3:][mask] = center - (coords - center)
    return out


def chiral_descriptor_from_smiles(smiles: str, device: torch.device) -> Tensor:
    info = chiral_tags_from_smiles(smiles)
    return global_chiral_descriptor(info).to(device)


def chiral_bias_from_descriptor(
    module: ChiralConditioning,
    desc: Tensor,
    batch_size: int,
) -> Tensor:
    if desc.ndim == 1:
        desc = desc.unsqueeze(0)
    if desc.shape[0] == 1 and batch_size > 1:
        desc = desc.expand(batch_size, -1)
    return module(desc)


_CHIRAL_TAG_CACHE: dict[str, Tensor | None] = {}


def _silence_rdkit():
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass


def _chiral_tag_int(Chem, atom) -> int:
    chiral = atom.GetChiralTag()
    if chiral in (Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,):
        return 1
    if chiral in (Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,):
        return 2
    if chiral != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
        return 3
    return 0


def _assign_stereo_best_effort(Chem, mol) -> None:
    """Prefer 3D CIP assignment for crystal coords; fall back to graph stereo."""
    try:
        if mol.GetNumConformers() > 0:
            Chem.AssignStereochemistryFrom3D(mol)
    except Exception:
        pass
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)


def chiral_asu_tags_from_crystal(crystal) -> Tensor | None:
    """Best-effort CIP-ish tags from RDKit mols on a (possibly batched) Crystal."""
    try:
        Chem, _ = _require_rdkit()
    except ImportError:
        return None
    cache_key = None
    try:
        cid = getattr(crystal, "csd_id", None)
        if isinstance(cid, (list, tuple)):
            cid = cid[0] if cid else None
        if isinstance(cid, str) and cid:
            cache_key = cid
            if cache_key in _CHIRAL_TAG_CACHE:
                cached = _CHIRAL_TAG_CACHE[cache_key]
                return cached.clone() if cached is not None else None
    except Exception:
        cache_key = None
    _silence_rdkit()
    try:
        mols = crystal.to_rdmol() if hasattr(crystal, "to_rdmol") else None
        if mols is None:
            if cache_key is not None:
                _CHIRAL_TAG_CACHE[cache_key] = None
            return None
        if not isinstance(mols, (list, tuple)):
            mols = [mols]
        tags_list = []
        for mol in mols:
            if mol is None:
                continue
            _assign_stereo_best_effort(Chem, mol)
            for atom in mol.GetAtoms():
                tags_list.append(_chiral_tag_int(Chem, atom))
        if not tags_list:
            if cache_key is not None:
                _CHIRAL_TAG_CACHE[cache_key] = None
            return None
        tags = torch.tensor(tags_list, dtype=torch.long)
        if cache_key is not None:
            _CHIRAL_TAG_CACHE[cache_key] = tags.detach().clone()
        return tags
    except Exception:
        if cache_key is not None:
            _CHIRAL_TAG_CACHE[cache_key] = None
        return None


def mirror_packed_coords(x: Tensor) -> Tensor:
    """Reflect fractional/cartesian packed coords about body COM (all bodies)."""
    out = x.clone()
    coords = out[:, 3:]
    center = coords.mean(dim=1, keepdim=True)
    out[:, 3:] = center - (coords - center)
    return out


def enantiomer_consistency_loss(
    pred_x1: Tensor,
    true_x1: Tensor,
    mask: Tensor | None = None,
    *,
    margin: float = 0.0,
) -> Tensor:
    """Penalize predictions closer to the mirrored GT than to the true enantiomer.

    loss = mean(relu(d_direct - d_mirror + margin)) over the batch.
    """
    B = pred_x1.shape[0]
    pred_c = pred_x1[:, 3:]
    true_c = true_x1[:, 3:]
    mirror_c = mirror_packed_coords(true_x1)[:, 3:]
    if mask is not None:
        m = mask.unsqueeze(-1).to(pred_c.dtype)
        denom = m.sum(dim=[1, 2]).clamp_min(1.0)
        d_direct = ((pred_c - true_c).pow(2) * m).sum(dim=[1, 2]) / denom
        d_mirror = ((pred_c - mirror_c).pow(2) * m).sum(dim=[1, 2]) / denom
    else:
        d_direct = (pred_c - true_c).pow(2).mean(dim=[1, 2])
        d_mirror = (pred_c - mirror_c).pow(2).mean(dim=[1, 2])
    return torch.relu(d_direct - d_mirror + margin).mean()


def batch_chiral_descriptors_from_crystals(
    crystals: list,
    device: torch.device,
) -> Tensor | None:
    """Stack 8-d descriptors; returns None if no crystal yields a descriptor."""
    descs = []
    for c in crystals:
        tags = chiral_asu_tags_from_crystal(c)
        if tags is None:
            descs.append(torch.zeros(8, dtype=torch.float32, device=device))
            continue
        info = ChiralInfo(
            asu_chiral=tags,
            n_chiral_centers=int((tags > 0).sum().item()),
            n_defined=int(((tags == 1) | (tags == 2)).sum().item()),
        )
        descs.append(global_chiral_descriptor(info).to(device))
    if not descs:
        return None
    return torch.stack(descs, dim=0)
