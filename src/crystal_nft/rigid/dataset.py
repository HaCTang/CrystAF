"""Rigidified training targets: teach the model to pack the conformer it will get.

The un-relaxed rigid route scores clash 45.36 against a 13.85 baseline. That is
not a general packing failure -- it is a MISMATCH. The model was trained to place
its own slightly-deformed molecules; at inference we substitute an ETKDG
conformer of a different shape into a packing tuned for the old one.

This wrapper removes the mismatch from the data side: each training crystal keeps
its lattice and its per-body centroids, but every molecule is replaced by the
rigidly-fitted ETKDG conformer. The model then learns packings for exactly the
conformers it is handed at inference, so the inference-time projection becomes a
small correction instead of a substitution.

What this CANNOT fix: the gas-phase-vs-crystal conformer gap, which is baked into
the rigidified targets themselves. Expect clash to improve and PDD to stay put.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from crystal_nft.rigid.kabsch import rigid_fit_bodies


class RigidTargetDataset(Dataset):
    """Wraps a `Subset` of the CSD train set, rigidifying x1 in place."""

    def __init__(self, base, cache_dir: str, logger=None):
        self.base = base
        self.cache: dict = {}
        for p in sorted(Path(cache_dir).glob("shard*.pt")):
            self.cache.update(torch.load(p, map_location="cpu", weights_only=False))
        self.n_hit = 0
        self.n_miss = 0
        if logger is not None:
            logger.info("RigidTargetDataset: %d cached molecule conformers", len(self.cache))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        c = self.base[i]
        try:
            return self._rigidify(c)
        except Exception:
            self.n_miss += 1
            return c

    def _rigidify(self, c):
        from clari.chem import Crystal
        from rdkit import Chem

        from crystal_nft.rigid.conformer import molecule_key

        mol = c.to_rdmol()
        try:
            Chem.SanitizeMol(mol)
            Chem.AssignStereochemistryFrom3D(mol)
        except Exception:
            pass
        idx_groups = Chem.GetMolFrags(mol, asMols=False)
        frag_mols = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)

        coords = c.coords.float()
        conf = coords.clone()
        ok = torch.zeros(coords.shape[0], dtype=torch.bool)
        hit = False
        for idxs, fm in zip(idx_groups, frag_mols, strict=False):
            k = molecule_key(fm)
            pos = self.cache.get(k) if k is not None else None
            # Lengths must match EXACTLY. Truncating to the shorter one is how a
            # stochastic dataset silently corrupts the target.
            if pos is None or pos.shape[0] != len(idxs):
                continue
            conf[list(idxs)] = pos.float()
            ok[list(idxs)] = True
            hit = True
        if not hit:
            self.n_miss += 1
            return c
        self.n_hit += 1
        fitted = rigid_fit_bodies(
            coords.unsqueeze(0), conf.unsqueeze(0),
            c.body_ids.unsqueeze(0), None, ok.unsqueeze(0),
        )[0]
        x = c.x.clone()
        x[3:3 + fitted.shape[0]] = fitted / float(Crystal.COORD_NORM)
        return c.replace(x=x)
