"""Conformer sources for the rigid-body route.

`oracle` (reference crystal geometry) is a ceiling. This module provides the
FAIR source: an RDKit ETKDGv3 conformer built from the molecular graph plus the
requested stereochemistry -- exactly MolCrystalFlow's setting, where conformers
come from OMEGA/ETKDGv2 and only the packing is generated.

What is and isn't taken from the reference:

* TAKEN: the R/S bit at each stereocentre. That is the tag the model is already
  conditioned on -- it is the *request*, molecular identity, not geometry.
* NOT TAKEN: any coordinate. The embedding is gas-phase distance geometry from
  the 2D graph, so torsions, packing-induced strain and the crystal conformation
  are all absent by construction. That absence is the honest cost of the route
  and is what the PDD column will show.

Embedding failures are counted and reported, never silently replaced by the
reference conformer -- that would reintroduce the oracle without saying so.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor


class ConformerStats:
    def __init__(self) -> None:
        self.bodies = 0
        self.embedded = 0
        self.failed = 0

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"ConformerStats(bodies={self.bodies}, embedded={self.embedded}, "
                f"failed={self.failed})")


_CACHE: dict[str, Tensor] = {}
STATS = ConformerStats()


def etkdg_conformer_coords(crystal, *, seed: int = 0xC0FFEE,
                           stats: Optional[ConformerStats] = None,
                           want_torsions: bool = False):
    """`(coords (N,3) Angstrom, ok (N,) bool)`, atom order matching `crystal`.

    `ok` is False for every atom of a body that failed to embed. Those bodies
    MUST be left alone by the caller: filling them with `crystal.coords` would
    quietly restore the reference conformer and turn the fair measurement back
    into the oracle for part of the data.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    st = stats if stats is not None else STATS
    mol = crystal.to_rdmol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    # Read the R/S bits off the reference geometry. This is the requested tag,
    # not the conformation.
    try:
        Chem.AssignStereochemistryFrom3D(mol)
    except Exception:
        pass

    out = crystal.coords.detach().clone().float()
    ok = torch.zeros(out.shape[0], dtype=torch.bool)
    bodies: List[Tuple[torch.Tensor, list]] = []
    frags = Chem.GetMolFrags(mol, asMols=False)
    frag_mols = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)

    for idxs, fm in zip(frags, frag_mols, strict=False):
        st.bodies += 1
        try:
            # SMILES alone is NOT a safe cache key: the coords are written back
            # positionally, so two bodies sharing a SMILES but differing in atom
            # ORDER would get a scrambled molecule. Bind the order into the key.
            order = tuple(a.GetAtomicNum() for a in fm.GetAtoms())
            key = Chem.MolToSmiles(fm, isomericSmiles=True) + repr(order)
        except Exception:
            key = None
        if key is not None and key in _CACHE:
            pos = _CACHE[key]
        else:
            pos = _embed(fm, seed)
            if pos is None:
                st.failed += 1
                continue
            if key is not None:
                _CACHE[key] = pos
        if pos.shape[0] != len(idxs):
            st.failed += 1
            continue
        st.embedded += 1
        out[list(idxs)] = pos.to(out.dtype)
        ok[list(idxs)] = True
        if want_torsions:
            from crystal_nft.rigid.torsion import rotatable_torsions

            from crystal_nft.rigid.torsion import intramolecular_bounds

            bodies.append((torch.tensor(list(idxs), dtype=torch.long),
                           rotatable_torsions(fm), intramolecular_bounds(fm)))
    if want_torsions:
        return out, ok, bodies
    return out, ok


def _embed(fm, seed: int) -> Optional[Tensor]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        m = Chem.Mol(fm)
        m.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.useSmallRingTorsions = True
        # Chirality must survive the embedding -- that is the entire point.
        params.enforceChirality = True
        if AllChem.EmbedMolecule(m, params) != 0:
            params.useRandomCoords = True
            params.maxIterations = 2000
            if AllChem.EmbedMolecule(m, params) != 0:
                return None
        c = m.GetConformer()
        return torch.tensor(c.GetPositions(), dtype=torch.float32)
    except Exception:
        return None


def molecule_key(fm) -> Optional[str]:
    """Identity of a molecule for conformer caching.

    Coordinates are written back POSITIONALLY, so two molecules sharing a SMILES
    but not an atom order are different objects here. The ordered atomic numbers
    are part of the key for that reason.
    """
    from rdkit import Chem

    try:
        order = tuple(a.GetAtomicNum() for a in fm.GetAtoms())
        return Chem.MolToSmiles(fm, isomericSmiles=True) + repr(order)
    except Exception:
        return None


def molecule_conformers(crystal, out: dict, *, seed: int = 0xC0FFEE,
                        stats: Optional[ConformerStats] = None) -> dict:
    """Embed every distinct molecule of `crystal` into `out` (key -> (n,3)).

    Keyed by molecule rather than by dataset index: the CSD train split uses
    `random_repr: True`, so a dataset index does NOT identify a fixed crystal.
    """
    from rdkit import Chem

    st = stats if stats is not None else STATS
    mol = crystal.to_rdmol()
    try:
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistryFrom3D(mol)
    except Exception:
        pass
    for fm in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
        st.bodies += 1
        k = molecule_key(fm)
        if k is None or k in out:
            continue
        pos = _embed(fm, seed)
        if pos is None:
            st.failed += 1
            continue
        st.embedded += 1
        out[k] = pos.half()
    return out
