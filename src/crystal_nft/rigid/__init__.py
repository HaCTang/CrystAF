"""MolCrystalFlow-style rigid-body handling for CrystAF, kept off the trunk.

The whole point, in one sentence: **SO(3) cannot change chirality**, so if a
molecule's internal geometry is supplied rather than generated, handedness stops
being something the model has to learn and becomes an invariant of the input.

MolCrystalFlow (Zeng et al.) represents a crystal as (lattice, per-molecule
centroid on a torus, per-molecule orientation on SO(3)) with the conformer held
rigid. Every one of those three modalities is chirality-preserving:

* lattice          -- acts on fractional coords, never on the molecule's frame
* centroid         -- translation
* orientation      -- proper rotation, det = +1

That is why MCF has no chirality problem to solve: it is structurally incapable
of emitting the wrong enantiomer. Fourteen CrystAF training runs spent ~20 PB
each trying to teach a Cartesian all-atom flow the same bit.

This package does NOT reimplement MCF. It applies MCF's *representation* to
CrystAF's existing samples, so the two can be compared on one protocol.
"""

from crystal_nft.rigid.kabsch import kabsch_proper, rigid_fit_bodies
from crystal_nft.rigid.projector import RigidProjector, active_rigid, get_active_rigid

__all__ = [
    "kabsch_proper",
    "rigid_fit_bodies",
    "RigidProjector",
    "active_rigid",
    "get_active_rigid",
]
