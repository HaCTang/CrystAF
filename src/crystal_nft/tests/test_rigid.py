"""Rigid-body projection: the properties the chirality argument rests on."""

import torch

from crystal_nft.rigid.kabsch import kabsch_proper, rigid_fit_bodies
from crystal_nft.rigid.projector import project_z


def _chiral_sign(p):
    """Signed volume of the first four atoms."""
    a, b, c, d = p[0], p[1], p[2], p[3]
    return torch.sign(torch.dot(torch.cross(b - a, c - a, dim=-1), d - a))


def test_kabsch_is_never_a_reflection():
    """Even when the best O(3) fit IS a reflection, we must return det=+1."""
    torch.manual_seed(0)
    x = torch.randn(1, 12, 3).double()
    x = x - x.mean(1, keepdim=True)
    mirror = x.clone()
    mirror[..., 2] *= -1.0          # exact mirror: O(3) fit would be det = -1
    r = kabsch_proper(x, mirror)
    assert torch.allclose(torch.linalg.det(r), torch.ones(1).double(), atol=1e-9)


def test_projection_forces_the_conformer_handedness():
    """A mirrored sample comes back with the CONFORMER's handedness, not its own.

    This is the property the whole MolCrystalFlow route depends on: chirality is
    an invariant of the supplied conformer, not something the sampler decides.
    """
    torch.manual_seed(1)
    conf = torch.randn(1, 8, 3)
    sampled = conf.clone()
    sampled[..., 2] *= -1.0                       # sample is the wrong enantiomer
    body = torch.zeros(1, 8, dtype=torch.long)
    assert _chiral_sign(sampled[0]) == -_chiral_sign(conf[0])
    out = rigid_fit_bodies(sampled, conf, body)
    assert _chiral_sign(out[0]) == _chiral_sign(conf[0])


def test_projection_preserves_intramolecular_geometry_exactly():
    """Every internal distance equals the conformer's -- so PoseBusters'
    intramolecular checks pass by construction, not by optimisation."""
    torch.manual_seed(2)
    conf = torch.randn(1, 10, 3)
    sampled = torch.randn(1, 10, 3)
    body = torch.zeros(1, 10, dtype=torch.long)
    out = rigid_fit_bodies(sampled, conf, body)
    d_conf = torch.cdist(conf[0], conf[0])
    d_out = torch.cdist(out[0], out[0])
    assert torch.max((d_conf - d_out).abs()) < 1e-5


def test_projection_keeps_the_sampled_centroid():
    """Packing stays the model's: only internal geometry is replaced."""
    torch.manual_seed(3)
    conf = torch.randn(1, 10, 3)
    sampled = torch.randn(1, 10, 3) + 5.0
    body = torch.zeros(1, 10, dtype=torch.long)
    out = rigid_fit_bodies(sampled, conf, body)
    assert torch.allclose(out[0].mean(0), sampled[0].mean(0), atol=1e-5)


def test_bodies_are_fitted_independently():
    """Two molecules in one cell get their own rotation."""
    torch.manual_seed(4)
    conf = torch.randn(1, 12, 3)
    sampled = torch.randn(1, 12, 3)
    body = torch.cat([torch.zeros(6), torch.ones(6)]).long().unsqueeze(0)
    out = rigid_fit_bodies(sampled, conf, body)
    for sl in (slice(0, 6), slice(6, 12)):
        d_conf = torch.cdist(conf[0, sl], conf[0, sl])
        d_out = torch.cdist(out[0, sl], out[0, sl])
        assert torch.max((d_conf - d_out).abs()) < 1e-5
        assert torch.allclose(out[0, sl].mean(0), sampled[0, sl].mean(0), atol=1e-5)


def test_project_z_is_a_noop_without_an_active_reference():
    z = torch.randn(2, 7, 3)
    assert torch.equal(project_z(z), z)


def test_relaxation_cannot_change_chirality_or_internal_geometry():
    """The relaxation has 6 DOF per body, so both rigid-route guarantees hold."""
    from crystal_nft.rigid.relax import relax_bodies

    torch.manual_seed(5)
    # two 5-atom bodies deliberately overlapping in a big cell
    a = torch.randn(5, 3) * 0.7
    b = a.clone() + torch.tensor([0.6, 0.0, 0.0])
    coords = torch.cat([a, b]).double()
    lattice = (torch.eye(3) * 30.0).double()
    body = torch.cat([torch.zeros(5), torch.ones(5)]).long()
    lb = torch.full((10, 10), 1.5).double()

    out = relax_bodies(coords, lattice, body, lb, steps=60)

    for sl in (slice(0, 5), slice(5, 10)):
        d0 = torch.cdist(coords[sl], coords[sl])
        d1 = torch.cdist(out[sl], out[sl])
        assert torch.max((d0 - d1).abs()) < 1e-6      # internal geometry frozen
        assert _chiral_sign(out[sl]) == _chiral_sign(coords[sl])   # SO(3) only


def test_relaxation_reduces_overlap():
    from crystal_nft.rigid.relax import relax_bodies

    torch.manual_seed(6)
    a = torch.randn(6, 3) * 0.6
    coords = torch.cat([a, a + torch.tensor([0.35, 0.0, 0.0])]).double()
    lattice = (torch.eye(3) * 40.0).double()
    body = torch.cat([torch.zeros(6), torch.ones(6)]).long()
    lb = torch.full((12, 12), 1.6).double()
    inter = body.unsqueeze(-1) != body.unsqueeze(-2)

    def worst(p):
        d = torch.cdist(p, p) + torch.eye(12).double() * 1e3
        return float(torch.relu(lb - d)[inter].max())

    before = worst(coords)
    after = worst(relax_bodies(coords, lattice, body, lb, steps=400, lr=0.05))
    assert after < before, (before, after)


def test_rodrigues_is_in_so3():
    from crystal_nft.rigid.relax import _rotation_from_axis_angle

    torch.manual_seed(7)
    r = _rotation_from_axis_angle(torch.randn(8, 3).double())
    assert torch.allclose(torch.linalg.det(r), torch.ones(8).double(), atol=1e-9)
    eye = torch.eye(3).double().expand(8, 3, 3)
    assert torch.allclose(r @ r.transpose(-1, -2), eye, atol=1e-9)


def test_bodies_without_a_conformer_are_left_untouched():
    """A failed ETKDG embedding must NOT be filled with reference geometry.

    Silently falling back would restore the oracle for part of the data while
    the run still called itself 'etkdg'.
    """
    torch.manual_seed(8)
    conf = torch.randn(1, 12, 3)
    sampled = torch.randn(1, 12, 3)
    body = torch.cat([torch.zeros(6), torch.ones(6)]).long().unsqueeze(0)
    ok = torch.cat([torch.ones(6), torch.zeros(6)]).bool().unsqueeze(0)
    out = rigid_fit_bodies(sampled, conf, body, None, ok)
    assert not torch.allclose(out[0, :6], sampled[0, :6])     # body 0 projected
    assert torch.equal(out[0, 6:], sampled[0, 6:])            # body 1 untouched


def _butane_like():
    """A 6-atom chain with one genuinely rotatable central bond."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles("CC(F)C(Cl)C"))
    AllChem.EmbedMolecule(m, randomSeed=7)
    return m


def test_torsions_preserve_bond_lengths_and_angles_exactly():
    """1-2 and 1-3 distances are invariant, so ETKDG's ideal geometry survives."""
    from crystal_nft.rigid.torsion import apply_torsions, rotatable_torsions

    m = _butane_like()
    x = torch.tensor(m.GetConformer().GetPositions(), dtype=torch.float64)
    tors = rotatable_torsions(m)
    assert len(tors) >= 1
    ang = torch.tensor([0.7] * len(tors), dtype=torch.float64)
    y = apply_torsions(x, tors, ang)

    adj = torch.zeros(m.GetNumAtoms(), m.GetNumAtoms(), dtype=torch.bool)
    for b in m.GetBonds():
        adj[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = True
        adj[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = True
    two = adj | (adj.double() @ adj.double() > 0)      # 1-2 and 1-3 pairs
    d0, d1 = torch.cdist(x, x), torch.cdist(y, y)
    assert torch.max((d0 - d1).abs()[two]) < 1e-8


def test_torsions_cannot_change_a_stereocentre():
    """No torsion angle, at any magnitude, flips the chiral volume sign."""
    from crystal_nft.rigid.torsion import apply_torsions, rotatable_torsions

    m = _butane_like()
    x = torch.tensor(m.GetConformer().GetPositions(), dtype=torch.float64)
    tors = rotatable_torsions(m)
    nbrs = [a.GetIdx() for a in m.GetAtomWithIdx(1).GetNeighbors()]
    quad = [1] + nbrs[:3]

    def sign(p):
        a, b, c, d = p[quad[0]], p[quad[1]], p[quad[2]], p[quad[3]]
        return torch.sign(torch.dot(torch.cross(b - a, c - a, dim=-1), d - a))

    s0 = sign(x)
    for mag in (0.3, 1.4, 2.9, -2.2, 5.0):
        y = apply_torsions(x, tors, torch.full((len(tors),), mag, dtype=torch.float64))
        assert sign(y) == s0, mag


def test_ring_bonds_are_never_rotatable():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from crystal_nft.rigid.torsion import rotatable_torsions

    m = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))
    AllChem.EmbedMolecule(m, randomSeed=3)
    assert rotatable_torsions(m) == []          # cyclohexane: no ring-closure risk


def test_torsion_fit_moves_toward_the_target():
    from crystal_nft.rigid.torsion import apply_torsions, fit_torsions_to, rotatable_torsions

    m = _butane_like()
    x = torch.tensor(m.GetConformer().GetPositions(), dtype=torch.float64)
    tors = rotatable_torsions(m)
    target = apply_torsions(x, tors, torch.tensor([1.1] * len(tors), dtype=torch.float64))

    def rmsd(p, q):
        from crystal_nft.rigid.kabsch import kabsch_proper
        p = p - p.mean(0, keepdim=True)
        q = q - q.mean(0, keepdim=True)
        R = kabsch_proper(p.unsqueeze(0), q.unsqueeze(0))[0]
        return float(((p @ R.transpose(0, 1) - q) ** 2).sum(-1).mean().sqrt())

    before = rmsd(x, target)
    after = rmsd(fit_torsions_to(x, target, tors, steps=300), target)
    assert after < before * 0.5, (before, after)


def test_torsion_fit_respects_the_steric_term():
    """With a steric term the fit must not fold the molecule onto itself."""
    from crystal_nft.rigid.torsion import (fit_torsions_to, intramolecular_bounds,
                                           rotatable_torsions)

    m = _butane_like()
    x = torch.tensor(m.GetConformer().GetPositions(), dtype=torch.float64)
    tors = rotatable_torsions(m)
    far, lb = intramolecular_bounds(m)
    # a deliberately collapsed target: RMSD alone would crush the molecule
    target = x * 0.35
    out = fit_torsions_to(x, target, tors, steps=200, steric=(far, lb))
    d = torch.cdist(out, out)
    worst = float(torch.relu(lb.double() - d)[far].max()) if bool(far.any()) else 0.0
    assert worst < 0.35, worst
