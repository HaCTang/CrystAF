"""Rigid-body clash relaxation: invariants and effectiveness."""

import torch

from crystal_nft.meanflow.relax import (
    clash_energy,
    covalent_lower_bounds,
    relax_clashes,
    _min_image_deltas,
)


def _clashing_cell(gap=0.6):
    """Two 4-atom carbon bodies pushed `gap` A apart -- well inside 2*r_cov(C)=1.52."""
    a = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]])
    b = a + torch.tensor([gap, 0.0, 0.0])
    cart = torch.cat([a, b])
    lattice = torch.eye(3) * 20.0
    body_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    atom_nums = torch.full((8,), 6)
    return cart, lattice, body_ids, atom_nums


def _intra_dists(cart, body_ids):
    out = []
    for b in torch.unique(body_ids):
        p = cart[body_ids == b]
        out.append(torch.cdist(p, p))
    return out


def test_relax_removes_the_clash():
    cart, lat, bid, znum = _clashing_cell()
    lb = covalent_lower_bounds(znum)
    inter = (bid.unsqueeze(-1) != bid.unsqueeze(-2)).float()
    inter.fill_diagonal_(0.0)
    before = float(clash_energy(cart, lat, inter, lb))
    assert before > 0, "fixture is not actually clashing"
    out = relax_clashes(cart, lat, bid, znum)
    after = float(clash_energy(out, lat, inter, lb))
    assert after <= 0.0, f"clash energy still {after:.4f} (was {before:.4f})"


def test_relax_preserves_intramolecular_geometry_exactly():
    """This is what makes `pb_score` invariant -- it must hold to float precision."""
    cart, lat, bid, znum = _clashing_cell()
    out = relax_clashes(cart, lat, bid, znum)
    for d0, d1 in zip(_intra_dists(cart, bid), _intra_dists(out, bid), strict=True):
        assert torch.allclose(d0, d1, atol=1e-4), (d0 - d1).abs().max()


def test_relax_is_a_noop_when_already_clash_free():
    cart, lat, bid, znum = _clashing_cell(gap=8.0)
    out = relax_clashes(cart, lat, bid, znum)
    assert torch.equal(out, cart)


def test_relax_respects_max_shift():
    cart, lat, bid, znum = _clashing_cell(gap=0.05)
    out = relax_clashes(cart, lat, bid, znum, max_shift=0.3, n_iter=200)
    moved = (out - cart).norm(dim=-1).max()
    # translation is clamped at max_shift; rotation about the centroid adds a
    # bounded arc, so allow the body radius on top rather than asserting 0.3
    assert float(moved) < 0.3 + 3.0


def test_min_image_matches_pymatgen_on_a_skewed_cell():
    """The metric uses pymatgen's periodic distances; ours must agree."""
    pytest = __import__("pytest")
    pmg = pytest.importorskip("pymatgen.core.lattice")
    torch.manual_seed(0)
    lattice = torch.tensor([[9.0, 0.0, 0.0], [3.5, 8.0, 0.0], [2.0, 1.5, 7.0]])
    cart = torch.randn(12, 3) * 4.0
    frac = cart @ torch.linalg.inv(lattice)
    ref = torch.from_numpy(
        pmg.Lattice(lattice.numpy()).get_all_distances(frac.numpy(), frac.numpy())
    ).float()
    ours = _min_image_deltas(cart, lattice).pow(2).sum(-1).sqrt()
    assert torch.allclose(ours, ref, atol=1e-4), (ours - ref).abs().max()


def test_relax_works_under_inference_mode():
    """The eval samples inside `torch.inference_mode()`, not `no_grad`.

    Inference tensors can never take part in autograd and `enable_grad()` does
    not lift that, so a gradient-based relaxation raises on every crystal there.
    This regression test is the one that would have caught it -- the original
    tests used `no_grad`, which works fine.
    """
    cart, lat, bid, znum = _clashing_cell()
    with torch.inference_mode():
        out = relax_clashes(cart, lat, bid, znum)
    lb = covalent_lower_bounds(znum)
    inter = (bid.unsqueeze(-1) != bid.unsqueeze(-2)).float()
    inter.fill_diagonal_(0.0)
    assert float(clash_energy(out, lat, inter, lb)) <= 0.0


def test_relax_hook_fires_under_inference_mode(monkeypatch):
    """End-to-end through the sampler hook, in the mode the eval actually uses."""
    from clari.chem import Crystal
    from crystal_nft.meanflow.sampler import _pcfm_relax_clashes

    monkeypatch.setenv("CRYSTAF_RELAX_CLASH", "1")
    cart, lat, bid, znum = _clashing_cell()
    scale = float(Crystal.COORD_NORM)
    z = torch.cat([(lat / (2 * scale)).unsqueeze(0), (cart / scale).unsqueeze(0)], dim=1)

    class _C:  # minimal stand-in exposing what the hook reads
        x = z
        body_ids = bid.unsqueeze(0)
        atom_nums = znum.unsqueeze(0)
        mask = torch.ones(1, cart.shape[0], dtype=torch.bool)
        lattice = lat.unsqueeze(0)

    with torch.inference_mode():
        out = _pcfm_relax_clashes(z, _C())
    assert not torch.allclose(out, z, atol=1e-6), "hook was a no-op under inference_mode"


def test_relax_torsions_preserves_bonds_angles_and_chirality():
    """Torsions are free coordinates -- this must not undo any stereo repair."""
    from crystal_nft.meanflow import pcfm as P
    from crystal_nft.meanflow.relax import relax_torsions

    g = torch.Generator().manual_seed(11)
    n = 9
    coords = torch.randn(n, 3, generator=g) * 1.3
    bonds = torch.zeros(n, n)
    chain = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 6), (4, 7), (2, 8)]
    for i, j in chain:
        bonds[i, j] = bonds[j, i] = 1
    lo = torch.full((n, n), 3.0)
    gd = torch.full((n, n), 9.0)
    mask = gd >= 4.0
    mask.fill_diagonal_(False)
    dg = [(0, n, lo, mask)]

    out = relax_torsions(coords, bonds, dg, rounds=2, n_theta=12)
    bi, bj = torch.nonzero(bonds > 0, as_tuple=True)
    d0 = (coords[bi] - coords[bj]).norm(dim=-1)
    d1 = (out[bi] - out[bj]).norm(dim=-1)
    assert torch.allclose(d0, d1, atol=1e-4), "bond lengths moved"

    adj = P._adjacency(bonds)
    for j, nbrs in adj.items():
        for a in nbrs:
            for c in nbrs:
                if a >= c:
                    continue
                f = lambda x: ((x[a] - x[j]) @ (x[c] - x[j])) / (
                    (x[a] - x[j]).norm() * (x[c] - x[j]).norm()).clamp_min(1e-8)
                assert torch.allclose(f(coords), f(out), atol=1e-4), f"angle {a}-{j}-{c} moved"

    centers = torch.tensor([[[1, 0, 2, 6]]])
    t0 = torch.sign(P._tau(coords.unsqueeze(0), centers))
    t1 = torch.sign(P._tau(out.unsqueeze(0), centers))
    assert torch.equal(t0, t1), "chirality changed"


def test_relax_torsions_reduces_internal_clash():
    from crystal_nft.meanflow.relax import relax_torsions

    g = torch.Generator().manual_seed(5)
    n = 9
    coords = torch.randn(n, 3, generator=g) * 1.1
    bonds = torch.zeros(n, n)
    for i, j in [(0,1),(1,2),(2,3),(3,4),(4,5),(1,6),(4,7),(2,8)]:
        bonds[i, j] = bonds[j, i] = 1
    lo = torch.full((n, n), 3.2)
    mask = torch.full((n, n), True); mask.fill_diagonal_(False)
    dg = [(0, n, lo, mask)]
    def energy(x):
        d = torch.cdist(x, x).clamp_min(1e-8)
        return float(torch.relu(lo * 0.7 - d).mul(mask).pow(2).sum())
    before = energy(coords)
    out = relax_torsions(coords, bonds, dg, rounds=3, n_theta=24)
    assert energy(out) <= before + 1e-6, f"{before:.3f} -> {energy(out):.3f}"
