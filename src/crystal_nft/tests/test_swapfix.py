"""Exact single-centre inversion via substituent swap."""

import torch

from crystal_nft.meanflow import pcfm as P


def _centre(sign=1.0):
    """One tetrahedral centre: atom 0 is C, atoms 1..4 substituents.

    Substituent 1 carries a 2-atom tail so branch logic is exercised.
    """
    c = torch.tensor([0.0, 0.0, 0.0])
    dirs = torch.tensor([
        [1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0],
    ])
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    lens = torch.tensor([1.5, 1.1, 1.4, 1.0]).unsqueeze(-1)
    p = c + dirs * lens * sign
    tail = p[0] + torch.tensor([0.3, 0.9, 0.2])          # atom 5, bonded to atom 1
    coords = torch.cat([c.unsqueeze(0), p, tail.unsqueeze(0)]).unsqueeze(0)
    bonds = torch.zeros(1, 6, 6)
    for i in (1, 2, 3, 4):
        bonds[0, 0, i] = bonds[0, i, 0] = 1
    bonds[0, 1, 5] = bonds[0, 5, 1] = 1
    centers = torch.tensor([[[0, 1, 2, 3]]])
    return coords, bonds, centers


def _cst(centers, target):
    return P.ChiralityConstraint(centers=centers, target=torch.tensor([[target]]))


def test_swap_inverts_the_centre():
    coords, bonds, centers = _centre()
    tau0 = float(P._tau(coords, centers)[0, 0])
    cst = _cst(centers, -torch.sign(torch.tensor(tau0)).item())
    out = P.swap_fix_stereocentres(coords, cst, bonds)
    tau1 = float(P._tau(out, centers)[0, 0])
    assert tau0 * tau1 < 0, f"tau {tau0:.3f} -> {tau1:.3f}, not inverted"


def test_swap_preserves_all_bond_lengths_exactly():
    """This is what makes it safe for PoseBusters: bonded distances are exact."""
    coords, bonds, centers = _centre()
    tau0 = float(P._tau(coords, centers)[0, 0])
    cst = _cst(centers, -torch.sign(torch.tensor(tau0)).item())
    out = P.swap_fix_stereocentres(coords, cst, bonds)
    bi, bj = torch.nonzero(bonds[0] > 0, as_tuple=True)
    d0 = (coords[0][bi] - coords[0][bj]).norm(dim=-1)
    d1 = (out[0][bi] - out[0][bj]).norm(dim=-1)
    assert torch.allclose(d0, d1, atol=1e-5), (d0 - d1).abs().max()


def test_swap_preserves_branch_internal_geometry():
    """Each branch moves rigidly and properly -- no internal mirroring."""
    coords, bonds, centers = _centre()
    tau0 = float(P._tau(coords, centers)[0, 0])
    cst = _cst(centers, -torch.sign(torch.tensor(tau0)).item())
    out = P.swap_fix_stereocentres(coords, cst, bonds)
    # atoms 1 and 5 are one branch; their mutual distance must be untouched
    d0 = (coords[0, 1] - coords[0, 5]).norm()
    d1 = (out[0, 1] - out[0, 5]).norm()
    assert torch.allclose(d0, d1, atol=1e-5)


def test_swap_is_a_noop_when_already_correct():
    coords, bonds, centers = _centre()
    tau0 = float(P._tau(coords, centers)[0, 0])
    cst = _cst(centers, torch.sign(torch.tensor(tau0)).item())
    out = P.swap_fix_stereocentres(coords, cst, bonds)
    assert torch.equal(out, coords)


def test_swap_skips_a_ring_pair():
    """If two substituents are joined by a ring through the centre, cutting them
    does not separate a branch, so that pair must not be swapped."""
    coords, bonds, centers = _centre()
    bonds[0, 1, 2] = bonds[0, 2, 1] = 1   # close a 3-ring: c-1-2-c
    bonds[0, 1, 3] = bonds[0, 3, 1] = 1   # and c-1-3-c
    bonds[0, 2, 3] = bonds[0, 3, 2] = 1   # and c-2-3-c -> every pair is a ring pair
    tau0 = float(P._tau(coords, centers)[0, 0])
    cst = _cst(centers, -torch.sign(torch.tensor(tau0)).item())
    out = P.swap_fix_stereocentres(coords, cst, bonds)
    assert torch.equal(out, coords), "ring-locked centre must be left alone"


def test_swap_uses_the_fourth_substituent_from_the_bond_graph():
    """`centers` stores only 3 of 4 substituents; a ring centre is unswappable
    among those three, and only the 4th (exocyclic) neighbour makes it fixable."""
    coords, bonds, centers = _centre()
    # a realistic ring centre: substituents 1 and 2 are ring atoms (joined to
    # each other off the centre), 3 and 4 are exocyclic
    bonds[0, 1, 2] = bonds[0, 2, 1] = 1
    tau0 = float(P._tau(coords, centers)[0, 0])
    cst = _cst(centers, -torch.sign(torch.tensor(tau0)).item())
    out = P.swap_fix_stereocentres(coords, cst, bonds)
    tau1 = float(P._tau(out, centers)[0, 0])
    # atoms 1,2 are ring-locked together; the only valid pair is the two
    # exocyclic substituents (3, 4) -- and 4 is not in `centers`
    assert tau0 * tau1 < 0, "should have swapped the two exocyclic substituents"
    bi, bj = torch.nonzero(bonds[0] > 0, as_tuple=True)
    d0 = (coords[0][bi] - coords[0][bj]).norm(dim=-1)
    d1 = (out[0][bi] - out[0][bj]).norm(dim=-1)
    assert torch.allclose(d0, d1, atol=1e-5), "bond lengths must stay exact"


def _two_centre_chain():
    """Two stereocentres joined by a rotatable bond, each with its own tail.

    atoms: 0=C1  1,2,3 = C1 substituents (3 is the linker) ... 4=C2, 5,6,7 = C2 subs
    """
    g = torch.Generator().manual_seed(3)
    coords = torch.randn(1, 10, 3, generator=g) * 1.4
    bonds = torch.zeros(1, 10, 10)
    def bond(i, j):
        bonds[0, i, j] = bonds[0, j, i] = 1
    for i in (1, 2, 3, 8):
        bond(0, i)          # centre 0
    for i in (5, 6, 7, 9):
        bond(4, i)          # centre 4
    bond(3, 4)              # linker: 0-3-4 chain, so 4's branch hangs off bond 3-4
    centers = torch.tensor([[[0, 1, 2, 8], [4, 5, 6, 9]]])
    return coords, bonds, centers


def test_reflect_fixes_one_centre_without_breaking_the_other():
    """The case whole-molecule inversion cannot handle: a mixed error pattern."""
    coords, bonds, centers = _two_centre_chain()
    tau = P._tau(coords, centers)[0]
    # ask for centre 0 as-is, centre 1 inverted -> only one is wrong
    target = torch.stack([torch.sign(tau[0]), -torch.sign(tau[1])]).unsqueeze(0)
    cst = P.ChiralityConstraint(centers=centers, target=target)
    out = P.reflect_fix_stereocentres(coords, cst, bonds)
    got = torch.sign(P._tau(out, centers)[0])
    assert torch.equal(got, target[0]), f"got {got.tolist()} want {target[0].tolist()}"


def test_reflect_preserves_every_bond_length_and_angle():
    coords, bonds, centers = _two_centre_chain()
    tau = P._tau(coords, centers)[0]
    target = torch.stack([torch.sign(tau[0]), -torch.sign(tau[1])]).unsqueeze(0)
    cst = P.ChiralityConstraint(centers=centers, target=target)
    out = P.reflect_fix_stereocentres(coords, cst, bonds)
    bi, bj = torch.nonzero(bonds[0] > 0, as_tuple=True)
    d0 = (coords[0][bi] - coords[0][bj]).norm(dim=-1)
    d1 = (out[0][bi] - out[0][bj]).norm(dim=-1)
    assert torch.allclose(d0, d1, atol=1e-4), (d0 - d1).abs().max()
    # bond angles: every path i-j-k through the graph
    adj = P._adjacency(bonds[0])
    for j, nbrs in adj.items():
        for a in nbrs:
            for c in nbrs:
                if a >= c:
                    continue
                def ang(x):
                    u = x[0, a] - x[0, j]; v = x[0, c] - x[0, j]
                    return (u @ v) / (u.norm() * v.norm()).clamp_min(1e-8)
                assert torch.allclose(ang(coords), ang(out), atol=1e-4), f"angle {a}-{j}-{c}"


def test_reflect_is_a_noop_when_all_centres_are_correct():
    coords, bonds, centers = _two_centre_chain()
    tau = P._tau(coords, centers)[0]
    cst = P.ChiralityConstraint(centers=centers, target=torch.sign(tau).unsqueeze(0))
    out = P.reflect_fix_stereocentres(coords, cst, bonds)
    assert torch.equal(out, coords)
