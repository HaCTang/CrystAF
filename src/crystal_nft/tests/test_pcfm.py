"""CPU tests for the PCFM inference-time chirality projection."""

import torch

from crystal_nft.meanflow.pcfm import (
    PCFM_RS_MARGIN,
    ChiralityConstraint,
    chirality_residual,
    project_chirality,
)


def _tetra(sign: float = 1.0, bond: float = 1.53) -> torch.Tensor:
    """An ideal tetrahedral centre; `sign < 0` gives the mirror image."""
    v = torch.tensor([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / 3**0.5
    if sign < 0:
        v = v * torch.tensor([-1.0, 1, 1])
    return torch.cat([torch.zeros(1, 3), v * bond], 0).unsqueeze(0)


def _cst(target: float = 1.0) -> ChiralityConstraint:
    return ChiralityConstraint(
        centers=torch.tensor([[[0, 1, 2, 3]]]), target=torch.tensor([[target]])
    )


def test_margin_matches_the_paper():
    assert PCFM_RS_MARGIN == 0.62


def test_residual_zero_for_correct_handedness():
    assert float(chirality_residual(_tetra(+1), _cst(+1)).max()) == 0.0


def test_residual_positive_for_wrong_handedness():
    assert float(chirality_residual(_tetra(-1), _cst(+1)).max()) > 1.0


def test_projection_fixes_a_wrongly_handed_centre():
    x = _tetra(-1)
    out = project_chirality(x, _cst(+1), n_iter=12, max_shift=2.0)
    assert float(chirality_residual(out, _cst(+1)).max()) < 1e-4


def test_projection_leaves_a_correct_centre_untouched():
    x = _tetra(+1)
    assert torch.equal(project_chirality(x, _cst(+1), n_iter=4), x)


def test_projection_respects_max_shift():
    x = _tetra(-1)
    out = project_chirality(x, _cst(+1), n_iter=1, max_shift=0.1)
    assert float((out - x).norm(dim=-1).max()) <= 0.1 + 1e-5


def test_inactive_targets_are_never_moved():
    """`target == 0` marks a centre with no canonical CIP frame: leave it alone."""
    x = _tetra(-1)
    cst = ChiralityConstraint(
        centers=torch.tensor([[[0, 1, 2, 3]]]), target=torch.tensor([[0.0]])
    )
    assert cst.n_active == 0
    assert torch.equal(project_chirality(x, cst, n_iter=4), x)


def test_batched_projection_is_independent_per_sample():
    x = torch.cat([_tetra(-1), _tetra(+1)], dim=0)
    cst = ChiralityConstraint(
        centers=torch.tensor([[0, 1, 2, 3]]).view(1, 1, 4).repeat(2, 1, 1),
        target=torch.tensor([[1.0], [1.0]]),
    )
    out = project_chirality(x, cst, n_iter=12, max_shift=2.0)
    assert float(chirality_residual(out, cst).max()) < 1e-4
    assert torch.allclose(out[1], x[1])  # the already-correct sample is untouched


# ---------------------------------------------------------------------------
# Global parity correction
# ---------------------------------------------------------------------------
def test_global_parity_flips_a_wrong_handed_cell():
    from crystal_nft.meanflow.pcfm import resolve_global_handedness

    x = _tetra(-1)
    out = resolve_global_handedness(x, _cst(+1))
    assert torch.allclose(out, -x)
    assert float(chirality_residual(out, _cst(+1)).max()) == 0.0


def test_global_parity_leaves_a_correct_cell_alone():
    from crystal_nft.meanflow.pcfm import resolve_global_handedness

    x = _tetra(+1)
    assert torch.allclose(resolve_global_handedness(x, _cst(+1)), x)


def test_global_parity_preserves_every_distance():
    """The reason it cannot cost PoseBusters or clash: it is an isometry."""
    from crystal_nft.meanflow.pcfm import resolve_global_handedness

    torch.manual_seed(0)
    x = torch.randn(1, 12, 3)
    cst = ChiralityConstraint(
        centers=torch.tensor([[[0, 1, 2, 3]]]), target=torch.tensor([[1.0]])
    )
    out = resolve_global_handedness(x, cst)
    assert torch.allclose(torch.cdist(x, x), torch.cdist(out, out), atol=1e-5)


def test_global_parity_uses_the_majority_of_centres():
    """A cell that is mostly right must not be flipped by one bad centre."""
    from crystal_nft.meanflow.pcfm import resolve_global_handedness

    # three centres: two already correct (+), one wrong (-)
    good = _tetra(+1)[0]
    bad = _tetra(-1)[0]
    coords = torch.cat([good, good + 10.0, bad + 20.0], dim=0).unsqueeze(0)
    cst = ChiralityConstraint(
        centers=torch.tensor([[[0, 1, 2, 3], [5, 6, 7, 8], [10, 11, 12, 13]]]),
        target=torch.ones(1, 3),
    )
    assert torch.allclose(resolve_global_handedness(coords, cst), coords)


# ---------------------------------------------------------------------------
# Per-body parity correction
# ---------------------------------------------------------------------------
def _two_body(sign_a: float, sign_b: float):
    """Two 5-atom molecules, 10 A apart, with independently chosen handedness."""
    a = _tetra(sign_a)[0]
    b = _tetra(sign_b)[0] + 10.0
    coords = torch.cat([a, b], dim=0).unsqueeze(0)
    cst = ChiralityConstraint(
        centers=torch.tensor([[[0, 1, 2, 3], [5, 6, 7, 8]]]),
        target=torch.ones(1, 2),
        body_ids=torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]),
    )
    return coords, cst


def test_body_parity_fixes_only_the_wrong_molecule():
    from crystal_nft.meanflow.pcfm import resolve_body_handedness

    coords, cst = _two_body(+1, -1)  # first molecule right, second wrong
    out = resolve_body_handedness(coords, cst, realign=False)
    assert torch.allclose(out[0, :5], coords[0, :5])  # correct body untouched
    assert not torch.allclose(out[0, 5:], coords[0, 5:])
    tau = chirality_residual(out, cst)
    assert float(tau.max()) < 1e-4  # both now satisfy the constraint


def test_body_parity_preserves_intramolecular_distances():
    """Why PoseBusters is invariant: PB scores each fragment, and every
    intramolecular distance survives an inversion about the body centroid."""
    from crystal_nft.meanflow.pcfm import resolve_body_handedness

    coords, cst = _two_body(-1, -1)
    out = resolve_body_handedness(coords, cst)
    for sl in (slice(0, 5), slice(5, 10)):
        assert torch.allclose(
            torch.cdist(coords[:, sl], coords[:, sl]),
            torch.cdist(out[:, sl], out[:, sl]),
            atol=1e-4,
        )


def test_body_parity_keeps_each_molecule_at_its_own_centroid():
    """The per-body route inverts about each molecule's own centroid.

    Pinned with `minimize_flips=False`: when *every* body is wrong the default
    now takes the free whole-cell inversion instead, which moves each centroid
    to its negation (still an exact isometry -- see the min_flips tests).
    """
    from crystal_nft.meanflow.pcfm import resolve_body_handedness

    coords, cst = _two_body(-1, -1)
    out = resolve_body_handedness(coords, cst, minimize_flips=False)
    for sl in (slice(0, 5), slice(5, 10)):
        assert torch.allclose(coords[:, sl].mean(1), out[:, sl].mean(1), atol=1e-4)


def test_body_parity_realign_reduces_displacement():
    """Kabsch realignment keeps the flipped molecule near the pose the model chose."""
    from crystal_nft.meanflow.pcfm import resolve_body_handedness

    torch.manual_seed(0)
    coords, cst = _two_body(-1, -1)
    coords = coords + 0.15 * torch.randn_like(coords)  # break the perfect symmetry
    plain = resolve_body_handedness(coords, cst, realign=False, minimize_flips=False)
    aligned = resolve_body_handedness(coords, cst, realign=True, minimize_flips=False)
    assert (aligned - coords).norm() <= (plain - coords).norm() + 1e-6


def test_body_parity_is_a_noop_without_body_ids():
    from crystal_nft.meanflow.pcfm import resolve_body_handedness

    x = _tetra(-1)
    assert torch.equal(resolve_body_handedness(x, _cst(+1)), x)


# ---------------------------------------------------------------------------
# Env gating
# ---------------------------------------------------------------------------
def test_mirror_fix_env_accepts_mode_names(monkeypatch):
    """`CRYSTAF_MIRROR_FIX` takes mode names, not just truthy spellings.

    Regression: an "in (1, true, yes)" gate silently turned `=body` into a no-op,
    so the constraint was never published to the sampler.
    """
    from crystal_nft.train.eval_clari_table1 import _pcfm_enabled

    monkeypatch.delenv("CRYSTAF_PCFM", raising=False)
    for value in ("body", "cell", "1", "true"):
        monkeypatch.setenv("CRYSTAF_MIRROR_FIX", value)
        assert _pcfm_enabled(), value
    for value in ("", "0", "false", "off", "no"):
        monkeypatch.setenv("CRYSTAF_MIRROR_FIX", value)
        assert not _pcfm_enabled(), value


def test_pcfm_env_enables_independently(monkeypatch):
    from crystal_nft.train.eval_clari_table1 import _pcfm_enabled

    monkeypatch.setenv("CRYSTAF_MIRROR_FIX", "0")
    monkeypatch.setenv("CRYSTAF_PCFM", "rs")
    assert _pcfm_enabled()
    monkeypatch.setenv("CRYSTAF_PCFM", "0")
    assert not _pcfm_enabled()


def test_bond_projection_runs_in_angstrom_not_normalized_units(monkeypatch):
    """`z[:, 3:]` is Cartesian/`COORD_NORM`; the DG bounds are in Angstrom.

    Projecting without converting reads a 1.5 A bond as 0.19 A and explodes the
    molecule (measured on Table1: pb 2.1%, clash 91.7%). Guard the conversion.
    """
    import torch
    from clari.chem import Crystal
    from crystal_nft.meanflow import pcfm as P
    from crystal_nft.meanflow.sampler import _pcfm_bond_project

    scale = float(Crystal.COORD_NORM)
    # two atoms exactly 1.5 A apart, expressed in the normalized frame
    z = torch.zeros(1, 5, 3)
    z[0, 3] = torch.tensor([0.0, 0.0, 0.0])
    z[0, 4] = torch.tensor([1.5 / scale, 0.0, 0.0])
    cst = P.BondConstraint(
        pairs=torch.tensor([[[0, 1]]]),
        lo=torch.tensor([[1.4]]),
        hi=torch.tensor([[1.6]]),
        valid=torch.tensor([[True]]),
    )
    monkeypatch.setenv("CRYSTAF_PCFM_BOND", "1")
    with P.active_bond_constraint(cst):
        out = _pcfm_bond_project(z)
    d = float((out[0, 4] - out[0, 3]).norm() * scale)
    # already inside [1.4, 1.6] -> projection must be a no-op, not a 10 A blowup
    assert abs(d - 1.5) < 1e-3, f"in-bounds bond moved to {d:.3f} A"

    # and a genuinely short bond must be lengthened *into* the window
    z[0, 4] = torch.tensor([0.8 / scale, 0.0, 0.0])
    monkeypatch.setenv("CRYSTAF_PCFM_BOND", "1")
    with P.active_bond_constraint(cst):
        out = _pcfm_bond_project(z)
    d = float((out[0, 4] - out[0, 3]).norm() * scale)
    assert 1.35 < d < 1.65, f"short bond projected to {d:.3f} A, want ~1.4-1.6"


def _two_body_cell(n_per_body=5, n_bodies=2, seed=0):
    """(coords, body_ids) for a cell of `n_bodies` well-separated rigid bodies."""
    import torch
    g = torch.Generator().manual_seed(seed)
    parts, ids = [], []
    for b in range(n_bodies):
        c = torch.randn(n_per_body, 3, generator=g)
        parts.append(c + torch.tensor([10.0 * b, 0.0, 0.0]))
        ids.append(torch.full((n_per_body,), b, dtype=torch.long))
    return torch.cat(parts).unsqueeze(0), torch.cat(ids).unsqueeze(0)


def test_min_flips_uses_free_cell_inversion_for_uniform_wrong_cell():
    """A uniform-but-wrong cell must cost zero body flips, not Z of them.

    Whole-cell inversion preserves every interatomic distance, so it is free in
    clash / PDD / volume; flipping each body individually is not. When every
    chiral body disagrees, `minimize_flips` must take the free route.
    """
    import torch
    from crystal_nft.meanflow import pcfm as P

    coords, body_ids = _two_body_cell()
    # one stereocentre per body, both currently disagreeing with their target
    centers = torch.tensor([[[0, 1, 2, 3], [5, 6, 7, 8]]])
    tau = P._tau(coords, centers)
    target = -torch.sign(tau)  # every body wrong
    cst = P.ChiralityConstraint(centers=centers, target=target, body_ids=body_ids)

    out = P.resolve_body_handedness(coords, cst, minimize_flips=True)
    # free route: the whole cell is negated, nothing else touched
    assert torch.allclose(out, -coords, atol=1e-5)
    # and it is an isometry -- all pairwise distances preserved exactly
    d0 = torch.cdist(coords[0], coords[0])
    d1 = torch.cdist(out[0], out[0])
    assert torch.allclose(d0, d1, atol=1e-5)
    # stereochemistry is actually fixed
    assert torch.equal(torch.sign(P._tau(out, centers)), torch.sign(target))


def test_min_flips_falls_back_to_body_flips_for_mixed_cell():
    """When only a minority disagrees, flipping those bodies is still cheaper."""
    import torch
    from crystal_nft.meanflow import pcfm as P

    coords, body_ids = _two_body_cell(n_bodies=4)
    centers = torch.tensor([[[0, 1, 2, 3], [5, 6, 7, 8],
                             [10, 11, 12, 13], [15, 16, 17, 18]]])
    tau = P._tau(coords, centers)
    sgn = torch.sign(tau)
    target = sgn.clone()
    target[0, 0] = -sgn[0, 0]  # exactly one of four bodies is wrong
    cst = P.ChiralityConstraint(centers=centers, target=target, body_ids=body_ids)

    out = P.resolve_body_handedness(coords, cst, minimize_flips=True)
    assert not torch.allclose(out, -coords, atol=1e-3), "should not invert the cell"
    # bodies 1..3 untouched, body 0 moved
    assert torch.allclose(out[0, 5:], coords[0, 5:], atol=1e-5)
    assert not torch.allclose(out[0, :5], coords[0, :5], atol=1e-3)
    assert torch.equal(torch.sign(P._tau(out, centers)), torch.sign(target))


def test_min_flips_reaches_same_stereo_as_the_old_path():
    """The optimisation must not change the stereochemical outcome, only its cost."""
    import torch
    from crystal_nft.meanflow import pcfm as P

    for seed in range(6):
        coords, body_ids = _two_body_cell(n_bodies=4, seed=seed)
        centers = torch.tensor([[[0, 1, 2, 3], [5, 6, 7, 8],
                                 [10, 11, 12, 13], [15, 16, 17, 18]]])
        g = torch.Generator().manual_seed(100 + seed)
        target = torch.where(torch.rand(1, 4, generator=g) < 0.5, -1.0, 1.0)
        cst = P.ChiralityConstraint(centers=centers, target=target, body_ids=body_ids)
        a = P.resolve_body_handedness(coords, cst, minimize_flips=False)
        b = P.resolve_body_handedness(coords, cst, minimize_flips=True)
        sa = torch.sign(P._tau(a, centers))
        sb = torch.sign(P._tau(b, centers))
        assert torch.equal(sa, sb), f"seed {seed}: stereo outcome changed"
        assert torch.equal(sb, torch.sign(target)), f"seed {seed}: target not met"


def test_min_flips_preserves_every_intramolecular_distance_vs_old_path():
    """Both parity routes must leave each molecule's internal geometry identical.

    That is what makes `pb_score` invariant: PoseBusters scores each fragment on
    intramolecular geometry alone. If the two routes disagreed here, switching
    the default would silently move PB.
    """
    import torch
    from crystal_nft.meanflow import pcfm as P

    for seed in range(8):
        coords, body_ids = _two_body_cell(n_bodies=4, seed=seed)
        centers = torch.tensor([[[0, 1, 2, 3], [5, 6, 7, 8],
                                 [10, 11, 12, 13], [15, 16, 17, 18]]])
        g = torch.Generator().manual_seed(200 + seed)
        target = torch.where(torch.rand(1, 4, generator=g) < 0.5, -1.0, 1.0)
        cst = P.ChiralityConstraint(centers=centers, target=target, body_ids=body_ids)
        old = P.resolve_body_handedness(coords, cst, minimize_flips=False)
        new = P.resolve_body_handedness(coords, cst, minimize_flips=True)
        for b in torch.unique(body_ids[0]).tolist():
            sel = body_ids[0] == b
            d_old = torch.cdist(old[0][sel], old[0][sel])
            d_new = torch.cdist(new[0][sel], new[0][sel])
            assert torch.allclose(d_old, d_new, atol=1e-4), (
                f"seed {seed} body {b}: intramolecular geometry differs by "
                f"{(d_old - d_new).abs().max():.2e}"
            )
