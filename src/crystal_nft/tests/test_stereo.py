"""CPU tests for CrystAF tetrahedral-stereochemistry support."""

import pytest
import torch
import torch.nn as nn

from crystal_nft.meanflow.stereo import (
    BatchStereo,
    TAG_R,
    TAG_S,
    active_stereo_tags,
    chiral_hinge_loss,
    chiral_signs,
    chiral_volumes,
    get_active_stereo_tags,
    stereo_agreement,
)


def _tetra(scale: float = 1.0) -> torch.Tensor:
    """Centre at the origin with three substituents along +x/+y/+z (plus a spectator)."""
    return scale * torch.tensor(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]]
    )


def test_chiral_volume_is_signed_and_scale_free():
    coords = _tetra()
    centers = torch.tensor([[0, 1, 2, 3]])
    v = chiral_volumes(coords, centers)
    assert torch.allclose(v, torch.tensor([1.0]))
    # Normalised triple product: independent of overall scale.
    assert torch.allclose(chiral_volumes(_tetra(3.7), centers), v)
    # Reflection flips the sign.
    mirrored = coords * torch.tensor([-1.0, 1, 1])
    assert torch.allclose(chiral_volumes(mirrored, centers), -v)


def test_chiral_volume_invariant_to_proper_rotation():
    coords = _tetra()
    centers = torch.tensor([[0, 1, 2, 3]])
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    rotated = coords @ q.T + torch.tensor([5.0, -2.0, 0.5])
    assert torch.allclose(
        chiral_volumes(rotated, centers), chiral_volumes(coords, centers), atol=1e-5
    )


def test_batched_gather_matches_per_sample():
    a = _tetra()
    b = a * torch.tensor([-1.0, 1, 1])
    coords = torch.stack([a, b])
    centers = torch.tensor([[0, 1, 2, 3]]).unsqueeze(0).repeat(2, 1, 1)
    got = chiral_volumes(coords, centers)
    assert torch.allclose(got[0], chiral_volumes(a, centers[0]))
    assert torch.allclose(got[1], chiral_volumes(b, centers[1]))
    assert got[0].item() > 0 > got[1].item()


def _spec(b: int = 2) -> BatchStereo:
    centers = torch.tensor([[0, 1, 2, 3]]).unsqueeze(0).repeat(b, 1, 1)
    return BatchStereo(
        centers=centers,
        valid=torch.ones(b, 1, dtype=torch.bool),
        atom_tags=torch.zeros(b, 5, dtype=torch.long),
        n_centers=b,
    )


def test_hinge_zero_when_handedness_matches():
    coords = torch.stack([_tetra(), _tetra()])
    assert chiral_hinge_loss(coords, coords, _spec()).item() == 0.0


def test_hinge_penalises_wrong_handedness():
    ref = torch.stack([_tetra(), _tetra()])
    pred = ref * torch.tensor([-1.0, 1, 1])
    loss = chiral_hinge_loss(pred, ref, _spec(), margin=0.2)
    assert abs(loss.item() - 1.2) < 1e-5  # relu(0.2 - (+1)*(-1))


def test_hinge_pushes_a_nearly_flat_centre_off_the_plane():
    """A near-planar centre is ambiguous; the margin must give it a gradient."""
    ref = torch.stack([_tetra(), _tetra()])
    flat = ref.clone()
    # Right handedness, but the substituent has swung almost into the plane
    # spanned by the other two (the volume is normalised, so shortening the
    # bond would not make it "flat" — swinging it in-plane does).
    flat[:, 3] = torch.tensor([0.6, 0.6, 0.05])
    flat.requires_grad_(True)
    loss = chiral_hinge_loss(flat, ref, _spec(), margin=0.2)
    assert 0 < loss.item() < 0.2  # inside the margin, not wrong-handed
    loss.backward()
    assert flat.grad.abs().sum() > 0


def test_hinge_is_finite_at_a_degenerate_centre():
    """Overlapping atoms must not produce NaN/inf in the loss or its gradient."""
    ref = torch.stack([_tetra(), _tetra()])
    degenerate = ref.clone()
    degenerate[:, 3] = degenerate[:, 0]  # substituent collapsed onto the centre
    degenerate.requires_grad_(True)
    loss = chiral_hinge_loss(degenerate, ref, _spec(), margin=0.2)
    assert torch.isfinite(loss).all()
    loss.backward()
    assert torch.isfinite(degenerate.grad).all()


def test_agreement_rate():
    ref = torch.stack([_tetra(), _tetra()])
    pred = ref.clone()
    pred[1] = pred[1] * torch.tensor([-1.0, 1, 1])
    rate, n = stereo_agreement(pred, ref, _spec())
    assert n == 2
    assert abs(rate - 0.5) < 1e-9


def test_padded_centres_are_ignored():
    ref = torch.stack([_tetra(), _tetra()])
    pred = ref * torch.tensor([-1.0, 1, 1])
    centers = torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]]).unsqueeze(0).repeat(2, 1, 1)
    spec = BatchStereo(
        centers=centers,
        valid=torch.tensor([[True, False], [True, False]]),
        atom_tags=torch.zeros(2, 5, dtype=torch.long),
        n_centers=2,
    )
    # Only the two valid centres contribute: mean of [1.2, 1.2], not [1.2, 1.2, x, x].
    assert abs(chiral_hinge_loss(pred, ref, spec, margin=0.2).item() - 1.2) < 1e-5
    rate, n = stereo_agreement(pred, ref, spec)
    assert n == 2 and rate == 0.0


def test_stereo_token_embedding_is_identity_at_init():
    from crystal_nft.meanflow.net import StereoTokenEmbedding

    inner = nn.Linear(24, 16, bias=False)
    emb = StereoTokenEmbedding(16)

    class _Dit(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_feats = inner

    dit = _Dit()
    emb.bind(dit)
    x = torch.randn(2, 7, 24)
    base = inner(x).clone()

    tags = torch.zeros(2, 7, dtype=torch.long)
    tags[0, 2] = TAG_R
    tags[1, 4] = TAG_S
    with active_stereo_tags(tags):
        assert torch.allclose(dit.embed_feats(x), base)  # zero-init ⇒ no-op
        with torch.no_grad():
            emb.emb.weight.normal_()
        out = dit.embed_feats(x)
    # Only the tagged rows moved; tag 0 stays exactly untouched.
    moved = (out - base).abs().sum(dim=-1) > 0
    assert moved[0, 2] and moved[1, 4]
    assert not moved[0, 3] and not moved[1, 0]


def test_stereo_tags_pad_for_lattice_rows():
    """`use_lattice_regs` prepends 3 rows; tags must shift with them."""
    from crystal_nft.meanflow.net import StereoTokenEmbedding

    inner = nn.Linear(24, 16, bias=False)
    emb = StereoTokenEmbedding(16)

    class _Dit(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_feats = inner

    dit = _Dit()
    emb.bind(dit)
    with torch.no_grad():
        emb.emb.weight.normal_()
    x = torch.randn(1, 3 + 5, 24)  # 3 lattice rows + 5 atoms
    base = inner(x).clone()
    tags = torch.zeros(1, 5, dtype=torch.long)
    tags[0, 0] = TAG_R  # first *atom*, i.e. row 3 of the padded tensor
    with active_stereo_tags(tags):
        out = dit.embed_feats(x)
    moved = (out - base).abs().sum(dim=-1) > 0
    assert moved[0, 3]
    assert not moved[0, 0] and not moved[0, 1] and not moved[0, 2]


def test_active_tags_context_restores():
    assert get_active_stereo_tags() is None
    with active_stereo_tags(torch.zeros(1, 2, dtype=torch.long)):
        assert get_active_stereo_tags() is not None
    assert get_active_stereo_tags() is None


# ---------------------------------------------------------------------------
# LoQI auxiliary edges
# ---------------------------------------------------------------------------
def _spec_with_frame(label_sign: float):
    from crystal_nft.meanflow.stereo import BatchStereo

    return BatchStereo(
        centers=torch.tensor([[[0, 1, 2, 3]]]),
        valid=torch.ones(1, 1, dtype=torch.bool),
        atom_tags=torch.zeros(1, 5, dtype=torch.long),
        n_centers=1,
        label_sign=torch.tensor([[label_sign]]),
        low_prio=torch.tensor([[4]]),
    )


def test_pair_edges_encode_a_directed_cycle():
    from crystal_nft.meanflow.stereo import (
        EDGE_CYCLE_BWD,
        EDGE_CYCLE_FWD,
        EDGE_LOW,
        build_stereo_pair_edges,
    )

    e = build_stereo_pair_edges(_spec_with_frame(1.0), 5)[0]
    # S keeps CIP order: p1->p2->p3->p1 forward, reverse entries backward.
    assert e[1, 2] == EDGE_CYCLE_FWD and e[2, 1] == EDGE_CYCLE_BWD
    assert e[2, 3] == EDGE_CYCLE_FWD and e[3, 2] == EDGE_CYCLE_BWD
    assert e[3, 1] == EDGE_CYCLE_FWD and e[1, 3] == EDGE_CYCLE_BWD
    # Lowest-priority substituent is linked undirected to all three.
    for other in (1, 2, 3):
        assert e[4, other] == EDGE_LOW and e[other, 4] == EDGE_LOW


def test_pair_edge_cycle_reverses_with_the_label():
    from crystal_nft.meanflow.stereo import EDGE_CYCLE_FWD, build_stereo_pair_edges

    s_edges = build_stereo_pair_edges(_spec_with_frame(1.0), 5)[0]
    r_edges = build_stereo_pair_edges(_spec_with_frame(-1.0), 5)[0]
    assert s_edges[1, 2] == EDGE_CYCLE_FWD
    assert r_edges[1, 3] == EDGE_CYCLE_FWD  # R reverses the cycle
    assert not torch.equal(s_edges, r_edges)


def test_pair_edges_skip_centres_without_a_canonical_frame():
    from crystal_nft.meanflow.stereo import build_stereo_pair_edges

    assert build_stereo_pair_edges(_spec_with_frame(0.0), 5) is None


def test_stereo_pair_embedding_is_identity_at_init():
    from crystal_nft.meanflow.net import StereoPairEmbedding
    from crystal_nft.meanflow.stereo import (
        StereoConditioning,
        build_stereo_pair_edges,
    )

    inner = nn.Embedding(32, 8)
    mod = StereoPairEmbedding(8)

    class _Dit(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_bonds = inner

    dit = _Dit()
    mod.bind(dit)
    bonds = torch.randint(1, 20, (1, 5, 5))
    base = inner(bonds).clone()
    edges = build_stereo_pair_edges(_spec_with_frame(1.0), 5)

    with active_stereo_tags(StereoConditioning(atom_tags=None, pair_edges=edges)):
        assert torch.allclose(dit.embed_bonds(bonds), base)  # zero-init ⇒ no-op
        with torch.no_grad():
            mod.emb.weight.normal_()
        out = dit.embed_bonds(bonds)
    moved = (out - base).abs().sum(-1) > 0
    assert moved[0, 1, 2]  # a cycle edge moved
    assert not moved[0, 0, 0]  # an untagged pair did not


# ---------------------------------------------------------------------------
# Mirror augmentation
# ---------------------------------------------------------------------------
def test_mirror_keeps_lattice_and_flips_only_selected_rows():
    from crystal_nft.meanflow.stereo import mirror_coords_in_place_x

    x = torch.randn(2, 3 + 5, 3)
    flip = torch.tensor([True, False])
    y = mirror_coords_in_place_x(x, flip)
    assert torch.equal(y[:, :3], x[:, :3])  # lattice untouched
    assert torch.equal(y[0, 3:], -x[0, 3:])
    assert torch.equal(y[1, 3:], x[1, 3:])


def test_mirror_preserves_all_interatomic_distances():
    """Why the frozen teacher is safe under the augmentation: pair features are
    distances, and inversion through the origin preserves every one of them."""
    from crystal_nft.meanflow.stereo import mirror_coords_in_place_x

    x = torch.randn(1, 3 + 6, 3)
    y = mirror_coords_in_place_x(x, torch.tensor([True]))
    assert torch.allclose(torch.cdist(x[:, 3:], x[:, 3:]), torch.cdist(y[:, 3:], y[:, 3:]))


def test_mirror_and_tag_swap_stay_consistent():
    from crystal_nft.meanflow.stereo import (
        chiral_signs,
        mirror_coords_in_place_x,
        swap_rs_tags,
    )

    x = torch.randn(1, 3 + 5, 3)
    centers = torch.tensor([[[0, 1, 2, 3]]])
    flip = torch.tensor([True])
    before = chiral_signs(x[:, 3:], centers)
    after = chiral_signs(mirror_coords_in_place_x(x, flip)[:, 3:], centers)
    assert torch.equal(after, -before)  # geometry flips
    tags = torch.tensor([[0, TAG_R, TAG_S, TAG_R, 0]])
    assert torch.equal(
        swap_rs_tags(tags, flip), torch.tensor([[0, TAG_S, TAG_R, TAG_S, 0]])
    )  # ...and so does the label, so the pair stays consistent


# ---------------------------------------------------------------------------
# Cache contract
# ---------------------------------------------------------------------------
class _StubCrystal:
    """Minimal stand-in exposing what `find_stereo_centers_full` reads."""

    def __init__(self, coords: torch.Tensor, csd_id: str = "STUB01"):
        self.coords = coords
        self.csd_id = csd_id
        self.num_atoms = coords.shape[0]
        self.bonds = torch.zeros(self.num_atoms, self.num_atoms, dtype=torch.long)
        for j in (1, 2, 3, 4):
            self.bonds[0, j] = self.bonds[j, 0] = 1


def test_cache_holds_only_graph_data_so_labels_track_coordinates(monkeypatch):
    """Regression: Clari's training collate permutes equivalent *bodies*
    (`Crystal.aligned_perm`) without touching `bonds`, so a csd_id-keyed cache of
    per-body R/S labels goes stale for a racemic cell.  Only the graph part may
    be cached; the labels must be re-derived from the coordinates every call."""
    from crystal_nft.meanflow import stereo as S

    coords = torch.tensor(
        [[0.0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    )
    calls = {"n": 0}
    centers = torch.tensor([[0, 1, 2, 3]])
    low = torch.tensor([4])
    frame = torch.tensor([True])

    def fake(crystal):
        calls["n"] += 1
        return centers.clone(), low.clone(), frame.clone()

    monkeypatch.setattr(S, "_find_centers_uncached", fake)
    S.clear_stereo_cache()

    _, tags_a, sign_a, _ = S.find_stereo_centers_full(_StubCrystal(coords))
    # Same csd_id, mirrored coordinates: served from cache, opposite handedness.
    mirrored = coords * torch.tensor([-1.0, 1, 1])
    _, tags_b, sign_b, _ = S.find_stereo_centers_full(_StubCrystal(mirrored))

    assert calls["n"] == 1, "the RDKit lookup should be cached"
    assert torch.equal(sign_b, -sign_a), "labels must follow the coordinates"
    assert {int(tags_a[0]), int(tags_b[0])} == {TAG_R, TAG_S}
    S.clear_stereo_cache()


def test_centres_without_a_cip_frame_are_marked_undefined(monkeypatch):
    from crystal_nft.meanflow import stereo as S
    from crystal_nft.meanflow.stereo import TAG_UNDEF

    coords = torch.tensor(
        [[0.0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    )
    monkeypatch.setattr(
        S,
        "_find_centers_uncached",
        lambda c: (
            torch.tensor([[0, 1, 2, 3]]),
            torch.tensor([-1]),
            torch.tensor([False]),  # CIP ranks tied -> not stereogenic
        ),
    )
    S.clear_stereo_cache()
    _, tags, sign, _ = S.find_stereo_centers_full(_StubCrystal(coords, "STUB02"))
    assert int(tags[0]) == TAG_UNDEF
    assert float(sign[0]) == 0.0  # excluded from PCFM / parity targets
    S.clear_stereo_cache()


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------
def _full_stub(device="cpu"):
    """Minimal crystal-shaped object with all fields `_CpuCell` reads."""

    class _S:
        pass

    o = _S()
    o.csd_id = "DEVICE01"
    o.num_atoms = 5
    o.coords = torch.tensor(
        [[0.0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], device=device
    )
    o.atom_nums = torch.tensor([6, 1, 6, 7, 8], device=device)
    o.atom_charges = torch.zeros(5, dtype=torch.long, device=device)
    o.body_ids = torch.zeros(5, dtype=torch.long, device=device)
    bonds = torch.zeros(5, 5, dtype=torch.long, device=device)
    for j in (1, 2, 3, 4):
        bonds[0, j] = bonds[j, 0] = 1
    o.bonds = bonds
    return o


def test_cpu_cell_snapshot_is_cpu_and_detached():
    from crystal_nft.meanflow import stereo as S

    cell = S._CpuCell(_full_stub())
    for t in (cell.atom_nums, cell.atom_charges, cell.bonds, cell.coords, cell.body_ids):
        assert t.device.type == "cpu" and not t.requires_grad
    assert S._body_signature(cell, 0, 5)
    assert S._body_spans(cell) == [(0, 5)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_perception_snapshots_cuda_tensors_to_cpu():
    """Regression: `_body_signature` called `.numpy()` straight on the batch,
    which raises on CUDA. The broad `except` turned that into "no
    stereocentres", so the whole stereo path was a silent no-op during training
    while every CPU test kept passing."""
    from crystal_nft.meanflow import stereo as S

    cell = S._CpuCell(_full_stub(device="cuda"))
    assert cell.atom_nums.device.type == "cpu"
    assert S._body_signature(cell, 0, 5)  # raised before the fix
    assert S._body_spans(cell) == [(0, 5)]


def test_perception_failures_are_reported_not_swallowed(caplog):
    """A swallowed exception silently means "achiral"; it must at least warn."""
    import logging

    from crystal_nft.meanflow import stereo as S

    class _Broken:
        csd_id = "BROKEN01"

        def __getattr__(self, name):
            raise RuntimeError("boom")

    S.clear_stereo_cache()
    S._WARNED_ERRORS.clear()
    with caplog.at_level(logging.WARNING, logger="crystal_nft.meanflow.stereo"):
        out = S._find_centers_uncached(_Broken())
    assert out == (None, None, None)
    assert any("stereo perception failed" in r.message for r in caplog.records)
    S._WARNED_ERRORS.clear()
    S.clear_stereo_cache()


# ---------------------------------------------------------------------------
# Hinge sample weighting (the fix for the x_r shortcut)
# ---------------------------------------------------------------------------
def test_hinge_sample_weight_selects_samples():
    """cont4 trained the tag to chance because at moderate r the noised state
    already reveals handedness, so the hinge was satisfiable without the tag.
    The weight lets the hinge be concentrated near r=0, where it is not."""
    ref = torch.stack([_tetra(), _tetra()])
    pred = ref * torch.tensor([-1.0, 1, 1])  # both wrong-handed
    spec = _spec(2)

    both = chiral_hinge_loss(pred, ref, spec, margin=0.2)
    only_first = chiral_hinge_loss(
        pred, ref, spec, margin=0.2, sample_weight=torch.tensor([1.0, 0.0])
    )
    assert abs(float(both) - 1.2) < 1e-5
    # Masking one sample must not change the *mean* over the surviving ones.
    assert abs(float(only_first) - 1.2) < 1e-5


def test_hinge_sample_weight_zero_everywhere_is_finite():
    """A batch entirely outside the r-window must not divide by zero."""
    ref = torch.stack([_tetra(), _tetra()])
    pred = ref * torch.tensor([-1.0, 1, 1])
    out = chiral_hinge_loss(
        pred, ref, _spec(2), margin=0.2, sample_weight=torch.zeros(2)
    )
    assert torch.isfinite(out) and float(out) == 0.0


def test_hinge_sample_weight_scales_partial_credit():
    """A soft ramp weights samples continuously rather than all-or-nothing."""
    ref = torch.stack([_tetra(), _tetra()])
    pred = torch.stack([_tetra() * torch.tensor([-1.0, 1, 1]), _tetra()])
    spec = _spec(2)
    # sample 0 is wrong-handed, sample 1 is correct; weighting only sample 0
    # must give the full penalty, weighting only sample 1 must give zero.
    w0 = chiral_hinge_loss(pred, ref, spec, margin=0.2, sample_weight=torch.tensor([1.0, 0.0]))
    w1 = chiral_hinge_loss(pred, ref, spec, margin=0.2, sample_weight=torch.tensor([0.0, 1.0]))
    assert float(w0) > float(w1)
    assert abs(float(w1)) < 1e-6
