"""Physics-Constrained Flow Matching (PCFM) projection for CrystAF sampling.

Cai et al., *Enforcing constraints in molecular and crystalline generative models
via physics-constrained flow matching* (ICLR AI4Mat 2026), extending Utkarsh
et al. (2025).  At an integration step the sampler looks ahead to an endpoint
estimate ``x̂(1)``, applies one Gauss–Newton projection towards the constraint
set, and re-aims the step at the corrected endpoint::

    x <- x - Jᵀ (J Jᵀ + εI)⁻¹ h(x),      J = ∇_x h(x)

For R/S tetrahedral chirality the paper's residual is a hinge on the normalised
signed volume at each stereocentre (their margin ``m_RS = 0.62``)::

    τ(x)   = (p₁-c)·[(p₂-c)×(p₃-c)] / (‖p₁-c‖‖p₂-c‖‖p₃-c‖)
    h_RS(x) = max(0, m_RS - tag·τ(x))

Nothing here is trained: it turns an existing CrystAF checkpoint into a
constrained sampler.  The Jacobian is written out in closed form rather than
taken with autograd — each residual touches only four atoms, and the projection
runs inside the sampling loop for every crystal at every step.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

_EPS = 1e-8
PCFM_RS_MARGIN = 0.62  # Cai et al., Sec. 3 "R/S tetrahedral chirality"


@dataclass
class ChiralityConstraint:
    """Target handedness at each tetrahedral stereocentre of a batch."""

    centers: Tensor  # (B, M, 4) long: [centre, p1, p2, p3] in CIP order
    target: Tensor  # (B, M) float: +1 / -1 desired sign of tau, 0 = inactive
    margin: float = PCFM_RS_MARGIN
    body_ids: Optional[Tensor] = None  # (B, N) long, which molecule each atom is in

    @property
    def n_active(self) -> int:
        return int((self.target != 0).sum().item())

    def to(self, device) -> "ChiralityConstraint":
        return ChiralityConstraint(
            centers=self.centers.to(device),
            target=self.target.to(device),
            margin=self.margin,
            body_ids=None if self.body_ids is None else self.body_ids.to(device),
        )


def _frames(coords: Tensor, centers: Tensor):
    """Unit substituent vectors and their lengths at every centre."""
    b, m, _ = centers.shape
    idx = centers.reshape(b, m * 4, 1).expand(b, m * 4, 3)
    p = torch.gather(coords, 1, idx).reshape(b, m, 4, 3)
    e = p[:, :, 1:, :] - p[:, :, :1, :]  # (B, M, 3, 3)
    n = e.norm(dim=-1).clamp_min(_EPS)  # (B, M, 3)
    return e / n.unsqueeze(-1), n


def chirality_residual(coords: Tensor, cst: ChiralityConstraint) -> Tensor:
    """``h_RS`` per centre, (B, M). Zero where the target is inactive."""
    e_hat, _ = _frames(coords, cst.centers)
    tau = (
        torch.linalg.cross(e_hat[:, :, 0], e_hat[:, :, 1]) * e_hat[:, :, 2]
    ).sum(-1)
    h = torch.relu(cst.margin - cst.target * tau)
    return torch.where(cst.target == 0, torch.zeros_like(h), h)


def _chirality_jacobian(coords: Tensor, cst: ChiralityConstraint) -> Tensor:
    """Dense ``∂h/∂coords`` of shape (B, M, N*3), built from the closed form.

    With ``ê_k`` the unit substituent vectors and ``g₁ = ê₂×ê₃`` (cyclically),
    ``∂τ/∂p_k = (g_k - τ ê_k)/‖e_k‖`` and ``∂τ/∂c = -Σ_k ∂τ/∂p_k``; the hinge
    contributes ``∂h/∂x = -tag ∂τ/∂x`` only where it is active.
    """
    b, m, _ = cst.centers.shape
    n_atoms = coords.shape[1]
    e_hat, norms = _frames(coords, cst.centers)
    e1, e2, e3 = e_hat[:, :, 0], e_hat[:, :, 1], e_hat[:, :, 2]
    tau = (torch.linalg.cross(e1, e2) * e3).sum(-1)  # (B, M)
    g = torch.stack(
        [
            torch.linalg.cross(e2, e3),
            torch.linalg.cross(e3, e1),
            torch.linalg.cross(e1, e2),
        ],
        dim=2,
    )  # (B, M, 3, 3)
    dtau_dp = (g - tau[..., None, None] * e_hat) / norms.unsqueeze(-1)
    dtau_dc = -dtau_dp.sum(dim=2, keepdim=True)  # (B, M, 1, 3)
    dtau = torch.cat([dtau_dc, dtau_dp], dim=2)  # (B, M, 4, 3)

    active = ((cst.target != 0) & (cst.margin - cst.target * tau > 0)).to(coords.dtype)
    dh = -(cst.target * active)[..., None, None] * dtau

    jac = coords.new_zeros(b, m, n_atoms, 3)
    idx = cst.centers.unsqueeze(-1).expand(b, m, 4, 3)
    # scatter_add (not scatter): a centre can also be a substituent of a
    # neighbouring stereocentre, and adjacent centres must accumulate.
    jac.scatter_add_(2, idx, dh)
    return jac.reshape(b, m, n_atoms * 3)


def project_chirality(
    coords: Tensor,
    cst: ChiralityConstraint,
    *,
    n_iter: int = 3,
    ridge: float = 1e-4,
    tol: float = 1e-6,
    max_shift: float = 1.0,
) -> Tensor:
    """Gauss–Newton projection of ``coords`` (B, N, 3) onto the R/S constraint set.

    ``max_shift`` caps the per-atom displacement of a single projection so a
    badly-conditioned centre cannot tear the packing apart.
    """
    if cst.n_active == 0:
        return coords
    x = coords.float()
    b, n_atoms, _ = x.shape
    m = cst.centers.shape[1]
    eye = torch.eye(m, device=x.device, dtype=x.dtype).unsqueeze(0)
    for _ in range(int(n_iter)):
        h = chirality_residual(x, cst)
        if float(h.max()) <= tol:
            break
        jac = _chirality_jacobian(x, cst)  # (B, M, 3N)
        jjt = jac @ jac.transpose(1, 2)  # (B, M, M)
        y = torch.linalg.solve(jjt + ridge * eye, h.unsqueeze(-1))  # (B, M, 1)
        delta = (jac.transpose(1, 2) @ y).reshape(b, n_atoms, 3)
        if max_shift > 0:
            norm = delta.norm(dim=-1, keepdim=True)
            scale = (max_shift / norm.clamp_min(max_shift)).clamp(max=1.0)
            delta = delta * scale
        x = x - delta
    return x.to(coords.dtype)


def constraint_from_batch_stereo(spec, *, margin: float = PCFM_RS_MARGIN, body_ids=None):
    """Build a :class:`ChiralityConstraint` from a :class:`BatchStereo`.

    Only centres with a canonical CIP frame are enforced.  Their target is the
    sign of ``tau`` in that frame, which is exactly the R/S label (verified to
    agree with RDKit's own label on 99.2% of CSD val stereocentres, the
    remainder being RDKit CIP edge cases) — one bit of molecular identity per
    centre, carrying no information about the target packing.
    """
    if spec is None or spec.label_sign is None:
        return None
    target = torch.where(spec.valid, spec.label_sign, torch.zeros_like(spec.label_sign))
    if int((target != 0).sum().item()) == 0:
        return None
    return ChiralityConstraint(
        centers=spec.centers,
        target=target,
        margin=float(margin),
        body_ids=body_ids,
    )


# --------------------------------------------------------------------------
# Global parity correction
# --------------------------------------------------------------------------
@torch.no_grad()
def _tau(coords: Tensor, centers: Tensor) -> Tensor:
    e_hat, _ = _frames(coords, centers)
    return (torch.linalg.cross(e_hat[:, :, 0], e_hat[:, :, 1]) * e_hat[:, :, 2]).sum(-1)


@torch.no_grad()
def resolve_global_handedness(coords: Tensor, cst: ChiralityConstraint) -> Tensor:
    """Invert a whole cell when its overall handedness is the wrong one.

    A crystal and its enantiomorph are both valid structures; when the input
    molecule's stereochemistry is specified, picking the matching one is part of
    the sampler (ET-Flow calls this post-hoc parity correction).  Inverting *all*
    coordinates through the origin preserves every interatomic distance — intra-
    and intermolecular — so ``pb_score``, ``clash_rate`` and the density are
    bit-for-bit unchanged while every stereocentre flips.

    Only decides on the *labels*, never on reference coordinates.  It cannot fix
    a cell whose centres disagree with each other (a racemic target where the
    model got the pattern wrong); that is what the projection is for.
    """
    if cst.n_active == 0:
        return coords
    tau = _tau(coords, cst.centers)
    active = cst.target != 0
    agree = ((torch.sign(tau) == torch.sign(cst.target)) & active).sum(1)
    n = active.sum(1).clamp_min(1)
    flip = (2 * agree < n).view(-1, 1, 1)
    return torch.where(flip, -coords, coords)


@torch.no_grad()
def _kabsch(p: Tensor, q: Tensor) -> Tensor:
    """Proper rotation (det = +1) best aligning centred ``p`` onto centred ``q``."""
    h = p.transpose(0, 1) @ q
    u, _, vt = torch.linalg.svd(h)
    d = torch.sign(torch.det(vt.transpose(0, 1) @ u.transpose(0, 1)))
    corr = torch.eye(3, device=p.device, dtype=p.dtype)
    corr[2, 2] = d
    return vt.transpose(0, 1) @ corr @ u.transpose(0, 1)


@torch.no_grad()
def resolve_body_handedness(
    coords: Tensor,
    cst: ChiralityConstraint,
    *,
    realign: bool = True,
    minimize_flips: bool = True,
) -> Tensor:
    """Invert individual molecules whose handedness disagrees with their labels.

    A chirality-blind generator picks each molecule's handedness independently,
    so a multi-molecule cell comes out mixed and a *global* inversion cannot fix
    it.  Inverting a single body through its own centroid flips exactly that
    molecule's stereocentres while preserving all of its internal distances — and
    PoseBusters scores each fragment on intramolecular geometry alone, so
    ``pb_score`` is invariant.  Only the packing (hence ``clash_rate`` and
    ``dist_pdd``) moves, and that is the whole cost of this operation.

    With ``realign`` the inverted body is then rotated by the proper rotation
    that best matches its original pose (Kabsch, det = +1), which keeps it as
    close to the packing site the model chose as an enantiomer can get.

    ``minimize_flips`` exploits the fact that inverting the *whole* cell is a
    true isometry — every distance, intra- and intermolecular, is preserved, so
    it is free in every Table 1 metric.  Flipping ``k`` of the ``Z`` chiral
    bodies and flipping the other ``Z - k`` then inverting the cell reach the
    same stereochemistry, so we take whichever needs fewer *body* flips.  The
    extreme case matters most: a uniform-but-wrong cell (``k = Z``) costs ``Z``
    damaging flips the old way and **zero** this way.
    """
    if cst.n_active == 0 or cst.body_ids is None:
        return coords
    out = coords.clone()
    tau = _tau(coords, cst.centers)
    ok = (torch.sign(tau) == torch.sign(cst.target)) & (cst.target != 0)
    active = cst.target != 0
    b_n = coords.shape[0]
    for b in range(b_n):
        if not bool(active[b].any()):
            continue
        bodies = cst.body_ids[b]
        centre_body = bodies[cst.centers[b, :, 0]]
        # per chiral body: does it currently disagree with its target?
        need: dict[int, bool] = {}
        for body in torch.unique(centre_body[active[b]]).tolist():
            sel = active[b] & (centre_body == body)
            n_sel = int(sel.sum())
            if n_sel == 0:
                continue
            need[body] = 2 * int(ok[b][sel].sum()) < n_sel
        if not need:
            continue
        n_wrong = sum(need.values())
        # flip the complement + invert the cell when that is the cheaper route
        invert_cell = bool(minimize_flips and 2 * n_wrong > len(need))
        for body, wrong in need.items():
            if wrong == invert_cell:
                continue  # cell inversion will (or already does) handle this one
            atoms = torch.nonzero(bodies == body, as_tuple=False).flatten()
            if atoms.numel() == 0:
                continue
            orig = coords[b, atoms]
            cen = orig.mean(dim=0, keepdim=True)
            inv = cen - (orig - cen)
            if realign:
                rot = _kabsch((inv - cen).double(), (orig - cen).double())
                inv = (inv - cen).double() @ rot.transpose(0, 1)
                inv = inv.to(orig.dtype) + cen
            out[b, atoms] = inv
        if invert_cell:
            out[b] = -out[b]
    return out


# --------------------------------------------------------------------------
# Exact single-centre inversion (substituent swap)
# --------------------------------------------------------------------------
def _adjacency(bonds: Tensor) -> list:
    """Neighbour lists from an (N, N) bond matrix."""
    idx = torch.nonzero(bonds > 0, as_tuple=False).tolist()
    adj: dict[int, list] = {}
    for u, v in idx:
        adj.setdefault(u, []).append(v)
    return adj


def _branch(adj, start: int, blocked: int) -> set:
    """Atoms reachable from ``start`` without passing through ``blocked``."""
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):  # noqa: SIM118
            if v == blocked or v in seen:
                continue
            seen.add(v)
            stack.append(v)
    return seen


@torch.no_grad()
def swap_fix_stereocentres(
    coords: Tensor,
    cst: ChiralityConstraint,
    bonds: Tensor,
    *,
    max_branch: int = 40,
) -> Tensor:
    """Invert individual wrong stereocentres *exactly*, by swapping two substituents.

    A whole-molecule inversion flips every centre at once, so it cannot repair a
    molecule whose centres are wrong in a mixed pattern -- which is where all of
    the post-flip residual lives. Swapping any two substituents of a tetrahedral
    centre inverts just that centre.

    The swap is done with the 180-degree rotation about the bisector of the two
    bond directions, ``R = 2 a a^T - I`` with ``a = normalise(u_i + u_j)``. That
    maps ``u_i <-> u_j`` and is a *proper* rotation (det = +1), so each branch is
    moved rigidly: its internal bond lengths and angles are preserved exactly,
    its own stereocentres keep their handedness, and both swapped substituents
    keep their exact bond length to the centre. Only the packing around the two
    branches changes -- which the clash relaxation afterwards can repair.

    Pairs whose branches are joined by a ring through the centre are skipped
    (cutting the two bonds does not separate them, so the "branch" is the whole
    molecule and rotating it is not a local edit).
    """
    if cst.n_active == 0:
        return coords
    out = coords.clone()
    b_n = coords.shape[0]
    for b in range(b_n):
        adj = _adjacency(bonds[b] if bonds.dim() == 3 else bonds)
        if not adj:
            continue
        for m in range(cst.centers.shape[1]):
            tgt = float(cst.target[b, m])
            if tgt == 0.0:
                continue
            cur = float(_tau(out[b : b + 1], cst.centers[b : b + 1, m : m + 1])[0, 0])
            if cur * tgt > 0:
                continue  # already correct
            c = int(cst.centers[b, m, 0])
            # Take substituents from the bond graph, not from `centers`, which
            # records only 3 of the 4. With three, 58% of centres have no
            # swappable pair at all -- for a ring centre two of the three are
            # ring atoms and every pair among them is ring-locked. Including the
            # fourth (usually H) drops that to 4%.
            subs = sorted(set(adj.get(c, ())))
            if len(subs) < 2:
                continue
            best = None
            other = set(subs)
            for ii in range(len(subs)):
                for jj in range(ii + 1, len(subs)):
                    pi, pj = subs[ii], subs[jj]
                    bi = _branch(adj, pi, c)
                    bj = _branch(adj, pj, c)
                    # Each branch must hold exactly one substituent of the centre.
                    # If a branch sweeps up a second one (a ring through the
                    # centre), rotating it turns several bonds together and
                    # det[R v1, R v2, R v3] = det(R) tau = +tau -- the centre does
                    # not invert. For a ring centre the swappable pair is the two
                    # exocyclic substituents.
                    if (bi & other) != {pi} or (bj & other) != {pj}:
                        continue
                    if bi & bj:
                        continue
                    size = max(len(bi), len(bj))
                    if size > max_branch:
                        continue
                    if best is None or size < best[0]:
                        best = (size, pi, pj, bi, bj)
            if best is None:
                continue
            _, pi, pj, bi, bj = best
            cen = out[b, c]
            ui = out[b, pi] - cen
            uj = out[b, pj] - cen
            ni, nj = ui.norm().clamp_min(_EPS), uj.norm().clamp_min(_EPS)
            a = ui / ni + uj / nj
            if float(a.norm()) < 1e-4:
                continue  # anti-parallel bonds: bisector undefined
            a = a / a.norm()
            rot = 2.0 * torch.outer(a, a) - torch.eye(3, device=a.device, dtype=a.dtype)
            atoms = torch.tensor(sorted(bi | bj), device=out.device, dtype=torch.long)
            moved = (out[b, atoms] - cen) @ rot.transpose(0, 1) + cen
            prev = out[b, atoms].clone()
            out[b, atoms] = moved
            new = float(_tau(out[b : b + 1], cst.centers[b : b + 1, m : m + 1])[0, 0])
            if new * tgt <= 0:
                out[b, atoms] = prev  # did not actually invert; leave it alone
    return out



def _pair_lb(atom_nums: Optional[Tensor], rows: Tensor, cols: Tensor) -> Optional[Tensor]:
    """(len(rows), len(cols)) sum of covalent radii, the same bound clash uses."""
    if atom_nums is None:
        return None
    from clari.chem.common import element_radii

    r = torch.tensor(
        [element_radii(int(z), "cov") for z in atom_nums.tolist()],
        dtype=torch.float32, device=atom_nums.device,
    )
    return r[rows].unsqueeze(-1) + r[cols].unsqueeze(0)


@torch.no_grad()
def reflect_fix_stereocentres(
    coords: Tensor,
    cst: ChiralityConstraint,
    bonds: Tensor,
    *,
    max_rounds: int = 6,
    atom_nums: Optional[Tensor] = None,
    n_theta: int = 24,
) -> Tensor:
    """Fix residual stereocentres by reflecting branches through their own bond.

    Reflecting the branch hanging off an acyclic bond ``u-v``, through a plane
    that *contains* that bond, inverts every stereocentre inside the branch while
    leaving the structure geometrically intact:

    * ``u`` and ``v`` both lie on the plane, so the bond length and every bond
      angle at the junction are preserved exactly;
    * a reflection is an isometry, so all distances inside the branch are exact;
    * its determinant is -1, so each enclosed stereocentre inverts.

    Whole-molecule inversion (:func:`resolve_body_handedness`) is the special
    case where the branch is everything, and a substituent swap is the special
    case of a one-atom branch. This generalises both: for a molecule whose
    centres are wrong in a mixed pattern, a branch containing just the wrong ones
    fixes them without disturbing the rest.

    Greedy: each round applies the reflection that removes the most errors, so a
    molecule needing several independent corrections converges over rounds.
    """
    if cst.n_active == 0:
        return coords
    out = coords.clone()
    for b in range(coords.shape[0]):
        adj = _adjacency(bonds[b] if bonds.dim() == 3 else bonds)
        if not adj:
            continue
        atom_nums_b = None
        if atom_nums is not None:
            atom_nums_b = atom_nums[b] if atom_nums.dim() == 2 else atom_nums
        active = [m for m in range(cst.centers.shape[1]) if float(cst.target[b, m]) != 0.0]
        if not active:
            continue
        cen_atom = {m: int(cst.centers[b, m, 0]) for m in active}
        for _ in range(int(max_rounds)):
            tau = _tau(out[b : b + 1], cst.centers[b : b + 1])[0]
            wrong = {m for m in active if float(tau[m]) * float(cst.target[b, m]) <= 0}
            if not wrong:
                break
            # only bonds inside a molecule that still has an error are worth trying
            live = set()
            for m in wrong:
                live |= _branch(adj, cen_atom[m], -1)
            best = None
            for u in sorted(live):
                for v in adj.get(u, ()):  # noqa: SIM118
                    if v <= u:
                        continue
                    br = _branch(adj, v, u)
                    if u in br:
                        continue  # ring bond: cutting it separates nothing
                    inside = {m for m in active if cen_atom[m] in br}
                    if not inside:
                        continue
                    gain = len(inside & wrong) - len(inside - wrong)
                    if gain > 0 and (best is None or gain > best[0]):
                        best = (gain, u, v, br)
            if best is None:
                break
            _, u, v, br = best
            org = out[b, u]
            d = out[b, v] - org
            dn = d.norm().clamp_min(_EPS)
            d = d / dn
            atoms = torch.tensor(sorted(br), device=out.device, dtype=torch.long)
            rel = out[b, atoms] - org
            tmp = torch.tensor([1.0, 0.0, 0.0], device=d.device, dtype=d.dtype)
            if float(torch.abs(d @ tmp)) > 0.9:
                tmp = torch.tensor([0.0, 1.0, 0.0], device=d.device, dtype=d.dtype)
            e1 = tmp - (tmp @ d) * d
            e1 = e1 / e1.norm().clamp_min(_EPS)
            e2 = torch.linalg.cross(d, e1)
            n0 = e1
            base = rel - 2.0 * (rel @ n0).unsqueeze(-1) * n0
            # Reflection fixes chirality but swings the branch into a new place,
            # where it can collide with the rest of its OWN molecule -- and the
            # clash relaxation downstream is rigid-body and intermolecular, so it
            # cannot undo that. Rotating about the anchoring bond afterwards is a
            # pure torsion change: it preserves every bond length and angle, so
            # `reflect then rotate(theta)` is valid for any theta. Scan theta and
            # keep the one that packs best against the rest of the structure.
            rest = torch.ones(out.shape[1], dtype=torch.bool, device=out.device)
            rest[atoms] = False
            if bool(rest.any()):
                other = out[b, rest]
                lb = _pair_lb(atom_nums_b, atoms, rest) if atom_nums_b is not None else None
                best_t, best_e = None, None
                for k in range(int(n_theta)):
                    th = 2.0 * math.pi * k / int(n_theta)
                    ct, st = math.cos(th), math.sin(th)
                    rot = (
                        ct * torch.eye(3, device=d.device, dtype=d.dtype)
                        + st * torch.tensor(
                            [[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]],
                            device=d.device, dtype=d.dtype)
                        + (1 - ct) * torch.outer(d, d)
                    )
                    cand = base @ rot.transpose(0, 1)
                    dist = torch.cdist(cand + org, other).clamp_min(_EPS)
                    thr = lb if lb is not None else torch.full_like(dist, 1.6)
                    e = torch.relu(thr - dist).pow(2).sum()
                    if best_e is None or float(e) < float(best_e):
                        best_e, best_t = e, cand
                out[b, atoms] = org + best_t
            else:
                out[b, atoms] = org + base
    return out


@torch.no_grad()
def fix_stereocentres_exact(
    coords: Tensor,
    cst: ChiralityConstraint,
    bonds: Tensor,
    *,
    rounds: int = 4,
    atom_nums: Optional[Tensor] = None,
    n_theta: int = 24,
) -> Tensor:
    """Alternate branch reflection and substituent swap until neither helps.

    The two exact operations are complementary and neither dominates: reflection
    handles a wrong *group* of centres separated by a rotatable bond but cannot
    touch a centre locked inside a fused ring system; the swap fixes exactly one
    centre but needs two clean exocyclic branches. Applying them once each in
    sequence leaves fixes on the table, because a swap can expose a branch whose
    reflection now has positive gain, and vice versa. Iterate to a fixed point.
    """
    out = coords
    for _ in range(int(rounds)):
        before = out
        out = reflect_fix_stereocentres(
            out, cst, bonds, atom_nums=atom_nums, n_theta=n_theta
        )
        out = swap_fix_stereocentres(out, cst, bonds)
        if torch.equal(out, before):
            break
    return out


# --------------------------------------------------------------------------
# Bond-length constraint (PCFM Sec. 3 "Bond length bounds")
# --------------------------------------------------------------------------
@dataclass
class BondConstraint:
    """RDKit distance-geometry bounds on every bonded pair.

    Unlike the chirality residual, this one targets a check Clari's PoseBusters
    config *actually runs* (``bond_lengths_within_bounds`` via its
    ``distance_geometry`` module), so satisfying it can move ``pb_score``.
    Cai et al. report a bond-length pass rate of 0.994 with this.

        h_ij = [relu((L_ij - d_ij)/L_ij), relu((d_ij - U_ij)/U_ij)]
    """

    pairs: Tensor  # (B, P, 2) long
    lo: Tensor  # (B, P)
    hi: Tensor  # (B, P)
    valid: Tensor  # (B, P) bool

    @property
    def n_active(self) -> int:
        return int(self.valid.sum().item())

    def to(self, device) -> "BondConstraint":
        return BondConstraint(
            pairs=self.pairs.to(device), lo=self.lo.to(device),
            hi=self.hi.to(device), valid=self.valid.to(device),
        )


def _pair_dists(coords: Tensor, pairs: Tensor):
    b, p, _ = pairs.shape
    i = pairs[..., 0].unsqueeze(-1).expand(b, p, 3)
    j = pairs[..., 1].unsqueeze(-1).expand(b, p, 3)
    xi = torch.gather(coords, 1, i)
    xj = torch.gather(coords, 1, j)
    d = xj - xi
    return d, d.norm(dim=-1).clamp_min(_EPS)


def bond_residual(coords: Tensor, cst: BondConstraint) -> Tensor:
    """(B, P) one-sided violation, positive only outside [lo, hi]."""
    _, dist = _pair_dists(coords, cst.pairs)
    lo_v = torch.relu((cst.lo - dist) / cst.lo.clamp_min(_EPS))
    hi_v = torch.relu((dist - cst.hi) / cst.hi.clamp_min(_EPS))
    h = lo_v + hi_v
    return torch.where(cst.valid, h, torch.zeros_like(h))


@torch.no_grad()
def project_bonds(
    coords: Tensor,
    cst: BondConstraint,
    *,
    n_iter: int = 4,
    tol: float = 1e-6,
    max_shift: float = 0.25,
) -> Tensor:
    """Push each bonded pair back inside its DG bounds.

    Each residual involves one pair, so the Gauss-Newton step is available in
    closed form: move both atoms along the bond by half the required correction.
    Overlapping pairs are handled by accumulating and averaging per atom, which
    is a Jacobi sweep rather than an exact solve -- hence several iterations.
    """
    if cst.n_active == 0:
        return coords
    x = coords.float()
    for _ in range(int(n_iter)):
        vec, dist = _pair_dists(x, cst.pairs)
        lo_v = (cst.lo - dist)
        hi_v = (dist - cst.hi)
        # signed correction: >0 means "lengthen", <0 means "shorten"
        corr = torch.where(lo_v > 0, lo_v, torch.where(hi_v > 0, -hi_v, torch.zeros_like(dist)))
        corr = torch.where(cst.valid, corr, torch.zeros_like(corr))
        if float(corr.abs().max()) <= tol:
            break
        unit = vec / dist.unsqueeze(-1)
        delta = 0.5 * corr.unsqueeze(-1) * unit  # each atom moves half
        upd = torch.zeros_like(x)
        cnt = torch.zeros(x.shape[:2], device=x.device, dtype=x.dtype)
        b, p, _ = cst.pairs.shape
        idx_i = cst.pairs[..., 0].unsqueeze(-1).expand(b, p, 3)
        idx_j = cst.pairs[..., 1].unsqueeze(-1).expand(b, p, 3)
        upd.scatter_add_(1, idx_i, -delta)
        upd.scatter_add_(1, idx_j, delta)
        ones = cst.valid.to(x.dtype)
        cnt.scatter_add_(1, cst.pairs[..., 0], ones)
        cnt.scatter_add_(1, cst.pairs[..., 1], ones)
        upd = upd / cnt.clamp_min(1.0).unsqueeze(-1)
        if max_shift > 0:
            n = upd.norm(dim=-1, keepdim=True)
            upd = upd * (max_shift / n.clamp_min(max_shift)).clamp(max=1.0)
        x = x + upd
    return x.to(coords.dtype)


# --------------------------------------------------------------------------
# Sampler-facing state
# --------------------------------------------------------------------------
_ACTIVE: Optional[ChiralityConstraint] = None
_ACTIVE_BOND = None


def set_active_bond_constraint(cst) -> None:
    global _ACTIVE_BOND
    _ACTIVE_BOND = cst


def get_active_bond_constraint():
    return _ACTIVE_BOND


class active_bond_constraint:
    """Context manager publishing the bond-length constraint to the sampler."""

    def __init__(self, cst):
        self.cst = cst
        self._prev = None

    def __enter__(self):
        global _ACTIVE_BOND
        self._prev = _ACTIVE_BOND
        _ACTIVE_BOND = self.cst
        return self.cst

    def __exit__(self, *exc):
        global _ACTIVE_BOND
        _ACTIVE_BOND = self._prev
        return False


def set_active_constraint(cst: Optional[ChiralityConstraint]) -> None:
    global _ACTIVE
    _ACTIVE = cst


def get_active_constraint() -> Optional[ChiralityConstraint]:
    return _ACTIVE


class active_constraint:
    """Context manager publishing a constraint to the CrystAF sampler."""

    def __init__(self, cst: Optional[ChiralityConstraint]):
        self.cst = cst
        self._prev: Optional[ChiralityConstraint] = None

    def __enter__(self):
        global _ACTIVE
        self._prev = _ACTIVE
        _ACTIVE = self.cst
        return self.cst

    def __exit__(self, *exc):
        global _ACTIVE
        _ACTIVE = self._prev
        return False
