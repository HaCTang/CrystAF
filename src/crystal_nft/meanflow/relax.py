"""Rigid-body clash relaxation.

``clash_rate`` in Table 1 is ``check_clashes_eval``: a *binary* per-crystal flag,
1 iff some intermolecular atom pair sits closer than the sum of the two covalent
radii, measured with periodic images.  One bad contact condemns the whole
crystal, so removing the worst few contacts is enough to flip a crystal's score.

The key structural fact this exploits: translating or rotating a whole molecular
body is an isometry *of that body*, so

* ``pb_score`` is exactly preserved -- PoseBusters scores each fragment on
  intramolecular geometry alone (``mol_pred`` only, no packing checks), and
* ``volume_error`` is exactly preserved -- the lattice is never touched.

Only ``clash_rate`` and ``dist_pdd`` can move at all.  We descend on the clash
energy and keep a spring to the original pose so the packing the model chose --
which is what ``dist_pdd`` measures -- is not thrown away.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["relax_clashes", "clash_energy", "covalent_lower_bounds"]

_EPS = 1e-8


def covalent_lower_bounds(atom_nums: Tensor) -> Tensor:
    """(N, N) sum of covalent radii -- exactly the bound ``check_clashes_eval`` uses."""
    from clari.chem.common import element_radii

    r = torch.tensor(
        [element_radii(int(z), "cov") for z in atom_nums.tolist()],
        dtype=torch.float32,
        device=atom_nums.device,
    )
    return r.unsqueeze(-1) + r


def _min_image_deltas(cart: Tensor, lattice: Tensor, n_shell: int = 1) -> Tensor:
    """(N, N, 3) minimum-image displacement vectors under periodicity.

    Wrapping the fractional difference into [-0.5, 0.5) is only the true minimum
    image for an orthogonal cell; for a skewed one the nearest image can sit in a
    neighbouring cell, so we additionally search the surrounding shell -- the
    same reason pymatgen's ``get_all_distances`` is used in the metric itself.
    """
    inv = torch.linalg.inv(lattice)
    d_cart = cart.unsqueeze(0) - cart.unsqueeze(1)  # (N, N, 3)
    d_frac = d_cart @ inv
    d_frac = d_frac - torch.round(d_frac)  # into [-0.5, 0.5)
    rng = torch.arange(-n_shell, n_shell + 1, device=cart.device, dtype=cart.dtype)
    shifts = torch.cartesian_prod(rng, rng, rng) @ lattice  # (S, 3)
    cand = (d_frac @ lattice).unsqueeze(2) + shifts.view(1, 1, -1, 3)  # (N, N, S, 3)
    idx = cand.pow(2).sum(-1).argmin(dim=-1, keepdim=True)  # (N, N, 1)
    return torch.gather(cand, 2, idx.unsqueeze(-1).expand(-1, -1, 1, 3)).squeeze(2)


def clash_energy(
    cart: Tensor,
    lattice: Tensor,
    inter: Tensor,
    lb: Tensor,
    *,
    slack: float = 0.0,
) -> Tensor:
    """Sum of squared violations of the covalent-radii bound over intermolecular pairs."""
    d = _min_image_deltas(cart, lattice).pow(2).sum(-1).clamp_min(_EPS).sqrt()
    viol = torch.relu((lb + slack) - d) * inter
    return viol.pow(2).sum() * 0.5  # each pair counted twice


def _rodrigues(omega: Tensor) -> Tensor:
    """(B, 3, 3) rotation matrices from axis-angle vectors, differentiably."""
    theta = omega.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    k = omega / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = torch.zeros_like(kx)
    kmat = torch.stack(
        [zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], dim=-1
    ).reshape(*omega.shape[:-1], 3, 3)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(kmat)
    s = torch.sin(theta).unsqueeze(-1)
    c = (1.0 - torch.cos(theta)).unsqueeze(-1)
    return eye + s * kmat + c * (kmat @ kmat)


def relax_clashes(
    cart: Tensor,
    lattice: Tensor,
    body_ids: Tensor,
    atom_nums: Tensor,
    *,
    n_iter: int = 60,
    lr: float = 0.05,
    slack: float = 0.05,
    spring: float = 0.02,
    max_shift: float = 1.5,
    early_stop: bool = True,
) -> Tensor:
    """Nudge whole molecules apart until no intermolecular covalent-radii overlap.

    Returns new Cartesian coordinates (same shape as ``cart``).  Every molecule
    moves only as a rigid body, so ``pb_score`` and ``volume_error`` are exactly
    unchanged by construction; ``spring`` and ``max_shift`` bound how far the
    packing may drift, which is what protects ``dist_pdd``.

    ``slack`` targets a small margin *inside* the metric's threshold so a
    borderline contact does not fall back over it.
    """
    n = cart.shape[0]
    if n == 0:
        return cart
    # The eval samples under `torch.inference_mode()`, which is stronger than
    # `no_grad`: every tensor born inside it is an *inference tensor* that can
    # never take part in autograd, and `enable_grad()` does not lift that.
    # Cloning just the inputs is not enough -- anything derived from them later
    # would still be created in inference mode and fail with "Inference tensors
    # cannot be saved for backward". The whole computation has to run outside.
    with torch.inference_mode(False), torch.enable_grad():
        cart = cart.detach().clone()
        lat = lattice.detach().clone().to(cart.dtype)
        body_ids = body_ids.detach().clone()
        atom_nums = atom_nums.detach().clone()

        lb = covalent_lower_bounds(atom_nums).to(cart.dtype)
        inter = (body_ids.unsqueeze(-1) != body_ids.unsqueeze(-2)).to(cart.dtype)
        inter.fill_diagonal_(0.0)

        if float(clash_energy(cart, lat, inter, lb, slack=slack)) <= 0.0:
            return cart  # already clean at the target margin

        uniq = torch.unique(body_ids)
        # (N, B) one-hot so per-body parameters broadcast to atoms
        sel = (body_ids.unsqueeze(-1) == uniq.unsqueeze(0)).to(cart.dtype)
        cen = (sel.T @ cart) / sel.sum(0).unsqueeze(-1).clamp_min(1.0)  # (B, 3)
        local = cart - sel @ cen  # atom position relative to its body centroid

        t = torch.zeros(len(uniq), 3, device=cart.device, dtype=cart.dtype, requires_grad=True)
        w = torch.zeros(len(uniq), 3, device=cart.device, dtype=cart.dtype, requires_grad=True)
        opt = torch.optim.Adam([t, w], lr=lr)

        best, best_e = cart, float("inf")
        for _ in range(int(n_iter)):
            rot = _rodrigues(w)  # (B, 3, 3)
            moved_local = torch.einsum("nb,bij,nj->ni", sel, rot, local)
            pos = moved_local + sel @ (cen + t)
            e = clash_energy(pos, lat, inter, lb, slack=slack)
            reg = spring * (t.pow(2).sum() + w.pow(2).sum())
            with torch.no_grad():
                # score against the *metric's* threshold, not the padded one
                hard = float(clash_energy(pos, lat, inter, lb, slack=0.0))
                if hard < best_e:
                    best_e, best = hard, pos.detach().clone()
                if early_stop and hard <= 0.0:
                    break
            opt.zero_grad(set_to_none=True)
            (e + reg).backward()
            opt.step()
            with torch.no_grad():
                t.clamp_(-max_shift, max_shift)

    return best


def _rot_about(axis: Tensor, theta: float) -> Tensor:
    """Rodrigues rotation matrix about a unit axis."""
    import math

    ct, st = math.cos(theta), math.sin(theta)
    k = torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        device=axis.device, dtype=axis.dtype,
    )
    return ct * torch.eye(3, device=axis.device, dtype=axis.dtype) + st * k + (1 - ct) * torch.outer(axis, axis)


@torch.no_grad()
def relax_torsions(
    coords: Tensor,
    bonds: Tensor,
    dg,
    *,
    rounds: int = 3,
    n_theta: int = 24,
    clash_scale: float = 0.7,
) -> Tensor:
    """Rotate about rotatable bonds to clear internal steric clashes.

    Torsions are free coordinates: turning a branch about its own bond changes
    only 1-4-and-longer distances, so every bond length, every bond angle and
    every stereocentre is preserved *exactly*. This can therefore repair the one
    thing the stereochemistry operations damage -- PoseBusters'
    ``internal_steric_clash`` -- without giving back any of the chirality they
    fixed, and without touching the checks they already leave intact.

    The objective is the metric's own criterion: RDKit distance-geometry lower
    bounds for pairs beyond 1-4, widened by ``threshold_clash`` (0.3), i.e.
    ``clash_scale = 0.7``.
    """
    import math

    from crystal_nft.meanflow.pcfm import _adjacency, _branch

    if not dg:
        return coords
    out = coords.clone()
    adj = _adjacency(bonds)
    for start, size, lo, mask in dg:
        lo = lo.to(out.device, out.dtype)
        mask = mask.to(out.device)
        idx = torch.arange(start, start + size, device=out.device)

        def energy(local: Tensor) -> Tensor:
            d = torch.cdist(local, local).clamp_min(_EPS)
            return torch.relu(lo * clash_scale - d).mul(mask).pow(2).sum()

        local = out[idx]
        if float(energy(local)) <= 0.0:
            continue  # already clean
        # rotatable bonds: acyclic, and both sides at least 2 atoms
        rot_bonds = []
        for u in range(size):
            for v in adj.get(start + u, ()):  # noqa: SIM118
                v -= start
                if v <= u or not (0 <= v < size):
                    continue
                br = _branch(
                    {a - start: [b - start for b in nb if start <= b < start + size]
                     for a, nb in adj.items() if start <= a < start + size},
                    v, u,
                )
                if u in br or len(br) < 2 or size - len(br) < 2:
                    continue
                rot_bonds.append((u, v, sorted(br)))
        if not rot_bonds:
            continue
        for _ in range(int(rounds)):
            improved = False
            for u, v, br in rot_bonds:
                cur = energy(local)
                if float(cur) <= 0.0:
                    break
                axis = local[v] - local[u]
                axis = axis / axis.norm().clamp_min(_EPS)
                sel = torch.tensor(br, device=out.device, dtype=torch.long)
                rel = local[sel] - local[u]
                best_e, best = cur, None
                for k in range(1, int(n_theta)):
                    r = _rot_about(axis, 2.0 * math.pi * k / int(n_theta))
                    trial = local.clone()
                    trial[sel] = rel @ r.transpose(0, 1) + local[u]
                    e = energy(trial)
                    if float(e) < float(best_e):
                        best_e, best = e, trial
                if best is not None:
                    local = best
                    improved = True
            if not improved:
                break
        out[idx] = local
    return out


def mmff_relax_bodies(
    crystal,
    *,
    max_displ: float = 0.15,
    force_constant: float = 50.0,
    max_its: int = 400,
    max_atoms: int = 200,
) -> Tensor:
    """Relax each molecule with MMFF under position restraints.

    The exact stereochemistry operations preserve bond lengths and angles but
    leave local strain: measured per check, they cost PoseBusters'
    ``internal_steric_clash`` (+2.96 points of fragments) and ``bond_angles``
    (+0.99). A restrained force field is the right instrument for that -- it
    relieves the strain while the position restraints hold every atom within
    ``max_displ`` of where the generator put it, so the packing survives.

    Returns new Cartesian coordinates. Molecules that fail to sanitise or type
    under MMFF are left untouched.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from crystal_nft.meanflow import stereo as _S

    cell = _S._CpuCell(crystal)
    coords = crystal.coords.clone()
    for start, size in _S._body_spans(cell):
        if size > max_atoms:
            continue
        try:
            mol = _S._body_rdmol(Chem, cell, start, size)
            if Chem.SanitizeMol(mol, catchErrors=True) != 0:
                continue
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is None:
                continue
            ff = AllChem.MMFFGetMoleculeForceField(mol, props)
            if ff is None:
                continue
            for i in range(size):
                ff.MMFFAddPositionConstraint(i, max_displ, force_constant)
            ff.Minimize(maxIts=int(max_its))
            new = torch.tensor(
                mol.GetConformer().GetPositions(), dtype=coords.dtype
            )
            if new.shape[0] == size and torch.isfinite(new).all():
                coords[start : start + size] = new
        except Exception:  # noqa: BLE001
            continue
    return coords
