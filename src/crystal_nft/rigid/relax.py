"""Rigid-body packing relaxation: 6 DOF per molecule, nothing else.

Why this is the right shape of fix. `check_clashes_eval` is BINARY per crystal --
one inter-body pair closer than the sum of covalent radii condemns the whole
structure. Substituting a proper conformer into a packing tuned for a slightly
deformed one produces exactly that: a handful of touching pairs, not a globally
bad packing. So this is a feasibility problem, and the smallest move that can
solve it is to slide and turn each molecule.

Critically, translations and rotations are the ONLY degrees of freedom here, so:

* every intramolecular distance is preserved exactly -> PB stays where the
  conformer put it;
* the rotation is built via the exponential map, so it is in SO(3) by
  construction -> chirality stays exactly 100%.

Nothing in this file can undo the two properties the rigid route buys.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _rotation_from_axis_angle(w: Tensor) -> Tensor:
    """(B, 3) axis-angle -> (B, 3, 3) in SO(3), via Rodrigues."""
    theta = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = w / theta
    K = torch.zeros(w.shape[0], 3, 3, dtype=w.dtype, device=w.device)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    th = theta.unsqueeze(-1)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return eye + torch.sin(th) * K + (1.0 - torch.cos(th)) * (K @ K)


def _image_shifts(lattice: Tensor, rng: int = 1) -> Tensor:
    """All lattice translations within +-rng cells: (n_img, 3) cartesian."""
    r = torch.arange(-rng, rng + 1, device=lattice.device, dtype=lattice.dtype)
    n = torch.cartesian_prod(r, r, r)
    return n @ lattice


def relax_bodies(
    coords: Tensor,
    lattice: Tensor,
    body_ids: Tensor,
    lower_bound: Tensor,
    *,
    margin: float = 0.05,
    steps: int = 250,
    lr: float = 0.02,
    anchor: float = 0.02,
    image_range: int = 1,
) -> Tensor:
    """Slide/turn each body to clear inter-body overlaps.

    `coords` (N, 3) cartesian Angstrom, `lattice` (3, 3) rows are lattice
    vectors, `lower_bound` (N, N) minimum allowed inter-body distance.
    Returns relaxed coords (N, 3).
    """
    # Eval runs under `torch.inference_mode`, and inference tensors cannot take
    # part in autograd even after the mode exits -- clone out of it explicitly
    # rather than letting the optimiser fail somewhere less obvious.
    with torch.inference_mode(False):
        coords = coords.detach().clone().double()
        lattice = lattice.detach().clone().double()
        lb = lower_bound.detach().clone().double()

        uids = torch.unique(body_ids)
        masks = [(body_ids == u) for u in uids]
        cents = torch.stack([coords[m].mean(0) for m in masks])       # (B, 3)
        inter = (body_ids.unsqueeze(-1) != body_ids.unsqueeze(-2))
        shifts = _image_shifts(lattice, image_range)                  # (I, 3)

        t = torch.zeros(len(uids), 3, dtype=coords.dtype,
                        device=coords.device, requires_grad=True)
        w = torch.zeros(len(uids), 3, dtype=coords.dtype,
                        device=coords.device, requires_grad=True)
        opt = torch.optim.Adam([t, w], lr=lr)

        def place() -> Tensor:
            R = _rotation_from_axis_angle(w)
            out = coords.clone()
            for i, m in enumerate(masks):
                out[m] = (coords[m] - cents[i]) @ R[i].transpose(0, 1) + cents[i] + t[i]
            return out

        best, best_v = coords, float("inf")
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            p = place()
            d = p.unsqueeze(0) - p.unsqueeze(1)                        # (N, N, 3)
            d = (d.unsqueeze(2) + shifts.view(1, 1, -1, 3)).norm(dim=-1)  # (N,N,I)
            dmin = d.min(dim=-1).values
            viol = torch.relu(lb + margin - dmin) * inter
            loss = (viol ** 2).sum() + anchor * (t.pow(2).sum() + w.pow(2).sum())
            v = float(viol.max().detach())
            if v < best_v:
                best_v, best = v, p.detach().clone()
            if v <= 0.0:
                break
            loss.backward()
            opt.step()
        return best.to(lower_bound.dtype)
