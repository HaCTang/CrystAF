"""Batched UMA force/stress relaxation, for building distillation targets.

This is *training-time* machinery: generated crystals are pulled downhill on the
UMA potential energy surface and the relaxed geometry becomes a regression
target. The inference chain is untouched -- nothing here runs at sampling time,
which is what separates it from the `CRYSTAF_RELAX_CLASH` repair chain.

Deliberately a fixed-step batched descent rather than ASE's FIRE/BFGS: those
optimise one structure at a time, and a distillation set needs thousands. Each
step is one batched UMA call for the whole list.

Step control is per structure and scale-free: positions move by at most
``max_disp`` angstrom per step, set by the largest force in that structure. A
raw ``x += lr * f`` cannot be tuned across a set where ``force_max`` spans
orders of magnitude (a clashing pair gives enormous forces while a settled
crystal gives ~0.1 eV/A).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["relax_atoms_uma", "RelaxStats"]


class RelaxStats:
    """What one relaxation pass did, for logging and sanity checks."""

    def __init__(self) -> None:
        self.steps = 0
        self.n_in = 0
        self.n_relaxed = 0
        self.fmax_before = float("nan")
        self.fmax_after = float("nan")
        self.mean_disp = float("nan")
        self.mean_vol_ratio = float("nan")

    def __repr__(self) -> str:  # pragma: no cover - logging only
        return (
            f"RelaxStats(steps={self.steps} n={self.n_relaxed}/{self.n_in} "
            f"fmax {self.fmax_before:.3f}->{self.fmax_after:.3f} "
            f"disp={self.mean_disp:.3f}A vol_ratio={self.mean_vol_ratio:.4f})"
        )


def _as_3x3_stress(s: np.ndarray) -> np.ndarray:
    """UMA may return 3x3 or 6-component Voigt stress."""
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    if s.size == 9:
        return s.reshape(3, 3)
    if s.size == 6:
        xx, yy, zz, yz, xz, xy = s
        return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=np.float64)
    raise ValueError(f"unexpected stress size {s.size}")


def _predict(predictor, task_name: str, batch_atoms: Sequence[Any]):
    """One batched UMA call -> (per-structure forces, per-structure stress)."""
    import torch
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

    data = [AtomicData.from_ase(a, task_name=task_name) for a in batch_atoms]
    batch = atomicdata_list_to_batch(data).to(predictor.device)
    with torch.no_grad():
        pred = predictor.predict(batch)

    def _get(keys):
        for k in keys:
            if k in pred:
                return pred[k]
        return None

    f = _get(("forces", "force", "forces_pred"))
    s = _get(("stress", "stress_pred"))
    forces = None if f is None else f.detach().float().cpu().numpy()
    stress = None if s is None else s.detach().float().cpu().numpy()

    out_f, out_s, off = [], [], 0
    for j, a in enumerate(batch_atoms):
        n = len(a)
        out_f.append(None if forces is None else forces[off : off + n])
        off += n
        out_s.append(None if stress is None else _as_3x3_stress(stress[j]))
    return out_f, out_s


def relax_atoms_uma(
    scorer,
    atoms_list: Sequence[Any],
    *,
    steps: int = 25,
    max_disp: float = 0.05,
    relax_cell: bool = True,
    cell_step: float = 0.05,
    max_strain: float = 0.01,
    lattice_scale: float = 1.0,
    batch_size: int | None = None,
) -> tuple[list[Any], RelaxStats]:
    """Pull each structure downhill on UMA forces; return relaxed copies.

    ``max_disp`` caps the largest per-atom displacement per step (angstrom) and
    ``max_strain`` the largest per-step cell strain, so a clashing structure
    with huge forces takes the same size step as a settled one.

    ``lattice_scale`` rescales the cell at the end **without moving atoms**
    (Cartesian coords fixed), which is the one volume correction that is safe to
    bake into a distillation target: molecular geometry, and hence PoseBusters,
    is untouched by construction. Use it instead of ``relax_cell`` -- measured
    on 36 generated structures, UMA's own cell relaxation drives signed volume
    error from +4.16% to +6.50% because UMA's equilibrium cell is *larger* than
    the experimental CSD reference, while relax + ``lattice_scale=0.985`` gives
    clash 0.000 at -0.46%.

    Entries that are ``None`` pass through untouched; a structure whose UMA call
    fails is returned at its last good geometry rather than dropped, so the
    caller's indexing is preserved.
    """
    out = [None if a is None else a.copy() for a in atoms_list]
    stats = RelaxStats()
    stats.n_in = sum(a is not None for a in atoms_list)
    stats.steps = int(steps)

    def _rescale() -> None:
        if abs(float(lattice_scale) - 1.0) <= 1e-9:
            return
        for a in out:
            if a is None:
                continue
            a.set_cell(np.asarray(a.get_cell()[:], dtype=np.float64) * float(lattice_scale),
                       scale_atoms=False)

    if stats.n_in == 0 or steps <= 0:
        # `steps=0` is still a valid request for a pure lattice rescale.
        _rescale()
        return out, stats

    idx = [i for i, a in enumerate(out) if a is not None]
    for i in idx:
        out[i].info.setdefault("spin", 0)
        out[i].info.setdefault("charge", 0)
    start_pos = {i: out[i].get_positions().copy() for i in idx}
    start_vol = {i: float(abs(np.linalg.det(out[i].get_cell()[:]))) for i in idx}

    bs = int(batch_size or getattr(scorer, "batch_size", 8) or 8)
    predictor = scorer.predictor
    task_name = scorer.task_name
    first_fmax: list[float] = []
    last_fmax: list[float] = []

    for step in range(int(steps)):
        step_fmax: list[float] = []
        for s0 in range(0, len(idx), bs):
            chunk = idx[s0 : s0 + bs]
            try:
                forces, stress = _predict(predictor, task_name, [out[i] for i in chunk])
            except Exception as exc:  # keep the last good geometry
                logger.warning("UMA relax step %d failed on a chunk: %s", step, exc)
                continue
            for k, i in enumerate(chunk):
                f = forces[k]
                if f is None or not np.all(np.isfinite(f)):
                    continue
                fmax = float(np.abs(f).max())
                step_fmax.append(fmax)
                if fmax > 1e-8:
                    out[i].set_positions(out[i].get_positions() + f * (max_disp / fmax))
                if relax_cell and stress[k] is not None and np.all(np.isfinite(stress[k])):
                    sig = stress[k]
                    strain = -cell_step * sig
                    smax = float(np.abs(strain).max())
                    if smax > max_strain:
                        strain *= max_strain / smax
                    cell = np.asarray(out[i].get_cell()[:], dtype=np.float64)
                    out[i].set_cell(cell @ (np.eye(3) + strain), scale_atoms=True)
        if step == 0 and step_fmax:
            first_fmax = step_fmax
        if step_fmax:
            last_fmax = step_fmax

    _rescale()

    disp = [float(np.linalg.norm(out[i].get_positions() - start_pos[i], axis=1).mean())
            for i in idx]
    vol = [float(abs(np.linalg.det(out[i].get_cell()[:])) / start_vol[i])
           for i in idx if start_vol[i] > 0]
    stats.n_relaxed = len(idx)
    stats.fmax_before = float(np.mean(first_fmax)) if first_fmax else float("nan")
    stats.fmax_after = float(np.mean(last_fmax)) if last_fmax else float("nan")
    stats.mean_disp = float(np.mean(disp)) if disp else float("nan")
    stats.mean_vol_ratio = float(np.mean(vol)) if vol else float("nan")
    return out, stats


def atoms_to_state_x(atoms: Any, *, coord_norm: float = 8.0):
    """ASE Atoms -> Clari's ``Crystal.x`` state rows.

    Inverse of `Crystal.to_ase` + the constructor's normalisation
    (`crystal.py`): ``x = cat([0.5 * lattice, coords]) / COORD_NORM``, so
    ``lattice = 2 * COORD_NORM * x[:3]`` and ``coords = COORD_NORM * x[3:]``.
    Returns a ``(3 + n_atoms, 3)`` float32 tensor on CPU.
    """
    import torch

    cell = np.asarray(atoms.get_cell()[:], dtype=np.float64)
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    rows = np.concatenate([0.5 * cell, pos], axis=0) / float(coord_norm)
    return torch.from_numpy(rows).to(torch.float32)
