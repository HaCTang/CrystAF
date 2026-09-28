"""Training-free physics injection at sampling time (UMA force guidance / relaxation).

These are the inference-time baselines the paper compares learned alignment
against. Both call the same black-box potential used as the post-training
reward (UMA-OMC single points), so the comparison isolates *where* the physics
enters -- in the weights, or in extra potential calls at every sample.

``CRYSTAF_UMA_GUIDE=<eta>``
    Endpoint force guidance inside the flow-map loop. After each jump whose
    source time is at least ``CRYSTAF_UMA_GUIDE_TSTART``, the endpoint estimate
    ``x1_hat = z_r + (1 - r) U`` is moved along UMA forces by ``eta * F``
    (Angstrom per eV/A), capped at ``CRYSTAF_UMA_GUIDE_MAXDISP`` Angstrom per atom,
    and the step is re-aimed at the corrected endpoint (the PCFM slot). One UMA
    call per guided jump.

``CRYSTAF_UMA_RELAX=<steps>``
    Post-hoc fixed-step UMA descent on the finished sample
    (`uma_relax.relax_atoms_uma`): at most ``CRYSTAF_UMA_RELAX_DISP`` Angstrom per
    atom per step, cell relaxation when ``CRYSTAF_UMA_RELAX_CELL=1``. One UMA call
    per step.

Every UMA structure evaluation is counted in ``STATS`` so the cost column is
measured, not estimated.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from clari.chem import Crystal

_OFF = ("", "0", "false", "no", "off")
_SCORER = None
# The eval samples on the main thread and assesses on a worker thread; both may
# call UMA (relaxation / diagnostics), so every potential call is serialized.
_UMA_LOCK = threading.RLock()
STATS = {"uma_structs": 0, "uma_calls": 0, "uma_sec": 0.0, "uma_failed_calls": 0}


def _uma_context():
    """UMA forces are -dE/dx by autograd; the eval samples under inference_mode.

    Leave inference mode, re-enable grad, and turn bf16 autocast off, or every
    call fails (and a caught failure would silently turn the method into a no-op).
    """
    from contextlib import ExitStack

    stack = ExitStack()
    if torch.cuda.is_available() and _env("CRYSTAF_UMA_DEVICE", "cuda") != "cpu":
        # CUDA's current device is per thread; the eval's assess worker thread
        # would otherwise default to GPU 0 and every rank would pile onto it.
        stack.enter_context(torch.cuda.device(int(os.environ.get("LOCAL_RANK", "0"))))
        torch.cuda.empty_cache()
    stack.enter_context(torch.inference_mode(False))
    stack.enter_context(torch.enable_grad())
    stack.enter_context(torch.autocast(device_type="cuda", enabled=False))
    return stack


def _env(name: str, default: str) -> str:
    return str(os.environ.get(name, default)).strip()


def guidance_enabled() -> bool:
    return _env("CRYSTAF_UMA_GUIDE", "0").lower() not in _OFF


def relax_enabled() -> bool:
    return _env("CRYSTAF_UMA_RELAX", "0").lower() not in _OFF


def get_scorer():
    global _SCORER
    with _UMA_LOCK:
        return _get_scorer_locked()


def _get_scorer_locked():
    global _SCORER
    if _SCORER is None:
        from crystal_nft.rewards.uma_scorer import UMAScorer

        # Built outside inference mode: weights loaded under it become inference
        # tensors, which autograd (UMA forces) then refuses to save for backward.
        with _uma_context():
            _SCORER = _build_scorer(UMAScorer)
    return _SCORER


def _build_scorer(UMAScorer):
    return UMAScorer(
            _env("CRYSTAF_UMA_MODEL", "uma-s-1p1"),
            device=_env("CRYSTAF_UMA_DEVICE", "cuda"),
            use_ultrafast=False,
            batch_size=int(_env("CRYSTAF_UMA_BS", "4")),
            checkpoint_path=_env(
                "CRYSTAF_UMA_CKPT",
                str(Path(__file__).resolve().parents[3] / "checkpoints" / "uma" / "uma-s-1p1.pt"),
            ),
        )


def _items(C: Crystal) -> list[Crystal]:
    return list(C.unbatch()) if getattr(C, "batched", False) else [C]


def _uma_forces(atoms_list):
    from crystal_nft.meanflow.uma_relax import _predict

    sc = get_scorer()
    for a in atoms_list:
        a.info.setdefault("spin", 0)
        a.info.setdefault("charge", 0)
    bs = max(1, int(sc.batch_size))
    forces, stresses = [], []
    t0 = time.perf_counter()
    for s0 in range(0, len(atoms_list), bs):
        chunk = atoms_list[s0 : s0 + bs]
        try:
            with _UMA_LOCK, _uma_context():
                f, s = _predict(sc.predictor, sc.task_name, [a.copy() for a in chunk])
        except Exception as exc:  # noqa: BLE001 -- counted and reported, never silent
            STATS["uma_failed_calls"] += 1
            if STATS["uma_failed_calls"] <= 3:
                import warnings

                warnings.warn(f"UMA guidance call failed: {exc}", RuntimeWarning, stacklevel=2)
            f, s = [None] * len(chunk), [None] * len(chunk)
        forces.extend(f)
        stresses.extend(s)
        STATS["uma_calls"] += 1
    STATS["uma_structs"] += len(atoms_list)
    STATS["uma_sec"] += time.perf_counter() - t0
    if STATS["uma_failed_calls"] >= 3 and STATS["uma_failed_calls"] == STATS["uma_calls"]:
        raise RuntimeError("every UMA guidance call has failed; refusing to report a no-op as a result")
    return forces, stresses


def force_guidance_step(
    *, z_prev: Tensor, z: Tensor, u: Tensor, t_from: Tensor, t_to: Tensor, C: Crystal
) -> Tensor:
    """Re-aim one flow-map jump at a UMA-force-corrected endpoint."""
    if not guidance_enabled():
        return z
    eta = float(_env("CRYSTAF_UMA_GUIDE", "0"))
    t_start = float(_env("CRYSTAF_UMA_GUIDE_TSTART", "0.5"))
    if float(t_from.min()) < t_start:
        return z
    denom = (1.0 - t_from).view(-1, *([1] * (z.ndim - 1)))
    if float(denom.min()) <= 1e-6:
        return z
    max_disp = float(_env("CRYSTAF_UMA_GUIDE_MAXDISP", "0.1"))
    scale = float(Crystal.COORD_NORM)
    x1_hat = z_prev + denom * u
    items = _items(C.replace(x=x1_hat))
    atoms = [it.to_ase() for it in items]
    for a in atoms:
        a.info.setdefault("spin", 0)
        a.info.setdefault("charge", 0)
    t0 = time.perf_counter()
    forces, nfail = _forces_robust(atoms)
    STATS["uma_sec"] += time.perf_counter() - t0
    STATS["uma_calls"] += 1
    STATS["uma_structs"] += len(atoms)
    STATS["uma_failed_structs"] = STATS.get("uma_failed_structs", 0) + nfail
    if STATS["uma_failed_structs"] > 0.10 * STATS["uma_structs"] and STATS["uma_structs"] > 200:
        raise RuntimeError(
            f"UMA guidance failed on {STATS['uma_failed_structs']} of {STATS['uma_structs']} "
            "structure-calls; refusing to report partly unguided samples"
        )
    delta = torch.zeros_like(x1_hat)
    for b, (it, f) in enumerate(zip(items, forces)):
        if f is None or not np.all(np.isfinite(f)):
            continue
        d = eta * f
        dmax = float(np.linalg.norm(d, axis=1).max()) if len(d) else 0.0
        if dmax > max_disp:
            d = d * (max_disp / dmax)
        n = d.shape[0]
        delta[b, 3 : 3 + n] = torch.from_numpy(d / scale).to(delta)
    dt = (t_to - t_from).view(-1, *([1] * (z.ndim - 1)))
    return z + (dt / denom) * delta


def _forces_robust(atoms_list):
    """UMA forces for a list; an OOM chunk is retried one structure at a time.

    Returns (forces, n_failed). A structure that still fails alone gets None.
    """
    import gc

    from crystal_nft.meanflow.uma_relax import _predict

    sc = get_scorer()
    bs = max(1, int(sc.batch_size))
    out = [None] * len(atoms_list)
    failed = 0

    def _one(chunk_idx):
        with _UMA_LOCK, _uma_context():
            f, _ = _predict(sc.predictor, sc.task_name, [atoms_list[i] for i in chunk_idx])
        return f

    for s0 in range(0, len(atoms_list), bs):
        idx = list(range(s0, min(s0 + bs, len(atoms_list))))
        try:
            f = _one(idx)
            for i, fi in zip(idx, f):
                out[i] = fi
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            gc.collect()
            torch.cuda.empty_cache()
            for i in idx:
                try:
                    out[i] = _one([i])[0]
                except RuntimeError as exc2:
                    if "out of memory" not in str(exc2).lower():
                        raise
                    gc.collect()
                    torch.cuda.empty_cache()
                    failed += 1
    return out, failed


def relax_post(z: Tensor, C: Crystal) -> Tensor:
    """Post-hoc fixed-step UMA descent on a finished sample (state in, state out).

    Same update as `uma_relax.relax_atoms_uma` (largest force component moves
    ``CRYSTAF_UMA_RELAX_DISP`` A per step, cell fixed) but OOM-safe, and every
    failed structure-step is counted: more than 1% failed aborts the run rather
    than silently reporting partly unrelaxed structures.
    """
    if not relax_enabled():
        return z
    steps = int(float(_env("CRYSTAF_UMA_RELAX", "0")))
    disp = float(_env("CRYSTAF_UMA_RELAX_DISP", "0.05"))
    scale = float(Crystal.COORD_NORM)
    items = _items(C.replace(x=z))
    atoms = [it.to_ase() for it in items]
    for a in atoms:
        a.info.setdefault("spin", 0)
        a.info.setdefault("charge", 0)
    t0 = time.perf_counter()
    fmax0 = []
    skip: set[int] = set()
    for step in range(steps):
        live = [i for i in range(len(atoms)) if i not in skip]
        if not live:
            break
        forces_live, _ = _forces_robust([atoms[i] for i in live])
        forces = [None] * len(atoms)
        for i, f in zip(live, forces_live):
            forces[i] = f
            if f is None:
                skip.add(i)  # out of GPU memory even alone: leave it as sampled
        for a, f in zip(atoms, forces):
            if f is None or not np.all(np.isfinite(f)):
                continue
            fm = float(np.abs(f).max())
            if step == 0:
                fmax0.append(fm)
            if fm > 1e-8:
                a.set_positions(a.get_positions() + f * (disp / fm))
    STATS["uma_sec"] += time.perf_counter() - t0
    STATS["uma_structs"] += steps * len(atoms)
    STATS["n_structures"] = STATS.get("n_structures", 0) + len(atoms)
    STATS["n_left_uncorrected"] = STATS.get("n_left_uncorrected", 0) + len(skip)
    if STATS["n_structures"] >= 200 and STATS["n_left_uncorrected"] > 0.10 * STATS["n_structures"]:
        raise RuntimeError(
            f"{STATS['n_left_uncorrected']} of {STATS['n_structures']} structures exceed GPU memory "
            "for UMA; too many to report the method as applied"
        )
    STATS.setdefault("relax_fmax_before", []).append(float(np.mean(fmax0)) if fmax0 else float("nan"))
    out = z.clone()
    for b, a in enumerate(atoms):
        pos = np.asarray(a.get_positions(), dtype=np.float64)
        n = pos.shape[0]
        if z.ndim == 3:
            out[b, 3 : 3 + n] = torch.from_numpy(pos / scale).to(out)
        else:
            out[3 : 3 + n] = torch.from_numpy(pos / scale).to(out)
    return out


def uma_diagnostics(crystals) -> list[dict]:
    """Per-structure UMA energy/molecule, forces and eligibility, for reporting.

    Not part of any method: this reads the potential's view of a finished sample
    so learned and inference-time physics can be compared in its own units.
    """
    from crystal_nft.adapters.clari_adapter import crystal_n_mols

    sc = get_scorer()
    atoms, n_mols = [], []
    for c in crystals:
        try:
            atoms.append(c.to_ase())
            n_mols.append(crystal_n_mols(c))
        except Exception:  # noqa: BLE001 -- recorded as ineligible, not dropped
            atoms.append(None)
            n_mols.append(1)
    with _UMA_LOCK, _uma_context():
        res = sc.score_atoms(atoms, n_mols=n_mols)
    out = []
    for r in res:
        ok = bool(r.valid) and np.isfinite(r.energy_per_mol) and r.energy_per_mol > -1e5
        out.append({
            "uma_valid": 1.0 if ok else 0.0,
            "uma_e_per_mol": float(r.energy_per_mol) if ok else None,
            "uma_fmax": float(r.force_max) if ok else None,
            "uma_fmean": float(r.force_mean_norm) if ok else None,
            "uma_stress": float(r.stress_norm) if ok else None,
        })
    return out
