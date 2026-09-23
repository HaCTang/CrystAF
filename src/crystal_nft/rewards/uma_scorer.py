"""UMA single-point scoring for organic molecular crystals (OMC)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

FAILED_ENERGY = float("inf")
ENERGY_KEYS = ("energy", "energy_pred", "y", "potential_energy")
FORCE_KEYS = ("forces", "force", "forces_pred")
STRESS_KEYS = ("stress", "stress_pred")


@dataclass
class ScoreResult:
    """Per-structure UMA single-point metrics."""

    energy: float
    energy_per_mol: float
    force_mean_norm: float
    force_max: float
    stress_norm: float
    clash: float
    density: float
    valid: bool
    n_mol: int = 1
    # Closest interatomic distance, in Angstrom. `clash` is only the binary
    # verdict (min_pair_distance < clash_cutoff); the margin is what lets a
    # reward penalise *approaching* the cutoff instead of only crossing it.
    min_pair_distance: float = float("nan")


def _build_inference_settings(
    *,
    use_ultrafast: bool,
    execution_mode: str,
    compile_model: bool,
):
    """Default fairchem backend unless `use_ultrafast` asks for a fast UMA-S mode.

    All reported runs use `use_ultrafast: false`, i.e. the default backend.
    """
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    settings_kwargs: dict[str, Any] = {
        "activation_checkpointing": False,
        "compile": compile_model,
    }

    modes_to_try: list[str] = []
    if use_ultrafast:
        modes_to_try.append(execution_mode)
        modes_to_try.extend(m for m in ("umas_fast_gpu", "umas_fast_pytorch") if m != execution_mode)

    for mode in modes_to_try:
        try:
            trial = dict(settings_kwargs)
            trial["execution_mode"] = mode
            if mode == "umas_fast_gpu":
                trial["merge_mole"] = True
            else:
                trial["merge_mole"] = False
            settings = InferenceSettings(**trial)
            logger.info("UMAScorer using execution_mode=%s", mode)
            return settings
        except Exception as exc:
            logger.debug("execution_mode=%s unavailable (%s)", mode, exc)

    logger.info("UMAScorer using default fairchem inference backend")
    return InferenceSettings(**settings_kwargs)


def _extract_tensor(predictions: dict, keys: Sequence[str]):
    for key in keys:
        if key in predictions:
            return predictions[key]
    return None


def _min_pair_distance(atoms) -> float:
    """Minimum pairwise distance under PBC (Angstrom)."""
    from ase.geometry import find_mic

    pos = atoms.get_positions()
    cell = atoms.cell
    n = len(pos)
    if n < 2:
        return float("inf")
    min_d = float("inf")
    for i in range(n):
        deltas = pos[i + 1 :] - pos[i]
        if cell is not None and getattr(atoms, "pbc", None) is not None and any(atoms.pbc):
            deltas, _ = find_mic(deltas, cell, atoms.pbc)
        dists = np.linalg.norm(deltas, axis=-1)
        if len(dists):
            min_d = min(min_d, float(dists.min()))
    return min_d


def _density_g_cm3(atoms) -> float:
    """Mass density in g/cm^3."""
    from ase.data import atomic_masses

    volume = float(atoms.get_volume())
    if volume <= 0:
        return 0.0
    mass_u = float(sum(atomic_masses[z] for z in atoms.get_atomic_numbers()))
    # u / Angstrom^3 -> g/cm^3
    return mass_u / volume * 1.66053906660


def _infer_n_mol(atoms, n_mol: int | None) -> int:
    if n_mol is not None and n_mol > 0:
        return int(n_mol)
    if "bb_indices" in getattr(atoms, "arrays", {}):
        return int(np.unique(atoms.arrays["bb_indices"]).size)
    return 1


class UMAScorer:
    """Batch single-point UMA energy / force / stress evaluator for OMC."""

    def __init__(
        self,
        model_name: str = "uma-s-1p2",
        *,
        device: str = "cuda",
        task_name: str = "omc",
        use_ultrafast: bool = False,
        execution_mode: str = "umas_fast_gpu",
        compile_model: bool = False,
        clash_cutoff: float = 0.8,
        density_min: float = 0.3,
        density_max: float = 3.0,
        batch_size: int = 8,
        checkpoint_path: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.task_name = task_name
        self.clash_cutoff = clash_cutoff
        self.density_min = density_min
        self.density_max = density_max
        self.batch_size = batch_size

        from fairchem.core.calculate.pretrained_mlip import (
            get_predict_unit,
            load_predict_unit,
        )

        # fairchem get_predict_unit only accepts "cpu" | "cuda" (not "cuda:N")
        predictor_device = "cpu" if str(device).startswith("cpu") else "cuda"

        settings = _build_inference_settings(
            use_ultrafast=use_ultrafast,
            execution_mode=execution_mode,
            compile_model=compile_model,
        )
        # Prefer an explicit local checkpoint (avoids HF network / offline cache issues)
        local_ckpt = checkpoint_path
        if local_ckpt is None and model_name in ("uma-s-1p1", "uma-s-1p2"):
            from pathlib import Path

            cand = Path(__file__).resolve().parents[3] / "checkpoints" / "uma" / f"{model_name}.pt"
            if cand.is_file():
                local_ckpt = str(cand)
        if local_ckpt:
            logger.info("UMAScorer loading local checkpoint %s", local_ckpt)
            self.predictor = load_predict_unit(
                local_ckpt,
                device=predictor_device,
                inference_settings=settings,
            )
        else:
            self.predictor = get_predict_unit(
                model_name,
                device=predictor_device,
                inference_settings=settings,
            )

    def _geometry_flags(self, atoms, n_mol: int) -> tuple[bool, float, float, float]:
        try:
            density = _density_g_cm3(atoms)
            min_d = _min_pair_distance(atoms)
        except Exception:
            return False, 1.0, 0.0, float("nan")
        clash = 1.0 if min_d < self.clash_cutoff else 0.0
        valid = (
            self.density_min <= density <= self.density_max
            and clash < 0.5
            and np.isfinite(density)
            and np.isfinite(min_d)
        )
        return valid, clash, density, float(min_d)

    def score_atoms(
        self,
        atoms_list: Sequence,
        *,
        n_mols: Sequence[int] | None = None,
    ) -> list[ScoreResult]:
        """Score a list of ASE Atoms with single-point UMA predictions."""
        import torch
        from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

        results: list[ScoreResult | None] = [None] * len(atoms_list)
        pending: list[tuple[int, Any, int]] = []

        for i, atoms in enumerate(atoms_list):
            n_mol = _infer_n_mol(atoms, None if n_mols is None else n_mols[i])
            if atoms is None:
                results[i] = ScoreResult(
                    energy=FAILED_ENERGY,
                    energy_per_mol=FAILED_ENERGY,
                    force_mean_norm=FAILED_ENERGY,
                    force_max=FAILED_ENERGY,
                    stress_norm=FAILED_ENERGY,
                    clash=1.0,
                    density=0.0,
                    valid=False,
                    n_mol=n_mol,
                )
                continue
            valid, clash, density, min_d = self._geometry_flags(atoms, n_mol)
            if not valid:
                results[i] = ScoreResult(
                    energy=FAILED_ENERGY,
                    energy_per_mol=FAILED_ENERGY,
                    force_mean_norm=FAILED_ENERGY,
                    force_max=FAILED_ENERGY,
                    stress_norm=FAILED_ENERGY,
                    clash=clash,
                    density=density,
                    min_pair_distance=min_d,
                    valid=False,
                    n_mol=n_mol,
                )
                continue
            atoms = atoms.copy()
            atoms.info.setdefault("spin", 0)
            atoms.info.setdefault("charge", 0)
            pending.append((i, atoms, n_mol))

        bs = max(1, self.batch_size)
        for start in range(0, len(pending), bs):
            chunk = pending[start : start + bs]
            while True:
                try:
                    atomic_data = [
                        AtomicData.from_ase(atoms, task_name=self.task_name)
                        for _, atoms, _ in chunk
                    ]
                    batch = atomicdata_list_to_batch(atomic_data).to(self.predictor.device)
                    with torch.no_grad():
                        pred = self.predictor.predict(batch)
                    break
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and len(chunk) > 1:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        mid = max(1, len(chunk) // 2)
                        # Re-queue smaller chunks
                        pending[start : start + len(chunk)] = chunk  # no-op keep
                        chunk_a, chunk_b = chunk[:mid], chunk[mid:]
                        # Process recursively by extending pending ahead
                        remaining = pending[start + len(chunk) :]
                        pending = pending[:start] + chunk_a + chunk_b + remaining
                        chunk = pending[start : start + mid]
                        bs = mid
                        continue
                    # Mark each as failed
                    for idx, atoms, n_mol in chunk:
                        _, clash, density, min_d = self._geometry_flags(atoms, n_mol)
                        results[idx] = ScoreResult(
                            energy=FAILED_ENERGY,
                            energy_per_mol=FAILED_ENERGY,
                            force_mean_norm=FAILED_ENERGY,
                            force_max=FAILED_ENERGY,
                            stress_norm=FAILED_ENERGY,
                            clash=clash,
                            density=density,
                            min_pair_distance=min_d,
                            valid=False,
                            n_mol=n_mol,
                        )
                    chunk = []
                    break

            if not chunk:
                continue

            energy_t = _extract_tensor(pred, ENERGY_KEYS)
            force_t = _extract_tensor(pred, FORCE_KEYS)
            stress_t = _extract_tensor(pred, STRESS_KEYS)
            if energy_t is None:
                raise KeyError(f"No energy key in predictor output: {sorted(pred)}")

            energies = energy_t.detach().float().cpu().reshape(-1).tolist()
            forces = None if force_t is None else force_t.detach().float().cpu().numpy()
            stresses = None if stress_t is None else stress_t.detach().float().cpu().numpy()

            # forces may be concatenated over atoms; split by natoms
            atom_counts = [len(atoms) for _, atoms, _ in chunk]
            force_offset = 0
            for j, (idx, atoms, n_mol) in enumerate(chunk):
                e = float(energies[j])
                if forces is not None:
                    n_atoms = atom_counts[j]
                    f = forces[force_offset : force_offset + n_atoms]
                    force_offset += n_atoms
                    f_norm = np.linalg.norm(f, axis=-1)
                    force_mean = float(f_norm.mean()) if len(f_norm) else FAILED_ENERGY
                    force_max = float(f_norm.max()) if len(f_norm) else FAILED_ENERGY
                else:
                    force_mean = FAILED_ENERGY
                    force_max = FAILED_ENERGY

                if stresses is not None:
                    s = stresses[j]
                    stress_norm = float(np.linalg.norm(s))
                else:
                    stress_norm = 0.0

                _, clash, density, min_d = self._geometry_flags(atoms, n_mol)
                results[idx] = ScoreResult(
                    energy=e,
                    energy_per_mol=e / max(n_mol, 1),
                    force_mean_norm=force_mean,
                    force_max=force_max,
                    stress_norm=stress_norm,
                    clash=clash,
                    density=density,
                    min_pair_distance=min_d,
                    valid=True,
                    n_mol=n_mol,
                )

        return [r if r is not None else ScoreResult(
            energy=FAILED_ENERGY,
            energy_per_mol=FAILED_ENERGY,
            force_mean_norm=FAILED_ENERGY,
            force_max=FAILED_ENERGY,
            stress_norm=FAILED_ENERGY,
            clash=1.0,
            density=0.0,
            valid=False,
        ) for r in results]
