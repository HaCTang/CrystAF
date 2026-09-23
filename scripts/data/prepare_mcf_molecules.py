#!/usr/bin/env python
"""Extract monomer XYZ templates from the Thurlemann train pickle for MolCrystalFlow NFT.

    python scripts/data/prepare_mcf_molecules.py \\
        --cache dataset/molcrystalflow/thurlemann23/preprocessed/normalized/train_molcrystal_normalized.pkl.gz \\
        --out-dir dataset/molcrystalflow/nft_molecules
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import write

# Inverse of molcrystalflow.data.dataset.ATOM_TYPE_TO_IDX
IDX_TO_Z = {
    0: 1,
    1: 6,
    2: 7,
    3: 8,
    4: 9,
    5: 16,
    6: 17,
    7: 15,
    8: 35,
    9: 53,
    10: 5,
    11: 14,
}


def _load_pkl(path: Path) -> list[dict]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _monomer_from_entry(entry: dict) -> tuple[Atoms, int, bool]:
    bb_num = entry["bb_num_vec"].numpy() if hasattr(entry["bb_num_vec"], "numpy") else np.asarray(entry["bb_num_vec"])
    z_value = int(len(bb_num))
    n_atoms = int(bb_num[0])
    local = entry["local_coords"]
    atom_types = entry["atom_types"]
    if hasattr(local, "numpy"):
        local = local.numpy()
        atom_types = atom_types.numpy()
    local = np.asarray(local)[:n_atoms]
    atom_types = np.asarray(atom_types)[:n_atoms]
    numbers = [IDX_TO_Z[int(t)] for t in atom_types]
    atoms = Atoms(numbers=numbers, positions=local)
    flips = entry.get("axis_flips")
    has_flip = False
    if flips is not None:
        flips = flips.numpy() if hasattr(flips, "numpy") else np.asarray(flips)
        has_flip = bool(np.any(flips[:, 0] > 0.5))
    return atoms, z_value, has_flip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Path to *_molcrystal_normalized.pkl.gz (train preferred)",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-mols", type=int, default=0, help="0 = all")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data = _load_pkl(args.cache)
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(data))
    if args.max_mols and args.max_mols < len(indices):
        indices = rng.choice(indices, size=args.max_mols, replace=False)
        indices.sort()

    xyz_dir = args.out_dir / "xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    molecules = []
    for i, idx in enumerate(indices):
        atoms, z_value, has_flip = _monomer_from_entry(data[int(idx)])
        mol_id = f"train_{int(idx):05d}"
        xyz_path = xyz_dir / f"{mol_id}.xyz"
        write(str(xyz_path), atoms)
        gt_volume = float(torch.linalg.det(data[int(idx)]["lattice_1"].float()).abs())
        molecules.append(
            {
                "id": mol_id,
                "xyz": str(xyz_path.resolve()),
                "z_value": z_value,
                "has_axis_flip": has_flip,
                "source_index": int(idx),
                "n_atoms": int(len(atoms)),
                "formula": atoms.get_chemical_formula(),
                "gt_volume": gt_volume,
                "gt_volume_per_z": gt_volume / z_value,
            }
        )
        if (i + 1) % 200 == 0:
            print(f"wrote {i + 1}/{len(indices)}")

    manifest = args.out_dir / "molecules.json"
    manifest.write_text(json.dumps({"molecules": molecules, "n": len(molecules)}, indent=2) + "\n")
    print(f"Wrote {len(molecules)} molecules -> {manifest}")
    z_hist: dict[int, int] = {}
    for m in molecules:
        z_hist[m["z_value"]] = z_hist.get(m["z_value"], 0) + 1
    print("Z histogram:", dict(sorted(z_hist.items())))


if __name__ == "__main__":
    main()
