#!/usr/bin/env python
"""Pack ConQuest-exported CIF/MOL2 files into csd_conquest.parquet.

Expects paired files sharing the same stem (= CSD refcode), e.g.:
  cifs/ACSALA.cif  +  mol2/ACSALA.mol2

Usage:
  python scripts/data/pack_conquest_parquet.py \\
    --cif-dir /path/to/cifs --mol2-dir /path/to/mol2s \\
    --out dataset/clari/raw/csd_conquest.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cif-dir", type=Path, required=True)
    p.add_argument("--mol2-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cif_map = {f.stem.upper(): f for f in args.cif_dir.glob("*.cif")}
    mol2_map = {f.stem.upper(): f for f in args.mol2_dir.glob("*.mol2")}
    ids = sorted(set(cif_map) & set(mol2_map))
    missing_cif = sorted(set(mol2_map) - set(cif_map))
    missing_mol2 = sorted(set(cif_map) - set(mol2_map))
    print(f"Paired refcodes: {len(ids)}")
    if missing_cif:
        print(f"MOL2 without CIF: {len(missing_cif)} (e.g. {missing_cif[:5]})")
    if missing_mol2:
        print(f"CIF without MOL2: {len(missing_mol2)} (e.g. {missing_mol2[:5]})")
    if not ids:
        raise SystemExit("No paired CIF/MOL2 files found")

    rows = []
    for cid in ids:
        rows.append(
            {
                "id": cid,
                "cif": cif_map[cid].read_text(errors="replace"),
                "mol2": mol2_map[cid].read_text(errors="replace"),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(args.out)
    print(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
