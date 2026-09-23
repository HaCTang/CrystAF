#!/usr/bin/env python
"""Export the Clari test families from an installed CSD into the two raw parquets
that `clari/scripts/data/1_process.py` consumes.

Replaces the ConQuest GUI export for the *test* split only. That is enough to
rebuild `test.pt` + `test_cifs.parquet`, which is all Table 3 needs, and it avoids
processing all ~1.4M CSD entries.

Two things drive the design:

* `1_process.py:split_dataset` holds out 1000 random *train* families for
  validation and will raise if fewer exist, so we export filler families too.
  Their content is irrelevant -- only `test.pt` is used downstream.
* There is no refcode-family API. `TextNumericSearch.add_identifier` does family
  lookup but costs ~9 s per call and ANDs when batched, so we enumerate
  `FAMILY` + `FAMILY01..99` instead and parallelise across processes.

    python scripts/data/export_csd_test_families.py --out-dir dataset/clari_2026/raw
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "CrystalGenModel" / "clari"))

SUFFIXES = [""] + [f"{i:02d}" for i in range(1, 100)]

META_SCHEMA = {
    "id": pl.Utf8,
    "deposition_date": pl.Date,
    "r_factor": pl.Float64,
    "has_3d_structure": pl.Boolean,
    "has_disorder": pl.Boolean,
    "is_powder_study": pl.Boolean,
    "pressure": pl.Utf8,
    "is_polymeric": pl.Boolean,
}
STRUCT_SCHEMA = {"id": pl.Utf8, "cif": pl.Utf8, "mol2": pl.Utf8}

_READER = None


def _reader():
    global _READER
    if _READER is None:
        from ccdc.io import EntryReader

        _READER = EntryReader("CSD")
    return _READER


def _dump_family(family: str) -> tuple[list[dict], list[dict]]:
    csd = _reader()
    meta, struct = [], []
    for suffix in SUFFIXES:
        refcode = family + suffix
        try:
            entry = csd.entry(refcode)
        except Exception:
            continue
        try:
            cif = entry.crystal.to_string("cif")
            mol2 = entry.molecule.to_string("mol2")
        except Exception:
            continue
        meta.append(
            {
                "id": entry.identifier,
                "deposition_date": entry.deposition_date,
                "r_factor": entry.r_factor,
                "has_3d_structure": entry.has_3d_structure,
                "has_disorder": entry.has_disorder,
                "is_powder_study": entry.is_powder_study,
                "pressure": None if entry.pressure is None else str(entry.pressure),
                "is_polymeric": entry.is_polymeric,
            }
        )
        struct.append({"id": entry.identifier, "cif": cif, "mol2": mol2})
    return meta, struct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "dataset" / "clari_2026" / "raw"))
    parser.add_argument(
        "--filler-metadata",
        default=str(ROOT / "dataset" / "clari" / "csd" / "metadata.parquet"),
        help="Existing metadata parquet, used only as a cheap source of non-test refcodes.",
    )
    parser.add_argument("--filler-families", type=int, default=3000)
    parser.add_argument("--num-workers", type=int, default=32)
    args = parser.parse_args()

    from clari.csd import AVAILABLE_CSD_SUBSETS, csd_fam

    test_fams = sorted({csd_fam(c) for c in AVAILABLE_CSD_SUBSETS["test"]})
    filler: list[str] = []
    if args.filler_families > 0:
        known = pl.read_parquet(args.filler_metadata).get_column("id").to_list()
        seen, test_set = set(), set(test_fams)
        for cid in known:
            fam = str(cid).strip().upper()[:6]
            if fam not in test_set and fam not in seen:
                seen.add(fam)
                filler.append(fam)
            if len(filler) >= args.filler_families:
                break
    families = test_fams + filler
    print(f"{len(test_fams)} test families + {len(filler)} filler = {len(families)}", flush=True)

    with Pool(args.num_workers) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(_dump_family, families, chunksize=4), 1):
            results.append(res)
            if i % 200 == 0:
                print(f"  {i}/{len(families)} families", flush=True)

    meta = [r for m, _ in results for r in m]
    struct = [r for _, s in results for r in s]
    found = {r["id"][:6] for r in meta}
    missing = [f for f in test_fams if f not in found]
    print(f"Exported {len(meta)} entries across {len(found)} families")
    if missing:
        print(f"WARNING: {len(missing)} test families not found: {missing[:10]}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(meta, schema=META_SCHEMA).write_parquet(out / "csd_metadata.parquet")
    pl.DataFrame(struct, schema=STRUCT_SCHEMA).write_parquet(out / "csd_conquest.parquet")
    print(f"Wrote {out}/csd_metadata.parquet and csd_conquest.parquet")


if __name__ == "__main__":
    os.environ.setdefault("CCDC_PYTHON_API_NO_QAPPLICATION", "1")
    main()
