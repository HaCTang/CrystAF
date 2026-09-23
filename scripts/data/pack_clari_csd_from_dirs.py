#!/usr/bin/env python
"""Build Clari raw parquet inputs from local ConQuest CIF + MOL2 dumps.

Creates:
  dataset/clari/raw/csd_metadata.parquet   (proxy metadata parsed from CIF headers)
  dataset/clari/raw/csd_conquest.parquet   (paired id/cif/mol2)

Optional --split-csv + --splits selects a subset (e.g. val,test) for faster eval builds.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date, datetime
from pathlib import Path

import polars as pl
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
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
CONQUEST_SCHEMA = {"id": pl.Utf8, "cif": pl.Utf8, "mol2": pl.Utf8}


def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = raw.strip().strip("'\"")
    if not s or s == "?":
        return None
    s = re.sub(r"\([0-9]+\)", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date(raw: str | None) -> date | None:
    if raw is None:
        return None
    s = raw.strip().strip("'\"")
    if not s or s == "?":
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    if re.fullmatch(r"\d{4}", s):
        return date(int(s), 1, 1)
    return None


def _header_map(cif_text: str) -> dict[str, str]:
    atom = cif_text.find("_atom_site_label")
    head = cif_text if atom < 0 else cif_text[:atom]
    out: dict[str, str] = {}
    for line in head.splitlines():
        line = line.strip()
        if not line.startswith("_") or " " not in line:
            continue
        key, val = line.split(None, 1)
        out[key.lower()] = val.strip()
    return out


def _proxy_metadata_from_cif(refcode: str, cif_text: str) -> dict:
    h = _header_map(cif_text)
    r = (
        _parse_float(h.get("_refine_ls_r_factor_gt"))
        or _parse_float(h.get("_refine_ls_r_factor_all"))
        or _parse_float(h.get("_refine_ls_r_factor_obs"))
    )
    # Clari filters with max_r_factor=10 (percent-like scale used by CCDC API).
    if r is not None and r <= 1.0:
        r = r * 100.0
    dep = _parse_date(h.get("_audit_creation_date")) or _parse_date(
        h.get("_journal_year")
    )
    pressure = h.get("_diffrn_ambient_pressure")
    if pressure is not None:
        pressure = pressure.strip().strip("'\"")
        if pressure in {"?", "", "ambient", "Ambi"}:
            pressure = None
    disorder = any("disorder" in k for k in h) or ("disorder" in cif_text[:2000].lower())
    return {
        "id": refcode,
        "deposition_date": dep,
        "r_factor": r,
        "has_3d_structure": "_atom_site_fract_x" in cif_text
        or "_atom_site_Cartn_x" in cif_text,
        "has_disorder": bool(disorder),
        "is_powder_study": False,
        "pressure": pressure,
        "is_polymeric": False,
    }


def _index_dir(path: Path, suffix: str) -> dict[str, Path]:
    suffix = suffix.lower()
    out: dict[str, Path] = {}
    with os.scandir(path) as it:
        for e in it:
            if not e.is_file():
                continue
            name = e.name
            if not name.lower().endswith(suffix):
                continue
            stem = Path(name).stem.upper()
            out[stem] = Path(e.path)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cif-dir", type=Path, default=ROOT / "dataset/clari/csd/C-cif")
    p.add_argument("--mol2-dir", type=Path, default=ROOT / "dataset/clari/csd/C-mol2")
    p.add_argument("--out-dir", type=Path, default=ROOT / "dataset/clari/raw")
    p.add_argument("--split-csv", type=Path, default=ROOT / "dataset/clari/hf/csd-split.csv")
    p.add_argument(
        "--splits",
        type=str,
        default="",
        help="Comma list of splits to keep (e.g. val,test or train,val,test). Empty=all paired.",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--chunksize", type=int, default=20000)
    args = p.parse_args()

    print(f"Indexing CIF dir: {args.cif_dir}", flush=True)
    cif_map = _index_dir(args.cif_dir, ".cif")
    print(f"  cif={len(cif_map)}", flush=True)
    print(f"Indexing MOL2 dir: {args.mol2_dir}", flush=True)
    mol2_map = _index_dir(args.mol2_dir, ".mol2")
    print(f"  mol2={len(mol2_map)}", flush=True)

    ids = sorted(set(cif_map) & set(mol2_map))
    print(
        f"Paired={len(ids)} | CIF-only={len(set(cif_map)-set(mol2_map))} | "
        f"MOL2-only={len(set(mol2_map)-set(cif_map))}",
        flush=True,
    )

    if args.splits.strip():
        wanted = {s.strip().lower() for s in args.splits.split(",") if s.strip()}
        split_df = pl.read_csv(args.split_csv)
        keep = {
            str(r["id"]).upper()
            for r in split_df.iter_rows(named=True)
            if str(r["split"]).lower() in wanted
        }
        before = len(ids)
        ids = [i for i in ids if i in keep]
        print(f"Split filter {sorted(wanted)}: {before} -> {len(ids)}", flush=True)

    if args.limit > 0:
        ids = ids[: args.limit]
        print(f"Limited to {len(ids)}", flush=True)
    if not ids:
        raise SystemExit("No paired CIF/MOL2 refcodes found")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "csd_metadata.parquet"
    conquest_path = args.out_dir / "csd_conquest.parquet"

    meta_chunks: list[pl.DataFrame] = []
    conquest_chunks: list[pl.DataFrame] = []
    meta_rows: list[dict] = []
    conquest_rows: list[dict] = []
    n_ok = 0
    n_fail = 0

    for ref in tqdm(ids, desc="Packing CIF+MOL2"):
        try:
            cif_text = cif_map[ref].read_text(errors="replace")
            mol2_text = mol2_map[ref].read_text(errors="replace")
            meta_rows.append(_proxy_metadata_from_cif(ref, cif_text))
            conquest_rows.append({"id": ref, "cif": cif_text, "mol2": mol2_text})
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            if n_fail <= 10:
                print(f"skip {ref}: {exc}", flush=True)
            continue

        if len(conquest_rows) >= args.chunksize:
            meta_chunks.append(pl.from_dicts(meta_rows, schema=META_SCHEMA))
            conquest_chunks.append(pl.from_dicts(conquest_rows, schema=CONQUEST_SCHEMA))
            meta_rows, conquest_rows = [], []

    if conquest_rows:
        meta_chunks.append(pl.from_dicts(meta_rows, schema=META_SCHEMA))
        conquest_chunks.append(pl.from_dicts(conquest_rows, schema=CONQUEST_SCHEMA))

    meta = pl.concat(meta_chunks) if meta_chunks else pl.DataFrame()
    conquest = pl.concat(conquest_chunks) if conquest_chunks else pl.DataFrame()
    meta.write_parquet(meta_path)
    conquest.write_parquet(conquest_path)
    summary = {
        "paired_ok": n_ok,
        "failed_reads": n_fail,
        "metadata": str(meta_path),
        "conquest": str(conquest_path),
        "n_meta": len(meta),
        "n_conquest": len(conquest),
        "splits": args.splits or "all_paired",
    }
    print(json.dumps(summary, indent=2), flush=True)
    (args.out_dir / "pack_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
