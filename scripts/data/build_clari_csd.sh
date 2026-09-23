#!/usr/bin/env bash
# Build Clari CSD tensors (train/val/test.pt) following Appendix A / Clari README.
#
# Prerequisites (ALL required):
#   1. Valid CCDC academic license with CSD database installed
#   2. csd-python-api installable from https://pip.ccdc.cam.ac.uk/
#   3. ConQuest (or pre-exported data/raw/csd_conquest.parquet with id,cif,mol2)
#
# Usage:
#   bash scripts/data/build_clari_csd.sh              # full pipeline if prereqs met
#   bash scripts/data/build_clari_csd.sh --check      # only diagnose environment
#   bash scripts/data/build_clari_csd.sh --process    # skip exports; only run 1_process
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLARI_ROOT="${ROOT}/CrystalGenModel/clari"
DATA_DIR="${CLARI_DATA_DIR:-${ROOT}/dataset/clari}"
RAW_DIR="${DATA_DIR}/raw"
CSD_DIR="${DATA_DIR}/csd"
MODE="${1:---all}"

mkdir -p "${RAW_DIR}" "${CSD_DIR}"
export CLARI_DATA_DIR="${DATA_DIR}"

echo "CLARI_DATA_DIR=${CLARI_DATA_DIR}"
echo "CLARI_ROOT=${CLARI_ROOT}"

check_env() {
  local ok=1
  echo "=== Environment check ==="

  if python3 -c "import ccdc" 2>/dev/null; then
    python3 -c "import ccdc; print('ccdc:', ccdc.__file__)"
  else
    echo "MISSING: python package 'ccdc' (csd-python-api)"
    ok=0
  fi

  if [[ -n "${CSDHOME:-}" && -d "${CSDHOME}" ]]; then
    echo "CSDHOME=${CSDHOME}"
  elif [[ -n "${CCDC_HOME:-}" && -d "${CCDC_HOME}" ]]; then
    echo "CCDC_HOME=${CCDC_HOME}"
  else
    echo "MISSING: CSDHOME / CCDC_HOME (CSD database install path)"
    ok=0
  fi

  if command -v conquest >/dev/null 2>&1 || command -v ConQuest >/dev/null 2>&1; then
    echo "ConQuest binary found: $(command -v conquest || command -v ConQuest)"
  else
    echo "NOTE: ConQuest CLI not on PATH (GUI export still OK if you already have csd_conquest.parquet)"
  fi

  if [[ -f "${RAW_DIR}/csd_metadata.parquet" ]]; then
    echo "FOUND metadata: ${RAW_DIR}/csd_metadata.parquet"
  else
    echo "PENDING: ${RAW_DIR}/csd_metadata.parquet"
  fi
  if [[ -f "${RAW_DIR}/csd_conquest.parquet" ]]; then
    echo "FOUND conquest: ${RAW_DIR}/csd_conquest.parquet"
  else
    echo "PENDING: ${RAW_DIR}/csd_conquest.parquet  (must come from ConQuest CIF+MOL2 export)"
  fi

  for f in config.json train.pt val.pt test.pt; do
    if [[ -f "${CSD_DIR}/${f}" ]]; then
      echo "FOUND ${CSD_DIR}/${f}"
    else
      echo "PENDING ${CSD_DIR}/${f}"
    fi
  done

  if [[ "${ok}" -ne 1 ]]; then
    cat <<'EOF'

BLOCKED: Cannot build CSD .pt on this machine yet.

Required (Clari README + paper Appendix A):
  1. CCDC academic license + installed CSD database
  2. Install API (on a machine that can auth to CCDC pip index):
       uv pip install csd-python-api --index-url https://pip.ccdc.cam.ac.uk/
     or: uv run -s CrystalGenModel/clari/scripts/data/0_metadata.py
  3. Export ALL CSD entries as CIF + MOL2 via ConQuest into
       dataset/clari/raw/csd_conquest.parquet  with columns: id, cif, mol2
     (Do NOT use csd-python-api for this step — it sanitizes bonds.)
  4. Re-run: bash scripts/data/build_clari_csd.sh --process

EOF
    return 1
  fi
  return 0
}

export_metadata() {
  echo "=== Exporting CSD metadata via csd-python-api ==="
  cd "${CLARI_ROOT}"
  # Prefer isolated script env from PEP 723; falls back to active env with ccdc
  if uv run -s scripts/data/0_metadata.py --help >/dev/null 2>&1; then
    uv run -s scripts/data/0_metadata.py --out "${RAW_DIR}/csd_metadata.parquet"
  else
    python scripts/data/0_metadata.py --out "${RAW_DIR}/csd_metadata.parquet"
  fi
}

print_conquest_instructions() {
  cat <<EOF

=== ConQuest export (manual; required by Clari) ===
ConQuest must dump every CSD entry as CIF + MOL2, then pack into parquet:

  columns: id (refcode), cif (text), mol2 (text)
  output:  ${RAW_DIR}/csd_conquest.parquet

Suggested workflow after ConQuest batch export of .cif/.mol2 pairs:
  python ${ROOT}/scripts/data/pack_conquest_parquet.py \\
    --cif-dir /path/to/cifs --mol2-dir /path/to/mol2s \\
    --out ${RAW_DIR}/csd_conquest.parquet

EOF
}

run_process() {
  echo "=== Running scripts.data.1_process ==="
  if [[ ! -f "${RAW_DIR}/csd_metadata.parquet" ]]; then
    echo "ERROR: missing ${RAW_DIR}/csd_metadata.parquet"
    exit 1
  fi
  if [[ ! -f "${RAW_DIR}/csd_conquest.parquet" ]]; then
    echo "ERROR: missing ${RAW_DIR}/csd_conquest.parquet"
    print_conquest_instructions
    exit 1
  fi
  cd "${CLARI_ROOT}"
  # 1_process writes under CLARI_DATA_DIR/csd by default via DATA_DIR
  uv run python -m scripts.data.1_process \
    --in_metadata "${RAW_DIR}/csd_metadata.parquet" \
    --in_cif_mol2 "${RAW_DIR}/csd_conquest.parquet" \
    --out "${CSD_DIR}" \
    --num_workers "${NUM_WORKERS:-16}" \
    --logging false
  echo "Done. Contents of ${CSD_DIR}:"
  ls -lah "${CSD_DIR}"
}

case "${MODE}" in
  --check)
    check_env || true
    ;;
  --process)
    run_process
    ;;
  --metadata)
    check_env
    export_metadata
    ;;
  --all|*)
    if ! check_env; then
      exit 1
    fi
    if [[ ! -f "${RAW_DIR}/csd_metadata.parquet" ]]; then
      export_metadata
    fi
    if [[ ! -f "${RAW_DIR}/csd_conquest.parquet" ]]; then
      print_conquest_instructions
      exit 1
    fi
    run_process
    ;;
esac
