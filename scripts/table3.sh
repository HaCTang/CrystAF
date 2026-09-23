#!/usr/bin/env bash
# Table 3 (CSP Sol@200) on the CSD-2026.1 test rebuild, 400 samples per target.
#
#   bash scripts/table3.sh sample     # sample-test + collision + UMA energies (GPU, clari env)
#   bash scripts/table3.sh compack    # official COMPACK on the UMA top-200 (CPU, ccdc env)
#   bash scripts/table3.sh report     # bootstrap Sol@200
#
# Needs dataset/clari_2026/csd/test.pt and test_cifs.parquet (see README, "Data")
# and, for `compack`, a licensed CSD Python API in .venvs/ccdc.
set -euo pipefail
source "$(dirname "$0")/../env/paths.sh"
cd "${CRYSTAF_ROOT}"

export CLARI_DATA_DIR="${CRYSTAF_ROOT}/dataset/clari_2026"
export CLARI_RESULTS_DIR="${CRYSTAF_ROOT}/runs/table3"
BIN="${CRYSTAF_ROOT}/CrystalGenModel/clari/.venv/bin"
CCDC_PY="${CRYSTAF_ROOT}/.venvs/ccdc/bin/python"
NUM_GPUS="${NUM_GPUS:-$(python3 -c 'import os,subprocess;v=os.environ.get("CUDA_VISIBLE_DEVICES");print(len(v.split(",")) if v else len(subprocess.check_output(["nvidia-smi","-L"]).splitlines()))')}"
NPROC="${NPROC:-$(nproc)}"
N=400
ROWS=(clari-m clari-l crystaf-uma crystaf-s)
SUBSETS=(oxtal teaching)

sample_row() {  # <row> <subset>
  local exp="$1_$2"
  local d="${CLARI_RESULTS_DIR}/${exp}"
  (
    case "$1" in
      clari-m) ckpt=checkpoints/clari-med.ckpt;   extra=(--n_steps 20) ;;
      clari-l) ckpt=checkpoints/clari-large.ckpt; extra=(--n_steps 20) ;;
      crystaf-*)
        ckpt=checkpoints/clari-med.ckpt; extra=()
        export CRYSTAF_MEANFLOW_CKPT="${CRYSTAF_ROOT}/checkpoints/crystaf/crystaf-uma-rl-seed929-epoch6.pt"
        export CRYSTAF_MEANFLOW_STEPS=32 MEANFLOW_SAMPLER_MODE=interval MEANFLOW_INTERVAL_RHO=1
        if [[ "$1" == crystaf-s ]]; then
          export CRYSTAF_MIRROR_FIX=body CRYSTAF_STEREO_REFLECT=1 CRYSTAF_MMFF=1 \
                 CRYSTAF_RELAX_CLASH=1 CRYSTAF_VOL_SCALE=0.9850
        fi ;;
    esac
    [[ -f "${d}/predictions.parquet" ]] || \
      "${BIN}/sample-test" "${CRYSTAF_ROOT}/${ckpt}" "${N}" "${exp}" --subset "$2" --num_gpus "${NUM_GPUS}" "${extra[@]}"
  )
  [[ -f "${d}/collision.csv" ]] || "${BIN}/collision" "${exp}"
  [[ -f "${d}/energies.csv" ]] || "${BIN}/compute-energies" "${exp}" --num_gpus "${NUM_GPUS}"
}

case "${1:-}" in
  sample)
    for r in "${ROWS[@]}"; do for s in "${SUBSETS[@]}"; do sample_row "$r" "$s"; done; done ;;
  compack)
    for r in "${ROWS[@]}"; do for s in "${SUBSETS[@]}"; do
      exp="${r}_${s}"
      [[ -f "${CLARI_RESULTS_DIR}/${exp}/compack.csv" ]] && continue
      "${CCDC_PY}" scripts/analysis/compack_shard.py "${exp}" --shard 0 --num_shards 1 \
        --num_processes "${NPROC}" --topk 200
      (cd CrystalGenModel/clari && "${CCDC_PY}" -s clari/evaluation/compack.py "${exp}" --num_processes 1)
    done; done ;;
  report)
    "${CLARI_PY}" scripts/analysis/table3_bootstrap.py "${ROWS[@]/%/_oxtal}" -k 200
    "${CLARI_PY}" scripts/analysis/table3_bootstrap.py "${ROWS[@]/%/_teaching}" -k 200 --teaching ;;
  *) echo "usage: $0 sample|compack|report" >&2; exit 1 ;;
esac
