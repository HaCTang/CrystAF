#!/usr/bin/env bash
# Structure recovery (Sol@k) on the CSD-2026.1 test rebuild: draw NS candidates per
# target, keep the K lowest in UMA energy, COMPACK against the experimental crystal.
#
#   bash scripts/table3.sh sample     # sample-test + collision + UMA energies (GPU, clari env)
#   bash scripts/table3.sh compack    # official COMPACK on the UMA top-K (CPU, ccdc env)
#   bash scripts/table3.sh report     # bootstrap Sol@K
#   bash scripts/table3.sh auc        # energy-enrichment AUC on OXtal (NS=400, top-200)
#
# NS=150 (default, K=30) is the main-text table; NS=400 (K=200) and NS=30 (K=30) are
# the appendix tables. Each NS is sampled independently. ROWS overrides the row list,
# e.g. ROWS="clari-l" NS=400 for the AUC comparison.
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
NS="${NS:-150}"
case "${NS}" in
  400) K="${K:-200}" ;;
  *) K="${K:-30}" ;;
esac
read -r -a ROWS <<< "${ROWS:-clari-m crystaf-base crystaf-base-chain crystaf-uma crystaf-uma-chain}"
SUBSETS=(oxtal teaching)
BASE="${CRYSTAF_ROOT}/checkpoints/crystaf/crystaf-cont3-step2000.pt"
UMA="${CRYSTAF_ROOT}/checkpoints/crystaf/crystaf-uma-rl-seed929-epoch6.pt"

sample_row() {  # <row> <subset>
  local exp="$1_$2_ns${NS}"
  local d="${CLARI_RESULTS_DIR}/${exp}"
  (
    case "$1" in
      clari-m) ckpt=checkpoints/clari-med.ckpt;   extra=(--n_steps 20) ;;
      clari-l) ckpt=checkpoints/clari-large.ckpt; extra=(--n_steps 20) ;;
      crystaf-*)
        ckpt=checkpoints/clari-med.ckpt; extra=()
        case "$1" in
          crystaf-base*) export CRYSTAF_MEANFLOW_CKPT="${BASE}" ;;
          crystaf-uma*) export CRYSTAF_MEANFLOW_CKPT="${UMA}" ;;
        esac
        export CRYSTAF_MEANFLOW_STEPS=32 MEANFLOW_SAMPLER_MODE=interval MEANFLOW_INTERVAL_RHO=1
        if [[ "$1" == *-chain ]]; then
          export CRYSTAF_MIRROR_FIX=body CRYSTAF_STEREO_REFLECT=1 CRYSTAF_MMFF=1 \
                 CRYSTAF_RELAX_CLASH=1 CRYSTAF_VOL_SCALE=0.9850
        fi ;;
      *) echo "unknown row $1" >&2; exit 1 ;;
    esac
    [[ -f "${d}/predictions.parquet" ]] || \
      "${BIN}/sample-test" "${CRYSTAF_ROOT}/${ckpt}" "${NS}" "${exp}" --subset "$2" --num_gpus "${NUM_GPUS}" "${extra[@]}"
  )
  [[ -f "${d}/collision.csv" ]] || "${BIN}/collision" "${exp}"
  [[ -f "${d}/energies.csv" ]] || "${BIN}/compute-energies" "${exp}" --num_gpus "${NUM_GPUS}"
}

case "${1:-}" in
  sample)
    for r in "${ROWS[@]}"; do for s in "${SUBSETS[@]}"; do sample_row "$r" "$s"; done; done ;;
  compack)
    for r in "${ROWS[@]}"; do for s in "${SUBSETS[@]}"; do
      exp="${r}_${s}_ns${NS}"
      [[ -f "${CLARI_RESULTS_DIR}/${exp}/compack.csv" ]] && continue
      "${CCDC_PY}" scripts/analysis/compack_shard.py "${exp}" --shard 0 --num_shards 1 \
        --num_processes "${NPROC}" --topk "${K}"
      (cd CrystalGenModel/clari && "${CCDC_PY}" -s clari/evaluation/compack.py "${exp}" --num_processes 1)
    done; done ;;
  report)
    "${CLARI_PY}" scripts/analysis/table3_bootstrap.py "${ROWS[@]/%/_oxtal_ns${NS}}" -k "${K}"
    "${CLARI_PY}" scripts/analysis/table3_bootstrap.py "${ROWS[@]/%/_teaching_ns${NS}}" -k "${K}" --teaching ;;
  auc)
    R="${CLARI_RESULTS_DIR}"
    "${CLARI_PY}" scripts/analysis/analyze_uma_compack.py \
      "${R}/crystaf-uma_oxtal_ns400" "${R}/clari-m_oxtal_ns400" "${R}/clari-l_oxtal_ns400" \
      --k 200 --output "${R}/uma_compack_enrichment.json" ;;
  *) echo "usage: $0 sample|compack|report|auc" >&2; exit 1 ;;
esac
