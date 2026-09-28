#!/usr/bin/env bash
# Inference-cost columns of the main table on ONE otherwise idle GPU: the first 4
# validation families x 20 samples, one sampler call of 20 per family; the first call
# (CUDA warm-up, UMA load) is dropped by scripts/analysis/cost_table.py.
#
#   bash scripts/cost.sh [gpu] [out_root]
set -euo pipefail
source "$(dirname "$0")/../env/paths.sh"
cd "${CRYSTAF_ROOT}"

export CUDA_VISIBLE_DEVICES="${1:-0}" NPROC=1 MAX_CRYSTALS=4
OUT="${2:-runs/cost}"
BASE=checkpoints/crystaf/crystaf-cont3-step2000.pt
UMA=checkpoints/crystaf/crystaf-uma-rl-seed929-epoch6.pt
CHAIN=(CRYSTAF_MIRROR_FIX=body CRYSTAF_STEREO_REFLECT=1 CRYSTAF_MMFF=1 CRYSTAF_RELAX_CLASH=1)
GUIDE_GENTLE=(CRYSTAF_UMA_GUIDE=0.01 CRYSTAF_UMA_GUIDE_MAXDISP=0.05 CRYSTAF_UMA_GUIDE_TSTART=0.5)
GUIDE_STRONG=(CRYSTAF_UMA_GUIDE=0.05 CRYSTAF_UMA_GUIDE_MAXDISP=0.2 CRYSTAF_UMA_GUIDE_TSTART=0.5)

run() {  # <tag> <ckpt> [KEY=VAL ...]
  local tag="$1" ckpt="$2"; shift 2
  rm -rf "${OUT:?}/${tag}"
  env "$@" bash scripts/eval_crystaf.sh "${ckpt}" 32 "${OUT}/${tag}"
}
run base_none "${BASE}"
run base_guide_gentle "${BASE}" "${GUIDE_GENTLE[@]}"
run base_guide_strong "${BASE}" "${GUIDE_STRONG[@]}"
run base_relax10 "${BASE}" CRYSTAF_UMA_RELAX=10
run base_relax50 "${BASE}" CRYSTAF_UMA_RELAX=50
run base_pcfm_bond "${BASE}" CRYSTAF_PCFM_BOND=1
run base_chain "${BASE}" "${CHAIN[@]}"
run uma_none "${UMA}"
run uma_relax50 "${UMA}" CRYSTAF_UMA_RELAX=50
run uma_chain "${UMA}" "${CHAIN[@]}"
rm -rf "${OUT:?}/clari_m_heun50"
SAMPLE_CHUNK=20 bash scripts/eval_clari.sh clari-med 50 "${OUT}/clari_m_heun50"

"${CLARI_PY}" scripts/analysis/cost_table.py "${OUT}"/*/
