#!/usr/bin/env bash
# MolCrystalFlow lattice-volume RMAD on the Thurlemann test split (single GPU).
#
#   bash scripts/eval_mcf.sh <out_dir> [nft_ckpt.pt]
#   bash scripts/eval_mcf.sh runs/eval/mcf_base
#   bash scripts/eval_mcf.sh runs/eval/mcf_uma checkpoints/crystaf/molcrystalflow-uma-nft-phaseB-epoch12.pt
set -euo pipefail
source "$(dirname "$0")/../env/paths.sh"
cd "${CRYSTAF_ROOT}"

OUT="${1:?usage: eval_mcf.sh <out_dir> [nft_ckpt.pt]}"
NFT="${2:-}"
MCF="${CRYSTAF_ROOT}/CrystalGenModel/MolCrystalFlow"
export PYTHONPATH="${MCF}:${MCF}/csp-pipeline${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=()
[[ -n "${NFT}" ]] && ARGS+=(--nft-ckpt "${NFT}")

exec "${MCF_PY}" -u -m crystal_nft.train.eval_mcf_fig3 \
  --base-ckpt "${MCF}/model-checkpoints/thurlemann23/best.ckpt" \
  --cache-dir dataset/molcrystalflow/thurlemann23/preprocessed/normalized \
  --out-dir "${OUT}" \
  --num-samples 10 \
  --num-gpus 1 \
  --skip-matching \
  "${ARGS[@]}"
