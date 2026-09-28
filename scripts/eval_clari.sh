#!/usr/bin/env bash
# Table 1 evaluation of a multi-step Clari model (Heun sampler).
#
#   bash scripts/eval_clari.sh <clari-med|clari-large> <n_steps> <out_dir> [nft_ckpt.pt]
#
# Same-protocol baselines: MAX_CRYSTALS=200 (default), n_steps 50.
# Distillation teacher row: n_steps 16 with checkpoints/crystaf/teacher-rank800-pbnft-epoch1.pt.
# Clari UMA-NFT rows use the full 1000-family validation set: MAX_CRYSTALS=0.
# Heun with T steps costs 2T-1 network evaluations.
set -euo pipefail
source "$(dirname "$0")/../env/paths.sh"
cd "${CRYSTAF_ROOT}"

MODEL="${1:?usage: eval_clari.sh <clari-med|clari-large> <n_steps> <out_dir> [nft_ckpt.pt]}"
STEPS="${2:?}"
OUT="${3:?}"
NFT="${4:-}"
NPROC="${NPROC:-$(python3 -c 'import os,subprocess;v=os.environ.get("CUDA_VISIBLE_DEVICES");print(len(v.split(",")) if v else len(subprocess.check_output(["nvidia-smi","-L"]).splitlines()))')}"
MAX="${MAX_CRYSTALS:-200}"

ARGS=(--sample-chunk "${SAMPLE_CHUNK:-1}")
[[ "${MAX}" != "0" ]] && ARGS+=(--max-crystals "${MAX}")
[[ -n "${NFT}" ]] && ARGS+=(--nft-ckpt "${NFT}")

mkdir -p "${OUT}"
exec "${CLARI_PY}" -m torch.distributed.run --standalone --nproc_per_node "${NPROC}" \
  -m crystal_nft.train.eval_clari_table1 \
  --checkpoint "checkpoints/${MODEL}.ckpt" \
  --clari-data-dir "${CLARI_DATA_DIR}" \
  --output-dir "${OUT}" \
  --samples 20 \
  --n-steps "${STEPS}" \
  --device cuda \
  --pack-size 1 \
  --metric-workers 1 \
  --amd-metric cityblock \
  "${ARGS[@]}"
