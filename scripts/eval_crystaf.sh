#!/usr/bin/env bash
# Table 1 evaluation of a CrystAF checkpoint (200 CSD val families x 20 samples).
#
#   bash scripts/eval_crystaf.sh <checkpoint.pt> <nfe> <out_dir>
#   bash scripts/eval_crystaf.sh checkpoints/crystaf/crystaf-uma-rl-seed929-epoch6.pt 32 runs/eval/uma_nfe32
#
# The power-law grid t_i = (i/N)^rho uses the reported rho per NFE:
# 8 -> 0.30, 16 -> 0.75, 32 and 50 -> 1. Override with MEANFLOW_INTERVAL_RHO.
# Sampling-time correctors are read from the environment, e.g. for "CrystAF-S":
#   CRYSTAF_MIRROR_FIX=body CRYSTAF_STEREO_REFLECT=1 CRYSTAF_MMFF=1 \
#   CRYSTAF_RELAX_CLASH=1 CRYSTAF_VOL_SCALE=0.9850 bash scripts/eval_crystaf.sh ...
# The headline numbers are summary.paper_bootstrap in <out_dir>/metrics.json.
set -euo pipefail
source "$(dirname "$0")/../env/paths.sh"
cd "${CRYSTAF_ROOT}"

CKPT="${1:?usage: eval_crystaf.sh <checkpoint.pt> <nfe> <out_dir>}"
NFE="${2:?}"
OUT="${3:?}"
NPROC="${NPROC:-$(python3 -c 'import os,subprocess;v=os.environ.get("CUDA_VISIBLE_DEVICES");print(len(v.split(",")) if v else len(subprocess.check_output(["nvidia-smi","-L"]).splitlines()))')}"

export MEANFLOW_SAMPLER_MODE=interval
export MEANFLOW_INTERVAL_SCHEDULE=power
if [[ -z "${MEANFLOW_INTERVAL_RHO:-}" ]]; then
  case "${NFE}" in
    8) export MEANFLOW_INTERVAL_RHO=0.30 ;;
    16) export MEANFLOW_INTERVAL_RHO=0.75 ;;
    *) export MEANFLOW_INTERVAL_RHO=1 ;;
  esac
fi

mkdir -p "${OUT}"
exec "${CLARI_PY}" -m torch.distributed.run --standalone --nproc_per_node "${NPROC}" \
  -m crystal_nft.train.eval_meanflow_table1 \
  --meanflow-ckpt "${CKPT}" \
  --meanflow-steps "${NFE}" \
  --checkpoint checkpoints/clari-med.ckpt \
  --clari-data-dir "${CLARI_DATA_DIR}" \
  --output-dir "${OUT}" \
  --max-crystals "${MAX_CRYSTALS:-200}" \
  --samples 20 \
  --seed 42 \
  --pack-size 8 \
  --sample-chunk 20 \
  --resume \
  --no-assess-in-subprocess \
  --assess-mem-gb 0 \
  --overlap-assess \
  --metric-workers 1
