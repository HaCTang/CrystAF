#!/usr/bin/env bash
# Launch one training stage on all visible GPUs.
#
#   bash scripts/train.sh <module> <config.yaml>
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh train_meanflow_nft configs/rl/uma_seed929.yaml
#
# <module> is one of: train_clari_nft, train_meanflow_csd, train_meanflow_nft,
# train_meanflow_umadistill, train_mcf_nft, train_mcf_grpo.
# NFT runs average weights across ranks at every epoch end, so use the GPU count
# given in the config header to reproduce a reported run exactly.
set -euo pipefail
source "$(dirname "$0")/../env/paths.sh"
cd "${CRYSTAF_ROOT}"

MODULE="${1:?usage: train.sh <module> <config.yaml>}"
CONFIG="${2:?usage: train.sh <module> <config.yaml>}"
NPROC="${NPROC:-$(python3 -c 'import os,subprocess;v=os.environ.get("CUDA_VISIBLE_DEVICES");print(len(v.split(",")) if v else len(subprocess.check_output(["nvidia-smi","-L"]).splitlines()))')}"

PY="${CLARI_PY}"
if [[ "${MODULE}" == train_mcf_* ]]; then
  PY="${MCF_PY}"
  export PYTHONPATH="${CRYSTAF_ROOT}/CrystalGenModel/MolCrystalFlow${PYTHONPATH:+:${PYTHONPATH}}"
fi

mkdir -p runs
exec "${PY}" -m torch.distributed.run --standalone --nproc_per_node "${NPROC}" \
  -m "crystal_nft.train.${MODULE}" --config "${CONFIG}"
