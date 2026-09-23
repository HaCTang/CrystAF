# Source this before running anything:  source env/paths.sh
export CRYSTAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CLARI_PY="${CRYSTAF_ROOT}/CrystalGenModel/clari/.venv/bin/python"
export MCF_PY="${CRYSTAF_ROOT}/.venvs/molcrystalflow/bin/python"
export CLARI_DATA_DIR="${CLARI_DATA_DIR:-${CRYSTAF_ROOT}/dataset/clari}"
export HF_HUB_DISABLE_XET=1
