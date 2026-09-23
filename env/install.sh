#!/usr/bin/env bash
# Build the two Python environments used in the paper.
#
#   bash env/install.sh          # submodules + clari env (CrystAF, Clari NFT)
#   bash env/install.sh mcf      # additionally the MolCrystalFlow env
#   bash env/install.sh ccdc     # additionally the CSD Python API env (Table 3 COMPACK)
#
# Requires git, uv and a CUDA 12.8 capable driver.
set -euo pipefail
source "$(dirname "$0")/paths.sh"
cd "${CRYSTAF_ROOT}"

FAIRCHEM="fairchem-core @ git+https://github.com/facebookresearch/fairchem.git@d05f188dd0a58430c3b800355eef50b750a17ad4#subdirectory=packages/fairchem-core"

echo "==> submodules"
git submodule update --init --recursive
if git -C CrystalGenModel/clari apply --check "${CRYSTAF_ROOT}/patches/clari.patch" 2>/dev/null; then
  git -C CrystalGenModel/clari apply "${CRYSTAF_ROOT}/patches/clari.patch"
  echo "    applied patches/clari.patch"
else
  echo "    patches/clari.patch already applied"
fi

echo "==> clari env (Python 3.13, torch 2.8.0, from clari/uv.lock)"
(cd CrystalGenModel/clari && uv sync --extra uma)
uv pip install --python "${CLARI_PY}" --no-deps --reinstall "${FAIRCHEM}"
uv pip install --python "${CLARI_PY}" --no-deps -e .

if [[ "${1:-}" == "ccdc" ]]; then
  echo "==> ccdc env for COMPACK (Python 3.11; needs a CCDC licence)"
  uv venv --python 3.11 .venvs/ccdc
  uv pip install --python .venvs/ccdc/bin/python \
    --index-url https://pip.ccdc.cam.ac.uk/ --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    csd-python-api==3.7.1 polars==1.44.2 tqdm==4.70.1 numpy==2.2.6
fi

if [[ "${1:-}" == "mcf" ]]; then
  echo "==> molcrystalflow env (Python 3.12, torch 2.7.1)"
  uv venv --seed --python 3.12 .venvs/molcrystalflow
  "${MCF_PY}" -m pip install -r env/requirements-molcrystalflow.txt
  "${MCF_PY}" -m pip install --no-deps "${FAIRCHEM}"
  "${MCF_PY}" -m pip install --no-deps -e CrystalGenModel/MolCrystalFlow -e .
fi

echo "==> check"
"${CLARI_PY}" -c "import clari, crystal_nft, fairchem.core, torch; print('clari env ok, torch', torch.__version__)"
[[ -x "${MCF_PY}" ]] && "${MCF_PY}" -c "import molcrystalflow, crystal_nft, torch; print('mcf env ok, torch', torch.__version__)" || true
