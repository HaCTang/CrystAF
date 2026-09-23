#!/usr/bin/env bash
# Download all weights and public data into checkpoints/ and dataset/.
#
#   bash env/download.sh
#
# UMA is gated: request access at https://huggingface.co/facebook/UMA and run
# `hf auth login` first. CSD data needs a CCDC licence (see README, "Data").
set -euo pipefail
source "$(dirname "$0")/paths.sh"
cd "${CRYSTAF_ROOT}"
mkdir -p checkpoints/crystaf checkpoints/uma dataset/molcrystalflow

"${CLARI_PY}" - <<'PY'
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

def fetch(repo, name, dest):
    dest = Path(dest)
    if dest.is_file():
        print("  have", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hf_hub_download(repo, name), dest)
    print("  got ", dest)

print("== Clari base models (the-matter-lab/clari)")
for f in ["clari-med.ckpt", "clari-large.ckpt"]:
    fetch("the-matter-lab/clari", f, f"checkpoints/{f}")

print("== CrystAF weights (Haocheng1/CrystAF)")
for f in [
    "crystaf-uma-rl-seed929-epoch6.pt",
    "crystaf-nft-mfpure-epoch12.pt",
    "crystaf-cont3-step2000.pt",
    "teacher-rank800-pbnft-epoch1.pt",
    "clari-m-uma-nft-epoch1.pt",
    "clari-l-uma-nft-epoch1.pt",
    "molcrystalflow-uma-nft-phaseB-epoch12.pt",
]:
    fetch("Haocheng1/CrystAF", f, f"checkpoints/crystaf/{f}")

print("== UMA (facebook/UMA, gated)")
fetch("facebook/UMA", "checkpoints/uma-s-1p1.pt", "checkpoints/uma/uma-s-1p1.pt")
# Table 3 energies: clari's compute-energies loads uma-s-1p2 through the fairchem hub cache.
from fairchem.core.calculate import pretrained_mlip
ckpt = pretrained_mlip._MODEL_CKPTS.checkpoints["uma-s-1p2"]
hf_hub_download(filename=ckpt.filename, repo_id=ckpt.repo_id,
                subfolder=ckpt.subfolder, revision=ckpt.revision)
for ref_type in ["atom_refs", "form_elem_refs"]:
    ref = getattr(ckpt, ref_type, None)
    if ref is not None:
        hf_hub_download(filename=ref["filename"], repo_id=ckpt.repo_id,
                        subfolder=ref.get("subfolder"), revision=ckpt.revision)
print("  cached uma-s-1p2")
PY

echo "== MolCrystalFlow checkpoints and Thurlemann data (Zenodo 19673190)"
ZENODO="https://zenodo.org/records/19673190/files"
MCF="CrystalGenModel/MolCrystalFlow"
if [[ ! -f "${MCF}/model-checkpoints/thurlemann23/best.ckpt" ]]; then
  wget -c "${ZENODO}/model-checkpoints.zip" -O /tmp/mcf-model-checkpoints.zip
  unzip -o /tmp/mcf-model-checkpoints.zip -d "${MCF}/"
fi
if [[ ! -d dataset/molcrystalflow/thurlemann23 ]]; then
  wget -c "${ZENODO}/thurlemann23.zip" -O /tmp/mcf-thurlemann23.zip
  unzip -o /tmp/mcf-thurlemann23.zip -d dataset/molcrystalflow/
fi
echo "done"
