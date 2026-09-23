# CrystAF: Physics-Aligned Reinforcement Learning for Molecular Crystal Flow Maps

Code, configurations and evaluation scripts for the paper.

CrystAF distills the all-atom Clari generator into a dual-time flow map
`z_t = z_r + (t - r) U(z_r, r, t)` and post-trains it with negative-aware fine-tuning (NFT),
using UMA energy / force / stress as a black-box physical reward. The same physics-aligned NFT
interface is also applied to multi-step Clari-M/L and to the rigid-body MolCrystalFlow.

```
src/crystal_nft/   meanflow/ (flow map, AnyFlow distillation, sampler, correctors)
                   nft/ rewards/ rigid/ adapters/ train/ tests/
configs/           one YAML per training stage (header states the reported GPU count)
scripts/           train.sh, eval_*.sh, table3.sh, data/, analysis/
env/               install.sh, download.sh, pinned requirements
patches/           clari.patch (checkpoint injection into sample-test, COMPACK fixes)
CrystalGenModel/   clari, MolCrystalFlow (git submodules, pinned commits)
```

## Setup

Requires Linux, CUDA 12.8 capable GPUs, `git` and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/HaCTang/CrystAF.git && cd CrystAF
bash env/install.sh          # clari env: Python 3.13, torch 2.8.0 (clari/uv.lock) + fairchem-core@d05f188
bash env/install.sh mcf      # MolCrystalFlow env: Python 3.12, torch 2.7.1 (env/requirements-molcrystalflow.txt)
bash env/install.sh ccdc     # optional, COMPACK for Table 3 (CSD Python API, needs a CCDC licence)
hf auth login                # UMA weights are gated: request access at huggingface.co/facebook/UMA
bash env/download.sh         # Clari, CrystAF, UMA and MolCrystalFlow weights + Thurlemann data
```

`install.sh` applies `patches/clari.patch` to the Clari submodule and installs this package into
each environment. CPU unit tests: `CrystalGenModel/clari/.venv/bin/python -m pytest`.

### Weights

`env/download.sh` places everything under `checkpoints/`. CrystAF weights are hosted at
[huggingface.co/Haocheng1/CrystAF](https://huggingface.co/Haocheng1/CrystAF).

| File (`checkpoints/crystaf/`) | Model |
|---|---|
| `crystaf-uma-rl-seed929-epoch6.pt` | CrystAF-UMA (reported model; also CrystAF-S with correctors) |
| `crystaf-nft-mfpure-epoch12.pt` | CrystAF-PB |
| `crystaf-cont3-step2000.pt` | CrystAF-base (distilled, before RL) |
| `teacher-rank800-pbnft-epoch1.pt` | PB-NFT Clari-M distillation teacher |
| `clari-m-uma-nft-epoch1.pt`, `clari-l-uma-nft-epoch1.pt` | UMA-NFT on Clari-M / Clari-L |
| `molcrystalflow-uma-nft-phaseB-epoch12.pt` | UMA-NFT on MolCrystalFlow |

### Data

The CSD is licensed and cannot be redistributed; building the Clari tensors needs a CCDC licence,
an installed CSD and a ConQuest CIF + MOL2 export (see the Clari README).

```bash
# Clari train/val/test split -> dataset/clari/csd/{config.json,train.pt,val.pt,test.pt}
bash scripts/data/build_clari_csd.sh --check
bash scripts/data/build_clari_csd.sh
# Table 3 test set rebuilt from CSD 2026.1 -> dataset/clari_2026/csd/{test.pt,test_cifs.parquet}
.venvs/ccdc/bin/python scripts/data/export_csd_test_families.py --out-dir dataset/clari_2026/raw
CLARI_DATA_DIR=dataset/clari_2026 bash scripts/data/build_clari_csd.sh --process
# MolCrystalFlow NFT molecules (public Thurlemann data, deterministic)
.venvs/molcrystalflow/bin/python scripts/data/prepare_mcf_molecules.py \
  --cache dataset/molcrystalflow/thurlemann23/preprocessed/normalized/train_molcrystal_normalized.pkl.gz \
  --out-dir dataset/molcrystalflow/nft_molecules
```

## Evaluation

All all-atom numbers use the fixed 200-family Clari validation subset x 20 samples and report
`summary.paper_bootstrap` from `metrics.json` (stereo is `mean_stereo_agreement_defined_pct`).
CrystAF must be sampled with the interval flow-map sampler (set by the scripts); the grid is
`t_i = (i/N)^rho` with rho = 0.30 / 0.75 / 1 / 1 for N = 8 / 16 / 32 / 50, one adapter for all N.

| Paper result | Command |
|---|---|
| Main table, Clari-M / Clari-L (Heun 50 = 99 NFE) | `bash scripts/eval_clari.sh clari-med 50 runs/eval/clari_m` (and `clari-large`) |
| Main table, CrystAF-base | `bash scripts/eval_crystaf.sh checkpoints/crystaf/crystaf-cont3-step2000.pt 32 runs/eval/base_32` |
| Main table, CrystAF-UMA | `bash scripts/eval_crystaf.sh checkpoints/crystaf/crystaf-uma-rl-seed929-epoch6.pt 32 runs/eval/uma_32` |
| Main table, CrystAF-PB | same with `crystaf-nft-mfpure-epoch12.pt` |
| Main table, CrystAF-S | CrystAF-UMA with `CRYSTAF_MIRROR_FIX=body CRYSTAF_STEREO_REFLECT=1 CRYSTAF_MMFF=1 CRYSTAF_RELAX_CLASH=1` |
| Main table, + cell calibration | CrystAF-S plus `CRYSTAF_VOL_SCALE=0.9850` (fitted on train: `scripts/analysis/calibrate_vol_scale.py`) |
| Sampling budgets (NFE 8 / 16 / 32 / 50) | replace `32` by the budget |
| Distillation teacher (Heun 16) | `bash scripts/eval_clari.sh clari-med 16 runs/eval/teacher checkpoints/crystaf/teacher-rank800-pbnft-epoch1.pt` |
| Generality, Clari-M/L +UMA-NFT (1000 families) | `MAX_CRYSTALS=0 bash scripts/eval_clari.sh clari-med 50 runs/eval/clari_m_nft checkpoints/crystaf/clari-m-uma-nft-epoch1.pt` |
| Generality, MolCrystalFlow RMAD | `bash scripts/eval_mcf.sh runs/eval/mcf_base` and `... runs/eval/mcf_nft checkpoints/crystaf/molcrystalflow-uma-nft-phaseB-epoch12.pt` |
| Table 3, Sol@200 | `bash scripts/table3.sh sample && bash scripts/table3.sh compack && bash scripts/table3.sh report` |

Use `python scripts/analysis/report_table1.py <out_dir> ...` to print result rows. Heun with T
steps costs 2T - 1 network evaluations; a flow-map jump costs one.

## Training

Each stage is `bash scripts/train.sh <module> <config>`; run on the GPU count in the config
header. NFT averages weights through the filesystem at each epoch end, so the number of ranks sets
the families per epoch.

```bash
T=scripts/train.sh
# 1. Distillation teacher: PB-rank NFT on Clari-M (7 GPUs)
bash $T train_clari_nft configs/teacher/pb_nft_s1.yaml
bash $T train_clari_nft configs/teacher/pb_nft_s2.yaml
# 2. AnyFlow distillation into CrystAF-base, 4 x 2000 steps (7 GPUs)
for s in 1 2 3 4; do bash $T train_meanflow_csd configs/distill/stage$s.yaml; done
# 3. Stage-2 RL. Every arm starts from the PB-rank warm-up (4 GPUs)
bash $T train_meanflow_nft configs/rl/pb_warmup_1.yaml
bash $T train_meanflow_nft configs/rl/pb_warmup_2.yaml
bash $T train_meanflow_nft configs/rl/uma_seed929.yaml     # CrystAF-UMA (epoch 6)
bash $T train_meanflow_nft configs/rl/pb_seed929.yaml      # CrystAF-PB  (epoch 12)
```

The RL configs read the released `checkpoints/crystaf/crystaf-cont3-step2000.pt` and
`teacher-rank800-pbnft-epoch1.pt`; point `meanflow_ckpt` / `clari_nft_init` at `runs/` to use your
own. Further configs:

| Config | Paper |
|---|---|
| `rl/uma_seed2029.yaml`, `rl/pb_seed2029.yaml` | second seeds |
| `rl/uma_rollout32.yaml`, `rl/uma_scaleup.yaml` (8 GPUs) | report-grid rollout, scale-up |
| `rl/uma_distill.yaml` (`train_meanflow_umadistill`, 5 GPUs) | UMA-relaxation distillation ablation |
| `clari_nft/clari_{m,l}_uma.yaml` (`train_clari_nft`) | generality, Clari-M/L |
| `mcf_nft/stage{1a,1b,2,3,4}.yaml` (`train_mcf_nft`) | generality, MolCrystalFlow NFT stages 1-4 |
| `mcf_grpo/stage{1,2,3,4}.yaml` (`train_mcf_grpo`) | Flow-GRPO baseline stages 1-4 |

## Licence

Code: MIT. CrystAF weights are derived from Clari and released under CC-BY-NC-4.0, following the
Clari model licence. Clari, MolCrystalFlow, fairchem/UMA and the CSD keep their own licences.

## Citation

```bibtex
@inproceedings{crystaf2027,
  title     = {{CrystAF}: Physics-Aligned Reinforcement Learning for Molecular Crystal Flow Maps},
  author    = {Anonymous},
  booktitle = {Submitted to the International Conference on Learning Representations},
  year      = {2027}
}
```
