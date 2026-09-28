# Where Should Physics Enter a Molecular Crystal Generator?

Code, configurations, evaluation scripts and figure data for the paper.

CrystAF is an all-atom crystal flow map `z_t = z_r + (t - r) U(z_r, r, t)`, distilled from Clari-M
with AnyFlow; one adapter serves 8-50 NFE. Holding CrystAF and the UMA-OMC potential fixed, the
paper compares three places where the same physical signal can enter:

- **Training time**: relax-and-distill, regressing `U` onto UMA-relaxed copies of the model's own samples.
- **Post-training**: negative-aware fine-tuning (NFT) with UMA energy / force / stress / eligibility
  advantages, converted to the instantaneous velocity through the flow-map identity
  `V = U - (t - r) D_r U`. No gradient passes through UMA.
- **Inference time**: UMA force guidance, fixed-cell UMA relaxation, PCFM bond projection, and a
  classical chain (parity inversion, restrained MMFF, declash, optional cell calibration).

The same post-training route is also applied to multi-step Clari-M and rigid-body MolCrystalFlow.

```
src/crystal_nft/   meanflow/ (flow map, AnyFlow distillation, sampler, correctors, UMA guidance/relaxation)
                   nft/ rewards/ rigid/ adapters/ train/ tests/
configs/           one YAML per training run (header states the reported GPU count)
scripts/           train.sh, eval_*.sh, cost.sh, table3.sh, data/, analysis/
figures/           figdata.json, cost_table.json (numbers behind every figure)
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
bash env/install.sh ccdc     # optional, COMPACK for structure recovery (CSD Python API, needs a CCDC licence)
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
| `crystaf-cont3-step2000.pt` | CrystAF-base (distilled; initializes every experiment) |
| `crystaf-uma-rl-seed929-epoch6.pt` | CrystAF-UMA (released model) |
| `teacher-rank800-pbnft-epoch1.pt` | Clari-M distillation teacher |
| `crystaf-nft-mfpure-epoch12.pt` | PB-rank reward comparison |
| `clari-m-uma-nft-epoch1.pt`, `clari-l-uma-nft-epoch1.pt` | UMA post-training on Clari-M / Clari-L |
| `molcrystalflow-uma-nft-phaseB-epoch12.pt` | UMA post-training on MolCrystalFlow |

### Data

The CSD is licensed and cannot be redistributed; building the Clari tensors needs a CCDC licence,
an installed CSD and a ConQuest CIF + MOL2 export (see the Clari README).

```bash
# Clari train/val/test split -> dataset/clari/csd/{config.json,train.pt,val.pt,test.pt}
bash scripts/data/build_clari_csd.sh --check
bash scripts/data/build_clari_csd.sh
# Structure-recovery test set rebuilt from CSD 2026.1 -> dataset/clari_2026/csd/{test.pt,test_cifs.parquet}
.venvs/ccdc/bin/python scripts/data/export_csd_test_families.py --out-dir dataset/clari_2026/raw
CLARI_DATA_DIR=dataset/clari_2026 bash scripts/data/build_clari_csd.sh --process
# MolCrystalFlow NFT molecules (public Thurlemann data, deterministic)
.venvs/molcrystalflow/bin/python scripts/data/prepare_mcf_molecules.py \
  --cache dataset/molcrystalflow/thurlemann23/preprocessed/normalized/train_molcrystal_normalized.pkl.gz \
  --out-dir dataset/molcrystalflow/nft_molecules
```

## Evaluation

All all-atom quality numbers use the first 200 families of the Clari validation split x 20 samples
and report `summary.paper_bootstrap` from `metrics.json` (stereo is
`mean_stereo_agreement_defined_pct`). CrystAF is sampled with the interval flow-map sampler (set by
the scripts) on `t_i = (i/N)^rho`, rho = 0.30 / 0.75 / 1 / 1 for N = 8 / 16 / 32 / 50; the main text
uses N = 32. Heun with T steps costs 2T - 1 network evaluations; a flow-map jump costs one.

```bash
E=scripts/eval_crystaf.sh; C=checkpoints/crystaf
bash $E $C/crystaf-cont3-step2000.pt 32 runs/eval/base            # CrystAF-base
bash $E $C/crystaf-uma-rl-seed929-epoch6.pt 32 runs/eval/uma      # CrystAF-UMA (released)
bash $E runs/ablation_full/checkpoint-nft-epoch4.pt 32 runs/eval/matched   # any trained arm
bash scripts/eval_clari.sh clari-med 50 runs/eval/clari_m         # Clari-M, Heun 50 = 99 NFE
```

Inference-time methods are environment variables on top of `eval_crystaf.sh` (same weights, same seeds):

| Main-table row | Environment |
|---|---|
| + UMA guidance, gentle | `CRYSTAF_UMA_GUIDE=0.01 CRYSTAF_UMA_GUIDE_MAXDISP=0.05 CRYSTAF_UMA_GUIDE_TSTART=0.5` |
| + UMA guidance, strong | `CRYSTAF_UMA_GUIDE=0.05 CRYSTAF_UMA_GUIDE_MAXDISP=0.2 CRYSTAF_UMA_GUIDE_TSTART=0.5` |
| + UMA relaxation (K) | `CRYSTAF_UMA_RELAX=10` or `50` (fixed cell, 0.05 A per step) |
| + PCFM projection | `CRYSTAF_PCFM_BOND=1` (6 Gauss-Newton iterations, 0.25 A cap, no chirality term) |
| + parity, MMFF, declash | `CRYSTAF_MIRROR_FIX=body CRYSTAF_STEREO_REFLECT=1 CRYSTAF_MMFF=1 CRYSTAF_RELAX_CLASH=1` |
| + cell calibration | add `CRYSTAF_VOL_SCALE=0.9850` (fitted on train: `scripts/analysis/calibrate_vol_scale.py`) |

`CRYSTAF_EVAL_UMA_SCORE=1` adds per-sample UMA single-point energy / force / stress diagnostics.
Print result rows with `python scripts/analysis/report_table1.py <out_dir> ...`.

| Paper result | Command |
|---|---|
| Main table, cost columns (1 GPU, first 4 families) | `bash scripts/cost.sh <gpu>`, then `scripts/analysis/cost_table.py` |
| Sampling budgets (NFE 8 / 16 / 32 / 50) | replace `32` by the budget |
| Distillation teacher (Heun 16) | `bash scripts/eval_clari.sh clari-med 16 runs/eval/teacher checkpoints/crystaf/teacher-rank800-pbnft-epoch1.pt` |
| Structure recovery, n_s = 150, k = 30 | `bash scripts/table3.sh sample && bash scripts/table3.sh compack && bash scripts/table3.sh report` |
| Structure recovery, appendix | same with `NS=400` (k = 200) or `NS=30` (k = 30) |
| Energy-enrichment AUC (OXtal) | `ROWS="clari-m clari-l crystaf-uma" NS=400 bash scripts/table3.sh sample` / `compack`, then `bash scripts/table3.sh auc` |
| Generality, Clari-M (1000 families) | `MAX_CRYSTALS=0 bash scripts/eval_clari.sh clari-med 50 runs/eval/clari_m_nft checkpoints/crystaf/clari-m-uma-nft-epoch1.pt` |
| Generality, MolCrystalFlow RMAD | `bash scripts/eval_mcf.sh runs/eval/mcf_base` and `... runs/eval/mcf_nft checkpoints/crystaf/molcrystalflow-uma-nft-phaseB-epoch12.pt` |

Structure-recovery rows: Clari-M (39 NFE), CrystAF-base and CrystAF-UMA (32 NFE), each with and
without the classical chain including cell calibration (`-chain`).

## Training

Each run is `bash scripts/train.sh <module> <config>` on the GPU count in the config header. NFT
averages weights through the filesystem at each epoch end, so the number of ranks sets the families
per epoch.

```bash
T=scripts/train.sh
# 1. Distillation teacher: PB-rank NFT on Clari-M (7 GPUs)
bash $T train_clari_nft configs/teacher/pb_nft_s1.yaml
bash $T train_clari_nft configs/teacher/pb_nft_s2.yaml
# 2. AnyFlow distillation into CrystAF-base, 4 x 2000 steps (7 GPUs)
for s in 1 2 3 4; do bash $T train_meanflow_csd configs/distill/stage$s.yaml; done
# 3. Post-training from CrystAF-base, full UMA signal (4 GPUs; read at epoch 4)
bash $T train_meanflow_nft configs/ablation/full.yaml
# 4. Training-time baseline: relax-and-distill (4 GPUs)
bash $T train_meanflow_umadistill configs/ablation/relax_distill.yaml
```

Configs read the released `checkpoints/crystaf/crystaf-cont3-step2000.pt` and
`teacher-rank800-pbnft-epoch1.pt`; point `meanflow_ckpt` / `clari_nft_init` at `runs/` to use your own.
Matched arms all start from CrystAF-base and are read at epoch 4 (epoch 6 in the appendix).

| Config (`train_meanflow_nft` unless noted) | Paper |
|---|---|
| `ablation/full.yaml` | full signal: E, F, stress, eligibility, volume (CrystAF-UMA, matched) |
| `ablation/efs.yaml` | E, F, stress, eligibility |
| `ablation/e_only.yaml` | E, eligibility |
| `ablation/feasibility.yaml` | eligibility, volume |
| `ablation/gate_only.yaml` | eligibility gate only |
| `ablation/dnft_on_u.yaml` | DiffusionNFT on `U` (no flow-map identity) |
| `ablation/relax_distill.yaml` (`train_meanflow_umadistill`) | relax-and-distill (24 candidates, 25 fixed-cell UMA steps) |
| `rl/uma_seed929.yaml`, `rl/uma_seed2029.yaml` | UMA reward, two seeds |
| `rl/pb_seed929.yaml`, `rl/pb_seed2029.yaml` | PB-rank reward comparison, two seeds |
| `rl/uma_rollout32.yaml`, `rl/uma_scaleup.yaml` (8 GPUs) | rollout-grid mismatch, scale-up |
| `rl/uma_distill.yaml` (`train_meanflow_umadistill`, 5 GPUs) | relaxation distillation on the PB-rank arm |
| `clari_nft/clari_{m,l}_uma.yaml` (`train_clari_nft`) | generality, Clari-M / Clari-L |
| `mcf_nft/stage{1a,1b,2,3,4}.yaml` (`train_mcf_nft`) | generality, MolCrystalFlow NFT stages 1-4 |
| `mcf_grpo/stage{1,2,3,4}.yaml` (`train_mcf_grpo`) | Flow-GRPO baseline stages 1-4 |

## Licence

Code: MIT. CrystAF weights are derived from Clari and released under CC-BY-NC-4.0, following the
Clari model licence. Clari, MolCrystalFlow, fairchem/UMA and the CSD keep their own licences.

## Citation

```bibtex
@inproceedings{tang2027physics,
  title     = {Where Should Physics Enter a Molecular Crystal Generator?},
  author    = {Tang, Haocheng and Wang, Junmei and Jin, Wengong},
  booktitle = {International Conference on Learning Representations},
  year      = {2027}
}
```
