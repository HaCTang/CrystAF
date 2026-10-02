# Where Should Physics Enter a Molecular Crystal Generator?

Code, configs, evaluation scripts and figure data (`figures/`) for the paper.

**CrystAF** (**Cryst**al**A**ny**F**low) is an all-atom crystal flow map `z_t = z_r + (t - r) U(z_r, r, t)` distilled from Clari-M.
With CrystAF and the UMA-OMC potential held fixed, the paper compares where the same physical
signal enters: at training time (relax-and-distill), in post-training (NFT through the flow-map
identity `V = U - (t - r) D_r U`), or at inference time (UMA guidance / relaxation, PCFM, a
classical correction chain). Weights: [huggingface.co/Haocheng1/CrystAF](https://huggingface.co/Haocheng1/CrystAF).

## Setup

Linux, CUDA 12.8 GPUs, `git`, [`uv`](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/HaCTang/CrystAF.git && cd CrystAF
bash env/install.sh          # clari env (+ patches/clari.patch, fairchem)
bash env/install.sh mcf      # MolCrystalFlow env
bash env/install.sh ccdc     # optional: COMPACK (needs a CCDC licence)
hf auth login                # UMA weights are gated (huggingface.co/facebook/UMA)
bash env/download.sh         # Clari, CrystAF, UMA, MolCrystalFlow weights -> checkpoints/
```

The CSD cannot be redistributed. Build the Clari split with `bash scripts/data/build_clari_csd.sh`
(CCDC licence + ConQuest export); the CSD 2026.1 recovery test set with
`scripts/data/export_csd_test_families.py` followed by
`CLARI_DATA_DIR=dataset/clari_2026 bash scripts/data/build_clari_csd.sh --process`.
Tests: `CrystalGenModel/clari/.venv/bin/python -m pytest`.

## Evaluation

200 validation families x 20 samples; read `summary.paper_bootstrap` in `metrics.json`. The
scripts use the interval flow-map sampler with rho = 0.30 / 0.75 / 1 for NFE 8 / 16 / 32-50.

```bash
bash scripts/eval_crystaf.sh checkpoints/crystaf/crystaf-uma-rl-seed929-epoch6.pt 32 runs/eval/uma
bash scripts/eval_crystaf.sh checkpoints/crystaf/crystaf-cont3-step2000.pt 32 runs/eval/base
bash scripts/eval_clari.sh clari-med 50 runs/eval/clari_m          # Heun 50 = 99 NFE
```

| Result | How |
|---|---|
| UMA guidance, gentle / strong | `CRYSTAF_UMA_GUIDE=0.01 CRYSTAF_UMA_GUIDE_MAXDISP=0.05` / `=0.05 ...=0.2`, both `CRYSTAF_UMA_GUIDE_TSTART=0.5` |
| UMA relaxation (K steps) | `CRYSTAF_UMA_RELAX=10` or `50` |
| PCFM projection | `CRYSTAF_PCFM_BOND=1` |
| Classical chain (+ calibration) | `CRYSTAF_MIRROR_FIX=body CRYSTAF_STEREO_REFLECT=1 CRYSTAF_MMFF=1 CRYSTAF_RELAX_CLASH=1` (+ `CRYSTAF_VOL_SCALE=0.9850`) |
| Cost columns | `bash scripts/cost.sh <gpu>` |
| Structure recovery | `bash scripts/table3.sh sample \| compack \| report`; `NS=150` (default), `400`, `30` |
| Energy-enrichment AUC | `ROWS="clari-m clari-l crystaf-uma" NS=400 bash scripts/table3.sh ...`, then `table3.sh auc` |
| Generality | `MAX_CRYSTALS=0 bash scripts/eval_clari.sh clari-med 50 <out> checkpoints/crystaf/clari-m-uma-nft-epoch1.pt`; `bash scripts/eval_mcf.sh <out> [ckpt]` |

Inference-time settings are environment variables on top of `eval_crystaf.sh`.

## Training

`bash scripts/train.sh <module> <config>` on the GPU count in the config header.

```bash
T=scripts/train.sh
bash $T train_clari_nft configs/teacher/pb_nft_s1.yaml && bash $T train_clari_nft configs/teacher/pb_nft_s2.yaml
for s in 1 2 3 4; do bash $T train_meanflow_csd configs/distill/stage$s.yaml; done   # CrystAF-base
bash $T train_meanflow_nft configs/ablation/full.yaml                                 # post-training
bash $T train_meanflow_umadistill configs/ablation/relax_distill.yaml                 # relax-and-distill
```

| Configs | Paper |
|---|---|
| `ablation/{full,efs,e_only,feasibility,gate_only}.yaml` | signal ablations (from CrystAF-base, read at epoch 4) |
| `ablation/dnft_on_u.yaml`, `ablation/relax_distill.yaml` | DiffusionNFT on `U`; relax-and-distill |
| `rl/{uma,pb}_seed{929,2029}.yaml` | UMA vs PB-rank reward, two seeds |
| `rl/uma_rollout32.yaml`, `rl/uma_scaleup.yaml`, `rl/uma_distill.yaml` | rollout grid, scale-up, relaxation distillation |
| `clari_nft/`, `mcf_nft/`, `mcf_grpo/` | generality: Clari-M/L, MolCrystalFlow NFT vs Flow-GRPO |

## Licence and citation

Code: MIT. Weights: CC-BY-NC-4.0 (derived from Clari). Submodules and the CSD keep their own licences.

```bibtex
@misc{tang2026physicsentermolecularcrystal,
      title={Where Should Physics Enter a Molecular Crystal Generator?}, 
      author={Haocheng Tang and Junmei Wang and Wengong Jin},
      year={2026},
      eprint={2609.36398},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2609.36398}, 
}
```
