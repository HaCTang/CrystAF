"""MolCrystalFlow adapter: packing sampling and NFT updates on rigid-body targets."""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch import Tensor

from crystal_nft.nft.loss import (
    combine_nft_and_kl,
    kl_velocity_loss,
    nft_mixture_predictions,
)

logger = logging.getLogger(__name__)

# Serialize FlowModule.model swaps when async prefetch samples on a worker thread
# while the main thread runs NFT updates on the same process/device.
_FLOW_SAMPLE_LOCK = __import__("threading").Lock()


def _ensure_packing_gen_on_path() -> Path:
    repo = Path(__file__).resolve().parents[3]
    mcf_root = repo / "CrystalGenModel" / "MolCrystalFlow"
    gen_dir = mcf_root / "csp-pipeline"
    for p in (str(mcf_root), str(gen_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return gen_dir


def load_mcf_bundle(
    ckpt_path: str,
    *,
    device: str = "cuda",
):
    """Load FlowModule plus frozen old/ref FlowModel copies."""
    _ensure_packing_gen_on_path()
    from molcrystalflow_gen.packing_gen import load_model

    flow, cfg = load_model(ckpt_path, device=device)
    flow.train()

    # Ensure inference cfg exists
    if flow._inference_cfg is None:
        flow._inference_cfg = OmegaConf.create({"num_samples": 1, "save_trajectories": False})
    else:
        flow._inference_cfg.save_trajectories = False

    old_model = copy.deepcopy(flow.model).eval()
    ref_model = copy.deepcopy(flow.model).eval()
    for p in old_model.parameters():
        p.requires_grad_(False)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    return {
        "flow": flow,
        "cfg": cfg,
        "model": flow.model,
        "old_model": old_model,
        "ref_model": ref_model,
        "device": device,
        "ckpt_path": ckpt_path,
    }


_FEATURE_CACHE: dict[str, dict] = {}


def extract_features_from_xyz(
    xyz_path: str, *, verbose: bool = False, use_cache: bool = True
) -> dict:
    """Extract MolCrystalFlow conditioning features from a monomer XYZ.

    Aux-feature extraction shells out to Open Babel/RDKit and is expensive, so
    results are memoized per path (features are deterministic for a given file).
    """
    if use_cache and xyz_path in _FEATURE_CACHE:
        return _FEATURE_CACHE[xyz_path]
    _ensure_packing_gen_on_path()
    from ase.io import read
    from molcrystalflow_gen.packing_gen import extract_molecule_features

    atoms = read(xyz_path)
    if isinstance(atoms, list):
        atoms = atoms[0]
    feats = extract_molecule_features(atoms, compute_aux_features=True, verbose=verbose)
    if use_cache:
        _FEATURE_CACHE[xyz_path] = feats
    return feats


@torch.inference_mode()
def sample_packings(
    flow,
    old_model: nn.Module,
    mol_features: dict,
    *,
    z_value: int,
    num_samples: int,
    has_axis_flip: bool = False,
    num_timesteps: int = 50,
    scaling: float = 9.0,
    exp_rate: float = 3.0,
    sampling_batch_size: int = 1,
    device: str = "cuda",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Sample K packings with the old policy.

    Returns a list of dicts with ASE atoms and training targets
    (trans_1, rotmats_1, lattice_1, axis_flip_state).
    """
    _ensure_packing_gen_on_path()
    from ase import Atoms
    from ase.data import chemical_symbols
    from molcrystalflow_gen.packing_gen import (
        IDX_TO_ATOM_TYPE,
        create_inference_batch,
        set_seed,
    )
    from torch_geometric.data import Batch

    set_seed(seed)
    train_model = flow.model
    with _FLOW_SAMPLE_LOCK:
        flow.model = old_model
        try:
            if has_axis_flip:
                samples_per_flip = num_samples // 2
                flip_states = [(0, samples_per_flip), (1, num_samples - samples_per_flip)]
            else:
                flip_states = [(0, num_samples)]

            records: list[dict[str, Any]] = []
            for flip_state, n_samples in flip_states:
                if n_samples == 0:
                    continue
                base_batch = create_inference_batch(
                    mol_features,
                    z_value,
                    has_axis_flip=has_axis_flip,
                    axis_flip_state=flip_state,
                    device="cpu",
                )
                base_data = (
                    base_batch.get_example(0)
                    if hasattr(base_batch, "get_example")
                    else base_batch.to_data_list()[0]
                )
                # FlowModule normally loops over num_samples serially. Instead,
                # replicate the conditioning graph and sample several structures in
                # one model forward pass. This substantially increases GPU occupancy.
                flow._inference_cfg.num_samples = 1
                flow._inference_cfg.save_trajectories = False
                if hasattr(flow, "interpolant"):
                    if hasattr(flow.interpolant, "_trans_cfg"):
                        flow.interpolant._trans_cfg.scaling = scaling
                    if hasattr(flow.interpolant, "_rots_cfg"):
                        flow.interpolant._rots_cfg.exp_rate = exp_rate

                remaining = n_samples
                requested_bs = max(1, int(sampling_batch_size))
                while remaining > 0:
                    batch_size = min(requested_bs, remaining)
                    while True:
                        try:
                            batch = Batch.from_data_list(
                                [base_data.clone() for _ in range(batch_size)]
                            ).to(device)
                            results = flow(batch, num_timesteps=num_timesteps)
                            break
                        except torch.cuda.OutOfMemoryError:
                            if batch_size <= 1:
                                raise
                            del batch
                            torch.cuda.empty_cache()
                            batch_size = max(1, batch_size // 2)
                            logger.warning(
                                "Sampling OOM; retrying with batch_size=%d", batch_size
                            )

                    # With num_samples=1, FlowModule returns concatenated graph
                    # outputs under the leading sampling dimension. Split them back
                    # into independent structures.
                    n_atoms = int(mol_features["num_atoms"]) * int(z_value)
                    cart = results["cart_coords"][0].detach().cpu().reshape(
                        batch_size, n_atoms, 3
                    )
                    types = results["atom_types"][0].detach().cpu().reshape(
                        batch_size, n_atoms
                    )
                    lattices = results["lattices"][0].detach().cpu().reshape(
                        batch_size, 3, 3
                    )
                    pred_trans = results["pred_trans"][0].detach().cpu().reshape(
                        batch_size, z_value, 3
                    )
                    pred_rotmats = results["pred_rotmats"][0].detach().cpu().reshape(
                        batch_size, z_value, 3, 3
                    )

                    for i in range(batch_size):
                        symbols = [
                            chemical_symbols[IDX_TO_ATOM_TYPE[int(t)]]
                            for t in types[i].numpy()
                        ]
                        atoms = Atoms(
                            symbols=symbols,
                            positions=cart[i].numpy(),
                            cell=lattices[i].numpy(),
                            pbc=True,
                        )
                        atoms.new_array(
                            "bb_indices",
                            np.repeat(
                                np.arange(z_value),
                                int(mol_features["num_atoms"]),
                            ),
                        )
                        records.append(
                            {
                                "atoms": atoms,
                                "trans_1": pred_trans[i],
                                "rotmats_1": pred_rotmats[i],
                                "lattice_1": lattices[i],
                                "axis_flip_state": flip_state,
                                "z_value": z_value,
                            }
                        )
                    remaining -= batch_size
            return records
        finally:
            flow.model = train_model


def _expand_r_to_nodes(r: Tensor, batch_index: Tensor) -> Tensor:
    """Expand per-graph r [G] to per-node [N] using PyG batch vector."""
    return r[batch_index]


def nft_train_step_mcf(
    flow,
    model: nn.Module,
    old_model: nn.Module,
    ref_model: nn.Module,
    mol_features: dict,
    records: Sequence[dict[str, Any]],
    r_weights: Tensor,
    *,
    beta: float = 0.1,
    kl_coef: float = 0.01,
    adv_clip_max: float = 5.0,
    has_axis_flip: bool = False,
    lattice_only: bool = False,
    device: str = "cuda",
) -> dict[str, Tensor]:
    """NFT update using generated rigid-body targets as clean samples."""
    from molcrystalflow.data import so3_utils
    from molcrystalflow_gen.packing_gen import create_inference_batch
    from torch_geometric.data import Batch

    training_cfg = flow._exp_cfg.training
    r_weights = r_weights.to(device=device, dtype=torch.float32)

    # Build one PyG batch from selected records (may have different flip states —
    # process as a list then Batch.from_data_list).
    data_list = []
    for rec in records:
        b = create_inference_batch(
            mol_features,
            int(rec["z_value"]),
            has_axis_flip=has_axis_flip,
            axis_flip_state=int(rec["axis_flip_state"]),
            device="cpu",
        )
        # create_inference_batch returns a Batch with one graph; extract Data
        data = b.get_example(0) if hasattr(b, "get_example") else b.to_data_list()[0]
        data.trans_1 = rec["trans_1"].float()
        data.rotmats_1 = rec["rotmats_1"].float()
        lat = rec["lattice_1"].float()
        if lat.ndim == 3 and lat.shape[0] == 1:
            lat = lat.squeeze(0)
        data.lattice_1 = lat.unsqueeze(0) if lat.ndim == 2 else lat
        data_list.append(data)

    batch = Batch.from_data_list(data_list).to(device)
    flow.interpolant.set_device(device)
    noisy = flow.interpolant.corrupt_batch(batch)

    gt_rotmats_1 = noisy["rotmats_1"]
    gt_lattice_1 = noisy["lattice_1"]
    gt_b_trans = noisy["b_trans"]
    rotmats_t = noisy["rotmats_t"]
    so3_t = noisy["so3_t"]
    l_t = noisy["l_t"]

    so3_norm_scale = 1 - torch.min(so3_t, torch.tensor(training_cfg.t_normalize_clip, device=device))
    l_norm_scale = 1 - torch.min(l_t, torch.tensor(training_cfg.t_normalize_clip, device=device))
    gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.float())

    forward_out = model(noisy)
    with torch.no_grad():
        old_out = old_model(noisy)
        ref_out = ref_model(noisy)

    # Translation head (per-node) — expand r
    r_nodes = _expand_r_to_nodes(r_weights, noisy.batch)
    # Align shapes: nft_vector_loss expects same leading dim for pred and r
    # We reduce per-node errors with r_nodes then mean.

    def _head_nft(fwd, old, target, r_vec, weight: float = 1.0):
        positive, negative = nft_mixture_predictions(fwd, old, beta=beta)
        # mean over feature dims -> [N]
        pos_err = ((positive - target) ** 2).mean(dim=tuple(range(1, target.ndim)))
        neg_err = ((negative - target) ** 2).mean(dim=tuple(range(1, target.ndim)))
        ori = r_vec * pos_err / beta + (1.0 - r_vec) * neg_err / beta
        return weight * ori.mean()

    # Scale targets like model_step
    trans_target = gt_b_trans * training_cfg.trans_scale
    fwd_trans = forward_out["pred_b_trans"] * training_cfg.trans_scale
    old_trans = old_out["pred_b_trans"] * training_cfg.trans_scale
    ref_trans = ref_out["pred_b_trans"] * training_cfg.trans_scale
    trans_policy = _head_nft(
        fwd_trans, old_trans, trans_target, r_nodes, training_cfg.translation_loss_weight
    )

    fwd_rot_vf = so3_utils.calc_rot_vf(rotmats_t, forward_out["pred_rotmats"])
    old_rot_vf = so3_utils.calc_rot_vf(rotmats_t, old_out["pred_rotmats"])
    ref_rot_vf = so3_utils.calc_rot_vf(rotmats_t, ref_out["pred_rotmats"])
    rot_target = gt_rot_vf / so3_norm_scale
    rot_policy = _head_nft(
        fwd_rot_vf / so3_norm_scale,
        old_rot_vf / so3_norm_scale,
        rot_target,
        r_nodes,
        training_cfg.rotation_loss_weights,
    )

    # Lattice is per-graph
    gt_lat = gt_lattice_1.view(gt_lattice_1.shape[0], -1) / l_norm_scale
    fwd_lat = forward_out["pred_lattice"].view(gt_lattice_1.shape[0], -1) / l_norm_scale
    old_lat = old_out["pred_lattice"].view(gt_lattice_1.shape[0], -1) / l_norm_scale
    ref_lat = ref_out["pred_lattice"].view(gt_lattice_1.shape[0], -1) / l_norm_scale
    lat_policy = _head_nft(
        fwd_lat, old_lat, gt_lat, r_weights, training_cfg.cell_loss_weight
    )

    if lattice_only:
        policy_loss = lat_policy
        kl = kl_velocity_loss(fwd_lat, ref_lat)
    else:
        policy_loss = trans_policy + rot_policy + lat_policy
        kl = (
            kl_velocity_loss(fwd_trans, ref_trans)
            + kl_velocity_loss(fwd_rot_vf, ref_rot_vf)
            + kl_velocity_loss(fwd_lat, ref_lat)
        ) / 3.0

    total = combine_nft_and_kl(
        policy_loss, kl, kl_coef=kl_coef, adv_clip_max=adv_clip_max
    )
    return {
        "loss": total,
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl.detach(),
        "trans_policy": trans_policy.detach(),
        "rot_policy": rot_policy.detach(),
        "lat_policy": lat_policy.detach(),
    }


def _per_sample_mse(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error over feature dims, keep leading sample dim."""
    dims = tuple(range(1, target.ndim))
    if not dims:
        return (pred - target) ** 2
    return ((pred - target) ** 2).mean(dim=dims)


def _grpo_clip_loss(
    err_theta: Tensor,
    err_old: Tensor,
    advantages: Tensor,
    *,
    clip_range: float,
    tau: float,
) -> tuple[Tensor, Tensor]:
    """Clipped GRPO surrogate using -MSE as a continuous logp proxy.

    ratio ~= exp(-(err_theta - err_old) / tau). Same form as Flow-GRPO's
    clipped importance ratio, adapted to flow-matching without SDE logprobs.
    """
    log_ratio = -(err_theta - err_old.detach()) / max(float(tau), 1e-8)
    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
    ratio = torch.exp(log_ratio)
    adv = advantages.to(dtype=ratio.dtype, device=ratio.device)
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    loss = torch.maximum(unclipped, clipped).mean()
    clipfrac = (torch.abs(ratio - 1.0) > clip_range).float().mean()
    return loss, clipfrac


def grpo_train_step_mcf(
    flow,
    model: nn.Module,
    old_model: nn.Module,
    ref_model: nn.Module,
    mol_features: dict,
    records: Sequence[dict[str, Any]],
    advantages: Tensor,
    *,
    clip_range: float = 0.2,
    tau: float = 1.0,
    kl_coef: float = 0.01,
    has_axis_flip: bool = False,
    lattice_only: bool = False,
    device: str = "cuda",
) -> dict[str, Tensor]:
    """GRPO-style clipped update on MCF flow heads (fair NFT baseline)."""
    from molcrystalflow.data import so3_utils
    from molcrystalflow_gen.packing_gen import create_inference_batch
    from torch_geometric.data import Batch

    training_cfg = flow._exp_cfg.training
    advantages = advantages.to(device=device, dtype=torch.float32)

    data_list = []
    for rec in records:
        b = create_inference_batch(
            mol_features,
            int(rec["z_value"]),
            has_axis_flip=has_axis_flip,
            axis_flip_state=int(rec["axis_flip_state"]),
            device="cpu",
        )
        data = b.get_example(0) if hasattr(b, "get_example") else b.to_data_list()[0]
        data.trans_1 = rec["trans_1"].float()
        data.rotmats_1 = rec["rotmats_1"].float()
        lat = rec["lattice_1"].float()
        if lat.ndim == 3 and lat.shape[0] == 1:
            lat = lat.squeeze(0)
        data.lattice_1 = lat.unsqueeze(0) if lat.ndim == 2 else lat
        data_list.append(data)

    batch = Batch.from_data_list(data_list).to(device)
    flow.interpolant.set_device(device)
    noisy = flow.interpolant.corrupt_batch(batch)

    gt_rotmats_1 = noisy["rotmats_1"]
    gt_lattice_1 = noisy["lattice_1"]
    gt_b_trans = noisy["b_trans"]
    rotmats_t = noisy["rotmats_t"]
    so3_t = noisy["so3_t"]
    l_t = noisy["l_t"]

    so3_norm_scale = 1 - torch.min(
        so3_t, torch.tensor(training_cfg.t_normalize_clip, device=device)
    )
    l_norm_scale = 1 - torch.min(
        l_t, torch.tensor(training_cfg.t_normalize_clip, device=device)
    )
    gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.float())

    forward_out = model(noisy)
    with torch.no_grad():
        old_out = old_model(noisy)
        ref_out = ref_model(noisy)

    # Graph-level advantages for node heads: scatter-mean after per-node MSE.
    def _node_err(pred: Tensor, target: Tensor) -> Tensor:
        per_node = _per_sample_mse(pred, target)
        n_graphs = int(advantages.shape[0])
        out = pred.new_zeros(n_graphs)
        counts = pred.new_zeros(n_graphs)
        out.index_add_(0, noisy.batch, per_node)
        counts.index_add_(0, noisy.batch, torch.ones_like(per_node))
        return out / counts.clamp_min(1.0)

    trans_target = gt_b_trans * training_cfg.trans_scale
    fwd_trans = forward_out["pred_b_trans"] * training_cfg.trans_scale
    old_trans = old_out["pred_b_trans"] * training_cfg.trans_scale
    ref_trans = ref_out["pred_b_trans"] * training_cfg.trans_scale

    fwd_rot_vf = so3_utils.calc_rot_vf(rotmats_t, forward_out["pred_rotmats"])
    old_rot_vf = so3_utils.calc_rot_vf(rotmats_t, old_out["pred_rotmats"])
    ref_rot_vf = so3_utils.calc_rot_vf(rotmats_t, ref_out["pred_rotmats"])
    rot_target = gt_rot_vf / so3_norm_scale

    gt_lat = gt_lattice_1.view(gt_lattice_1.shape[0], -1) / l_norm_scale
    fwd_lat = forward_out["pred_lattice"].view(gt_lattice_1.shape[0], -1) / l_norm_scale
    old_lat = old_out["pred_lattice"].view(gt_lattice_1.shape[0], -1) / l_norm_scale
    ref_lat = ref_out["pred_lattice"].view(gt_lattice_1.shape[0], -1) / l_norm_scale

    if lattice_only:
        lat_loss, clipfrac = _grpo_clip_loss(
            _per_sample_mse(fwd_lat, gt_lat),
            _per_sample_mse(old_lat, gt_lat),
            advantages,
            clip_range=clip_range,
            tau=tau,
        )
        policy_loss = training_cfg.cell_loss_weight * lat_loss
        kl = kl_velocity_loss(fwd_lat, ref_lat)
    else:
        trans_loss, cf_t = _grpo_clip_loss(
            _node_err(fwd_trans, trans_target),
            _node_err(old_trans, trans_target),
            advantages,
            clip_range=clip_range,
            tau=tau,
        )
        rot_loss, cf_r = _grpo_clip_loss(
            _node_err(fwd_rot_vf / so3_norm_scale, rot_target),
            _node_err(old_rot_vf / so3_norm_scale, rot_target),
            advantages,
            clip_range=clip_range,
            tau=tau,
        )
        lat_loss, cf_l = _grpo_clip_loss(
            _per_sample_mse(fwd_lat, gt_lat),
            _per_sample_mse(old_lat, gt_lat),
            advantages,
            clip_range=clip_range,
            tau=tau,
        )
        policy_loss = (
            training_cfg.translation_loss_weight * trans_loss
            + training_cfg.rotation_loss_weights * rot_loss
            + training_cfg.cell_loss_weight * lat_loss
        )
        clipfrac = (cf_t + cf_r + cf_l) / 3.0
        kl = (
            kl_velocity_loss(fwd_trans, ref_trans)
            + kl_velocity_loss(fwd_rot_vf, ref_rot_vf)
            + kl_velocity_loss(fwd_lat, ref_lat)
        ) / 3.0

    total = policy_loss + float(kl_coef) * kl
    return {
        "loss": total,
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl.detach(),
        "clipfrac": clipfrac.detach(),
    }
