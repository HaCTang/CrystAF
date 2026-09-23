"""DiffusionNFT-style forward-process losses for flow-matching generators."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def return_decay(step: int, decay_type: int) -> float:
    """Old-policy EMA decay schedule from DiffusionNFT."""
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.0075, 0.999
    else:
        raise ValueError(f"Unknown decay_type={decay_type}")
    if step < flat:
        return 0.0
    return min((step - flat) * uprate, uphold)


@torch.no_grad()
def copy_params(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    for s, d in zip(src.parameters(), dst.parameters(), strict=True):
        d.data.copy_(s.data)


@torch.no_grad()
def sync_old_policy(
    train_module: torch.nn.Module,
    old_module: torch.nn.Module,
    *,
    step: int,
    decay_type: int = 1,
) -> float:
    decay = return_decay(step, decay_type)
    for src, tgt in zip(train_module.parameters(), old_module.parameters(), strict=True):
        tgt.data.copy_(tgt.data * decay + src.data * (1.0 - decay))
    return decay


def adaptive_x_weight(pred_x: Tensor, true_x: Tensor, eps: float = 1e-5) -> Tensor:
    """Per-sample adaptive weight used by DiffusionNFT (detached)."""
    with torch.no_grad():
        w = (
            torch.abs(pred_x.double() - true_x.double())
            .mean(dim=tuple(range(1, true_x.ndim)), keepdim=True)
            .clamp(min=eps)
        )
    return w.to(pred_x.dtype)


def nft_mixture_predictions(
    forward_pred: Tensor,
    old_pred: Tensor,
    *,
    beta: float,
) -> tuple[Tensor, Tensor]:
    """Implicit positive / negative velocity (or endpoint) mixtures."""
    positive = beta * forward_pred + (1.0 - beta) * old_pred.detach()
    negative = (1.0 + beta) * old_pred.detach() - beta * forward_pred
    return positive, negative


def nft_reconstruction_loss(
    forward_pred: Tensor,
    old_pred: Tensor,
    *,
    xt: Tensor,
    clean: Tensor,
    t: Tensor,
    r: Tensor,
    beta: float,
    time_convention: str = "clari",
    reduce: bool = True,
    mask: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """
    DiffusionNFT policy loss in reconstruction space.

    time_convention:
      - "clari": xt = (1-t)*noise + t*clean, clean_hat = xt + (1-t)*v
      - "sd3":   xt = (1-t)*clean + t*noise, clean_hat = xt - t*v
    """
    if t.ndim == 1:
        t_exp = t.view(-1, *([1] * (clean.ndim - 1)))
    else:
        t_exp = t

    positive, negative = nft_mixture_predictions(forward_pred, old_pred, beta=beta)

    if time_convention == "clari":
        pos_clean = xt + (1.0 - t_exp) * positive
        neg_clean = xt + (1.0 - t_exp) * negative
    elif time_convention == "sd3":
        pos_clean = xt - t_exp * positive
        neg_clean = xt - t_exp * negative
    else:
        raise ValueError(f"Unknown time_convention={time_convention}")

    pos_w = adaptive_x_weight(pos_clean, clean)
    neg_w = adaptive_x_weight(neg_clean, clean)

    pos_err = (pos_clean - clean) ** 2 / pos_w
    neg_err = (neg_clean - clean) ** 2 / neg_w

    if mask is not None:
        # mask: (B, N) or broadcastable to coord dims after lattice rows
        m = mask
        while m.ndim < pos_err.ndim:
            m = m.unsqueeze(-1)
        pos_err = pos_err * m
        neg_err = neg_err * m
        denom = m.mean(dim=tuple(range(1, pos_err.ndim))).clamp(min=1e-6)
        pos_loss = pos_err.mean(dim=tuple(range(1, pos_err.ndim))) / denom
        neg_loss = neg_err.mean(dim=tuple(range(1, neg_err.ndim))) / denom
    else:
        pos_loss = pos_err.mean(dim=tuple(range(1, pos_err.ndim)))
        neg_loss = neg_err.mean(dim=tuple(range(1, neg_err.ndim)))

    r = r.reshape(-1)
    ori = r * pos_loss / beta + (1.0 - r) * neg_loss / beta
    policy = ori.mean() if reduce else ori
    return {
        "policy_loss": policy,
        "unweighted_policy_loss": ori.mean().detach(),
        "positive_loss": pos_loss.mean().detach(),
        "negative_loss": neg_loss.mean().detach(),
    }


def nft_vector_loss(
    forward_pred: Tensor,
    old_pred: Tensor,
    target: Tensor,
    *,
    r: Tensor,
    beta: float,
    reduce: bool = True,
) -> dict[str, Tensor]:
    """
    NFT mixture applied directly in prediction space (for MCF endpoint / VF heads).

    L = r * ||pos - target||^2 / beta + (1-r) * ||neg - target||^2 / beta
    """
    positive, negative = nft_mixture_predictions(forward_pred, old_pred, beta=beta)
    pos_w = adaptive_x_weight(positive, target)
    neg_w = adaptive_x_weight(negative, target)
    pos_loss = ((positive - target) ** 2 / pos_w).mean(dim=tuple(range(1, target.ndim)))
    neg_loss = ((negative - target) ** 2 / neg_w).mean(dim=tuple(range(1, target.ndim)))
    r = r.reshape(-1)
    # If target is per-node (PyG), r may need expansion — caller should align shapes.
    if pos_loss.shape[0] != r.shape[0]:
        # scatter-style: assume r is per-graph and pred is flat — caller should pass aligned r
        raise ValueError(
            f"r batch ({r.shape[0]}) != loss batch ({pos_loss.shape[0]}); "
            "align advantages to prediction rows before calling nft_vector_loss"
        )
    ori = r * pos_loss / beta + (1.0 - r) * neg_loss / beta
    policy = ori.mean() if reduce else ori
    return {
        "policy_loss": policy,
        "unweighted_policy_loss": ori.mean().detach(),
        "positive_loss": pos_loss.mean().detach(),
        "negative_loss": neg_loss.mean().detach(),
    }


def kl_velocity_loss(forward_pred: Tensor, ref_pred: Tensor) -> Tensor:
    return F.mse_loss(forward_pred, ref_pred.detach())


def combine_nft_and_kl(
    policy_loss: Tensor,
    kl_loss: Tensor,
    *,
    kl_coef: float,
    adv_clip_max: float = 5.0,
) -> Tensor:
    """Match DiffusionNFT scaling: policy_loss * adv_clip_max + kl_coef * kl."""
    return policy_loss * adv_clip_max + kl_coef * kl_loss
