"""Lightweight AMD/PDD assess worker for spawn children.

Must not import eval/train modules: those pull the full model stack and already
use ~40-50 GiB virtual address space. A 32 GiB RLIMIT_AS then makes every
subsequent import raise MemoryError.
"""

from __future__ import annotations

import os

# Parent owns GPUs; assess is CPU-only. Set before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch


def _json_safe_metrics(metrics: dict) -> dict:
    clean = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = (
                value.detach().cpu().item()
                if value.numel() == 1
                else value.detach().cpu().tolist()
            )
        elif isinstance(value, (np.floating, np.integer)):
            value = value.item()
        clean[key] = value
    return clean


def _current_vm_bytes() -> int:
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def _apply_as_headroom(extra_bytes: int) -> None:
    """Cap address space at current VmSize + extra (never below current usage)."""
    if not extra_bytes or extra_bytes <= 0:
        return
    import resource

    current = _current_vm_bytes()
    cap = current + int(extra_bytes)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


def assess_subprocess_entry(
    result_queue,
    pred,
    true_crystal,
    amd_metric: str,
    mem_limit_bytes: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        from clari.pipelines.utils.metrics import assess_crystals_eval

        _apply_as_headroom(int(mem_limit_bytes or 0))
        with torch.inference_mode():
            metrics = assess_crystals_eval(
                pred, true_crystal, amd_metric=amd_metric
            )
        result_queue.put(("ok", _json_safe_metrics(metrics)))
    except Exception as exc:  # noqa: BLE001 — must report any child failure
        try:
            result_queue.put(("err", f"{type(exc).__name__}: {exc}"))
        except Exception:  # noqa: BLE001
            os._exit(1)
