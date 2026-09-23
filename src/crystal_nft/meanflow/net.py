"""Wrap Clari DiT for MeanFlow dual-time conditioning u(x, r, t).

AnyFlow-style TIME-EMBEDDING mixing: the flow map is parameterised by
modifying the time embedding inside the DiT, NOT by adding a correction
to the output.  Following NVlabs/AnyFlow:

    rt_emb = (1 - gate) * time_embedder(t) + gate * delta_embedder(r)

where ``time_embedder`` is the original timestep embedding (base) and
``delta_embedder`` is a deep copy with separate weights.  The gate is a
fixed scalar (default 0.25).  The entire DiT processes this mixed
embedding, so all transformer layers see the dual-time information.
"""

from __future__ import annotations

import os

import copy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def _infer_time_embed_dim(base: nn.Module) -> int:
    proj = getattr(base, "proj", None)
    if proj is not None and hasattr(proj, "out_features"):
        return int(proj.out_features)
    param = next(base.parameters(), None)
    device = param.device if param is not None else torch.device("cpu")
    dtype = param.dtype if param is not None else torch.float32
    with torch.no_grad():
        y = base(torch.zeros(1, 1, device=device, dtype=dtype))
    return int(y.shape[-1])


class MeanFlowTimeEmbedding(nn.Module):
    """Proxy that replaces ``dit.embed_timestep`` for dual-time conditioning.

    Mixes the base time embedding (for ``t``) with a delta time embedding
    (for ``r``) via a convex combination controlled by ``gate``, plus an
    optional additive residual of ``base(t)``:

        rt_emb = (1 - gate) * base(t) + gate * delta(r) + residual * base(t)
                 then (1 - α) * that + α * delta(t)

    ``residual=0, α=0`` keeps the original AnyFlow mix.  With ``gate=1``
    that is exactly ``delta(r)``.  Raising ``α`` blends in ``delta(t)``
    from the *same* student embedder (r=t remains ``delta(t)``).

    When ``r`` is not set (``set_r(None)``), falls back to ``base(t_col)``
    so the model behaves identically to the original DiT.
    """

    def __init__(
        self,
        base: nn.Module,
        gate_value: float = 0.25,
        *,
        time_parameterization: str = "legacy",
    ):
        super().__init__()
        parameterization = str(time_parameterization).lower().strip()
        if parameterization not in ("legacy", "clari_forward"):
            raise ValueError(
                "time_parameterization must be legacy|clari_forward, "
                f"got {parameterization!r}"
            )
        self.time_parameterization = parameterization
        self.base = base
        self.delta = copy.deepcopy(base)
        self.register_buffer(
            "gate", torch.tensor([gate_value], dtype=torch.float32)
        )
        # Additive base(t) residual on top of the convex mix:
        #   e = (1-g)*base(t) + g*delta(r) + residual * base(t)
        # residual=0 keeps the current mix (gate=1.0 => delta(r) only).
        self.register_buffer(
            "time_residual", torch.tensor([0.0], dtype=torch.float32)
        )
        # Same-embed endpoint mix (student delta, NOT frozen teacher base):
        #   e = (1-α)*mix + α*delta(t)
        # With gate=1, mix=delta(r), so e=(1-α)*delta(r)+α*delta(t).
        # α=0 is the current gate=1.0 student. r=t ⇒ e=delta(t) for any α.
        self.register_buffer(
            "endpoint_mix", torch.tensor([0.0], dtype=torch.float32)
        )
        # Zero-init map of frozen base(t-r)-base(0). Identity at step 0 so a
        # resumed mix-0.25 checkpoint is unchanged. Using base (not student
        # delta) keeps Δt in the graph without sending grads into delta.
        # Optional masks: dt_min / r_min / r_max select which (r, t) jumps
        # receive the residual. Default r_min=0, r_max=1 is a no-op.
        dim = _infer_time_embed_dim(base)
        self.interval_proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.interval_proj.weight)
        hidden = 2 * int(dim)
        self.interval_mlp = nn.Sequential(
            nn.Linear(int(dim), hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, int(dim), bias=False),
        )
        nn.init.zeros_(self.interval_mlp[-1].weight)
        self.register_buffer(
            "interval_mix", torch.tensor([0.0], dtype=torch.float32)
        )
        self.register_buffer(
            "interval_dt_min", torch.tensor([0.0], dtype=torch.float32)
        )
        self.register_buffer(
            "interval_r_max", torch.tensor([1.0], dtype=torch.float32)
        )
        self.register_buffer(
            "interval_r_min", torch.tensor([0.0], dtype=torch.float32)
        )
        self._r: Optional[Tensor] = None
        self._t: Optional[Tensor] = None

    def set_times(self, r: Tensor | None, t: Tensor | None) -> None:
        self._r = r
        self._t = t

    def set_r(self, r: Tensor | None) -> None:
        self.set_times(r, None)

    def forward(self, t_col: Tensor) -> Tensor:
        base_emb = self.base(t_col)
        if self._r is None:
            return base_emb
        other = self._r
        if self.time_parameterization == "clari_forward":
            if self._t is None:
                raise RuntimeError("clari_forward time embedding requires both r and t")
            other = self._t
        other_col = _time_column(other, t_col)
        if other_col.shape != t_col.shape:
            other_col = other_col.expand_as(t_col).contiguous()
        # Always run base AND delta so DDP sees both, including gate=1 / residual=0.
        delta_r = self.delta(other_col)
        delta_t = self.delta(t_col)
        gate = self.gate.to(base_emb.dtype)
        mix = (1.0 - gate) * base_emb + gate * delta_r
        res = getattr(self, "time_residual", None)
        if res is not None:
            mix = mix + res.to(base_emb.dtype) * base_emb
        alpha = getattr(self, "endpoint_mix", None)
        if alpha is None:
            return mix
        alpha = alpha.to(mix.dtype)
        mix = (1.0 - alpha) * mix + alpha * delta_t
        imix = getattr(self, "interval_mix", None)
        proj = getattr(self, "interval_proj", None)
        if (
            imix is not None
            and proj is not None
            and self._r is not None
            and self._t is not None
        ):
            r_col = _time_column(self._r, t_col)
            t_for_dt = _time_column(self._t, t_col)
            if r_col.shape != t_col.shape:
                r_col = r_col.expand_as(t_col).contiguous()
            if t_for_dt.shape != t_col.shape:
                t_for_dt = t_for_dt.expand_as(t_col).contiguous()
            dt_col = (t_for_dt - r_col).clamp_min(0.0)
            zero_col = torch.zeros_like(dt_col)
            # Frozen teacher embed: Δt stays live for finite-difference JVP,
            # but student delta is not in this path (cont18 collapse).
            interval = self.base(dt_col) - self.base(zero_col)
            residual = proj(interval)
            mlp = getattr(self, "interval_mlp", None)
            if mlp is not None:
                residual = residual + mlp(interval)
            add = imix.to(mix.dtype) * residual
            dt_min = getattr(self, "interval_dt_min", None)
            r_lim = getattr(self, "interval_r_max", None)
            r_floor = getattr(self, "interval_r_min", None)
            if dt_min is not None or r_lim is not None or r_floor is not None:
                jump_gate = torch.ones_like(dt_col, dtype=mix.dtype)
                if dt_min is not None:
                    jump_gate = jump_gate * (dt_col > dt_min.to(dt_col.dtype)).to(mix.dtype)
                if r_lim is not None:
                    jump_gate = jump_gate * (r_col <= r_lim.to(r_col.dtype)).to(mix.dtype)
                if r_floor is not None:
                    jump_gate = jump_gate * (r_col >= r_floor.to(r_col.dtype)).to(mix.dtype)
                add = add * jump_gate
            mix = mix + add
        return mix


def _time_column(value: Tensor, like: Tensor | None = None) -> Tensor:
    if value.dim() == 0:
        value = value.unsqueeze(0)
    out = value.unsqueeze(-1)
    if like is not None:
        out = out.to(device=like.device, dtype=like.dtype)
    return out


class DualTimeResidualConditioner(nn.Module):
    """Zero-init dual-time residual that is exactly zero when ``r == t``."""

    def __init__(
        self,
        time_embed: nn.Module,
        dim_cond: int,
        *,
        feature_mode: str = "both",
        gate_value: float = 1.0,
    ):
        super().__init__()
        mode = str(feature_mode).lower().strip()
        if mode not in ("interval", "both"):
            raise ValueError(f"dual_time_feature_mode must be interval|both, got {mode!r}")
        self.feature_mode = mode
        self.embed = copy.deepcopy(time_embed)
        n_features = 1 if mode == "interval" else 2
        hidden = 2 * int(dim_cond)
        self.adapter = nn.Sequential(
            nn.Linear(n_features * int(dim_cond), hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, int(dim_cond), bias=False),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        self.register_buffer(
            "dual_gate", torch.tensor([float(gate_value)], dtype=torch.float32)
        )
        self._r: Optional[Tensor] = None
        self._t: Optional[Tensor] = None

    def set_times(self, r: Tensor, t: Tensor) -> None:
        self._r = r
        self._t = t

    def forward(self, like: Tensor) -> Tensor:
        if self._r is None or self._t is None:
            return torch.zeros_like(like)
        r_col = _time_column(self._r, like)
        t_col = _time_column(self._t, like)
        dt_col = (t_col - r_col).clamp_min(0.0)
        zero_col = torch.zeros_like(dt_col)
        interval = self.embed(dt_col) - self.embed(zero_col)
        features = [interval]
        if self.feature_mode == "both":
            endpoint = self.embed(r_col) - self.embed(t_col)
            features.insert(0, endpoint)
        dual = self.adapter(torch.cat(features, dim=-1))
        return self.dual_gate.to(dtype=dual.dtype, device=dual.device) * dual


class DualTimeStemAdapter(nn.Module):
    """Preserve Clari's original condition mean, then add a dual-time residual."""

    def __init__(self, inner: nn.Module, conditioner: DualTimeResidualConditioner):
        super().__init__()
        self.inner = inner
        self.conditioner = conditioner

    def forward(self, x: Tensor) -> Tensor:
        base = self.inner(x)
        return base + self.conditioner(base)


class MeanFlowDiTWrapper(nn.Module):
    """Flow-map wrapper on top of Clari DiT (AnyFlow time-embedding variant)."""

    def __init__(
        self,
        dit: nn.Module,
        gate_value: float = 0.25,
        *,
        conditioning_mode: str = "mix",
        dual_time_feature_mode: str = "both",
        dual_gate_value: float = 1.0,
        time_parameterization: str = "legacy",
    ):
        super().__init__()
        self.dit = dit
        mode = str(conditioning_mode).lower().strip()
        if mode not in ("mix", "dual_residual"):
            raise ValueError(f"conditioning_mode must be mix|dual_residual, got {mode!r}")
        self.conditioning_mode = mode
        self.time_parameterization = str(time_parameterization).lower().strip()
        if self.time_parameterization not in ("legacy", "clari_forward"):
            raise ValueError(
                "time_parameterization must be legacy|clari_forward, "
                f"got {self.time_parameterization!r}"
            )
        if mode == "mix":
            base_emb = dit.embed_timestep
            proxy = MeanFlowTimeEmbedding(
                base_emb,
                gate_value=gate_value,
                time_parameterization=self.time_parameterization,
            )
            for p in proxy.base.parameters():
                p.requires_grad_(False)
            dit.embed_timestep = proxy
            object.__setattr__(self, "_time_proxy", proxy)
            object.__setattr__(self, "embed_deltat", proxy.delta)
            object.__setattr__(self, "delta_mlp", proxy.delta)
        else:
            conditioner = DualTimeResidualConditioner(
                dit.embed_timestep,
                int(dit.dim_cond),
                feature_mode=dual_time_feature_mode,
                gate_value=dual_gate_value,
            )
            dit.stem_cond = DualTimeStemAdapter(dit.stem_cond, conditioner)
            object.__setattr__(self, "_dual_time", conditioner)
            object.__setattr__(self, "embed_deltat", conditioner.embed)
            object.__setattr__(self, "delta_mlp", conditioner.embed)
        self.self_cond = getattr(dit, "self_cond", False)
        self._chiral_bias: Optional[nn.Module] = None
        self._stereo_tokens: Optional[nn.Module] = None
        self._stereo_pairs: Optional[nn.Module] = None
        self._stereo_cond: Optional[nn.Module] = None
        self._stereo_node_mod: Optional[nn.Module] = None
        self._stereo_pair_mod: Optional[nn.Module] = None

    @property
    def dim_cond(self) -> int:
        return int(self.dit.dim_cond)

    def attach_chiral_bias(self, module: nn.Module) -> None:
        self._chiral_bias = module

    def attach_stereo_tokens(self, module: "StereoTokenEmbedding") -> None:
        """Register the per-atom CIP token embedding and hook it onto the DiT."""
        self._stereo_tokens = module
        module.bind(self.dit)

    def attach_stereo_pairs(self, module: "StereoPairEmbedding") -> None:
        """Register the LoQI auxiliary-edge embedding and hook it onto the DiT."""
        self._stereo_pairs = module
        module.bind(self.dit)

    def attach_stereo_pair_mod(self, module: "StereoPairMod") -> None:
        """Register the LoQI directed-cycle post-tanh pair modulation."""
        self._stereo_pair_mod = module
        module.bind(self.dit)

    def attach_stereo_node_mod(self, module: "StereoNodeMod") -> None:
        """Register the per-atom post-tanh modulation and hook it onto the DiT."""
        self._stereo_node_mod = module
        module.bind(self.dit)

    def attach_stereo_chiral_branch(self, module: "StereoChiralBranch") -> None:
        self._stereo_chiral_branch = module

    def attach_stereo_cond(self, module: "StereoCondEmbedding") -> None:
        """Register the FiLM/``cond`` stereo path (applied after the token tanh)."""
        self._stereo_cond = module

    def _set_times(self, r: Tensor, t: Tensor) -> None:
        if self.conditioning_mode == "mix":
            self._time_proxy.set_times(r, t)
            interval = (t - r).clamp_min(0.0)
            for module in self.dit.modules():
                setter = getattr(module, "set_interval_gate", None)
                if callable(setter):
                    setter(interval)
        else:
            self._dual_time.set_times(r, t)
            interval = (t - r).clamp_min(0.0)
            for module in self.dit.modules():
                setter = getattr(module, "set_interval_gate", None)
                if callable(setter):
                    setter(interval)

    def forward(
        self,
        x: Tensor,
        xsc: Tensor | None,
        r: Tensor,
        t: Tensor,
        f,
    ) -> Tensor:
        return self.crystal_forward(x=x, xsc=xsc, r=r, t=t, f=f)

    def crystal_forward(
        self,
        x: Tensor,
        xsc: Tensor | None,
        r: Tensor,
        t: Tensor,
        f,
    ) -> Tensor:
        self._set_times(r, t)
        dit_time = r if self.time_parameterization == "clari_forward" else t
        # Publish the time to the FiLM hooks. They sit on `dit.node_mod` /
        # `dit.pair_bias`, which never receive t, so without this they fire with
        # the same strength at every point of the trajectory.
        _publish_stereo_time(self.dit, dit_time)
        try:
            out = self.dit.crystal_forward(x=x, xsc=xsc, t=dit_time, f=f)
        finally:
            _publish_stereo_time(self.dit, None)
        # Parity-sensitive residual must be applied INSIDE this forward. Doing
        # it in `interface.forward` after net(...) returns puts the branch
        # parameters outside the DDP-wrapped module, and their autograd hooks
        # then fire after the reducer has finalised:
        #   RuntimeError: Expected to mark a variable ready only once.
        br = getattr(self, "_stereo_chiral_branch", None)
        if br is not None:
            geo = x
            if bool(getattr(br, "use_endpoint_geometry", False)):
                # The branch reads a tetrahedron: V, s*V and three edge lengths.
                # On the raw state x_t those are meaningless noise at small t, so
                # the branch can only act LATE -- once atoms are nearly in place,
                # where moving them distorts bonds/angles. That is why PB sits at
                # ~69 for every configuration with an effective residual, while
                # clash (intermolecular) moves freely, and why every softening
                # that shrank the residual shrank chirality by the same amount.
                # Flipping a stereocentre is a PERMUTATION, not a rotation -- no
                # rotation can change chirality -- so a late correction is
                # necessarily a large rearrangement.
                # The endpoint estimate x_hat1 = x_t + (1-t) u is a plausible
                # crystal at ANY t, so feeding the branch that geometry lets it
                # steer the trajectory from the start, when displacement is free.
                tt = dit_time.reshape(-1)
                if tt.numel() == 1:
                    tt = tt.expand(out.shape[0])
                sh = [-1] + [1] * (out.ndim - 1)
                geo = x + (1.0 - tt[: out.shape[0]]).view(*sh) * out.detach()
            out = br(out, geo, dit_time)
        return out

    def crystal_forward_velocity(
        self,
        x: Tensor,
        xsc: Tensor | None,
        t: Tensor,
        f,
    ) -> Tensor:
        return self.crystal_forward(x, xsc, r=t, t=t, f=f)


class InstantaneousDiT(nn.Module):
    """DDP-safe wrapper around an unwrapped Clari DiT (r = t score net).

    ``forward(x, xsc, t, f)`` matches the non-flow-map branch in
    ``MeanFlowCrystalInterface`` so a DDP container can own the call.
    """

    def __init__(self, dit: nn.Module):
        super().__init__()
        self.dit = dit
        self.self_cond = getattr(dit, "self_cond", False)

    def crystal_forward(self, x, xsc, t, f):
        return self.dit.crystal_forward(x=x, xsc=xsc, t=t, f=f)

    def forward(self, x, xsc, t, f):
        return self.crystal_forward(x, xsc, t, f)


def is_time_mix_buffer_key(name: str) -> bool:
    """True for mix scalars; false for AdaLN ``*.gate.weight``."""
    if name.endswith(".gate.weight") or name.endswith(".gate.bias"):
        return False
    return (
        name.endswith("embed_timestep.gate")
        or name.endswith("embed_timestep.time_residual")
        or name.endswith("embed_timestep.endpoint_mix")
        or name.endswith("_time_proxy.gate")
        or name.endswith("_time_proxy.time_residual")
        or name.endswith("_time_proxy.endpoint_mix")
        or name.endswith("conditioner.dual_gate")
        or name.endswith("_dual_time.dual_gate")
        or         name.endswith("embed_timestep.interval_mix")
        or name.endswith("_time_proxy.interval_mix")
        or name.endswith("embed_timestep.interval_dt_min")
        or name.endswith("_time_proxy.interval_dt_min")
        or name.endswith("embed_timestep.interval_r_max")
        or name.endswith("_time_proxy.interval_r_max")
        or name.endswith("embed_timestep.interval_r_min")
        or name.endswith("_time_proxy.interval_r_min")
        or name.endswith("nfe8_mix")
        or name in ("gate", "time_residual", "endpoint_mix", "dual_gate", "interval_mix")
    )


def scheduled_mix_value(
    step: int,
    start: float,
    end: float,
    nsteps: int,
    default: float,
) -> float:
    """Linear anneal. ``nsteps<=0`` keeps ``default``. Last train step hits ``end``."""
    n = int(nsteps)
    if n <= 0:
        return float(default)
    if n == 1:
        return float(end)
    u = min(1.0, max(0.0, float(step) / float(n - 1)))
    return float(start) + (float(end) - float(start)) * u


def time_mix_from_cfg(cfg: dict, step: int) -> tuple[float, float, float]:
    gate_default = float(cfg.get("gate_value", 0.25))
    gate_start = cfg.get("gate_anneal_start")
    gate_end = cfg.get("gate_anneal_end")
    gate = scheduled_mix_value(
        int(step),
        float(gate_default if gate_start is None else gate_start),
        float(gate_default if gate_end is None else gate_end),
        int(cfg.get("gate_anneal_steps", 0) or 0),
        gate_default,
    )
    res_default = float(cfg.get("time_residual", 0.0))
    res_start = cfg.get("time_residual_start")
    res_end = cfg.get("time_residual_end")
    residual = scheduled_mix_value(
        int(step),
        float(res_default if res_start is None else res_start),
        float(res_default if res_end is None else res_end),
        int(cfg.get("time_residual_anneal_steps", 0) or 0),
        res_default,
    )
    ep_default = float(cfg.get("endpoint_mix", 0.0))
    ep_start = cfg.get("endpoint_mix_start")
    ep_end = cfg.get("endpoint_mix_end")
    endpoint = scheduled_mix_value(
        int(step),
        float(ep_default if ep_start is None else ep_start),
        float(ep_default if ep_end is None else ep_end),
        int(cfg.get("endpoint_mix_anneal_steps", 0) or 0),
        ep_default,
    )
    return gate, residual, endpoint


def get_time_proxy(net: nn.Module) -> Optional[MeanFlowTimeEmbedding]:
    core = net.module if hasattr(net, "module") else net
    proxy = getattr(core, "_time_proxy", None)
    if isinstance(proxy, MeanFlowTimeEmbedding):
        return proxy
    dit = getattr(core, "dit", None)
    emb = getattr(dit, "embed_timestep", None) if dit is not None else None
    return emb if isinstance(emb, MeanFlowTimeEmbedding) else None


def get_dual_time_conditioner(net: nn.Module) -> Optional[DualTimeResidualConditioner]:
    core = net.module if hasattr(net, "module") else net
    dual = getattr(core, "_dual_time", None)
    if isinstance(dual, DualTimeResidualConditioner):
        return dual
    dit = getattr(core, "dit", None)
    stem = getattr(dit, "stem_cond", None) if dit is not None else None
    while stem is not None:
        if isinstance(stem, DualTimeStemAdapter):
            return stem.conditioner
        stem = getattr(stem, "inner", None)
    return None


def set_time_mix(
    net: nn.Module,
    *,
    gate: float | None = None,
    residual: float | None = None,
    endpoint: float | None = None,
    interval: float | None = None,
    interval_dt_min: float | None = None,
    interval_r_max: float | None = None,
    interval_r_min: float | None = None,
    dual_gate: float | None = None,
) -> None:
    proxy = get_time_proxy(net)
    with torch.no_grad():
        if proxy is not None and gate is not None:
            proxy.gate.fill_(float(gate))
        if proxy is not None and residual is not None and hasattr(proxy, "time_residual"):
            proxy.time_residual.fill_(float(residual))
        if proxy is not None and endpoint is not None and hasattr(proxy, "endpoint_mix"):
            proxy.endpoint_mix.fill_(float(endpoint))
        if proxy is not None and interval is not None and hasattr(proxy, "interval_mix"):
            proxy.interval_mix.fill_(float(interval))
        if (
            proxy is not None
            and interval_dt_min is not None
            and hasattr(proxy, "interval_dt_min")
        ):
            proxy.interval_dt_min.fill_(float(interval_dt_min))
        if (
            proxy is not None
            and interval_r_max is not None
            and hasattr(proxy, "interval_r_max")
        ):
            proxy.interval_r_max.fill_(float(interval_r_max))
        if (
            proxy is not None
            and interval_r_min is not None
            and hasattr(proxy, "interval_r_min")
        ):
            proxy.interval_r_min.fill_(float(interval_r_min))
        dual = get_dual_time_conditioner(net)
        if dual is not None and dual_gate is not None:
            dual.dual_gate.fill_(float(dual_gate))


def live_time_mix_meta(net: nn.Module) -> dict:
    proxy = get_time_proxy(net)
    out: dict = {}
    if proxy is not None:
        out["gate_live"] = float(proxy.gate.reshape(-1)[0].detach().cpu())
        if hasattr(proxy, "time_residual"):
            out["time_residual_live"] = float(proxy.time_residual.reshape(-1)[0].detach().cpu())
        if hasattr(proxy, "endpoint_mix"):
            out["endpoint_mix_live"] = float(proxy.endpoint_mix.reshape(-1)[0].detach().cpu())
        if hasattr(proxy, "interval_mix"):
            out["interval_mix_live"] = float(proxy.interval_mix.reshape(-1)[0].detach().cpu())
        if hasattr(proxy, "interval_r_min"):
            out["interval_r_min_live"] = float(proxy.interval_r_min.reshape(-1)[0].detach().cpu())
        if hasattr(proxy, "interval_r_max"):
            out["interval_r_max_live"] = float(proxy.interval_r_max.reshape(-1)[0].detach().cpu())
    dual = get_dual_time_conditioner(net)
    if dual is not None:
        out["dual_gate_live"] = float(dual.dual_gate.reshape(-1)[0].detach().cpu())
    return out


def restore_time_mix_buffers(
    net: nn.Module,
    state: dict | None,
    meta: dict | None = None,
) -> int:
    """Put online mix scalars back after an EMA ``load_state_dict``."""
    n = 0
    if state:
        cur = net.state_dict()
        patch = {
            k: v
            for k, v in state.items()
            if is_time_mix_buffer_key(k) and k in cur and torch.is_tensor(v)
        }
        if patch:
            net.load_state_dict(patch, strict=False)
            n += len(patch)
    if meta:
        gate = meta.get("gate_live")
        res = meta.get("time_residual_live")
        ep = meta.get("endpoint_mix_live")
        dual_gate = meta.get("dual_gate_live")
        if gate is not None or res is not None or ep is not None or dual_gate is not None:
            set_time_mix(
                net,
                gate=None if gate is None else float(gate),
                residual=None if res is None else float(res),
                endpoint=None if ep is None else float(ep),
                dual_gate=None if dual_gate is None else float(dual_gate),
            )
            n += 1
    return n


def load_ema_preserving_time_mix(net: nn.Module, payload: dict) -> bool:
    ema = payload.get("ema_state_dict")
    if not ema:
        return False
    net.load_state_dict(ema, strict=False)
    restore_time_mix_buffers(net, payload.get("net_state_dict"), payload.get("meta"))
    return True


_STEREO_HOOKS: dict = {}


def _stereo_gain(buf: Tensor) -> float:
    """Conditioning gain; ``CRYSTAF_STEREO_GAIN`` overrides for eval-time probes."""
    import os

    env = os.environ.get("CRYSTAF_STEREO_GAIN", "").strip()
    if env:
        return float(env)
    return float(buf)


class StereoTokenEmbedding(nn.Module):
    """Zero-init per-atom CIP tag embedding added to Clari's ``embed_feats`` output.

    Clari builds its atom tokens as ``tanh((0.5/K) * sum(h_k))``; appending a new
    term to that sum would rescale every pretrained term.  Instead this adds into
    ``embed_feats``' output via a forward hook, so ``K`` is unchanged, no existing
    state-dict key is renamed, and the zero-init embedding makes step 0 exactly
    equal to the un-conditioned model.

    Tags come from :func:`crystal_nft.meanflow.stereo.build_batch_stereo` and are
    published for the current batch through ``active_stereo_tags``.
    """

    def __init__(self, dim: int, n_tags: int = 4, gain: float = 1.0):
        super().__init__()
        self.emb = nn.Embedding(int(n_tags), int(dim))
        nn.init.zeros_(self.emb.weight)
        self.register_buffer("gain", torch.tensor(float(gain)))

    def bind(self, dit: nn.Module) -> None:
        """(Re)register the forward hook on ``dit.embed_feats``. Idempotent.

        The handle lives in a module-level registry rather than on ``self`` so
        that ``copy.deepcopy`` of the wrapper (NFT old/ref copies, frozen
        teachers) does not try to clone a live hook.
        """
        handle = _STEREO_HOOKS.pop(id(self), None)
        if handle is not None:
            handle.remove()
        _STEREO_HOOKS[id(self)] = dit.embed_feats.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, args, output: Tensor) -> Tensor:
        from crystal_nft.meanflow.stereo import get_active_stereo_tags

        tags = get_active_stereo_tags()
        if tags is None:
            return output
        b, n = output.shape[0], output.shape[1]
        tags = tags.to(output.device)
        if tags.ndim == 1:
            tags = tags.unsqueeze(0)
        # Self-conditioning and CD passes run on a prefix of the batch.
        if tags.shape[0] == 1 and b > 1:
            tags = tags.expand(b, -1)
        elif tags.shape[0] > b:
            tags = tags[:b]
        elif tags.shape[0] < b:
            return output
        # `use_lattice_regs` prepends 3 lattice rows to the atom axis.
        if tags.shape[1] == n - 3:
            tags = torch.nn.functional.pad(tags, (3, 0), value=0)
        elif tags.shape[1] > n:
            tags = tags[:, :n]
        elif tags.shape[1] < n:
            tags = torch.nn.functional.pad(tags, (0, n - tags.shape[1]), value=0)
        # Tag 0 (achiral atom / padding) must contribute *exactly* nothing, so an
        # achiral molecule sees a bit-identical model.  Keeping the masked term in
        # the graph (rather than returning early) also keeps the parameter "used"
        # for DDP when a batch happens to contain no stereocentre.
        keep = (tags > 0).unsqueeze(-1).to(output.dtype)
        # Clari's token stem sums ~5 branches whose per-dim RMS is 0.9 (element,
        # feats) to 8.8 (cart). A zero-init embedding trained at this lr plateaus
        # near RMS 0.04 -- about 5% of one branch -- so without a gain the tag is
        # simply drowned and flipping it moves 3.5% of centres.
        g = _stereo_gain(self.gain)
        return output + (self.emb(tags).to(output.dtype) * keep * g)




def _publish_stereo_time(dit: nn.Module, t) -> None:
    """Stash the current time on the DiT for the stereo FiLM hooks."""
    dit._stereo_cur_t = None if t is None else t.detach()


def _stereo_time_window(module: nn.Module, output: Tensor) -> Optional[Tensor]:
    """``(B, 1, 1)`` multiplier for a FiLM head, or None when unwindowed.

    Measured on 64 crystals with Heun-50: the FiLM
    heads -- not the parity branch -- are what produce handedness, and they are
    the entire PB cost (heads dead 89.94, heads live 74.74). Their benefit is
    concentrated early: the endpoint estimate is already 0.90 correct by t=0.48
    and then DECAYS to 0.64, so everything the heads do past that point buys no
    chirality while still pushing an almost-finished crystal off-manifold.
    This lets a run keep the early half and drop the late half.
    """
    t_lo = float(getattr(module, "t_min", 0.0))
    t_hi = float(getattr(module, "t_max", 1.0))
    if t_lo <= 0.0 and t_hi >= 1.0:
        return None
    dit = getattr(module, "_bound_dit", None)
    t = None if dit is None else getattr(dit, "_stereo_cur_t", None)
    if t is None:
        return None
    tt = t.reshape(-1).to(output.dtype)
    if tt.numel() == 1 and output.shape[0] > 1:
        tt = tt.expand(output.shape[0])
    elif tt.numel() != output.shape[0]:
        return None
    w = ((tt >= t_lo).to(output.dtype)
         * ((t_hi - tt) / 0.10).clamp(0.0, 1.0))
    return w.view(-1, 1, 1)


class StereoNodeMod(nn.Module):
    """Per-atom CIP modulation applied *after* Clari's token tanh.

    This is the path the other two do not provide.

    * ``StereoTokenEmbedding`` adds into ``embed_feats``, i.e. one of ~5 branches
      *inside* ``tanh((0.5/K) sum(h_k))``. That tanh is already saturated by
      ``embed_cart`` (per-dim RMS 8.8), so the tag survives at ~0.06% of the
      output -- measured, not estimated.
    * ``StereoCondEmbedding`` drives ``node_mod`` after the tanh and is strong
      (~0.45%), but it is **per-crystal**. Its input is the cell descriptor
      ``[mean_sign, frac_R, frac_S, has]``, which for a racemic cell is *exactly
      invariant* under swapping every R<->S. 57% of chiral CSD crystals are
      racemic 50/50, so on most of the data the strong path is blind by
      construction to the flip the model is being asked to perform.

    So this module puts the per-atom tag where ``cond`` already works: a
    zero-init FiLM (scale, shift) on the post-tanh node features, indexed by the
    atom's own tag. Zero-init means step 0 is bit-identical to the base model,
    and tag 0 (achiral/padding) is masked to contribute exactly nothing.
    """

    def __init__(self, dim: int, n_tags: int = 4, gain: float = 1.0,
                 t_min: float = 0.0, t_max: float = 1.0):
        super().__init__()
        self.scale = nn.Embedding(int(n_tags), int(dim))
        self.shift = nn.Embedding(int(n_tags), int(dim))
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        self.register_buffer("gain", torch.tensor(float(gain)))
        self.t_min = float(t_min)
        self.t_max = float(t_max)

    def bind(self, dit: nn.Module) -> None:
        handle = _STEREO_HOOKS.pop(("nodemod", id(self)), None)
        if handle is not None:
            handle.remove()
        # plain __dict__ write: assigning an nn.Module through the normal path
        # would register the DiT as a CHILD of this module and recurse forever.
        self.__dict__["_bound_dit"] = dit
        _STEREO_HOOKS[("nodemod", id(self))] = dit.node_mod.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, args, output: Tensor) -> Tensor:
        from crystal_nft.meanflow.stereo import get_active_stereo_tags

        tags = get_active_stereo_tags()
        if tags is None:
            return output
        b, n = output.shape[0], output.shape[1]
        tags = tags.to(output.device)
        if tags.ndim == 1:
            tags = tags.unsqueeze(0)
        if tags.shape[0] == 1 and b > 1:
            tags = tags.expand(b, -1)
        elif tags.shape[0] > b:
            tags = tags[:b]
        elif tags.shape[0] < b:
            return output
        if tags.shape[1] == n - 3:  # use_lattice_regs prepends 3 lattice rows
            tags = torch.nn.functional.pad(tags, (3, 0), value=0)
        elif tags.shape[1] > n:
            tags = tags[:, :n]
        elif tags.shape[1] < n:
            tags = torch.nn.functional.pad(tags, (0, n - tags.shape[1]), value=0)
        keep = (tags > 0).unsqueeze(-1).to(output.dtype)
        # Bounded FiLM. The unbounded form, output * (1 + w*g) + w*g, is fine
        # when a trainable backbone can absorb the modulation, but under
        # student_tune=stereo_only nothing can: at gain=20 the embedding reached
        # ~0.5 by step 80 and the ~11x node amplification blew the AnyFlow loss
        # from 0.23 to 9.88 (coord 0.07 -> 3.78). tanh caps the modulation at
        # +-gain and is still exactly 0 at zero-init, so step 0 remains
        # bit-identical to the base model.
        g = _stereo_gain(self.gain)
        sc = torch.tanh(self.scale(tags).to(output.dtype)) * keep * g
        sh = torch.tanh(self.shift(tags).to(output.dtype)) * keep * g
        w = _stereo_time_window(self, output)
        if w is not None:
            sc = sc * w
            sh = sh * w
        return output * (1.0 + sc) + sh


class StereoPairEmbedding(nn.Module):
    """Zero-init LoQI auxiliary-edge embedding added to Clari's ``embed_bonds``.

    Clari's ``bonds`` matrix already carries bond order *and* topological hop
    count, so overwriting entries to hold stereo edges would destroy real
    information.  This adds a separate zero-init embedding on top of the bond
    pair features instead, leaving ``bonds`` untouched.
    """

    def __init__(self, dim_pair: int, n_types: int = 4, gain: float = 1.0):
        super().__init__()
        self.emb = nn.Embedding(int(n_types), int(dim_pair))
        nn.init.zeros_(self.emb.weight)
        self.register_buffer("gain", torch.tensor(float(gain)))

    def bind(self, dit: nn.Module) -> None:
        handle = _STEREO_HOOKS.pop(id(self), None)
        if handle is not None:
            handle.remove()
        _STEREO_HOOKS[id(self)] = dit.embed_bonds.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, args, output: Tensor) -> Tensor:
        from crystal_nft.meanflow.stereo import get_active_stereo_pair_edges

        edges = get_active_stereo_pair_edges()
        if edges is None:
            return output
        b, n = output.shape[0], output.shape[1]
        edges = edges.to(output.device)
        if edges.shape[0] == 1 and b > 1:
            edges = edges.expand(b, -1, -1)
        elif edges.shape[0] > b:
            edges = edges[:b]
        elif edges.shape[0] != b:
            return output
        if edges.shape[1] == n - 3:  # `use_lattice_regs` prepends 3 lattice rows
            edges = torch.nn.functional.pad(edges, (3, 0, 3, 0), value=0)
        elif edges.shape[1] > n:
            edges = edges[:, :n, :n]
        elif edges.shape[1] < n:
            pad = n - edges.shape[1]
            edges = torch.nn.functional.pad(edges, (0, pad, 0, pad), value=0)
        keep = (edges > 0).unsqueeze(-1).to(output.dtype)
        g = _stereo_gain(self.gain)
        return output + (self.emb(edges).to(output.dtype) * keep * g)




class StereoPairMod(nn.Module):
    """LoQI directed-cycle edges modulating pair features *after* Clari's pair tanh.

    A per-atom tag saying "this centre is R" is meaningless on its own: R/S is
    defined **relative to the CIP priority ordering** of the substituents. That
    ordering is what LoQI's directed cycle encodes, and it lives in the pair
    features -- which reach the output at 0.00022 relative, the weakest of the
    three paths, for the same reason the token path does: ``embed_bonds`` is one
    branch inside ``tanh((0.5/K) sum(pair_k))``.

    So the model could be told a centre is R while having no usable signal for
    *which* arrangement is R -- exactly the observed "influence without
    direction" (flip_response 0.40, follow_rate 0.50). This puts the edge types
    where ``pair_mod`` already works: a zero-init FiLM on post-tanh pair features.
    """

    def __init__(self, dim_pair: int, n_edges: int = 4, gain: float = 1.0,
                 t_min: float = 0.0, t_max: float = 1.0):
        super().__init__()
        self.scale = nn.Embedding(int(n_edges), int(dim_pair))
        self.shift = nn.Embedding(int(n_edges), int(dim_pair))
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        self.register_buffer("gain", torch.tensor(float(gain)))
        self.t_min = float(t_min)
        self.t_max = float(t_max)

    def bind(self, dit: nn.Module) -> None:
        handle = _STEREO_HOOKS.pop(("pairmod", id(self)), None)
        if handle is not None:
            handle.remove()
        self.__dict__["_bound_dit"] = dit
        _STEREO_HOOKS[("pairmod", id(self))] = dit.pair_mod.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, args, output: Tensor) -> Tensor:
        from crystal_nft.meanflow.stereo import get_active_stereo_pair_edges

        e = get_active_stereo_pair_edges()
        if e is None:
            return output
        b, n = output.shape[0], output.shape[1]
        e = e.to(output.device)
        if e.ndim == 2:
            e = e.unsqueeze(0)
        if e.shape[0] == 1 and b > 1:
            e = e.expand(b, -1, -1)
        elif e.shape[0] > b:
            e = e[:b]
        elif e.shape[0] < b:
            return output
        if e.shape[1] == n - 3:  # lattice registers
            e = torch.nn.functional.pad(e, (3, 0, 3, 0), value=0)
        elif e.shape[1] > n:
            e = e[:, :n, :n]
        elif e.shape[1] < n:
            pad = n - e.shape[1]
            e = torch.nn.functional.pad(e, (0, pad, 0, pad), value=0)
        keep = (e > 0).unsqueeze(-1).to(output.dtype)
        # Bounded FiLM. The unbounded form, output * (1 + w*g) + w*g, is fine
        # when a trainable backbone can absorb the modulation, but under
        # student_tune=stereo_only nothing can: at gain=20 the embedding reached
        # ~0.5 by step 80 and the ~11x node amplification blew the AnyFlow loss
        # from 0.23 to 9.88 (coord 0.07 -> 3.78). tanh caps the modulation at
        # +-gain and is still exactly 0 at zero-init, so step 0 remains
        # bit-identical to the base model.
        g = _stereo_gain(self.gain)
        sc = torch.tanh(self.scale(e).to(output.dtype)) * keep * g
        w = _stereo_time_window(self, output)
        if w is not None:
            sc = sc * w
        sh = torch.tanh(self.shift(e).to(output.dtype)) * keep * g
        return output * (1.0 + sc) + sh


class StereoCondEmbedding(nn.Module):
    """Zero-init stereo -> ``cond`` bias, i.e. FiLM modulation of the whole DiT.

    Every previous attempt injected the CIP tag additively into ``embed_feats``,
    *before* Clari's ``tanh((0.5/K) * sum(h_k))``.  Measured, that lands at ~5% of
    one branch on 2 atoms in 68, and the tanh is already near saturation from
    ``embed_cart`` (RMS 8.8) -- so the signal barely propagates.

    ``cond`` is a different animal: it drives ``node_mod(h, cond)`` and
    ``pair_mod(pair, cond)`` *after* the tanh, multiplicatively, on every token.
    It is per-crystal rather than per-atom, so it can only express the cell's
    overall handedness -- fine for an enantiopure (Sohncke) cell, not for a
    racemate -- but it is a far stronger path.
    """

    def __init__(self, dim_cond: int, in_dim: int = 4, gain: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(dim_cond)),
            nn.SiLU(),
            nn.Linear(int(dim_cond), int(dim_cond)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("gain", torch.tensor(float(gain)))

    def forward(self, desc: Tensor) -> Tensor:
        return self.net(desc) * _stereo_gain(self.gain)


def wrap_dit_for_meanflow(
    dit: nn.Module,
    gate_value: float = 0.25,
    *,
    conditioning_mode: str = "mix",
    dual_time_feature_mode: str = "both",
    dual_gate_value: float = 1.0,
    time_parameterization: str = "legacy",
) -> MeanFlowDiTWrapper:
    if isinstance(dit, MeanFlowDiTWrapper):
        patch_dit_chiral_stem(dit.dit)
        return dit
    emb = getattr(dit, "embed_timestep", None)
    if isinstance(emb, MeanFlowTimeEmbedding):
        dit.embed_timestep = emb.base
    stem = getattr(dit, "stem_cond", None)
    if isinstance(stem, DualTimeStemAdapter):
        dit.stem_cond = stem.inner
    patch_dit_chiral_stem(dit)
    return MeanFlowDiTWrapper(
        dit,
        gate_value=gate_value,
        conditioning_mode=conditioning_mode,
        dual_time_feature_mode=dual_time_feature_mode,
        dual_gate_value=dual_gate_value,
        time_parameterization=time_parameterization,
    )


def patch_dit_chiral_stem(dit: nn.Module) -> None:
    if getattr(dit, "_mf_chiral_patched", False):
        return
    orig_stem = dit.stem_cond

    class _StemWithChiral(nn.Module):
        def __init__(self, inner, owner):
            super().__init__()
            self.inner = inner
            object.__setattr__(self, "_owner_dit", owner)

        def forward(self, x):
            bias = getattr(self._owner_dit, "_mf_chiral_bias", None)
            if bias is not None:
                if bias.ndim == 2 and x.ndim == 2:
                    x = x + bias
                elif bias.ndim == 2:
                    x = x + bias.unsqueeze(1)
            return self.inner(x)

    dit.stem_cond = _StemWithChiral(orig_stem, dit)
    dit._mf_chiral_patched = True


class StereoChiralBranch(nn.Module):
    """Parity-sensitive velocity residual at each stereocentre.

    Why this exists. Every other stereo module here (`StereoTokenEmbedding`,
    `StereoNodeMod`, `StereoPairMod`, `StereoCondEmbedding`) is a SCALAR feature
    modulation. A scalar bias can shift the conditional mean -- measured, the
    one-shot endpoint estimate reaches 0.90 tag-following -- but it has no
    reflection-odd mechanism, so it cannot systematically produce the coordinated
    displacement of four substituents that flips a signed volume. With
    `stereo_mirror_p=0.5` the data is symmetric under reflection+tag-swap, so the
    tag is the ONLY tie-breaker and a scalar bias is a very weak instrument for
    it. Measured consequence: tag control is ~0.90 on the one-shot endpoint and
    ~0.52 on anything actually sampled, at every NFE from 8 to 50.

    The fix is geometric. Under a reflection ``M`` (det = -1),
    ``(Mu) x (Mv) = -M(u x v)``: a cross product of relative vectors is a
    PSEUDOvector, so a velocity built from cross products carries the opposite
    parity to a true displacement and can act with a definite sign relative to
    the requested handedness.

    For each stereocentre ``[c, n1, n2, n3]`` take the edge vectors
    ``u_k = x_{n_k} - x_c`` and emit

        dv_r = a_r (u1 x u2) + b_r (u2 x u3) + g_r (u3 x u1),  r in {c,n1,n2,n3}

    with the nine coefficients per centre predicted from rotation-invariant
    inputs only: the normalised signed volume ``V``, the target sign ``s``,
    their product ``sV`` (negative exactly when the centre is currently wrong),
    the three edge lengths, and the time ``t``. Output layer is zero-init, so an
    untrained branch is exactly a no-op.

    Note the tetrahedral index depends on all four substituents -- moving only
    the centre cannot change it -- which is why the residual is applied to the
    neighbours too.
    """

    def __init__(self, hidden: int = 64, scale: float = 1.0, gate_margin: float = 0.0,
                 use_endpoint_geometry: bool = False,
                 bond_preserving: bool = False, t_max: float = 1.0,
                 t_min: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 12),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("gate_margin", torch.tensor(float(gate_margin)))
        self.use_endpoint_geometry = bool(use_endpoint_geometry)
        self.bond_preserving = bool(bond_preserving)
        self.t_max = float(t_max)
        self.t_min = float(t_min)

    def forward(self, pred: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        from crystal_nft.meanflow.stereo import get_active_stereo_geometry

        if os.environ.get("CRYSTAF_BRANCH_OFF", "0") == "1":
            # Diagnostic switch: run the trained model with the parity residual
            # disabled at inference. Separates "the branch's action costs PB"
            # from "training damaged the backbone" -- PB has been pinned at
            # 74-75 across three different geometric fixes to the residual, so
            # the two need to be told apart before another fix is attempted.
            return pred
        centers, label_sign, valid = get_active_stereo_geometry()
        if centers is None or label_sign is None or centers.numel() == 0:
            return pred
        coords = xt[:, 3:, :]  # (B, N, 3), lattice rows excluded
        b = coords.shape[0]
        m = int(centers.shape[1])
        if centers.shape[0] not in (1, b):
            return pred
        if centers.shape[0] == 1 and b > 1:
            centers = centers.expand(b, -1, -1)
        idx = centers.reshape(b, m * 4, 1).expand(b, m * 4, 3)
        pts = torch.gather(coords, 1, idx).reshape(b, m, 4, 3)
        u = pts[:, :, 1:, :] - pts[:, :, :1, :]  # (B, M, 3, 3)
        n = u.norm(dim=-1).clamp_min(1e-6)
        e = u / n.unsqueeze(-1)
        v_norm = (torch.linalg.cross(e[:, :, 0], e[:, :, 1]) * e[:, :, 2]).sum(-1)
        s = label_sign.to(pred.dtype)
        if s.shape[0] == 1 and b > 1:
            s = s.expand(b, -1)
        # Time arrives as (B,) from training but can be a scalar / size-1 tensor
        # from the sampling path; normalise instead of assuming (B,).
        tt = t.reshape(-1).to(pred.dtype)
        if tt.numel() == 1:
            tt = tt.expand(b)
        tt = tt[:b].reshape(b, 1).expand(b, m)
        feats = torch.stack(
            [v_norm.to(pred.dtype), s, (s * v_norm).to(pred.dtype),
             n[:, :, 0].to(pred.dtype), n[:, :, 1].to(pred.dtype),
             n[:, :, 2].to(pred.dtype), tt],
            dim=-1,
        )
        co = self.net(feats).reshape(b, m, 4, 3)  # (B, M, 4 atoms, 3 coeffs)
        c12 = torch.linalg.cross(u[:, :, 0], u[:, :, 1])
        c23 = torch.linalg.cross(u[:, :, 1], u[:, :, 2])
        c31 = torch.linalg.cross(u[:, :, 2], u[:, :, 0])
        basis = torch.stack([c12, c23, c31], dim=2)  # (B, M, 3, 3)
        dv = torch.einsum("bmac,bmcd->bmad", co, basis)  # (B, M, 4, 3)
        if bool(getattr(self, "bond_preserving", False)):
            # Make the residual a ROTATION of substituents, not a free
            # displacement. For neighbour k with bond vector u_k, only the
            # component perpendicular to u_k leaves |u_k| unchanged to first
            # order: u1 x u2 and u3 x u1 are both perpendicular to u1, but
            # u2 x u3 is NOT, so that term stretches the bond. And the residual
            # was also applied to the CENTRE, which perturbs all four bonds at
            # once. PB checks bond lengths and angles, so this is precisely the
            # damage profile: the branch's mere presence cost ~3.7 PB (82.26 ->
            # 78.55) even with a weak hinge and the correct-centre gate on.
            #
            # Project each neighbour's residual onto the plane perpendicular to
            # its own bond, and hold the centre fixed. Chirality can still be
            # flipped -- swapping two substituents is a motion along arcs, which
            # is exactly what perpendicular displacement generates.
            nrm = u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            e_n = u / nrm  # (B,M,3,3)
            d_n = dv[:, :, 1:, :]
            d_n = d_n - (d_n * e_n).sum(-1, keepdim=True) * e_n
            # Perpendicular displacement preserves |u| only to FIRST order: a
            # step d grows the bond by |d|^2/(2|u|), and over 50 integration
            # steps that accumulates -- measured, bond_lengths still failed
            # 47.78% vs 15.56% for the baseline even with the projection on.
            # Subtract the second-order radial term so the motion is a rotation
            # to second order.
            d_n = d_n - (d_n.pow(2).sum(-1, keepdim=True) / (2.0 * nrm)) * e_n
            dv = torch.cat([torch.zeros_like(dv[:, :, :1, :]), d_n], dim=2)
        w = valid.to(pred.dtype) if valid is not None else torch.ones_like(s)
        if w.shape[0] == 1 and b > 1:
            w = w.expand(b, -1)
        if float(self.gate_margin) > 0.0:
            # Act ONLY where the handedness is wrong or near-planar. Without
            # this the residual displaces the atoms of every stereocentre,
            # including the ones already correct -- pure structural damage for
            # no chirality gain. relu(m - s*V) is 0 once a centre is correctly
            # handed and clearly non-planar, so a converged centre is left alone.
            w = w * torch.relu(
                float(self.gate_margin) - (s * v_norm.to(pred.dtype))
            ).clamp(max=1.0)
        w = (w * (label_sign != 0).to(pred.dtype)).unsqueeze(-1).unsqueeze(-1)
        t_min = float(getattr(self, "t_min", 0.0))
        if t_min > 0.0:
            # Narrow the window in which the residual fires. Its PB cost tracks
            # the total displacement it applies, and at t_max=0.85 it acts on 42
            # of 50 sampling steps, so the displacements accumulate. Handedness
            # is decided in t in [0.5, 0.9] (trajectory trace), so firing for ~5
            # steps there should set it while leaving ~45 steps of ordinary
            # dynamics to relax the geometry.
            dv = dv * ((tt >= t_min).float()).unsqueeze(-1).unsqueeze(-1)
        t_max = float(getattr(self, "t_max", 1.0))
        if t_max < 1.0:
            # Switch the residual OFF before the endpoint. Handedness is decided
            # in t in [0.5, 0.9] (trajectory trace); the last steps are geometry
            # relaxation, and that is where PB is determined. A residual still
            # firing at t ~ 1 perturbs the finished structure -- measured, all
            # three intramolecular checks roughly tripled vs baseline:
            # bond_angles 14.4 -> 50.0, bond_lengths 15.6 -> 47.8,
            # internal_steric_clash 17.8 -> 43.3.
            ramp = ((t_max - tt) / 0.10).clamp(0.0, 1.0)
            dv = dv * ramp.unsqueeze(-1).unsqueeze(-1)
        dv = dv * w * float(self.scale)
        # Stash the mean squared displacement so the loss can penalise it.
        # Nothing else constrains how FAR the branch moves atoms -- only whether
        # the chirality hinge is satisfied -- so it distorts bond lengths and
        # angles, which is what PB measures. Measured: clash recovered to 13.99
        # once off-manifold rows left the MSE, but PB stayed at 69.0 vs the
        # 82.26 baseline, and the branch is the common factor across v17/v18.
        self._last_sqnorm = (dv.float() ** 2).sum(-1).mean()
        out = pred.clone()
        tgt = out[:, 3:, :]
        tgt.scatter_add_(
            1, centers.reshape(b, m * 4, 1).expand(b, m * 4, 3),
            dv.reshape(b, m * 4, 3).to(tgt.dtype),
        )
        out = torch.cat([out[:, :3, :], tgt], dim=1)
        return out
