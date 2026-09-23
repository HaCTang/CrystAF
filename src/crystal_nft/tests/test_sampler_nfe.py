"""The sampling budget of each sampler, in network evaluations.

Clari's Heun sampler is second order: it calls ``interface.pred`` twice per
step except on the last, so ``num_steps=T`` costs ``2T - 1`` evaluations, not
``T``. Every "Heun 50" baseline in this repo is therefore a 99-evaluation
sample, and comparing it to the flow map's ``NFE=32`` as "50 vs 32" understates
the gap by a factor of two. This got into a paper figure once; the test is here
so it cannot again.
"""

import torch

from clari.pipelines.base.interfaces import Interface
import clari.pipelines.base.samplers as samplers_mod
from clari.pipelines.base.samplers import EulerSampler, HeunSampler


class _CountingInterface(Interface):
    """Counts network evaluations; the values it returns do not matter."""

    def __init__(self) -> None:
        self.calls = 0

    def pred(self, *, net, xt, xsc, t, f, **kwargs):
        self.calls += 1
        return torch.zeros_like(xt)

    def score(self, *, xt, t, pred, scale):
        return torch.zeros_like(xt)

    def estimate_x1(self, *, xt, t, pred):
        return torch.zeros_like(xt)


class _Batch:
    """The two attributes ``Sampler.sample`` touches on a Crystal."""

    def __init__(self, x: torch.Tensor) -> None:
        self.x = x
        self.mask = torch.ones(x.shape[:-1])

    @property
    def device(self):
        return self.x.device

    def replace(self, **kwargs):
        out = _Batch(kwargs.get("x", self.x))
        out.mask = self.mask
        return out


def _count(sampler, steps: int, monkeypatch) -> int:
    monkeypatch.setattr(samplers_mod, "zero_com_suffix", lambda x, w=None: x)
    interface = _CountingInterface()
    sampler(num_steps=steps, stochasticity="none").sample(
        interface, None, _Batch(torch.randn(2, 6, 3)), sample_prior=False
    )
    return interface.calls


def test_heun_costs_two_evaluations_per_step_except_the_last(monkeypatch):
    for steps in (2, 8, 16, 32, 50):
        assert _count(HeunSampler, steps, monkeypatch) == 2 * steps - 1


def test_the_reported_baseline_budgets(monkeypatch):
    # The numbers the paper quotes for the Clari baselines and the teacher.
    assert _count(HeunSampler, 50, monkeypatch) == 99
    assert _count(HeunSampler, 16, monkeypatch) == 31


def test_euler_costs_one_evaluation_per_step(monkeypatch):
    # The contrast that makes the flow map's NFE column mean what it says.
    for steps in (8, 16, 32):
        assert _count(EulerSampler, steps, monkeypatch) == steps


def test_interface_does_not_shortcut_the_last_step(monkeypatch):
    # ``Sampler.step`` takes a one-evaluation ``get_final`` path when the
    # interface overrides it. Clari's SiTInterface does not, so 2T-1 holds.
    from clari.pipelines.base.interfaces import SiTInterface

    assert SiTInterface.get_final is Interface.get_final
