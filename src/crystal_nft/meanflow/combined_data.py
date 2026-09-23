"""Mixed Clari CSD + MolCrystalFlow training datasets."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset

from clari.chem import Crystal
from clari.datamodules.csd import CrystalDataset


class MCFCrystalDataset(Dataset):
    """Flat or family-grouped list of Crystal samples from MCF cache."""

    def __init__(
        self,
        crystals: Sequence[Crystal],
        *,
        group_by_fam: bool = True,
        random_repr: bool = True,
        augment: bool = True,
    ):
        self.crystals = list(crystals)
        self.random_repr = random_repr
        self.augment = augment
        if group_by_fam:
            fam_groups: dict[str, list[int]] = defaultdict(list)
            for i, c in enumerate(self.crystals):
                cid = c.csd_id if isinstance(c.csd_id, str) else str(c.csd_id)
                fam_groups[cid].append(i)
            self.classes = [sorted(v) for _, v in sorted(fam_groups.items())]
        else:
            self.classes = [[i] for i in range(len(self.crystals))]

    def __len__(self) -> int:
        return len(self.classes)

    def __getitem__(self, idx: int) -> Crystal:
        if self.random_repr:
            i = random.choice(self.classes[idx])
        else:
            i = self.classes[idx][0]
        c = self._normalize_bodies(self.crystals[i])
        if self.augment:
            return c.augment()
        return c.wrapped(mode="com")

    @staticmethod
    def _normalize_bodies(c: Crystal) -> Crystal:
        _, inv = torch.unique(c.body_ids, return_inverse=True)
        if torch.equal(c.body_ids, inv.long()):
            return c
        return c.replace(body_ids=inv.long())


class MixedCrystalDataset(Dataset):
    """Sample Clari or MCF each step with configurable probability."""

    def __init__(self, clari_ds: Dataset, mcf_ds: Dataset | None, *, mcf_prob: float = 0.25):
        self.clari_ds = clari_ds
        self.mcf_ds = mcf_ds
        self.mcf_prob = float(mcf_prob) if mcf_ds is not None else 0.0
        self._len = len(clari_ds)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Crystal:
        if self.mcf_ds is not None and random.random() < self.mcf_prob:
            j = random.randrange(len(self.mcf_ds))
            return self.mcf_ds[j]
        return self.clari_ds[idx % len(self.clari_ds)]


def build_train_dataset(
    clari_data_dir,
    *,
    split_opts,
    mcf_cache: str | None = None,
    mcf_prob: float = 0.25,
    mixed: bool = True,
) -> Dataset:
    import json
    import os

    from clari.datamodules.csd import DEFAULT_SPLIT_OPTS
    import torch_geometric as pyg

    os.environ["CLARI_DATA_DIR"] = str(clari_data_dir)
    root = clari_data_dir / "csd"
    with open(root / "config.json") as f:
        json.load(f)
    split_opts = split_opts or DEFAULT_SPLIT_OPTS
    train_opts = dict(split_opts.get("train", DEFAULT_SPLIT_OPTS["train"]))
    pyg_path = root / "train.pt"
    crystals, slices, _ = pyg.io.fs.torch_load(pyg_path)
    crystals = pyg.data.Data.from_dict(crystals)
    clari_ds = CrystalDataset(crystals, slices, **train_opts)

    mcf_ds = None
    if mcf_cache and os.path.isfile(mcf_cache):
        mcf_list = torch.load(mcf_cache, weights_only=False)
        mcf_ds = MCFCrystalDataset(mcf_list, group_by_fam=True, random_repr=True, augment=True)

    if mcf_ds is None:
        return clari_ds
    if mixed:
        return MixedCrystalDataset(clari_ds, mcf_ds, mcf_prob=mcf_prob)
    return ConcatDataset([clari_ds, mcf_ds])
