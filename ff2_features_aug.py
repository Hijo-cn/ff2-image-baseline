#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FF2 阶段 1b:带增强的冻结特征提取

核心思路:
    冻结特征范式下,"数据增强"应该在【特征提取阶段】做,而不是训练阶段。
    对每张训练图生成 N 个增强版本,各提一次特征 → 训练样本从 593 变 593*(1+N)。

    ⚠️ 只对 train 增强;test 只保留原图,防止泄漏。

用法:
    TMPDIR=/tmp python ff2_features_aug.py --backbone dinov2l --img-size 518 --aug-train 10

产出:
    features/feats_aug_<backbone>_<size>_x<N>.npz
    额外字段: base_idx(原图行号) / aug_id(0=原图, 1..N=增强)

依赖: 复用 ff2_features.py 的 load_index / build_model
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ff2_features import (load_index, build_model, snap_size, pool_features,
                          BACKBONES, FEATDIR, DATA)  # noqa: E402


def make_aug(size, strength="current"):
    """增强强度。

    weak    : 仅轻度几何(±5°、3% 平移、±4% 缩放),不加任何光度变化
    current : 几何 + 强光度(当前主用配方)
    """
    if strength == "weak":
        return T.Compose([
            T.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.96, 1.04),
                           interpolation=T.InterpolationMode.BICUBIC, fill=0),
        ])
    return T.Compose([
        T.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.92, 1.08),
                       interpolation=T.InterpolationMode.BICUBIC, fill=0),
        T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.20, hue=0.04),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.25),
        T.RandomGrayscale(p=0.10),
    ])


class AugDS(Dataset):
    """entries: list of (row_idx, aug_id)"""

    def __init__(self, rows, entries, size, mean, std, strength="current"):
        self.rows = rows
        self.entries = entries
        self.size = size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.aug = make_aug(size, strength)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        row_idx, aug_id = self.entries[i]
        p = self.rows[row_idx]["path"]
        im = Image.open(p).convert("RGB")

        # 先等比缩放 + 中心裁到 size(和原始管线一致)
        w, h = im.size
        s = self.size / min(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        im = im.resize((nw, nh), Image.BICUBIC)
        if nw != self.size or nh != self.size:
            left, top = (nw - self.size) // 2, (nh - self.size) // 2
            im = im.crop((left, top, left + self.size, top + self.size))

        # 增强(固定种子,可复现)
        if aug_id > 0:
            st = np.random.get_state()
            np.random.seed((row_idx * 1000 + aug_id) % (2**31 - 1))
            torch.manual_seed((row_idx * 1000 + aug_id) % (2**31 - 1))
            try:
                im = self.aug(im)
            finally:
                np.random.set_state(st)

        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
        x = (x - self.mean) / self.std
        return x, row_idx, aug_id


@torch.no_grad()
def run(model, rows, entries, size, mean, std, device, bs=16, workers=6, strength="current"):
    ds = AugDS(rows, entries, size, mean, std, strength)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                    pin_memory=True)
    n = len(entries)
    cls_out = pm_out = None
    b_idx = np.zeros(n, dtype=np.int32)
    a_idx = np.zeros(n, dtype=np.int32)
    t0 = time.time()
    for bi, (x, ri, ai) in enumerate(dl):
        x = x.to(device, non_blocking=True)
        cls, pm = pool_features(model.forward_features(x))
        if cls_out is None:
            D = cls.shape[1]
            cls_out = np.zeros((n, D), dtype=np.float32)
            pm_out = np.zeros((n, D), dtype=np.float32)
        sl = slice(bi * bs, min((bi + 1) * bs, n))
        cls_out[sl] = cls.float().cpu().numpy()
        pm_out[sl] = pm.float().cpu().numpy()
        b_idx[sl] = ri.numpy()
        a_idx[sl] = ai.numpy()
        if bi % 20 == 0:
            print(f"  [{min((bi+1)*bs, n)}/{n}] {time.time()-t0:.0f}s", flush=True)
    print(f"  done in {time.time()-t0:.0f}s")
    return cls_out, pm_out, b_idx, a_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="skull_ventral",
                    help="metadata.csv 里的 image_type,例如 skull_ventral / skull_dorsal / skull_lateral / mandible_lateral / teeth")
    ap.add_argument("--backbone", default="dinov2l")
    ap.add_argument("--img-size", type=int, default=0)
    ap.add_argument("--aug-train", type=int, default=10, help="每个样本的增强份数")
    ap.add_argument("--aug-all", action="store_true",
                    help="对所有样本增强(用于 5 折 CV,任何样本都可能进训练折)")
    ap.add_argument("--aug-strength", default="current", choices=["weak", "current"],
                    help="weak=仅轻度几何;current=几何+强光度(主用)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.makedirs(FEATDIR, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    rows = load_index(args.view)
    entries = []
    for i, r in enumerate(rows):
        entries.append((i, 0))                      # 原图(全部行)
        if args.aug_all or r["split"] == "train":   # 默认只增强 train
            for a in range(1, args.aug_train + 1):
                entries.append((i, a))
    print(f"base images: {len(rows)}   feature entries: {len(entries)}")

    model, mean, std, default_size = build_model(args.backbone)
    size = args.img_size or default_size
    size = snap_size(model, size)
    model = model.to(device)
    print(f"backbone: {BACKBONES[args.backbone][0]}  img_size={size}  "
          f"aug_strength={args.aug_strength}")

    cls_f, pm_f, b_idx, a_idx = run(model, rows, entries, size, mean, std, device,
                                    bs=args.batch_size, workers=args.workers,
                                    strength=args.aug_strength)

    species = [f'{r["genus"]} {r["species"]}' for r in rows]
    genus = [r["genus"] for r in rows]
    sp_uniq = sorted(set(species))
    g_uniq = sorted(set(genus))
    sfx = "" if args.aug_strength == "current" else f"_{args.aug_strength}"
    vt = "" if args.view == "skull_ventral" else "_" + args.view.replace("skull_", "").replace("mandible_lateral", "mand")
    out = args.out or f"feats_aug_{args.backbone}_{size}_x{args.aug_train}" \
                      f"{'_all' if args.aug_all else ''}{sfx}{vt}.npz"
    if not os.path.isabs(out):
        out = os.path.join(FEATDIR, out)

    np.savez_compressed(
        out,
        cls=cls_f, patchmean=pm_f, base_idx=b_idx, aug_id=a_idx,
        split=np.array([rows[i]["split"] for i in b_idx]),
        specimen=np.array([rows[i]["specimen"] for i in b_idx]),
        species=np.array([species[i] for i in b_idx]),
        genus=np.array([genus[i] for i in b_idx]),
        is_damaged=np.array([rows[i]["is_damaged"] for i in b_idx]),
        sp_idx=np.array([sp_uniq.index(species[i]) for i in b_idx]),
        g_idx=np.array([g_uniq.index(genus[i]) for i in b_idx]),
        sp_uniq=np.array(sp_uniq), g_uniq=np.array(g_uniq),
        backbone=args.backbone, img_size=size, aug_train=args.aug_train,
    )
    n_tr = int((np.array([rows[i]["split"] for i in b_idx]) == "train").sum())
    print(f"\nwrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    print(f"  total {len(b_idx)}  train {n_tr}  test {len(b_idx)-n_tr}")
    print(f"  cls {cls_f.shape}")


if __name__ == "__main__":
    main()
