#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FF2 图像基线 —— 阶段 1:冻结特征提取(单视图)

第一版只做 skull_ventral(腹面)+ DINOv2-L/14,把管线跑通。

用法:
    python ff2_features.py --backbone dinov2l --img-size 518
    python ff2_features.py --backbone dinov2l --img-size 518 --pool both --out feats_dinov2l_518.npz

产出:
    <out>.npz  含 cls / patchmean / split / species_idx / genus_idx / specimen / is_damaged
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = os.environ.get("FF2_ROOT", "")
DATA = os.path.join(ROOT, "datasets", "yanshu_600_4cranium_1teeth_2measurement")
FEATDIR = os.path.join(ROOT, "features")

BACKBONES = {
    "dinov3h": ("vit_huge_plus_patch16_dinov3.lvd1689m", 512),
    # name            timm id / hf-hub id                            img default
    "dinov2l":      ("vit_large_patch14_dinov2.lvd142m",              518),
    "dinov2b":      ("vit_base_patch14_dinov2.lvd142m",               518),
    "dinov3l":      ("vit_large_patch16_dinov3.lvd1689m",             518),
    "bioclip2":     ("hf-hub:imageomics/bioclip-2",                   224),
    "siglip2l":     ("vit_base_patch16_siglip_384.v2_webli",          384),
    "in21k_vitl":   ("vit_large_patch16_224.augreg_in21k_ft_in1k",    518),
    "convnextl":    ("convnext_large.fb_in22k_ft_in1k",               384),
    "efficientnetb7": ("tf_efficientnet_b7.ap_in1k",                  600),
}


def load_index(view="skull_ventral"):
    """读 metadata.csv,只保留指定视角。"""
    csvp = os.path.join(DATA, "metadata.csv")
    rows = []
    with open(csvp, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["image_type"] != view:
                continue
            rel = r["image_path"].replace("/", os.sep)
            fname = r["image_name"]
            base = os.path.splitext(fname)[0]
            is_damaged = 1 if base.split("#")[-1].lower() == "p" else 0
            # image_path 形如 cranium/train/Genus/...  第 2 段才是 split
            seg = r["image_path"].replace("\\", "/").split("/")
            split = seg[1] if len(seg) > 1 and seg[1] in ("train", "test") else ""
            rows.append({
                "split": split,
                "genus": r["genus"],
                "species": r["species"],
                "specimen": r["specimen"],
                "path": os.path.join(DATA, rel),
                "is_damaged": is_damaged,
            })
    return rows


class ImgDS(Dataset):
    def __init__(self, rows, size, mean, std):
        self.rows = rows
        self.size = size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p = self.rows[i]["path"]
        im = Image.open(p).convert("RGB")
        w, h = im.size
        # 原图已是方形(600x600),直接等比缩放到 size
        s = self.size / min(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        im = im.resize((nw, nh), Image.BICUBIC)
        # center crop 到 size x size
        if nw != self.size or nh != self.size:
            left = (nw - self.size) // 2
            top = (nh - self.size) // 2
            im = im.crop((left, top, left + self.size, top + self.size))
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
        x = (x - self.mean) / self.std
        return x, i


class _OCAdapter(torch.nn.Module):
    """把 open_clip 模型包成有 forward_features 的接口。

    open_clip 只给 pool 后的向量,所以返回 (B, 1, D);
    下游 cls = f[:,0],patchmean 退化为同一向量。
    """

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward_features(self, x):
        # 不加 no_grad:推理时外层已有 no_grad 上下文,训练基准需要梯度
        return self.m.encode_image(x).unsqueeze(1)


def _build_openclip(hub_id, default_size):
    import open_clip
    import torchvision.transforms as T

    model, _, preprocess = open_clip.create_model_and_transforms(hub_id)
    model.eval()

    mean = std = None
    for t in getattr(preprocess, "transforms", []):
        if isinstance(t, T.Normalize):
            mean = list(t.mean)
            std = list(t.std)
    if mean is None:
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

    size = getattr(model.visual, "image_size", None) or default_size
    if isinstance(size, (tuple, list)):
        size = size[0]
    return _OCAdapter(model), mean, std, int(size)


def pool_features(f):
    """把各种 forward_features 输出统一成 (cls, patchmean)。

    ViT      : (B, 1+N, D)  -> cls = token0, patchmean = 其余均值
    CNN NCHW : (B, C, H, W) -> 两者都取空间均值
    CNN NHWC : (B, H, W, C) -> 同上
    """
    if f.dim() == 3:
        cls = f[:, 0]
        pm = f[:, 1:].mean(dim=1) if f.shape[1] > 1 else f[:, 0]
        return cls, pm
    if f.dim() == 4:
        # 判断通道维在前还是在后
        if f.shape[1] > f.shape[-1]:
            v = f.mean(dim=(2, 3))      # NCHW
        else:
            v = f.mean(dim=(1, 2))      # NHWC
        return v, v
    raise RuntimeError(f"unexpected feature shape {tuple(f.shape)}")


def snap_size(model, size):
    """把输入尺寸吸到 patch size 的整数倍(ViT 的硬性要求)。"""
    pe = getattr(model, "patch_embed", None)
    ps = getattr(pe, "patch_size", None)
    if ps is None:
        return size
    ps = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
    if size % ps != 0:
        new = (size // ps) * ps
        print(f"  img_size {size} 不是 patch({ps}) 的整数倍 → 调整为 {new}")
        return new
    return size


def build_model(name):
    import timm
    if name not in BACKBONES:
        raise SystemExit(f"unknown backbone {name}. choices: {list(BACKBONES)}")
    timm_id, default_size = BACKBONES[name]

    # open_clip 格式(仓库里没有 config.json,不能走 timm)
    if timm_id.startswith("hf-hub:") and name in ("bioclip2", "bioclip1"):
        return _build_openclip(timm_id, default_size)

    # dynamic_img_size 只对 ViT 有效;CNN(ConvNeXt/EffNet)不接受这个参数
    try:
        model = timm.create_model(timm_id, pretrained=True, num_classes=0,
                                  dynamic_img_size=True)
    except TypeError:
        model = timm.create_model(timm_id, pretrained=True, num_classes=0)
    model.eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    mean, std = cfg["mean"], cfg["std"]
    return model, mean, std, default_size


@torch.no_grad()
def extract(model, rows, size, mean, std, device, bs=16, workers=8):
    ds = ImgDS(rows, size, mean, std)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                    pin_memory=True, persistent_workers=(workers > 0))
    n = len(rows)
    cls_out = None
    pm_out = None
    t0 = time.time()
    for bi, (x, idx) in enumerate(dl):
        x = x.to(device, non_blocking=True)
        cls, pm = pool_features(model.forward_features(x))
        if cls_out is None:
            D = cls.shape[1]
            cls_out = np.zeros((n, D), dtype=np.float32)
            pm_out = np.zeros((n, D), dtype=np.float32)
        cls_out[idx.numpy()] = cls.float().cpu().numpy()
        pm_out[idx.numpy()] = pm.float().cpu().numpy()
        if bi % 10 == 0:
            done = min((bi + 1) * bs, n)
            print(f"  [{done}/{n}] {time.time()-t0:.1f}s", flush=True)
    print(f"  done in {time.time()-t0:.1f}s")
    return cls_out, pm_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2l")
    ap.add_argument("--img-size", type=int, default=0, help="0 = 用骨干默认")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--view", default="skull_ventral")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(FEATDIR, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(device)}")

    rows = load_index(args.view)
    print(f"images ({args.view}): {len(rows)}")
    from collections import Counter
    if any(r["split"] not in ("train", "test") for r in rows):
        bad = Counter(r["split"] for r in rows)
        raise SystemExit(f"split 解析失败,拿到的是 {dict(bad)},检查 metadata.csv 的 image_path")
    print("  split:", dict(Counter(r["split"] for r in rows)))
    print("  specimens:", len({r["specimen"] for r in rows}))
    for sp in ("train", "test"):
        sub = [r for r in rows if r["split"] == sp]
        print(f"    {sp}: {len(sub)} imgs, "
              f"{len({(r['genus'], r['species']) for r in sub})} species, "
              f"{len({r['genus'] for r in sub})} genera, "
              f"damaged={sum(r['is_damaged'] for r in sub)}")

    model, mean, std, default_size = build_model(args.backbone)
    size = args.img_size or default_size
    size = snap_size(model, size)
    model = model.to(device)
    print(f"backbone: {BACKBONES[args.backbone][0]}  img_size={size}")

    cls_f, pm_f = extract(model, rows, size, mean, std, device,
                          bs=args.batch_size, workers=args.workers)

    # 标签编码
    species = [f'{r["genus"]} {r["species"]}' for r in rows]
    genus = [r["genus"] for r in rows]
    sp_uniq = sorted(set(species))
    g_uniq = sorted(set(genus))
    sp_idx = np.array([sp_uniq.index(s) for s in species])
    g_idx = np.array([g_uniq.index(g) for g in genus])

    out = args.out or f"feats_{args.backbone}_{size}.npz"
    if not os.path.isabs(out):
        out = os.path.join(FEATDIR, out)
    np.savez_compressed(
        out,
        cls=cls_f, patchmean=pm_f,
        split=np.array([r["split"] for r in rows]),
        specimen=np.array([r["specimen"] for r in rows]),
        species=np.array(species), genus=np.array(genus),
        is_damaged=np.array([r["is_damaged"] for r in rows]),
        sp_idx=sp_idx, g_idx=g_idx,
        sp_uniq=np.array(sp_uniq), g_uniq=np.array(g_uniq),
        backbone=args.backbone, img_size=size,
    )
    print(f"\nwrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    print(f"  cls: {cls_f.shape}   patchmean: {pm_f.shape}")
    print(f"  species: {len(sp_uniq)}   genera: {len(g_uniq)}")


if __name__ == "__main__":
    main()
