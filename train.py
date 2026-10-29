#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FF2 基线 —— 微调训练脚本(v1:单骨干、单折 pilot)

设计要点
--------
1. **划分只读 splits.json** —— 不在运行时随机生成,保证全组一致
2. **标签只从 splits.json 的 class_names 取** —— 不依赖文件夹名排序
3. **只训腹面单视图** —— 多视图属于另一条线,基线阶段锁死
4. **输出直接对接 evaluation.py** —— 吐 preds/*.npz,不自己算指标
5. **保留每折权重** —— 事后换评测口径/TTA 只需重跑推理,不用重训

用法
----
    # pilot:单折、默认配置
    python -u train.py --backbone dinov3l --fold 0 \
        --splits splits/splits_v1.json \
        --preds preds --weights weights

    # 冒烟测试(只跑 3 个 epoch、50 张图,确认管线通)
    python -u train.py --backbone dinov3l --fold 0 --smoke

    # 全 5 折
    for k in 0 1 2 3 4; do
      python -u train.py --backbone dinov3l --fold $k \
        --splits splits/splits_v1.json --preds preds --weights weights
    done
"""
import argparse
import csv
import json
import os
import sys
import time
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from ff2_features import BACKBONES
except Exception:
    BACKBONES = {
        "dinov3l": ("vit_large_patch16_dinov3.lvd1689m", 512),
        "dinov2l": ("vit_large_patch14_dinov2.lvd142m", 518),
        "bioclip2": ("hf-hub:imageomics/bioclip-2", 224),
        "efficientnetb7": ("tf_efficientnet_b7.ap_in1k", 600),
        "convnextl": ("convnext_large.fb_in22k_ft_in1k", 384),
    }

ROOT = os.environ.get("FF2_ROOT", "")
DATA = os.path.join(ROOT, "datasets", "yanshu_600_4cranium_1teeth_2measurement")


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ------------------------------------------------------------------ 数据

def load_multi_index(views):
    """返回 {specimen: {view: path}}, {specimen: meta}。支持多视角。"""
    wanted = {VIEW_TYPES[v]: v for v in views if v in VIEW_TYPES}
    idx, meta = {}, {}
    csvp = os.path.join(DATA, "metadata.csv")
    for r in csv.DictReader(open(csvp, encoding="utf-8-sig", newline="")):
        v = wanted.get(r["image_type"])
        if v is None:
            continue
        sid = r["specimen"]
        if sid not in idx:
            idx[sid] = {}
            meta[sid] = {"genus": r["genus"],
                         "species": f'{r["genus"]} {r["species"]}',
                         "is_damaged": 0}
        idx[sid][v] = os.path.join(DATA, r["image_path"].replace("/", os.sep))
        fn = os.path.splitext(r["image_name"])[0]
        if fn.split("#")[-1].lower() == "p":
            meta[sid]["is_damaged"] = 1
    return idx, meta


VIEW_TYPES = {
    "ventral": "skull_ventral",
    "dorsal": "skull_dorsal",
    "lateral": "skull_lateral",
    "mand": "mandible_lateral",
    "teeth": "teeth_skull_ventral",
}


def load_specimen_index(view="skull_ventral"):
    """specimen -> {path, genus, species, is_damaged}(单视角,兼容旧代码)"""
    csvp = os.path.join(DATA, "metadata.csv")
    idx = {}
    for r in csv.DictReader(open(csvp, encoding="utf-8-sig", newline="")):
        if r["image_type"] != view:
            continue
        sid = r["specimen"]
        fn = os.path.splitext(r["image_name"])[0]
        dmg = 1 if fn.split("#")[-1].lower() == "p" else 0
        if sid in idx:
            idx[sid]["is_damaged"] = max(idx[sid]["is_damaged"], dmg)
            continue
        idx[sid] = {
            "path": os.path.join(DATA, r["image_path"].replace("/", os.sep)),
            "genus": r["genus"],
            "species": f'{r["genus"]} {r["species"]}',
            "is_damaged": dmg,
        }
    return idx


def make_aug(strength, size):
    if strength == "none":
        return None
    if strength == "weak":
        return T.Compose([
            T.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.96, 1.04),
                           interpolation=T.InterpolationMode.BICUBIC, fill=0),
        ])
    if strength == "current":
        return T.Compose([
            T.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.92, 1.08),
                           interpolation=T.InterpolationMode.BICUBIC, fill=0),
            T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.20, hue=0.04),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.25),
            T.RandomGrayscale(p=0.10),
        ])
    return None


class VentralDS(Dataset):
    def __init__(self, items, size, mean, std, train=False, aug="weak"):
        self.items = items                     # list of (path, label)
        self.size = size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.train = train
        self.aug = make_aug(aug, size) if train else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        im = Image.open(path).convert("RGB")
        w, h = im.size
        s = self.size / min(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        im = im.resize((nw, nh), Image.BICUBIC)
        if nw != self.size or nh != self.size:
            left, top = (nw - self.size) // 2, (nh - self.size) // 2
            im = im.crop((left, top, left + self.size, top + self.size))
        if self.aug is not None:
            im = self.aug(im)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
        x = (x - self.mean) / self.std
        return x, label, i


# ------------------------------------------------------------------ 模型

def build(timm_id, n_classes, size):
    import timm
    try:
        model = timm.create_model(timm_id, pretrained=True, num_classes=n_classes,
                                  dynamic_img_size=True)
    except TypeError:
        model = timm.create_model(timm_id, pretrained=True, num_classes=n_classes)
    cfg = timm.data.resolve_data_config({}, model=model)
    return model, cfg["mean"], cfg["std"]


class LoRALinear(nn.Module):
    """把 nn.Linear 包成 LoRA:原权重冻结,只学低秩旁路。"""

    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.r = rank
        self.scaling = alpha / rank
        # 关键:跟随原权重的 device/dtype(模型可能已经 .to(cuda))
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features,
                                               device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank,
                                               device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)          # 初始时旁路为 0,不改变原模型

    def forward(self, x):
        out = self.base(x)
        lora = (x @ self.lora_A.to(x.dtype).t()) @ self.lora_B.to(x.dtype).t()
        return out + lora * self.scaling


def apply_lora(model, rank=8, alpha=16, targets=("qkv", "proj", "fc1", "fc2")):
    n = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name.split(".")[-1] not in targets:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(module, rank, alpha))
        n += 1
    return n


class DualHeadNet(nn.Module):
    """在原模型上加一个 2048 维头:CLS + patch 均值拼接(和冻结探针一致)"""

    def __init__(self, base, n_classes):
        super().__init__()
        self.base = base
        self.num_prefix = int(getattr(base, "num_prefix_tokens", 1) or 1)
        D = int(base.num_features)
        base.head = nn.Identity()          # 原头不用了
        self._tok = None
        base.norm.register_forward_hook(self._hook)
        self.cls_head = nn.Linear(2 * D, n_classes)

    def _hook(self, m, inp, out):
        self._tok = out

    def forward(self, x):
        _ = self.base(x)                   # 触发 hook
        tok = self._tok
        if tok is None or tok.dim() != 3:
            raise RuntimeError("没拿到 token 序列,检查 hook 位置")
        cls = tok[:, 0]
        patch = tok[:, self.num_prefix:].mean(dim=1)
        return self.cls_head(torch.cat([cls, patch], dim=1))


def backbone_of(model):
    return getattr(model, "base", model)


def unfreeze_head(model):
    """把分类头(含 DualHead 的 cls_head)设为可训练。"""
    names = ("head", "fc", "classifier", "cls_head")
    for holder in (model, backbone_of(model)):
        for nm in names:
            sub = getattr(holder, nm, None)
            if sub is not None and hasattr(sub, "parameters"):
                for p in sub.parameters():
                    p.requires_grad = True


def build_llrd_groups(net, base_lr, head_lr, wd, llrd):
    """按层深衰减学习率的参数分组;llrd=1.0 时退化为两层(主干/头)。"""
    import re as _re
    base = backbone_of(net)
    blocks = getattr(base, "blocks", [])
    n = len(blocks)
    buckets = {}
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if "cls_head" in name or name.startswith("head"):
            lr = head_lr
        else:
            m = _re.search(r"\.blocks\.(\d+)\.", name)
            if m:
                lr = base_lr * (llrd ** (n - 1 - int(m.group(1))))
            else:
                lr = base_lr * (llrd ** (n + 1))
        buckets.setdefault(round(lr, 12), []).append(p)
    groups = []
    for lr, ps in sorted(buckets.items(), reverse=True):
        d = [p for p in ps if p.ndim > 1]
        nd = [p for p in ps if p.ndim <= 1]
        if d:
            groups.append({"params": d, "lr": lr, "weight_decay": wd})
        if nd:
            groups.append({"params": nd, "lr": lr, "weight_decay": 0.0})
    return groups


def param_groups(model, lr_backbone, lr_head, wd):
    head_prefixes = ("head", "classifier", "fc", "cls_head")
    bb, hd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (hd if any(n.startswith(h) or f".{h}" in n for h in head_prefixes) else bb).append((n, p))
    no_decay = ("bias", "norm", "ln_", "bn")
    groups = []
    for named, lr in ((bb, lr_backbone), (hd, lr_head)):
        d, nd = [], []
        for n, p in named:
            (nd if any(k in n for k in no_decay) else d).append(p)
        if d:
            groups.append({"params": d, "lr": lr, "weight_decay": wd})
        if nd:
            groups.append({"params": nd, "lr": lr, "weight_decay": 0.0})
    return groups


# ------------------------------------------------------------------ 训练

def run_epoch(model, loader, device, opt=None, scaler=None, class_w=None, sched=None):
    train = opt is not None
    model.train(train)
    tot_loss, correct, n = 0.0, 0, 0
    skipped = 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", enabled=(scaler is not None)):
                logits = model(x)
                loss = F.cross_entropy(logits.float(), y, weight=class_w)
            if train:
                opt.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    skipped += 1          # 跳过坏 batch,避免 NaN 污染权重
                    continue
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                if sched is not None:
                    sched.step()          # 每 batch 一次(OneCycleLR 的正确用法)
        tot_loss += loss.item() * len(y)
        correct += int((logits.argmax(1) == y).sum())
        n += len(y)
    return tot_loss / max(n, 1), correct / max(n, 1), skipped


@torch.no_grad()
def predict(model, loader, device, n_classes):
    model.eval()
    probs = np.zeros((len(loader.dataset), n_classes), dtype=np.float32)
    labels = np.zeros(len(loader.dataset), dtype=np.int64)
    for x, y, i in loader:
        x = x.to(device)
        with torch.autocast("cuda", enabled=True):
            logits = model(x).float()
        probs[i.numpy()] = torch.softmax(logits, dim=1).cpu().numpy()
        labels[i.numpy()] = y.numpy()
    return probs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov3l")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--splits", default="splits/splits_v1.json")
    ap.add_argument("--preds", default="preds")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone 学习率")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="从训练折再切出验证集做早停;0=不切,固定 epoch")
    ap.add_argument("--patience", type=int, default=0,
                    help="验证集连续多少 epoch 不提升就停;0=不停")
    ap.add_argument("--aug", default="weak", choices=["none", "weak", "current"])
    ap.add_argument("--unfreeze-last", type=int, default=-1,
                    help="只训练最后 N 个 Transformer block + norm + head;-1=全解冻")
    ap.add_argument("--tag", default="", help="输出文件名前缀,用于区分不同配置的实验")
    ap.add_argument("--genus", default="",
                    help="只在该属内部训练(复刻 HISNET-FF 的按属分解);空=51 类扁平")
    ap.add_argument("--sched", default="none", choices=["none", "onecycle"],
                    help="none=常数学习率(实测更稳);onecycle=单周期退火")
    ap.add_argument("--peft", default="none", choices=["none", "lora", "bitfit"],
                    help="lora=低秩旁路;bitfit=只训 bias;都冻结主干")
    ap.add_argument("--opt", default="adamw", choices=["adamw", "sgd"],
                    help="优化器;sgd 用于复刻 HISNET-FF 的配方")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lpft", type=int, default=0,
                    help="LP-FT 第一阶段 epoch 数:先冻结主干只训 head,再解冻微调")
    ap.add_argument("--no-save-weights", dest="save_weights", action="store_false",
                    default=True,
                    help="不保存 .pth(ViT-L 的 state_dict 约 1.2GB,写 NFS 很慢)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--img-size", type=int, default=0)
    ap.add_argument("--views", default="ventral",
                    help="用哪些视角,逗号分隔:ventral,dorsal,lateral,mand,teeth")
    ap.add_argument("--dual-head", action="store_true",
                    help="头改用 CLS+patch均值 拼接(2048 维),与冻结探针保持一致")
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="线性 warmup 的 epoch 数(0=不 warmup)")
    ap.add_argument("--llrd", type=float, default=1.0,
                    help="逐层学习率衰减系数,如 0.75;1.0=不衰减")
    ap.add_argument("--record-epoch-test", action="store_true",
                    help="诊断用:每一轮都记录测试集准确率,用于量化'在测试集上挑最好一轮'的虚高")
    ap.add_argument("--smoke", action="store_true", help="冒烟测试:3 epoch / 50 张图")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sp = json.load(open(args.splits, encoding="utf-8"))
    class_names = sp["class_names"]
    rank = {c: i for i, c in enumerate(class_names)}

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    spec_idx, _spec_meta = load_multi_index(views)
    fold = sp["folds"][args.fold]
    print(f"视角: {views}")

    def build_items(indices, limit=None):
        items = []
        for gi in indices:
            s = sp["specimens"][gi]
            sid = s["id"]
            if sid not in spec_idx:
                continue
            if s["species"] not in rank:
                continue
            if args.genus and s["genus"] != args.genus:   # 按属过滤
                continue
            for v in views:                                # 每个视角一个样本
                p = spec_idx[sid].get(v)
                if p:
                    items.append((p, s["species"]))
        if limit:
            items = items[:limit]
        return items

    def build_groups(indices):
        """测试时按标本分组,组内多视角"""
        groups = []
        for gi in indices:
            s = sp["specimens"][gi]
            sid = s["id"]
            if sid not in spec_idx or s["species"] not in rank:
                continue
            if args.genus and s["genus"] != args.genus:
                continue
            paths = [spec_idx[sid][v] for v in views if v in spec_idx[sid]]
            if paths:
                groups.append({"specimen": sid, "species": s["species"],
                               "genus": s["genus"], "is_damaged": s["is_damaged"],
                               "paths": paths})
        return groups

    train_items = build_items(fold["train"])
    test_groups = build_groups(fold["test"])

    # 按属模式下,标签重映射到该属内部的 0..k-1
    if args.genus:
        present = sorted({sp_name for _, sp_name in train_items}
                         | {g["species"] for g in test_groups})
        if len(present) < 2:
            sys.exit(f"属 {args.genus} 在 fold {args.fold} 里类别数 < 2,无法训练")
        local = {s: i for i, s in enumerate(present)}
        train_items = [(p, local[s]) for p, s in train_items]
        for g in test_groups:
            g["label"] = local[g["species"]]
        class_names_use = present
        print(f"按属模式: {args.genus}  类别数={len(present)}")
    else:
        train_items = [(p, rank[s]) for p, s in train_items]
        for g in test_groups:
            g["label"] = rank[g["species"]]
        class_names_use = class_names
    n_classes = len(class_names_use)

    # 从训练折里切验证集(可选)
    val_items = []
    if args.val_frac > 0:
        by_cls = {}
        for it in train_items:
            by_cls.setdefault(it[1], []).append(it)
        rng = random.Random(args.seed)
        for c, lst in by_cls.items():
            rng.shuffle(lst)
            k = max(1, int(round(len(lst) * args.val_frac))) if len(lst) >= 2 else 0
            val_items += lst[:k]
            by_cls[c] = lst[k:]
        train_items = [it for lst in by_cls.values() for it in lst]

    if args.smoke:
        args.epochs = 3
        train_items = train_items[:50]
        test_groups = test_groups[:15]
        val_items = []

    n_img_test = sum(len(g["paths"]) for g in test_groups)
    print(f"fold {args.fold}: train {len(train_items)} 张  val {len(val_items)}  "
          f"test {len(test_groups)} 标本 / {n_img_test} 张  classes {n_classes}")

    timm_id, dsize = BACKBONES[args.backbone]
    model, mean, std = build(timm_id, n_classes, args.img_size or dsize)
    size = args.img_size or dsize
    pe = getattr(getattr(model, "patch_embed", None), "patch_size", None)
    if pe is not None:
        pe = pe[0] if isinstance(pe, (tuple, list)) else int(pe)
        if size % pe:
            size = (size // pe) * pe
    if args.dual_head:
        model = DualHeadNet(model, n_classes)
    model = model.to(device)
    print(f"backbone: {timm_id}  img_size={size}  amp={args.amp}"
          + ("  [dual-head 2048d]" if args.dual_head else ""))

    # PEFT:LoRA —— 主干冻结,只训低秩旁路 + head
    if args.peft == "lora":
        n_rep = apply_lora(model, args.lora_rank, args.lora_alpha)
        for name, p in model.named_parameters():
            p.requires_grad = ("lora_A" in name) or ("lora_B" in name)
        for nm in ("norm", "head", "fc", "classifier"):
            sub = getattr(model, nm, None)
            if sub is not None and hasattr(sub, "parameters"):
                for p in sub.parameters():
                    p.requires_grad = True
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"LoRA: 替换 {n_rep} 个 Linear,rank={args.lora_rank}  "
              f"可训练 {n_tr/1e6:.2f}M / {n_all/1e6:.1f}M "
              f"({n_tr/n_all*100:.2f}%)")

    # PEFT:BitFit —— 只训 bias(约 0.3M)
    if args.peft == "bitfit":
        for name, p in model.named_parameters():
            p.requires_grad = ("bias" in name)
        for nm in ("head", "fc", "classifier"):
            sub = getattr(model, nm, None)
            if sub is not None and hasattr(sub, "parameters"):
                for p in sub.parameters():
                    p.requires_grad = True
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"BitFit: 只训 bias + head  可训练 {n_tr/1e6:.2f}M / "
              f"{n_all/1e6:.1f}M ({n_tr/n_all*100:.2f}%)")

    # 部分解冻:只训最后 N 个 block + norm + head(小数据微调的常规做法)
    if args.unfreeze_last >= 0:
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            print("⚠️ 该模型没有 .blocks,忽略 --unfreeze-last")
        else:
            for p in model.parameters():
                p.requires_grad = False
            k = min(args.unfreeze_last, len(blocks))
            for blk in list(blocks)[len(blocks) - k:] if k else []:
                for p in blk.parameters():
                    p.requires_grad = True
            for name in ("norm", "head", "fc", "classifier"):
                sub = getattr(model, name, None)
                if sub is not None and hasattr(sub, "parameters"):
                    for p in sub.parameters():
                        p.requires_grad = True
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_all = sum(p.numel() for p in model.parameters())
            print(f"部分解冻:最后 {k}/{len(blocks)} blocks  "
                  f"可训练 {n_tr/1e6:.1f}M / {n_all/1e6:.1f}M")

    mk = lambda items, tr: VentralDS(items, size, mean, std, train=tr, aug=args.aug)
    dl_tr = DataLoader(mk(train_items, True), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=False)

    # 测试:展平成图片列表,记下每张图属于哪个标本
    flat_test = [(p, g["label"], gi) for gi, g in enumerate(test_groups) for p in g["paths"]]
    dl_te = DataLoader(mk([(p, lab) for p, lab, _ in flat_test], False),
                       batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)
    dl_va = (DataLoader(mk(val_items, False), batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True) if val_items else None)

    class_w = None
    if args.class_weight == "balanced":
        cnt = Counter(lbl for _, lbl in train_items)
        w = np.array([len(train_items) / (n_classes * max(cnt.get(c, 0), 1))
                      for c in range(n_classes)], dtype=np.float32)
        class_w = torch.tensor(w, device=device)

    # LP-FT 第一阶段:冻结主干,只训 head(得到一个好的分类头)
    if args.lpft > 0:
        for p in model.parameters():
            p.requires_grad = False
        unfreeze_head(model)
        opt1 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.head_lr, weight_decay=0.0)
        print(f"LP-FT 第一阶段:冻结主干,只训 head {args.lpft} epoch")
        for ep in range(1, args.lpft + 1):
            trl, tra, _ = run_epoch(model, dl_tr, device, opt1, None, class_w, None)
            print(f"  [head] ep {ep:02d}/{args.lpft}  loss {trl:.4f}  acc {tra*100:5.1f}%",
                  flush=True)
        for p in model.parameters():
            p.requires_grad = True
        print("LP-FT 第二阶段:解冻主干,全网络统一小学习率")

    # LP-FT 第二阶段:head 与主干用同一个(小的)学习率,否则 AdamW 一步就把 head 踢飞
    hlr2 = args.lr if args.lpft > 0 else args.head_lr
    if args.opt == "sgd":
        groups = build_llrd_groups(model, args.lr, hlr2, args.wd, args.llrd) \
            if args.llrd != 1.0 else param_groups(model, args.lr, hlr2, args.wd)
        opt = torch.optim.SGD(groups, momentum=0.9, nesterov=True)
        print("optimizer: SGD(momentum=0.9, nesterov)")
    else:
        if args.llrd != 1.0:
            groups = build_llrd_groups(model, args.lr, hlr2, args.wd, args.llrd)
            print(f"LLRD: {len(groups)} 个参数组,衰减系数 {args.llrd},"
                  f"lr 范围 {groups[-1]['lr']:.2e} ~ {groups[0]['lr']:.2e}")
        else:
            groups = param_groups(model, args.lr, hlr2, args.wd)
        opt = torch.optim.AdamW(groups)
    sched = None
    if args.sched == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[g["lr"] for g in opt.param_groups],
            total_steps=max(1, args.epochs * len(dl_tr)),
            pct_start=max(args.warmup / max(args.epochs, 1), 0.05))
    elif args.warmup_epochs > 0:
        warm = max(1, args.warmup_epochs * len(dl_tr))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm))
        print(f"warmup: {args.warmup_epochs} epoch 线性升温,之后常数")
    print(f"scheduler: {args.sched}"
          + ("(常数学习率)" if args.sched == "none" and args.warmup_epochs == 0 else ""))
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    labels_flat = np.array([lab for _, lab, _ in flat_test])
    yg_all = np.array([g["label"] for g in test_groups])

    def quick_test():
        """当前模型在测试集上的图片级 / 标本级准确率(仅诊断用)"""
        model.eval()
        pr = np.zeros((len(flat_test), n_classes), dtype=np.float32)
        with torch.no_grad():
            for x, _, i in dl_te:
                x = x.to(device)
                with torch.autocast("cuda", enabled=True):
                    logits = model(x).float()
                pr[i.numpy()] = torch.softmax(logits, 1).cpu().numpy()
        ia = float((pr.argmax(1) == labels_flat).mean())
        P = np.zeros((len(test_groups), n_classes), dtype=np.float32)
        for gi in range(len(test_groups)):
            sel = [j for j, (_, _, g) in enumerate(flat_test) if g == gi]
            P[gi] = pr[sel].mean(0)
        sa = float((P.argmax(1) == yg_all).mean())
        model.train()
        return ia, sa

    test_curve = []

    os.makedirs(args.logs, exist_ok=True)
    logp = os.path.join(args.logs, f"{args.backbone}_fold{args.fold}.log")
    logf = open(logp, "w", encoding="utf-8")

    best_val, best_state = -1.0, None
    bad = 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        trl, tra, nskip = run_epoch(model, dl_tr, device, opt, scaler, class_w, sched)
        msg = f"ep {ep:03d}/{args.epochs}  train_loss {trl:.4f}  train_acc {tra*100:5.1f}%"
        if nskip:
            msg += f"  (跳过 {nskip} 个 NaN batch)"
        improved = False
        if dl_va is not None:
            _, va, _ = run_epoch(model, dl_va, device)
            msg += f"  val_acc {va*100:5.1f}%"
            if va > best_val:
                best_val, improved, bad = va, True, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                msg += "  *"
            else:
                bad += 1
        msg += f"  {time.time()-t0:.0f}s"
        if args.record_epoch_test:
            ia, sa = quick_test()
            test_curve.append((ep, ia, sa))
            msg += f"  TEST_img {ia*100:5.1f}%  TEST_spec {sa*100:5.1f}%"
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()
        if dl_va is not None and args.patience > 0 and bad >= args.patience:
            print(f"early stop: {args.patience} epoch 无提升")
            break

    if args.record_epoch_test and test_curve:
        last = test_curve[-1]
        bi = max(test_curve, key=lambda r: r[1])
        bs = max(test_curve, key=lambda r: r[2])
        print("\n=== 模型选择偏差诊断(在测试集上挑最好一轮 vs 最后一轮)===")
        print(f"  图片级: 最后一轮 {last[1]*100:5.2f}%   最好一轮 {bi[1]*100:5.2f}% "
              f"(ep {bi[0]})   虚高 +{(bi[1]-last[1])*100:.2f}")
        print(f"  标本级: 最后一轮 {last[2]*100:5.2f}%   最好一轮 {bs[2]*100:5.2f}% "
              f"(ep {bs[0]})   虚高 +{(bs[2]-last[2])*100:.2f}")
        logf.write("\n=== 模型选择偏差诊断 ===\n")
        for ep, ia, sa in test_curve:
            logf.write(f"  ep{ep:03d} img {ia*100:6.2f}% spec {sa*100:6.2f}%\n")
        logf.flush()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored best val_acc={best_val*100:.1f}%")

    probs_flat, labels_flat = predict(model, dl_te, device, n_classes)

    # 按标本聚合:同一标本的多视角概率取平均
    probs = np.zeros((len(test_groups), n_classes), dtype=np.float32)
    for gi in range(len(test_groups)):
        sel = [i for i, (_, _, g) in enumerate(flat_test) if g == gi]
        probs[gi] = probs_flat[sel].mean(axis=0)
    labels = np.array([g["label"] for g in test_groups], dtype=np.int64)

    # 先落盘预测,再存权重(避免后者失败把结果一起丢掉)
    os.makedirs(args.preds, exist_ok=True)
    os.makedirs(args.weights, exist_ok=True)
    base = args.tag or (f"{args.backbone}_ftgenus_{args.genus}" if args.genus
                        else f"{args.backbone}_ft")
    tag = f"{base}_fold{args.fold}"
    pfile = os.path.join(args.preds, tag + ".npz")

    test_specs = test_groups
    spec_ids = [s.get("specimen") or s.get("id") for s in test_specs]
    print(f"保存 {len(spec_ids)} 个标本的预测 → {pfile}")
    np.savez_compressed(
        pfile,
        probs=probs, labels=labels,
        class_names=np.array(class_names_use),
        specimen=np.array([s.get("specimen", s.get("id")) for s in test_specs]),
        fold=np.full(len(labels), args.fold, dtype=np.int64),
        genus=np.array([s["genus"] for s in test_specs]),
        is_damaged=np.array([s["is_damaged"] for s in test_specs], dtype=np.int64),
        meta=json.dumps({
            "name": tag, "backbone": args.backbone, "train": "fine-tune",
            "genus_mode": args.genus or "(flat 51-class)",
            "lr": args.lr, "head_lr": args.head_lr, "wd": args.wd,
            "epochs": args.epochs, "batch": args.batch, "amp": args.amp,
            "class_weight": args.class_weight, "val_frac": args.val_frac,
            "img_size": size, "aug": args.aug,
            "unfreeze_last": args.unfreeze_last, "patience": args.patience,
            "peft": args.peft,
            "lora_rank": args.lora_rank if args.peft == "lora" else None,
            "splits_sha256": sp.get("sha256", ""), "seed": args.seed,
        }, ensure_ascii=False),
    )
    wfile = os.path.join(args.weights, tag + ".pth")
    if args.save_weights:
        torch.save(model.state_dict(), wfile)
    else:
        wfile = "(未保存)"
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0
    print(f"\nwrote {pfile}\nwrote {wfile}\npeak VRAM {peak:.2f} GB")
    logf.close()


if __name__ == "__main__":
    main()
