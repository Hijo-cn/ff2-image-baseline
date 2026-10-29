#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把「每个视角单独训练出来的模型」按标本做晚期融合(概率平均)

配合 train.py --views <单个视角> 使用:
    为每个视角各训一个模型 → 本脚本按标本平均概率 → 交给 evaluation.py

用法
----
    python ff2_fuse_views.py --splits splits/splits_v1.json \
        --out preds/dino_fix21_fused_fold0.npz \
        preds/dino_fix21_ventral_fold0.npz \
        preds/dino_fix21_dorsal_fold0.npz \
        preds/dino_fix21_lateral_fold0.npz \
        preds/dino_fix21_mand_fold0.npz \
        preds/dino_fix21_teeth_fold0.npz
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="各视角的预测文件")
    ap.add_argument("--splits", default="splits/splits_v1.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--weights", default="",
                    help="可选:逗号分隔的权重,顺序与文件一致;默认等权")
    args = ap.parse_args()

    names = None
    per_spec = defaultdict(list)
    labels_of = {}
    genus_of = {}
    dmg_of = {}
    view_acc = {}

    for f in args.files:
        d = np.load(f, allow_pickle=True)
        cn = [str(x) for x in d["class_names"]]
        if names is None:
            names = cn
        elif cn != names:
            raise SystemExit(f"{f} 的 class_names 与其它文件不一致")
        P = np.asarray(d["probs"], dtype=np.float32)
        lab = np.asarray(d["labels"]).astype(int)
        spec = [str(x) for x in d["specimen"]]
        for i, sid in enumerate(spec):
            per_spec[sid].append(P[i])
            labels_of[sid] = int(lab[i])
            if "genus" in d.files:
                genus_of[sid] = str(d["genus"][i])
            if "is_damaged" in d.files:
                dmg_of[sid] = int(d["is_damaged"][i])
        acc = float((P.argmax(1) == lab).mean())
        tag = os.path.basename(f).replace("_fold0.npz", "")
        view_acc[tag] = (acc, len(spec))
        print(f"  {tag:32s} n={len(spec):4d}  top-1={acc*100:5.1f}%")

    if args.weights:
        w = np.array([float(x) for x in args.weights.split(",")], dtype=np.float32)
        if len(w) != len(args.files):
            raise SystemExit("权重个数与文件数不一致")
        w = w / w.sum()
    else:
        w = None

    sids = [s for s in per_spec if s in labels_of]
    # 按 splits 顺序排列,便于与其它文件对齐
    sp = json.load(open(args.splits, encoding="utf-8"))
    order = {s["id"]: i for i, s in enumerate(sp["specimens"])}
    sids.sort(key=lambda s: order.get(s, 10 ** 9))

    fused = np.zeros((len(sids), len(names)), dtype=np.float32)
    for i, sid in enumerate(sids):
        plist = per_spec[sid]
        if w is None or len(plist) != len(w):
            fused[i] = np.mean(plist, axis=0)
        else:
            fused[i] = np.sum([ww * pp for ww, pp in zip(w, plist)], axis=0)

    labels = np.array([labels_of[s] for s in sids], dtype=np.int64)
    acc = float((fused.argmax(1) == labels).mean())

    print("\n" + "-" * 60)
    for k, (a, n) in view_acc.items():
        print(f"  {k:32s} {a*100:5.1f}%  (n={n})")
    print(f"  {'融合后':32s} {acc*100:5.1f}%  (n={len(sids)})")
    print("-" * 60)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        probs=fused, labels=labels,
        class_names=np.array(names),
        specimen=np.array(sids),
        fold=np.array([args.fold] * len(sids), dtype=np.int64),
        genus=np.array([genus_of.get(s, "") for s in sids]),
        is_damaged=np.array([dmg_of.get(s, 0) for s in sids], dtype=np.int64),
        meta=json.dumps({"name": os.path.splitext(os.path.basename(args.out))[0],
                         "fusion": "late prob-average",
                         "views": [os.path.basename(f) for f in args.files],
                         "per_view_top1": {k: v[0] for k, v in view_acc.items()}},
                        ensure_ascii=False),
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
