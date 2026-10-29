#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征级融合:把一个标本多个视角的特征直接拼接,训一个分类器

与概率平均的区别
----------------
概率平均:每视角一个分类器 → 各自出概率 → 平均(丢失视角间交叉信息)
特征拼接:5 × 2048 维直接拼成 10240 维 → 一个分类器(保留交叉信息)

用法
----
    python -u ff2_feat_concat.py --splits splits/splits_v1.json \
        --out preds/featcat_oof.npz --aug-per-spec 3
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

ROOT = os.environ.get("FF2_ROOT", "/mnt/nfs_share/EC_DATABASE_UGREEN/yanhaiqiao/FF2")
FEATDIR = os.path.join(ROOT, "features")

FILES = {
    "ventral": "feats_aug_dinov3l_512_x10_all.npz",
    "dorsal": "feats_aug_dinov3l_512_x10_all_dorsal.npz",
    "lateral": "feats_aug_dinov3l_512_x10_all_lateral.npz",
    "mand": "feats_aug_dinov3l_512_x10_all_mand.npz",
    "teeth": "feats_aug_dinov3l_512_x10_all_teeth.npz",
}


def load_view(name, class_names, aug_per_spec, seed=0):
    p = os.path.join(FEATDIR, FILES[name])
    if not os.path.exists(p):
        return {}
    d = np.load(p, allow_pickle=True)
    X = normalize(np.concatenate([d["cls"], d["patchmean"]], axis=1).astype(np.float64),
                  norm="l2")
    sp_uniq = [str(x) for x in d["sp_uniq"]]
    sp_idx = d["sp_idx"].astype(int)
    spec = np.array([str(x) for x in d["specimen"]])
    aug = d["aug_id"].astype(int) if "aug_id" in d.files else np.zeros(len(spec), int)
    cidx = {c: i for i, c in enumerate(class_names)}
    lab = np.array([cidx.get(sp_uniq[i], -1) for i in sp_idx])
    by = defaultdict(list)
    for i in range(len(spec)):
        by[spec[i]].append(i)
    rng = np.random.default_rng(seed)
    out = {}
    for sid, rows in by.items():
        o = np.array(rows)
        orig = o[aug[o] == 0]
        rest = o[aug[o] > 0]
        keep = max(aug_per_spec - 1, 0)
        if len(rest) > keep:
            rest = rng.choice(rest, keep, replace=False)
        sel = np.concatenate([orig[:1], rest]) if keep else orig[:1]
        sel = sel[lab[sel] >= 0]
        if len(sel):
            out[sid] = (X[sel], int(lab[sel[0]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits/splits_v1.json")
    ap.add_argument("--out", default="preds/featcat_oof.npz")
    ap.add_argument("--aug-per-spec", type=int, default=3)
    ap.add_argument("--c", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sp = json.load(open(args.splits, encoding="utf-8"))
    class_names = sp["class_names"]
    C = len(class_names)
    views = list(FILES)

    print("读取特征...")
    vd = {v: load_view(v, class_names, args.aug_per_spec, args.seed) for v in views}
    dims = {}
    for v in views:
        if vd[v]:
            dims[v] = next(iter(vd[v].values()))[0].shape[1]
        print(f"  {v:8s}: {len(vd[v]):4d} 标本, {dims.get(v,0)} 维")

    total_dim = sum(dims.values())
    print(f"拼接后维度: {total_dim}\n")

    def build(s):
        """一个标本在所有视角上的特征拼接;缺失视角用零向量补"""
        parts = []
        for v in views:
            if s in vd[v]:
                X, _ = vd[v][s]
                parts.append(X.mean(axis=0))
            else:
                parts.append(np.zeros(dims[v]))
        return np.concatenate(parts)

    n = len(sp["specimens"])
    P = np.zeros((n, C), dtype=np.float32)
    y_all = np.full(n, -1, dtype=np.int64)
    fold_all = np.zeros(n, dtype=np.int64)
    idx_of = {sp["specimens"][i]["id"]: i for i in range(n)}

    for k, fold in enumerate(sp["folds"]):
        tr = [sp["specimens"][i]["id"] for i in fold["train"]]
        te = [sp["specimens"][i]["id"] for i in fold["test"]]
        tr = [s for s in tr if all(s in vd[v] for v in views) or s in vd["ventral"]]
        te = [s for s in te if s in vd["ventral"]]

        Xtr = np.stack([build(s) for s in tr])
        ytr = np.array([vd["ventral"][s][1] for s in tr])
        Xtr = normalize(Xtr, norm="l2")
        clf = LogisticRegression(C=args.c, max_iter=2000,
                                 class_weight="balanced").fit(Xtr, ytr)

        Xte = normalize(np.stack([build(s) for s in te]), norm="l2")
        pr = clf.predict_proba(Xte)
        for i, s in enumerate(te):
            gi = idx_of[s]
            P[gi, clf.classes_] = pr[i]
            y_all[gi] = vd["ventral"][s][1]
            fold_all[gi] = k
        acc = (pr.argmax(1) == np.array([vd["ventral"][s][1] for s in te])).mean()
        print(f"fold {k}: train {len(tr)} test {len(te)}  top-1 {acc*100:.2f}%", flush=True)

    ok = y_all >= 0
    print(f"\n特征拼接 5 折 top-1 = {(P[ok].argmax(1) == y_all[ok]).mean()*100:.2f}%")
    print("(对照:概率平均冻结 89.5%)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    idx = np.where(ok)[0]
    np.savez_compressed(
        args.out, probs=P[idx], labels=y_all[idx],
        class_names=np.array(class_names),
        specimen=np.array([sp["specimens"][i]["id"] for i in idx]),
        fold=fold_all[idx],
        genus=np.array([sp["specimens"][i]["genus"] for i in idx]),
        is_damaged=np.array([sp["specimens"][i]["is_damaged"] for i in idx], dtype=np.int64),
        meta=json.dumps({"name": "featcat", "fusion": "feature concatenation",
                         "dim": total_dim}, ensure_ascii=False),
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
