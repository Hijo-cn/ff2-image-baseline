#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专用属分类器:18 类,每视角一个,概率平均

背景
----
我们现在的属级概率 = 51 个种的概率按属求和。这会被"属内种数多"的属放大
(Mogera 9 个种 vs Parascaptor 3 个种),导致属级偏置。

FF1 是单独训练 18 类属分类器。本脚本在冻结特征上做同样的事,看差距有多大。

用法
----
    python -u ff2_genus_probe.py --splits splits/splits_v1.json \
        --out preds/genus_probe_oof.npz --aug-per-spec 3
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
    ap.add_argument("--out", default="preds/genus_probe_oof.npz")
    ap.add_argument("--aug-per-spec", type=int, default=3)
    ap.add_argument("--c", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--views", default="ventral,dorsal,lateral,mand,teeth")
    args = ap.parse_args()

    sp = json.load(open(args.splits, encoding="utf-8"))
    class_names = sp["class_names"]
    C = len(class_names)
    views = [v.strip() for v in args.views.split(",") if v.strip()]

    spec2sp = {s["id"]: s["species"] for s in sp["specimens"]}
    spec2g = {s["id"]: s["genus"] for s in sp["specimens"]}
    genera = sorted({s["genus"] for s in sp["specimens"]})
    gidx = {g: i for i, g in enumerate(genera)}
    G = len(genera)

    print("读取特征...")
    vd = {v: load_view(v, class_names, args.aug_per_spec, args.seed) for v in views}
    for v in views:
        print(f"  {v:8s}: {len(vd[v]):4d} 标本")

    def fit(X, y):
        return LogisticRegression(C=args.c, max_iter=2000,
                                  class_weight="balanced").fit(X, y)

    n = len(sp["specimens"])
    Pg_all = np.zeros((n, G), dtype=np.float32)
    Ps_all = np.zeros((n, C), dtype=np.float32)     # 种概率求和,用于对照
    yg_all = np.full(n, -1, dtype=np.int64)
    glab = {s["id"]: gidx[s["genus"]] for s in sp["specimens"]}

    idx_of = {s["id"]: i for i, s in enumerate(sp["specimens"])}
    Pg_all[:] = 0
    for k, fold in enumerate(sp["folds"]):
        tr = [sp["specimens"][i]["id"] for i in fold["train"]]
        te = [sp["specimens"][i]["id"] for i in fold["test"]]
        tr = [s for s in tr if s in vd[views[0]]]
        te = [s for s in te if s in vd[views[0]]]
        acc_v = {v: np.zeros((len(te), G)) for v in views}
        cnt_v = {v: np.zeros(len(te)) for v in views}
        for v in views:
            items = [vd[v][s] for s in tr if s in vd[v]]
            if not items:
                continue
            X = np.concatenate([x for x, _ in items])
            y = np.concatenate([[glab[s]] * len(vd[v][s][0]) for s in tr if s in vd[v]])
            clf = fit(X, y)
            for j, s in enumerate(te):
                if s not in vd[v]:
                    continue
                Xs, _ = vd[v][s]
                pr = clf.predict_proba(Xs).mean(axis=0)
                acc_v[v][j][clf.classes_] = pr
                cnt_v[v][j] += 1
        for j, s in enumerate(te):
            tot = np.zeros(G)
            for v in views:
                if cnt_v[v][j] > 0:
                    tot += acc_v[v][j]
            i = idx_of[s]
            Pg_all[i] = tot
            yg_all[i] = glab[s]

    ok = yg_all >= 0
    acc = (Pg_all[ok].argmax(1) == yg_all[ok]).mean()
    print(f"\n专用属分类器(18 类,5 折)= {acc*100:.2f}%")
    print("对照:种概率求和的属级(旧基线)= 96.5%(cran4)/ 97.2%(含牙齿)")
    print("FF1 属级 = 98.8%")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    idx = np.where(ok)[0]
    np.savez_compressed(
        args.out,
        probs=Pg_all[idx],
        labels=yg_all[idx],
        class_names=np.array(genera),
        specimen=np.array([sp["specimens"][i]["id"] for i in idx]),
        fold=np.zeros(len(idx), dtype=np.int64),
        genus=np.array([sp["specimens"][i]["genus"] for i in idx]),
        is_damaged=np.array([sp["specimens"][i]["is_damaged"] for i in idx], dtype=np.int64),
        meta=json.dumps({"name": "genus_probe", "n_classes": G}, ensure_ascii=False),
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
