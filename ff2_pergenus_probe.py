#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专用属级 + 属内分类器(冻结特征上的两级体系)

与现有做法的区别
----------------
现状:一个 51 类分类器,再用 oracle 属路由把别的属概率清零。
      属级概率 = 属内各种概率之和。

本脚本:
  ① 单独训一个 18 类**属分类器**(不再靠种概率相加)
  ② 每个多型属单独训一个 3~9 类**属内分类器**(只在该属样本上训练)
  ③ 两级串联:属 → 属内种

这样属级和属内各自有专门的判别边界,不会被别的属的样本干扰。

用法
----
    python -u ff2_pergenus_probe.py --splits splits/splits_v1.json \
        --outdir preds/pg --aug-per-spec 3
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

ROOT = os.environ.get("FF2_ROOT", "")
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
    ap.add_argument("--outdir", default="preds/pg")
    ap.add_argument("--aug-per-spec", type=int, default=3)
    ap.add_argument("--c", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sp = json.load(open(args.splits, encoding="utf-8"))
    class_names = sp["class_names"]
    C = len(class_names)
    spec2cls = {s["id"]: s["species"] for s in sp["specimens"]}
    spec2gen = {s["id"]: s["genus"] for s in sp["specimens"]}
    views = list(FILES)

    vd = {v: load_view(v, class_names, args.aug_per_spec, args.seed) for v in views}

    def fit(X, y):
        return LogisticRegression(C=args.c, max_iter=2000,
                                  class_weight="balanced").fit(X, y)

    def pooled_probs(v, clf, s):
        X, _ = vd[v][s]
        pr = clf.predict_proba(X).mean(axis=0)
        out = np.zeros(C)
        out[clf.classes_] = pr
        return out

    n = len(sp["specimens"])
    P_species = np.zeros((n, C), dtype=np.float32)   # 属内专用
    P_genus = np.zeros((n, C), dtype=np.float32)     # 专用属分类器(展开到种)
    y_all = np.full(n, -1, dtype=np.int64)
    g_all = np.full(n, -1, dtype=np.int64)
    fold_all = np.zeros(n, dtype=np.int64)

    genera = sorted({s["genus"] for s in sp["specimens"]})
    gidx = {g: i for i, g in enumerate(genera)}
    spec2gidx = {s["id"]: gidx[s["genus"]] for s in sp["specimens"]}

    for k, fold in enumerate(sp["folds"]):
        tr = [sp["specimens"][i]["id"] for i in fold["train"]]
        te = [sp["specimens"][i]["id"] for i in fold["test"]]
        tr = [s for s in tr if s in vd[views[0]]]
        te = [s for s in te if s in vd[views[0]]]

        # ---------- ① 专用属分类器 ----------
        clfs_g = {}
        for v in views:
            items = [vd[v][s] for s in tr if s in vd[v]]
            X = np.concatenate([x for x, _ in items])
            yg = np.concatenate([[spec2gidx[s]] * len(vd[v][s][0])
                                 for s in tr if s in vd[v]])
            clfs_g[v] = fit(X, yg)

        # ---------- ② 每个多型属单独的属内分类器 ----------
        by_gen = defaultdict(list)
        for s in tr:
            by_gen[spec2gen[s]].append(s)
        poly = {g: ss for g, ss in by_gen.items()
                if len({spec2cls[x] for x in ss}) >= 2}
        clfs_sp = {}
        for g, ss in poly.items():
            for v in views:
                items = [vd[v][s] for s in ss if s in vd[v]]
                if not items:
                    continue
                X = np.concatenate([x for x, _ in items])
                y = np.concatenate([[vd[v][s][1]] * len(vd[v][s][0])
                                    for s in ss if s in vd[v]])
                clfs_sp[(g, v)] = fit(X, y)

        # ---------- 测试 ----------
        idx_of = {sp["specimens"][i]["id"]: i for i in range(n)}

        for s in te:
            gi = idx_of[s]
            y_all[gi] = class_names.index(spec2cls[s])
            g_all[gi] = spec2gidx[s]
            fold_all[gi] = k

            # 专用属分类器 → 展开成"该属所有种均分"
            pg = np.zeros(len(genera))
            for v in views:
                if s not in vd[v]:
                    continue
                Xs, _ = vd[v][s]
                pr = clfs_g[v].predict_proba(Xs).mean(axis=0)
                pg[clfs_g[v].classes_] += pr
            for c_i, cname in enumerate(class_names):
                pg_c = pg[gidx[cname.split(" ")[0]]]
                P_genus[gi, c_i] = pg_c

            # 属内专用分类器
            g_true = spec2gen[s]
            if g_true in poly:
                acc = np.zeros(C)
                cnt = 0
                for v in views:
                    if (g_true, v) not in clfs_sp or s not in vd[v]:
                        continue
                    Xs, _ = vd[v][s]
                    pr = clfs_sp[(g_true, v)].predict_proba(Xs).mean(axis=0)
                    acc[clfs_sp[(g_true, v)].classes_] += pr
                    cnt += 1
                if cnt:
                    P_species[gi] = acc / cnt

    ok = (y_all >= 0) & (P_species.sum(1) > 0)
    flat_pg = (P_species[ok].argmax(1) == y_all[ok]).mean()
    ok2 = (g_all >= 0) & (P_genus.sum(1) > 0)
    gen_acc = (P_genus[ok2].argmax(1) == g_all[ok2]).mean()

    print(f"\n专用属分类器(18 类)  genus top-1 = {gen_acc*100:.2f}%")
    print(f"专用属内分类器        hier_oracle = {flat_pg*100:.2f}%")
    print(f"(对照:概率平均 89.5 / 96.6 / 92.2)")

    os.makedirs(args.outdir, exist_ok=True)
    idx = np.where(ok)[0]
    np.savez_compressed(
        os.path.join(args.outdir, "pergenus_species_oof.npz"),
        probs=P_species[idx], labels=y_all[idx],
        class_names=np.array(class_names),
        specimen=np.array([sp["specimens"][i]["id"] for i in idx]),
        fold=fold_all[idx],
        genus=np.array([sp["specimens"][i]["genus"] for i in idx]),
        is_damaged=np.array([sp["specimens"][i]["is_damaged"] for i in idx], dtype=np.int64),
        meta=json.dumps({"name": "pergenus_species", "splits_sha256": sp.get("sha256", "")},
                        ensure_ascii=False),
    )
    print(f"wrote {args.outdir}/pergenus_species_oof.npz")


if __name__ == "__main__":
    main()
