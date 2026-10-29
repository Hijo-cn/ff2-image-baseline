#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多型属内部:按种上色的 UMAP + 定量对照

回答一个问题:
    "属内种到底是『有子结构但被挤住』还是『完全没有子结构』?"

对每个多型属:
  1. 只取该属的标本原图(aug==0)
  2. 在该子集上单独拟合 UMAP(cosine),按种上色
  3. 同时给出两个定量指标:
       - 属内轮廓系数(cosine):空间上分不分得开
       - 属内 5 折逻辑回归准确率:监督信号能不能把它分开

两者对比即可判断:
   高准确率 + 低轮廓 → 信息存在,但只在全局线性方向里,局部邻域不可见
   低准确率 + 低轮廓 → 视觉信息确实到顶

用法:
    python ff2_umap_genus.py --feats feats_aug_dinov3l_512_x10_all.npz --outdir ~/ff2_reports
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

ROOT = os.environ.get("FF2_ROOT", "/mnt/nfs_share/EC_DATABASE_UGREEN/yanhaiqiao/FF2")
FEATDIR = os.path.join(ROOT, "features")


def l2(x):
    return normalize(np.asarray(x, dtype=np.float64), norm="l2")


def within_genus_cv(X, y, n_splits=5, seeds=(0, 1, 2)):
    counts = np.bincount(y)
    k = min(n_splits, int(counts[counts > 0].min()))
    if k < 2:
        return float("nan")
    accs = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            if len(np.unique(y[tr])) < 2:
                continue
            clf = LogisticRegression(C=10, max_iter=3000,
                                     class_weight="balanced").fit(X[tr], y[tr])
            accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="feats_aug_dinov3l_512_x10_all.npz")
    ap.add_argument("--pool", default="cls+patch")
    ap.add_argument("--outdir", default=os.path.expanduser("~/ff2_reports"))
    args = ap.parse_args()

    outdir = os.path.abspath(os.path.expanduser(args.outdir))
    os.makedirs(outdir, exist_ok=True)

    fp = args.feats if os.path.isabs(args.feats) else os.path.join(FEATDIR, args.feats)
    d = np.load(fp, allow_pickle=True)
    print(f"loaded {fp}")

    if args.pool == "cls":
        X = d["cls"]
    elif args.pool == "patch":
        X = d["patchmean"]
    else:
        X = np.concatenate([d["cls"], d["patchmean"]], axis=1)
    X = l2(X)

    aug = d["aug_id"].astype(int) if "aug_id" in d.files else np.zeros(len(X), dtype=int)
    X0 = X[aug == 0]
    sp = np.array([str(s) for s in d["species"]])[aug == 0]
    gn = np.array([str(g) for g in d["genus"]])[aug == 0]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    rows = []
    for g in sorted(set(gn)):
        m = gn == g
        n_sp = len(set(sp[m]))
        if n_sp < 2:
            continue
        Xg, spg = X0[m], sp[m]
        n = len(Xg)
        n_sp = len(set(spg))

        # 定量
        try:
            sil = float(silhouette_score(Xg, spg, metric="cosine"))
        except Exception:
            sil = float("nan")
        cv = within_genus_cv(Xg, np.unique(spg, return_inverse=True)[1])

        # UMAP(子集小,邻域参数随之调小)
        nn = int(min(15, max(5, n // 8)))
        Z = umap.UMAP(n_neighbors=nn, min_dist=0.1, metric="cosine",
                      random_state=0, n_jobs=1).fit_transform(Xg)

        fig, ax = plt.subplots(figsize=(7.5, 6), dpi=170)
        cmap = plt.get_cmap("tab10")
        for i, s in enumerate(sorted(set(spg))):
            mm = spg == s
            ax.scatter(Z[mm, 0], Z[mm, 1], s=52, alpha=0.9,
                       color=cmap(i % 10), edgecolors="white", linewidths=0.4,
                       label=f"{s.split(' ', 1)[-1]} ({mm.sum()})")
        ax.set_title(f"{g} — {n} specimens, {n_sp} species\n"
                     f"within-genus silhouette = {sil:.3f}   |   "
                     f"5-fold logistic acc = {cv*100:.1f}%", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7.5, frameon=False, loc="center left",
                  bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        p = os.path.join(outdir, f"umap_genus_{g}.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)

        rows.append({"genus": g, "n": n, "n_species": n_sp,
                     "silhouette": sil, "cv_acc": cv, "png": p})
        print(f"  {g:14s} n={n:3d} sp={n_sp}  sil={sil:+.3f}  "
              f"CV={cv*100:5.1f}%  -> {os.path.basename(p)}")

    print()
    print(f"{'genus':14s} {'n':>4s} {'#sp':>4s} {'silhouette':>11s} {'CV acc':>8s}")
    print("-" * 48)
    for r in sorted(rows, key=lambda x: x["cv_acc"]):
        print(f"{r['genus']:14s} {r['n']:4d} {r['n_species']:4d} "
              f"{r['silhouette']:+11.3f} {r['cv_acc']*100:7.1f}%")

    jp = os.path.join(outdir, "umap_genus_summary.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {jp}")


if __name__ == "__main__":
    main()
