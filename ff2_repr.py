#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FF2 表征质量分析(师兄指定的四项)

对每个骨干的冻结特征,输出:
  1. Linear Probe   —— 5 折分组 CV × 多种子
  2. kNN            —— 两种口径(标本原型 / 全部增强副本)
  3. 类内/类间距离   —— 判别比 + Fisher 比 + 轮廓系数 + 类中心 LOO 准确率
  4. UMAP           —— 只画 746 个标本的原图特征,按属上色

用法:
    python ff2_repr.py                 # 跑全部已存在的 npz
    python ff2_repr.py --only dinov3l  # 只跑一个
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

ROOT = os.environ.get("FF2_ROOT", "")
FEATDIR = os.path.join(ROOT, "features")
REPDIR = os.path.join(ROOT, "reports")

# ViT 系用 cls+patch;CNN 与 open_clip 只有池化向量,用 cls
POOL_BY_PREFIX = {
    "dinov2l": "cls+patch",
    "dinov3l": "cls+patch",
    "in21k_vitl": "cls+patch",
    "siglip2l": "cls+patch",
    "bioclip2": "cls",
    "convnextl": "cls",
    "efficientnetb7": "cls",
}


def l2(x):
    return normalize(np.asarray(x, dtype=np.float64), norm="l2")


def acc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((a == b).mean()) if len(a) else float("nan")


def pick_pool(tag):
    for k, v in POOL_BY_PREFIX.items():
        if tag.startswith(k):
            return v
    return "cls"


def load_X(d, pool):
    if pool == "cls":
        return d["cls"]
    if pool == "patch":
        return d["patchmean"]
    return np.concatenate([d["cls"], d["patchmean"]], axis=1)


# ---------------- 1. Linear probe(多种子) ----------------

def linear_probe_multi(X, y, base, aug, seeds, C=100, n_splits=5):
    base_ids = np.unique(base)
    b_y = np.array([y[base == b][0] for b in base_ids])
    per_seed = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []
        for tr_b, te_b in skf.split(base_ids, b_y):
            tr = np.isin(base, base_ids[tr_b])
            te = np.isin(base, base_ids[te_b]) & (aug == 0)   # 只在原图上评测
            clf = LogisticRegression(C=C, max_iter=2000,
                                     class_weight="balanced").fit(X[tr], y[tr])
            folds.append(acc(y[te], clf.predict(X[te])))
        per_seed.append(folds)
    per_seed = np.array(per_seed)          # (n_seed, n_fold)
    return {
        "per_seed_mean": per_seed.mean(axis=1).tolist(),
        "mean": float(per_seed.mean()),
        "std_over_folds": float(per_seed.std(axis=1).mean()),
        "std_over_seeds": float(per_seed.mean(axis=1).std()),
    }


# ---------------- 2. kNN 两种口径 ----------------

def knn_two_modes(X, y, base, aug, seeds, ks=(1, 3, 5, 10, 20), n_splits=5):
    base_ids = np.unique(base)
    b_y = np.array([y[base == b][0] for b in base_ids])
    prot_acc = {k: [] for k in ks}
    rep_acc = {k: [] for k in ks}

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_b, te_b in skf.split(base_ids, b_y):
            tr_set, te_set = set(base_ids[tr_b]), set(base_ids[te_b])
            mtr = np.array([b in tr_set for b in base])
            mte = np.array([b in te_set for b in base]) & (aug == 0)

            # --- 口径 A:每标本取增强副本均值作为原型 ---
            uniq = np.unique(base[mtr])
            P = np.stack([X[mtr][base[mtr] == u].mean(0) for u in uniq])
            P = l2(P)
            yp = np.array([y[mtr][base[mtr] == u][0] for u in uniq])
            S = X[mte] @ P.T
            order = np.argsort(-S, axis=1)
            for k in ks:
                kk = min(k, len(uniq))
                pred = yp[order[:, :kk]]
                # 多票
                votes = np.array([np.bincount(p, minlength=y.max() + 1).argmax()
                                  for p in pred])
                prot_acc[k].append(acc(y[mte], votes))

            # --- 口径 B:全部增强副本直接当邻居 ---
            Xr, yr = X[mtr], y[mtr]
            S2 = X[mte] @ Xr.T
            order2 = np.argsort(-S2, axis=1)
            for k in ks:
                kk = min(k, len(Xr))
                pred = yr[order2[:, :kk]]
                votes = np.array([np.bincount(p, minlength=y.max() + 1).argmax()
                                  for p in pred])
                rep_acc[k].append(acc(y[mte], votes))

    return ({f"k{k}": float(np.mean(v)) for k, v in prot_acc.items()},
            {f"k{k}": float(np.mean(v)) for k, v in rep_acc.items()})


# ---------------- 3. 类内 / 类间距离 ----------------

def distance_metrics(X, y):
    X0 = l2(X)
    classes = np.unique(y)
    cents = np.stack([X0[y == c].mean(0) for c in classes])
    cents = l2(cents)

    # 类内余弦距离
    intra = []
    for i, c in enumerate(classes):
        pts = X0[y == c]
        intra.append(1.0 - (pts @ cents[i]))
    intra = np.concatenate(intra)

    # 类间余弦距离(类中心两两)
    inter = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            inter.append(1.0 - float(cents[i] @ cents[j]))
    inter = np.array(inter)

    ratio = float(inter.mean() / max(intra.mean(), 1e-12))

    # Fisher 比:类间散度 / 类内散度(欧氏,基于 L2 归一化特征)
    mu = X0.mean(0)
    Sb = sum(len(X0[y == c]) * np.sum((cents[i] - mu) ** 2)
             for i, c in enumerate(classes))
    Sw = sum(np.sum((X0[y == c] - cents[i]) ** 2)
             for i, c in enumerate(classes))
    fisher = float(Sb / max(Sw, 1e-12))

    # 轮廓系数
    try:
        sil = float(silhouette_score(X0, y, metric="cosine"))
    except Exception:
        sil = float("nan")

    # 类中心最近邻准确率(不做任何训练)
    S = X0 @ cents.T
    pred = np.array([classes[j] for j in np.argmax(S, axis=1)])
    centroid_acc = acc(y, pred)

    return {
        "intra_cos_dist": float(intra.mean()),
        "inter_cos_dist": float(inter.mean()),
        "discrim_ratio": ratio,
        "fisher_ratio": fisher,
        "silhouette_cosine": sil,
        "centroid_acc": centroid_acc,
    }


# ---------------- 4. UMAP ----------------

def umap_plot(X0, g_idx, g_uniq, tag, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                        random_state=0, n_jobs=1)
    Z = reducer.fit_transform(X0)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    cmap = plt.get_cmap("tab20")
    ncol = len(g_uniq)
    for i, g in enumerate(g_uniq):
        m = g_idx == i
        ax.scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.85,
                   color=cmap(i % 20), label=f"{g} ({m.sum()})", linewidths=0)
    ax.set_title(f"UMAP of frozen features — {tag}  (n={len(X0)} specimens, coloured by genus)",
                 fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, markerscale=1.4, loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return Z


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑名字包含该子串的骨干")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--c", type=float, default=100.0)
    ap.add_argument("--no-umap", action="store_true")
    ap.add_argument("--outdir", default="",
                    help="输出目录,默认 $FF2_ROOT/reports;可指向本地盘绕开 NAS 权限问题")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    repdir = os.path.abspath(os.path.expanduser(args.outdir)) if args.outdir else REPDIR
    os.makedirs(repdir, exist_ok=True)
    print(f"输出目录: {repdir}")
    files = sorted(glob.glob(os.path.join(FEATDIR, "feats_aug_*_x10_all.npz")))
    if args.only:
        files = [f for f in files if args.only in os.path.basename(f)]
    if not files:
        raise SystemExit("没有找到 npz 文件")

    all_res = {}
    jp = os.path.join(repdir, "repr_all.json")
    if os.path.exists(jp):
        try:
            with open(jp, encoding="utf-8") as fh:
                all_res = json.load(fh)
            print(f"已从 {jp} 载入 {len(all_res)} 条历史结果(将合并)")
        except Exception:
            all_res = {}
    for fp in files:
        tag = os.path.basename(fp).replace("feats_aug_", "").replace("_x10_all.npz", "")
        pool = pick_pool(tag)
        print(f"\n{'=' * 66}\n[{tag}]  pool={pool}  file={os.path.basename(fp)}\n{'=' * 66}")

        d = None
        try:
            d = np.load(fp, allow_pickle=True)
            X = l2(load_X(d, pool))
        except Exception as e:
            print(f"  !! 跳过(读取失败: {type(e).__name__}: {e})")
            continue
        y_sp = d["sp_idx"].astype(int)
        y_gen = d["g_idx"].astype(int)
        g_uniq = [str(x) for x in d["g_uniq"]]
        if "base_idx" in d.files:
            base, aug = d["base_idx"].astype(int), d["aug_id"].astype(int)
        else:
            base, aug = np.arange(len(y_sp)), np.zeros(len(y_sp), dtype=int)

        X0 = X[aug == 0]           # 746 个标本原图
        sp0, gen0 = y_sp[aug == 0], y_gen[aug == 0]

        print("  [1/4] linear probe (multi-seed) ...", flush=True)
        lp_sp = linear_probe_multi(X, y_sp, base, aug, seeds, C=args.c)
        lp_gen = linear_probe_multi(X, y_gen, base, aug, seeds, C=args.c)

        print("  [2/4] kNN ...", flush=True)
        knn_prot, knn_rep = knn_two_modes(X, y_sp, base, aug, seeds)

        print("  [3/4] distance metrics ...", flush=True)
        dm = distance_metrics(X0, sp0)

        umap_path = None
        if not args.no_umap:
            print("  [4/4] UMAP ...", flush=True)
            umap_path = os.path.join(repdir, f"umap_{tag}.png")
            umap_plot(X0, gen0, g_uniq, tag, umap_path)
            print(f"        -> {umap_path}")

        r = {"tag": tag, "pool": pool,
             "species_mean": lp_sp["mean"], "species_std": lp_sp["std_over_seeds"],
             "genus_mean": lp_gen["mean"], "genus_std": lp_gen["std_over_seeds"],
             "knn_proto": knn_prot, "knn_replica": knn_rep, **dm,
             "umap_png": umap_path}
        all_res[tag] = r

        print(f"  species {lp_sp['mean']*100:.2f}±{lp_sp['std_over_seeds']*100:.2f}  "
              f"genus {lp_gen['mean']*100:.2f}±{lp_gen['std_over_seeds']*100:.2f}  "
              f"判别比 {dm['discrim_ratio']:.3f}  Fisher {dm['fisher_ratio']:.3f}")

    # 写 JSON
    jp = os.path.join(repdir, "repr_all.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(all_res, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {jp}")

    # 写 CSV
    cp = os.path.join(repdir, "repr_summary.csv")
    cols = ["tag", "pool", "species_mean", "species_std", "genus_mean", "genus_std",
            "knn_proto_k1", "knn_proto_k5", "knn_proto_k10",
            "knn_replica_k1", "knn_replica_k5", "knn_replica_k10",
            "discrim_ratio", "fisher_ratio", "silhouette_cosine", "centroid_acc"]
    with open(cp, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for tag, r in all_res.items():
            row = [tag, r["pool"],
                   f"{r['species_mean']:.4f}", f"{r['species_std']:.4f}",
                   f"{r['genus_mean']:.4f}", f"{r['genus_std']:.4f}",
                   f"{r['knn_proto']['k1']:.4f}", f"{r['knn_proto']['k5']:.4f}",
                   f"{r['knn_proto']['k10']:.4f}",
                   f"{r['knn_replica']['k1']:.4f}", f"{r['knn_replica']['k5']:.4f}",
                   f"{r['knn_replica']['k10']:.4f}",
                   f"{r['discrim_ratio']:.4f}", f"{r['fisher_ratio']:.4f}",
                   f"{r['silhouette_cosine']:.4f}", f"{r['centroid_acc']:.4f}"]
            fh.write(",".join(row) + "\n")
    print(f"wrote {cp}")


if __name__ == "__main__":
    main()
