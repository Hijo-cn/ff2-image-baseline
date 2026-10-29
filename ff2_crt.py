#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cRT / τ-normalization:长尾下的分类器再平衡(表征完全冻结)

背景
----
Kang et al., "Decoupling Representation and Classifier for Long-Tailed
Recognition" (ICLR 2020) 指出:长尾问题里"学表征"和"学分类器"应该分开。
表征已经够好时,只需**重训分类器**并做类别均衡,就能显著提升小类表现。

我们的现状:top-1 84.18% 但 macro-F1 只有 0.7577 —— 典型的"大类占优"。

本脚本在完全冻结的 DINOv3 特征上,比较四种分类器:

    base        逻辑回归 + class_weight=balanced(现有基线)
    base_tau    同上,再对分类器权重做 τ-normalization
    cbs         类别均衡采样重训(每类等量),class_weight=None
    cbs_tau*    同上 + τ-normalization(τ=0.5 / 1.0)

关键实现:τ-normalization 只改权重向量,**不需要重新训练** ——
所以每个 fold 只拟合 2 次(cbs 与 base),其余靠后处理得到。

用法
----
    python -u ff2_crt.py --feats feats_aug_dinov3l_512_x10_all.npz \
        --splits splits/splits_v1.json --outdir preds/crt
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

ROOT = os.environ.get("FF2_ROOT", "")
FEATDIR = os.path.join(ROOT, "features")


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def tau_normalize(W, tau):
    """分类器权重 τ-normalization:压低大类权重的模长。"""
    n = np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
    m = n.mean()
    return W / (n ** tau) * (m ** tau)


def class_balanced_index(y, rng):
    """类别均衡采样:每类取总样本数/类数,少类有放回、多类无放回。"""
    cls, cnt = np.unique(y, return_counts=True)
    target = int(len(y) / len(cls))
    out = []
    for c in cls:
        idx = np.flatnonzero(y == c)
        if len(idx) >= target:
            out.append(rng.choice(idx, target, replace=False))
        else:
            out.append(rng.choice(idx, target, replace=True))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="feats_aug_dinov3l_512_x10_all.npz")
    ap.add_argument("--splits", default="splits/splits_v1.json")
    ap.add_argument("--outdir", default="preds/crt")
    ap.add_argument("--c", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fp = args.feats if os.path.isabs(args.feats) else os.path.join(FEATDIR, args.feats)
    sp = json.load(open(args.splits, encoding="utf-8"))
    class_names = sp["class_names"]

    d = np.load(fp, allow_pickle=True)
    X = normalize(np.concatenate([d["cls"], d["patchmean"]], axis=1).astype(np.float64),
                  norm="l2")
    y = d["sp_idx"].astype(int)
    spec = np.array([str(x) for x in d["specimen"]])
    aug = d["aug_id"].astype(int) if "aug_id" in d.files else np.zeros(len(y), dtype=int)
    is_orig = aug == 0

    spec2row = {}
    for i in np.where(is_orig)[0]:
        spec2row[spec[i]] = i

    variants = ["base", "base_tau1", "cbs", "cbs_tau05", "cbs_tau1"]
    acc = {v: {} for v in variants}

    for k, fold in enumerate(sp["folds"]):
        tr_ids = {sp["specimens"][i]["id"] for i in fold["train"]}
        mtr = np.array([s in tr_ids for s in spec])
        te_specs = [sp["specimens"][i]["id"] for i in fold["test"]]
        te_rows = np.array([spec2row[s] for s in te_specs if s in spec2row])
        te_pos = [j for j, s in enumerate(te_specs) if s in spec2row]

        Xtr, ytr = X[mtr], y[mtr]
        Xte = X[te_rows]

        rng = np.random.default_rng(args.seed + k)

        # ---- 拟合 1:现有基线(class_weight=balanced) ----
        clf_b = LogisticRegression(C=args.c, max_iter=3000, class_weight="balanced")
        clf_b.fit(Xtr, ytr)
        Pb = softmax(Xte @ clf_b.coef_.T + clf_b.intercept_)

        # ---- 拟合 2:类别均衡采样重训(class_weight=None) ----
        bidx = class_balanced_index(ytr, rng)
        clf_c = LogisticRegression(C=args.c, max_iter=3000)
        clf_c.fit(Xtr[bidx], ytr[bidx])
        Wc = clf_c.coef_
        Pc = softmax(Xte @ Wc.T + clf_c.intercept_)

        got = {
            "base": Pb,
            "base_tau1": softmax(Xte @ tau_normalize(clf_b.coef_, 1.0).T + clf_b.intercept_),
            "cbs": Pc,
            "cbs_tau05": softmax(Xte @ tau_normalize(Wc, 0.5).T + clf_c.intercept_),
            "cbs_tau1": softmax(Xte @ tau_normalize(Wc, 1.0).T + clf_c.intercept_),
        }

        n_tr_spec = len(tr_ids)
        n_te = len(te_rows)
        print(f"fold {k}: train {n_tr_spec} specimens / {mtr.sum()} rows, "
              f"balanced sample {len(bidx)}, test {n_te}")
        for v in variants:
            acc[v][k] = (got[v], te_pos, te_specs)

    os.makedirs(args.outdir, exist_ok=True)
    cidx = {c: i for i, c in enumerate(class_names)}
    labels = np.array([cidx[s["species"]] for s in sp["specimens"]], dtype=np.int64)

    for v in variants:
        n_tot = len(sp["specimens"])
        probs = np.zeros((n_tot, len(class_names)), dtype=np.float32)
        fold_arr = np.zeros(n_tot, dtype=np.int64)
        for k, (P, te_pos, te_specs) in acc[v].items():
            for local_j, gpos in enumerate(te_pos):
                gidx = sp["folds"][k]["test"][gpos]
                probs[gidx] = P[local_j]
                fold_arr[gidx] = k

        tag = f"crt_{v}"
        p = os.path.join(args.outdir, tag + ".npz")
        np.savez_compressed(
            p,
            probs=probs, labels=labels,
            class_names=np.array(class_names),
            specimen=np.array([s["id"] for s in sp["specimens"]]),
            fold=fold_arr,
            genus=np.array([s["genus"] for s in sp["specimens"]]),
            is_damaged=np.array([s["is_damaged"] for s in sp["specimens"]], dtype=np.int64),
            meta=json.dumps({"name": tag, "backbone": "dinov3l (frozen)",
                             "train": "classifier only", "variant": v, "C": args.c,
                             "splits_sha256": sp.get("sha256", "")}, ensure_ascii=False),
        )
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
