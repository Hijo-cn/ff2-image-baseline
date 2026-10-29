#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多视角 Stacking 融合(替代概率平均)

动机
----
简单概率平均隐含假设:所有视角对所有类别同等可靠。
但实际是"牙齿对 Talpa 有用、对 Euroscaptor 没用"。
Stacking 把 5 个视角的概率拼成 5×C 的元特征,训一个二级分类器,
由它自己学出各个视角在各种情况下的可靠度。

防泄漏
------
外层:固定 5 折(splits.json),测试标本绝不参与训练
内层:在训练折内部再切 5 折,得到训练标本的**无偏**元特征
     (否则一级模型在训练集上的概率是过拟合的,二级会被带偏)

用法
----
    python -u ff2_stacking.py --splits splits/splits_v1.json \
        --out preds/stack_frozen_oof.npz --aug-per-spec 3
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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


def view_probs(vd, spec_ids, clf):
    """按标本平均概率"""
    P = np.zeros((len(spec_ids), clf.classes_.max() + 1))
    for i, s in enumerate(spec_ids):
        if s not in vd:
            continue
        X, _ = vd[s]
        pr = clf.predict_proba(X).mean(axis=0)
        P[i, clf.classes_] = pr
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits/splits_v1.json")
    ap.add_argument("--out", default="preds/stack_frozen_oof.npz")
    ap.add_argument("--aug-per-spec", type=int, default=3)
    ap.add_argument("--c", type=float, default=100.0)
    ap.add_argument("--meta-c", type=float, default=1.0)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sp = json.load(open(args.splits, encoding="utf-8"))
    class_names = sp["class_names"]
    C = len(class_names)
    views = list(FILES)

    print("读取特征...")
    vd = {v: load_view(v, class_names, args.aug_per_spec, args.seed) for v in views}
    for v in views:
        print(f"  {v:8s}: {len(vd[v]):4d} 标本")

    def fit(X, y):
        return LogisticRegression(C=args.c, max_iter=2000, class_weight="balanced").fit(X, y)

    n = len(sp["specimens"])
    P_out = np.zeros((n, C), dtype=np.float32)
    y_out = np.full(n, -1, dtype=np.int64)
    fold_out = np.zeros(n, dtype=np.int64)

    for k, fold in enumerate(sp["folds"]):
        tr_specs = [sp["specimens"][i]["id"] for i in fold["train"]]
        te_specs = [sp["specimens"][i]["id"] for i in fold["test"]]
        tr_lab = {s: vd[views[0]][s][1] for s in tr_specs if s in vd[views[0]]}
        tr_specs = [s for s in tr_specs if s in tr_lab]
        y_tr = np.array([tr_lab[s] for s in tr_specs])

        # ---------- 内层 OOF:得到训练标本的无偏元特征 ----------
        meta_tr = np.zeros((len(tr_specs), len(views) * C))
        skf = StratifiedKFold(n_splits=args.inner, shuffle=True, random_state=args.seed)
        for itr, iva in skf.split(np.zeros(len(tr_specs)), y_tr):
            s_tr = [tr_specs[i] for i in itr]
            for j, v in enumerate(views):
                items = [vd[v][s] for s in s_tr if s in vd[v]]
                if not items:
                    continue
                X = np.concatenate([x for x, _ in items])
                yy = np.concatenate([[l] * len(x) for x, l in items])
                clf = fit(X, yy)
                for ii in iva:
                    s = tr_specs[ii]
                    if s not in vd[v]:
                        continue
                    Xs, _ = vd[v][s]
                    pr = clf.predict_proba(Xs).mean(axis=0)
                    meta_tr[ii, j * C:(j + 1) * C][clf.classes_] = pr

        meta_clf = LogisticRegression(C=args.meta_c, max_iter=2000,
                                      class_weight="balanced").fit(meta_tr, y_tr)

        # ---------- 外层:全量训练一级模型,预测测试 ----------
        clfs = {}
        for v in views:
            items = [vd[v][s] for s in tr_specs if s in vd[v]]
            X = np.concatenate([x for x, _ in items])
            yy = np.concatenate([[l] * len(x) for x, l in items])
            clfs[v] = fit(X, yy)

        meta_te = np.zeros((len(te_specs), len(views) * C))
        for j, v in enumerate(views):
            for i, s in enumerate(te_specs):
                if s not in vd[v]:
                    continue
                Xs, _ = vd[v][s]
                pr = clfs[v].predict_proba(Xs).mean(axis=0)
                meta_te[i, j * C:(j + 1) * C][clfs[v].classes_] = pr

        pr_te = meta_clf.predict_proba(meta_te)
        for i, gi in enumerate(fold["test"]):
            P_out[gi, meta_clf.classes_] = pr_te[i]
            y_out[gi] = class_names.index(sp["specimens"][gi]["species"])
            fold_out[gi] = k
        acc = (pr_te.argmax(1) == np.array([y_out[i] for i in fold["test"]])).mean()
        print(f"fold {k}: 训练 {len(tr_specs)} 测试 {len(te_specs)}  top-1 {acc*100:.2f}%", flush=True)

    ok = y_out >= 0
    acc = (P_out[ok].argmax(1) == y_out[ok]).mean()
    print(f"\nStacking 5 折合并 top-1 = {acc*100:.2f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    idx = [i for i in range(n) if ok[i]]
    np.savez_compressed(
        args.out,
        probs=P_out[idx], labels=y_out[idx],
        class_names=np.array(class_names),
        specimen=np.array([sp["specimens"][i]["id"] for i in idx]),
        fold=fold_out[idx],
        genus=np.array([sp["specimens"][i]["genus"] for i in idx]),
        is_damaged=np.array([sp["specimens"][i]["is_damaged"] for i in idx], dtype=np.int64),
        meta=json.dumps({"name": "stack_frozen", "fusion": "stacking",
                         "views": views, "aug_per_spec": args.aug_per_spec,
                         "splits_sha256": sp.get("sha256", "")}, ensure_ascii=False),
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
