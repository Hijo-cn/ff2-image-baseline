#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多视角池化 / 融合实验

背景
----
HISNET-FF 的真实设置:每个标本 4 张颅骨图(背/腹/侧/下颌),用 ImageFolder
平铺成独立样本训练,**单图输入**;测试时对每张图分别预测、分别计分。
牙齿每标本 1 张,文件名与腹面相同,所以"融合"实际是 腹面 + 牙齿。

本脚本在同一套固定划分下比较:

    单视角      腹面 / 背面 / 侧面 / 下颌 / 牙齿
    池化        多个视角一起训练,得到一个模型
    晚期融合    每个视角一个模型,预测时概率平均

并同时报两种口径:

    image_level     每张图单独计分          ← 对齐 FF 的 93.9
    specimen_level  同一标本多张图概率平均   ← 对齐我们的 84.18

用法
----
    python -u ff2_multiview.py --splits splits/splits_v1.json \
        --outdir preds/mv --aug-per-spec 3
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
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

CRANIUM = ["ventral", "dorsal", "lateral", "mand"]
CRANIUM3 = ["ventral", "dorsal", "lateral"]          # 去掉下颌(单独最弱)

CONFIGS = [
    ("ventral_only", ["ventral"], "pool"),
    ("dorsal_only", ["dorsal"], "pool"),
    ("lateral_only", ["lateral"], "pool"),
    ("mand_only", ["mand"], "pool"),
    ("teeth_only", ["teeth"], "pool"),
    ("cranium4_pool", CRANIUM, "pool"),
    ("cranium4_late", CRANIUM, "late"),
    ("cranium4_teeth_pool", CRANIUM + ["teeth"], "pool"),
    ("cranium4_teeth_late", CRANIUM + ["teeth"], "late"),
    ("ventral_teeth_pool", ["ventral", "teeth"], "pool"),
    # ---- 消融:去掉下颌 ----
    ("cranium3_late", CRANIUM3, "late"),
    ("cranium3_teeth_late", CRANIUM3 + ["teeth"], "late"),
    # ---- 消融:按内层验证精度加权融合 ----
    ("cranium4_weighted", CRANIUM, "weighted"),
    ("cranium4_teeth_weighted", CRANIUM + ["teeth"], "weighted"),
]


def load_view(name, class_names, aug_per_spec, seed=0):
    """读一个视角的特征 → {specimen: (X, label)}"""
    p = os.path.join(FEATDIR, FILES[name])
    if not os.path.exists(p):
        print(f"  ⚠️ 缺少 {name}: {p}")
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

    by_spec = defaultdict(list)
    for i in range(len(spec)):
        by_spec[spec[i]].append(i)

    rng = np.random.default_rng(seed)
    out = {}
    for sid, rows in by_spec.items():
        order = np.array(rows)
        orig = order[aug[order] == 0]
        rest = order[aug[order] > 0]
        keep = max(aug_per_spec - 1, 0)
        if len(rest) > keep:
            rest = rng.choice(rest, keep, replace=False)
        sel = np.concatenate([orig[:1], rest]) if keep else orig[:1]
        sel = sel[lab[sel] >= 0]
        if len(sel) == 0:
            continue
        out[sid] = (X[sel], int(lab[sel[0]]))
    return out


def stack(items):
    X = np.concatenate([x for x, _ in items], axis=0)
    y = np.concatenate([[lab] * len(x) for x, lab in items])
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits/splits_v1.json")
    ap.add_argument("--outdir", default="preds/mv")
    ap.add_argument("--aug-per-spec", type=int, default=3)
    ap.add_argument("--c", type=float, default=100.0)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="",
                    help="只跑指定配置(逗号分隔),例如 cranium4_teeth_late,cranium4_late")
    args = ap.parse_args()

    sp = json.load(open(args.splits, encoding="utf-8"))
    class_names = sp["class_names"]
    C = len(class_names)
    folds = sp["folds"]

    print(f"读取特征(每标本每视角 {args.aug_per_spec} 条)...")
    view_data = {}
    for v in FILES:
        vd = load_view(v, class_names, args.aug_per_spec, args.seed)
        view_data[v] = vd
        n = sum(len(x[0]) for x in vd.values())
        print(f"  {v:8s}: {len(vd):4d} 个标本, {n:5d} 条特征")

    def fit(X, y):
        return LogisticRegression(C=args.c, max_iter=args.max_iter,
                                  class_weight="balanced").fit(X, y)

    results = []
    only = set(x.strip() for x in args.only.split(",") if x.strip())
    # specimen -> 所属折(用于保存 OOF 文件)
    spec_fold = {}
    for k, fold in enumerate(folds):
        for i in fold["test"]:
            spec_fold[sp["specimens"][i]["id"]] = k

    for name, views, mode in CONFIGS:
        if only and name not in only:
            continue
        if any(not view_data.get(v) for v in views):
            print(f"[{name}] 跳过:缺视角")
            continue

        img_true, img_pred = [], []
        spec_probs = defaultdict(list)
        spec_lab = {}

        for k, fold in enumerate(folds):
            tr = {sp["specimens"][i]["id"] for i in fold["train"]}
            te = [sp["specimens"][i]["id"] for i in fold["test"]]

            w = None
            if mode == "weighted":
                # 在训练折内部再切一份,用来估计各视角的权重(不碰测试集)
                tl = sorted(tr)
                rng = np.random.default_rng(args.seed + 1000 + k)
                rng.shuffle(tl)
                nb = max(1, len(tl) // 5)
                tb = set(tl[:nb])
                ta = set(tl[nb:])
                w = {}
                for v in views:
                    it = [view_data[v][s] for s in ta if s in view_data[v]]
                    if not it:
                        w[v] = 1e-3
                        continue
                    clf_v = fit(*stack(it))
                    hit = []
                    for s in tb:
                        if s not in view_data[v]:
                            continue
                        Xs, lab = view_data[v][s]
                        hit.append(int(clf_v.predict_proba(Xs).mean(0).argmax() == lab))
                    w[v] = max(np.mean(hit) - 1.0 / C, 1e-3) if hit else 1e-3
                tw = sum(w.values())
                w = {kk: vv / tw for kk, vv in w.items()}
                print(f"    fold{k} 视角权重: "
                      + "  ".join(f"{kk}={vv:.3f}" for kk, vv in w.items()), flush=True)

            if mode == "pool":
                items = [view_data[v][s] for v in views for s in tr if s in view_data[v]]
                clf = fit(*stack(items))
                get = lambda v: clf
            else:
                clfs = {}
                for v in views:
                    it = [view_data[v][s] for s in tr if s in view_data[v]]
                    clfs[v] = fit(*stack(it))
                get = lambda v: clfs[v]

            for s in te:
                for v in views:
                    if s not in view_data[v]:
                        continue
                    Xs, lab = view_data[v][s]
                    P = get(v).predict_proba(Xs)
                    pv = P.mean(axis=0)
                    spec_probs[s].append(w[v] * pv if w else pv)
                    img_pred.extend(P.argmax(1).tolist())
                    img_true.extend([lab] * len(Xs))
                    spec_lab[s] = lab

        S = np.stack([np.mean(spec_probs[s], axis=0) for s in spec_probs])
        yS = np.array([spec_lab[s] for s in spec_probs])
        img_true = np.array(img_true)
        img_pred = np.array(img_pred)
        predS = S.argmax(1)

        r = {
            "name": name, "mode": mode, "views": views,
            "n_img": int(len(img_true)), "n_spec": int(len(S)),
            "img_top1": float((img_pred == img_true).mean()),
            "img_macro_f1": float(f1_score(img_true, img_pred, average="macro",
                                           labels=list(range(C)), zero_division=0)),
            "spec_top1": float((predS == yS).mean()),
            "spec_macro_f1": float(f1_score(yS, predS, average="macro",
                                            labels=list(range(C)), zero_division=0)),
        }
        results.append(r)
        print(f"[{name:22s}] img {r['img_top1']*100:5.1f}%  spec {r['spec_top1']*100:5.1f}%  "
              f"(spec n={r['n_spec']}, img n={r['n_img']})", flush=True)

        # 保存标本级 OOF 预测(供 evaluation.py 出逐属/逐类报告)
        os.makedirs(args.outdir, exist_ok=True)
        n_all = len(sp["specimens"])
        P = np.zeros((n_all, C), dtype=np.float32)
        got = np.zeros(n_all, dtype=bool)
        sid2idx = {s["id"]: i for i, s in enumerate(sp["specimens"])}
        for sid, plist in spec_probs.items():
            gi = sid2idx[sid]
            P[gi] = np.mean(plist, axis=0)
            got[gi] = True
        ytrue = np.array([class_names.index(s["species"]) if s["species"] in class_names else -1
                          for s in sp["specimens"]], dtype=np.int64)
        ytrue[~got] = -1
        keep = got & (ytrue >= 0)
        npz = os.path.join(args.outdir, f"mv_{name}_spec.npz")
        np.savez_compressed(
            npz,
            probs=P[keep], labels=ytrue[keep],
            class_names=np.array(class_names),
            specimen=np.array([s["id"] for s, k in zip(sp["specimens"], keep) if k]),
            fold=np.array([spec_fold.get(s["id"], -1) for s, k in zip(sp["specimens"], keep) if k],
                          dtype=np.int64),
            genus=np.array([s["genus"] for s, k in zip(sp["specimens"], keep) if k]),
            is_damaged=np.array([s["is_damaged"] for s, k in zip(sp["specimens"], keep) if k],
                                dtype=np.int64),
            meta=json.dumps({"name": f"mv_{name}", "mode": mode, "views": views,
                             "backbone": "dinov3l (frozen)", "fusion": "late" if mode == "late" else "pool",
                             "aug_per_spec": args.aug_per_spec,
                             "splits_sha256": sp.get("sha256", "")}, ensure_ascii=False),
        )
        print(f"    wrote {npz}")

    print("\n" + "=" * 84)
    print(f"{'配置':24s} {'图片级 top-1':>13s} {'标本级 top-1':>13s} {'标本级 macroF1':>15s}")
    print("-" * 84)
    for r in sorted(results, key=lambda x: -x["spec_top1"]):
        print(f"{r['name']:24s} {r['img_top1']*100:12.1f}% {r['spec_top1']*100:12.1f}% "
              f"{r['spec_macro_f1']:15.4f}")
    print("=" * 84)

    os.makedirs(args.outdir, exist_ok=True)
    jp = os.path.join(args.outdir, "multiview_results.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {jp}")


if __name__ == "__main__":
    main()
