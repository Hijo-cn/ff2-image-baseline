#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
冻结特征 → 统一 preds 格式(护栏基线)

作用:
    用 splits.json 的固定划分,在冻结特征上跑 5 折,输出一个 out-of-fold 预测文件,
    交给 evaluation.py 评测。这样护栏数字和其它人的微调结果走完全相同的评测链路。

输出:
    preds/frozen_<tag>_oof.npz,含 evaluation.py 需要的全部字段

用法:
    python ff2_frozen_to_preds.py \
        --feats feats_aug_dinov3l_512_x10_all.npz \
        --splits splits/splits_v1.json \
        --out preds/frozen_dinov3l_oof.npz
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

ROOT = os.environ.get("FF2_ROOT", "")
FEATDIR = os.path.join(ROOT, "features")

POOL = {"dinov3l": "cls+patch", "dinov2l": "cls+patch",
        "in21k": "cls+patch", "siglip2l": "cls+patch",
        "bioclip2": "cls", "convnextl": "cls", "efficientnetb7": "cls"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--c", type=float, default=100.0)
    args = ap.parse_args()

    fp = args.feats if os.path.isabs(args.feats) else os.path.join(FEATDIR, args.feats)
    tag = os.path.basename(fp).replace("feats_aug_", "").replace("_x10_all", "").replace(".npz", "")
    pool = POOL.get(tag.split("_")[0], "cls+patch")

    splits = json.load(open(args.splits, encoding="utf-8"))
    print(f"splits: {args.splits}  sha256={splits.get('sha256','?')[:16]}...  "
          f"folds={splits['n_folds']}")

    d = np.load(fp, allow_pickle=True)
    X = {"cls": d["cls"], "patch": d["patchmean"]}.get(
        pool, np.concatenate([d["cls"], d["patchmean"]], axis=1))
    X = normalize(np.asarray(X, dtype=np.float64), norm="l2")

    y = d["sp_idx"].astype(int)
    class_names = [str(x) for x in d["sp_uniq"]]
    genus = np.array([str(x) for x in d["genus"]])
    spec = np.array([str(x) for x in d["specimen"]])
    dmg = d["is_damaged"].astype(int)
    aug = d["aug_id"].astype(int) if "aug_id" in d.files else np.zeros(len(y), dtype=int)

    # splits 的 class_names 顺序应与特征一致
    if splits["class_names"] != class_names:
        print("⚠️ splits 与特征的 class_names 不完全一致,按特征顺序继续")
        print(f"   splits 前 3:{splits['class_names'][:3]}")
        print(f"   特征   前 3:{class_names[:3]}")

    # 标本级信息(取每个标本的原图行)
    is_orig = aug == 0
    spec2row = {}
    for i in np.where(is_orig)[0]:
        spec2row[spec[i]] = i

    n_spec_expected = len(splits["specimens"])
    missing = [s["id"] for s in splits["specimens"] if s["id"] not in spec2row]
    if missing:
        print(f"⚠️ splits 里有 {len(missing)} 个标本在特征文件中找不到: {missing[:5]}")

    n = len(splits["specimens"])
    probs = np.zeros((n, len(class_names)), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    fold_arr = np.zeros(n, dtype=np.int64)
    spec_arr = np.empty(n, dtype=object)
    genus_arr = np.empty(n, dtype=object)
    dmg_arr = np.zeros(n, dtype=np.int64)

    for k, fold in enumerate(splits["folds"]):
        te_specs = [splits["specimens"][i]["id"] for i in fold["test"]]
        tr_specs = set(splits["specimens"][i]["id"] for i in fold["train"])

        mtr = np.array([s in tr_specs for s in spec])
        te_rows = np.array([spec2row[s] for s in te_specs if s in spec2row])
        te_pos = np.array([i for i, s in enumerate(te_specs) if s in spec2row])

        clf = LogisticRegression(C=args.c, max_iter=3000,
                                 class_weight="balanced").fit(X[mtr], y[mtr])
        pr = clf.predict_proba(X[te_rows])

        # 对齐到全局标本顺序
        fold_spec_ids = [splits["specimens"][i]["id"] for i in fold["test"]]
        for local_j, gpos in enumerate(te_pos):
            gidx = fold["test"][gpos]
            probs[gidx] = pr[local_j]
            labels[gidx] = y[te_rows[local_j]]
            fold_arr[gidx] = k
            spec_arr[gidx] = splits["specimens"][gidx]["id"]
            genus_arr[gidx] = splits["specimens"][gidx]["genus"]
            dmg_arr[gidx] = splits["specimens"][gidx]["is_damaged"]

        print(f"  fold {k}: train {mtr.sum()} rows / {len(tr_specs)} specimens, "
              f"test {len(te_rows)} specimens")

    # 检查每个标本都被预测了一次
    got = (probs.sum(axis=1) > 0)
    if not got.all():
        print(f"⚠️ 有 {int((~got).sum())} 个标本没有被任何一折预测到")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    meta = {"name": f"frozen_{tag}", "backbone": tag, "train": "linear probe (frozen)",
            "C": args.c, "pool": pool, "aug": "x10_all",
            "splits_sha256": splits.get("sha256", "")}
    np.savez_compressed(
        args.out,
        probs=probs.astype(np.float32),
        labels=labels,
        class_names=np.array(class_names),
        specimen=np.array(spec_arr.astype(str)),
        fold=fold_arr,
        genus=np.array(genus_arr.astype(str)),
        is_damaged=dmg_arr,
        meta=json.dumps(meta, ensure_ascii=False),
    )
    print(f"\nwrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")
    print(f"  meta: {meta}")


if __name__ == "__main__":
    main()
