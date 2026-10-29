#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把逐折预测合并成一个 out-of-fold 文件

train.py 每折输出一个 preds/<tag>_fold<k>.npz。要算 5 折均值±标准差,
需要把它们拼成一个"每个标本恰好被预测一次"的文件,再交给 evaluation.py。

用法:
    python ff2_merge_folds.py --splits splits/splits_v1.json \
        --preds "preds/dino_lpft5e5_fold*.npz" \
        --out preds/dino_lpft5e5_oof.npz
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sp = json.load(open(args.splits, encoding="utf-8"))
    class_names = sp["class_names"]
    cidx = {c: i for i, c in enumerate(class_names)}
    n = len(sp["specimens"])

    probs = np.zeros((n, len(class_names)), dtype=np.float32)
    fold_arr = np.full(n, -1, dtype=np.int64)
    filled = np.zeros(n, dtype=bool)

    files = sorted(glob.glob(args.preds))
    if not files:
        raise SystemExit(f"没有匹配到文件: {args.preds}")

    # splits 里 specimen id -> 全局下标
    sid2idx = {s["id"]: i for i, s in enumerate(sp["specimens"])}
    meta_any = {}

    for f in files:
        d = np.load(f, allow_pickle=True)
        if "meta" in d.files:
            try:
                meta_any = json.loads(str(d["meta"]))
            except Exception:
                pass
        fold = int(np.unique(d["fold"])[0]) if "fold" in d.files else -1
        cn = [str(x) for x in d["class_names"]]
        assert cn == class_names, f"{f} 的 class_names 与 splits 不一致"
        P = np.asarray(d["probs"], dtype=np.float32)
        ids = [str(x) for x in d["specimen"]]
        k = 0
        for sid in ids:
            gi = sid2idx[sid]
            probs[gi] = P[k]
            fold_arr[gi] = fold
            filled[gi] = True
            k += 1
        print(f"  {os.path.basename(f):44s} fold={fold}  {len(ids)} 个样本")

    missing = int((~filled).sum())
    print(f"\n覆盖 {int(filled.sum())}/{n} 个标本" + (f",缺 {missing}" if missing else ""))
    if missing:
        raise SystemExit("有标本没有被任何一折预测到,检查 glob 是否漏了折")

    labels = np.array([cidx[s["species"]] for s in sp["specimens"]], dtype=np.int64)
    meta_any["name"] = os.path.splitext(os.path.basename(args.out))[0]
    meta_any["splits_sha256"] = sp.get("sha256", "")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        probs=probs, labels=labels,
        class_names=np.array(class_names),
        specimen=np.array([s["id"] for s in sp["specimens"]]),
        fold=fold_arr,
        genus=np.array([s["genus"] for s in sp["specimens"]]),
        is_damaged=np.array([s["is_damaged"] for s in sp["specimens"]], dtype=np.int64),
        meta=json.dumps(meta_any, ensure_ascii=False),
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
