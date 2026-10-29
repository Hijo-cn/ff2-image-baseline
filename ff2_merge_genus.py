#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
把「按属分解」的多模型预测合并成一个统一口径的概率矩阵

用途
----
HISNET-FF 的结构是"先判属,再在属内判种"。本脚本复刻它的**属内部分**:
把 7 个多型属各自的模型输出,拼回一张 51 类的概率矩阵,再交给 evaluation.py。

为了让"属内微调"和"冻结基线"可比,两种产物都做 **oracle 属掩码**:
只保留真实属对应的那些物种列,其余置零。
这样 top-1 就等于"属内条件精度",和之前报的 hier_oracle 同口径。

两种模式
--------
1. 合并按属模型:
   python ff2_merge_genus.py --splits splits/splits_v1.json \
       --preds "preds/dino_ftgenus_*_fold0.npz" \
       --out preds/dino_genus_oracle_fold0.npz

2. 把已有的 51 类预测做同样的 oracle 掩码(作为对照):
   python ff2_merge_genus.py --splits splits/splits_v1.json \
       --mask preds/frozen_dinov3l_oof.npz --fold 0 \
       --out preds/frozen_dinov3l_oracle_fold0.npz
"""
import argparse
import glob
import json
import os

import numpy as np


def load_splits(path, fold):
    sp = json.load(open(path, encoding="utf-8"))
    class_names = sp["class_names"]
    g2s = {}
    for s in sp["specimens"]:
        g2s.setdefault(s["genus"], set()).add(s["species"])
    test_ids = [sp["specimens"][i]["id"] for i in sp["folds"][fold]["test"]]
    meta = {s["id"]: s for s in sp["specimens"]}
    return sp, class_names, g2s, test_ids, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--preds", default="", help="按属模型的 glob(单折)")
    ap.add_argument("--mask", default="", help="把已有的 51 类预测做 oracle 掩码")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sp, class_names, g2s, test_ids, meta = load_splits(args.splits, args.fold)
    cidx = {c: i for i, c in enumerate(class_names)}
    C = len(class_names)

    if args.mask:
        # ---------- 模式 2:给已有 51 类预测做 oracle 掩码 ----------
        d = np.load(args.mask, allow_pickle=True)
        probs = np.asarray(d["probs"], dtype=np.float32)
        spec = [str(x) for x in d["specimen"]]
        ncl = [str(x) for x in d["class_names"]]
        assert ncl == class_names, "class_names 顺序和 splits 不一致"

        rows, keep = [], []
        test_set = set(test_ids)
        for i, sid in enumerate(spec):
            if sid not in test_set:              # 只取当前 fold 的测试标本
                continue
            g = meta[sid]["genus"]
            if len(g2s.get(g, [])) < 2:          # 单型属不参与属内比较
                continue
            p = np.zeros(C, dtype=np.float32)
            for s_name in g2s[g]:
                p[cidx[s_name]] = probs[i, cidx[s_name]]
            rows.append(p)
            keep.append(i)
        if not rows:
            raise SystemExit("没有可用的样本")
        P = np.stack(rows)
        spec_out = [spec[i] for i in keep]
        labels = np.array([cidx[meta[s]["species"]] for s in spec_out], dtype=np.int64)
        print(f"oracle 掩码: {len(spec_out)} 个多型属测试样本")
    else:
        # ---------- 模式 1:合并按属模型 ----------
        files = sorted(glob.glob(args.preds))
        if not files:
            raise SystemExit(f"没有匹配到文件: {args.preds}")
        per_spec = {}      # specimen -> (probs over its genus species, local class names)
        folds = set()
        for f in files:
            d = np.load(f, allow_pickle=True)
            local_names = [str(x) for x in d["class_names"]]
            g = None
            if "meta" in d.files:
                try:
                    g = json.loads(str(d["meta"])).get("genus_mode")
                except Exception:
                    g = None
            if not g or g.startswith("("):
                g = local_names[0].split(" ", 1)[0]
            if "fold" in d.files:
                folds.update(np.unique(d["fold"]).tolist())
            print(f"  {os.path.basename(f):48s} genus={g:14s} classes={len(local_names)}")
            probs = np.asarray(d["probs"], dtype=np.float32)
            # 标本列表以 splits 为准重建(train.py 早期版本 specimen 字段未按属过滤)
            cand = [sid for sid in test_ids if meta[sid]["genus"] == g]
            if len(cand) == len(probs):
                ids = cand
            else:
                ids = ([str(x) for x in d["specimen"]]
                       if "specimen" in d.files and len(d["specimen"]) == len(probs)
                       else None)
                if ids is None:
                    raise SystemExit(
                        f"{os.path.basename(f)}: probs {len(probs)} 行,但 splits 里属 {g} "
                        f"的测试标本 {len(cand)} 个,specimen 字段也不匹配 —— 无法对齐")
            for i, sid in enumerate(ids):
                per_spec[sid] = (probs[i], local_names)

        if len(folds) > 1:
            raise SystemExit(f"glob 里混了多个 fold: {sorted(folds)}")

        spec_out, rows = [], []
        for sid in test_ids:
            if sid not in per_spec:
                continue
            p_local, local_names = per_spec[sid]
            p = np.zeros(C, dtype=np.float32)
            for j, nm in enumerate(local_names):
                if nm in cidx:
                    p[cidx[nm]] = p_local[j]
            rows.append(p)
            spec_out.append(sid)
        if not rows:
            raise SystemExit("合并后没有任何样本")
        P = np.stack(rows)
        labels = np.array([cidx[meta[s]["species"]] for s in spec_out], dtype=np.int64)
        print(f"合并: {len(spec_out)} 个样本,覆盖 {len({meta[s]['genus'] for s in spec_out})} 个属")

    # oracle 掩码:只保留真实属的物种列
    for i, sid in enumerate(spec_out):
        g = meta[sid]["genus"]
        allowed = {cidx[s] for s in g2s[g]}
        for c in range(C):
            if c not in allowed:
                P[i, c] = 0.0
        P[i] /= max(P[i].sum(), 1e-12)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        probs=P.astype(np.float32),
        labels=labels,
        class_names=np.array(class_names),
        specimen=np.array(spec_out),
        fold=np.full(len(spec_out), args.fold, dtype=np.int64),
        genus=np.array([meta[s]["genus"] for s in spec_out]),
        is_damaged=np.array([meta[s]["is_damaged"] for s in spec_out], dtype=np.int64),
        meta=json.dumps({"name": os.path.splitext(os.path.basename(args.out))[0],
                         "protocol": "oracle-genus routing",
                         "fold": args.fold,
                         "n_polytypic_test": len(spec_out)},
                        ensure_ascii=False),
    )
    print(f"\nwrote {args.out}  ({os.path.getsize(args.out)/1e6:.2f} MB)")
    print("口径提醒:这是 oracle 属路由下的属内条件精度,"
          "和 evaluation.py 报的 hier_oracle 同口径,不能和 51 类扁平 top-1 直接比。")


if __name__ == "__main__":
    main()
