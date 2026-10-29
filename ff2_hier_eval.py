#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
层级评测:一次算清四种口径,便于和 HISNET-FF 论文对比

口径说明
--------
flat          51 类直接分类(最严)
genus         18 类属级
hier_oracle   已知正确属,属内分种(= 论文的 hierarchical 口径)
hier_e2e      先用预测的属,再在属内分种(端到端,更接近实际部署)

用法
----
    python ff2_hier_eval.py preds/dino_fix21_fused_oof.npz
    python ff2_hier_eval.py preds/a.npz preds/b.npz ...
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    files = sys.argv[1:]

    print(f"{'配置':36s} {'flat':>8s} {'genus':>8s} {'hier_oracle':>12s} {'hier_e2e':>9s}")
    print("-" * 78)

    for f in files:
        if not os.path.exists(f):
            print(f"{os.path.basename(f):36s}  (文件不存在)")
            continue
        d = np.load(f, allow_pickle=True)
        P = np.asarray(d["probs"], dtype=np.float64)
        y = np.asarray(d["labels"]).astype(int)
        names = [str(x) for x in d["class_names"]]
        C = len(names)

        sp2g = {c: c.split(" ", 1)[0] for c in names}
        genera = sorted(set(sp2g.values()))
        gidx = {g: i for i, g in enumerate(genera)}
        M = np.zeros((C, len(genera)))
        for j, c in enumerate(names):
            M[j, gidx[sp2g[c]]] = 1

        yg = np.array([gidx[sp2g[names[i]]] for i in y])
        Pg = P @ M

        flat = (P.argmax(1) == y).mean()
        genus = (Pg.argmax(1) == yg).mean()

        # oracle 属路由:只保留真实属的物种列
        Po = P.copy()
        for i in range(len(y)):
            keep = M[:, yg[i]] > 0
            Po[i, ~keep] = 0.0
            s = Po[i].sum()
            if s > 0:
                Po[i] /= s
        hier_o = (Po.argmax(1) == y).mean()

        # 端到端:预测属 → 该属内取最大
        gp = Pg.argmax(1)
        Pe = P.copy()
        for i in range(len(y)):
            keep = M[:, gp[i]] > 0
            Pe[i, ~keep] = 0.0
        hier_e = (Pe.argmax(1) == y).mean()

        tag = os.path.basename(f).replace(".npz", "")
        print(f"{tag:36s} {flat*100:7.1f}% {genus*100:7.1f}% "
              f"{hier_o*100:11.1f}% {hier_e*100:8.1f}%")

    print("-" * 78)
    print("对照 FF1 论文:颅骨(4视角) 90.5 / 98.8 / 93.9；颅骨+牙齿 96.5(hier)")


if __name__ == "__main__":
    main()
