#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FF2 图像基线 —— 探针评测(支持增强特征)

输入: features/feats_*.npz(有无 base_idx/aug_id 都能跑)
输出: reports/probe_<tag>.json + 终端表格

口径说明:
  - hier_or  = 已知正确属(oracle 路由)下的物种精度 —— 和论文 93.9% 可比
  - hier_e2e = 真实端到端(属判错则种必错)
  - 超参 CV 按 base_idx 分组,增强副本不会跨折泄漏
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

ROOT = os.environ.get("FF2_ROOT", "/mnt/nfs_share/EC_DATABASE_UGREEN/yanhaiqiao/FF2")
FEATDIR = os.path.join(ROOT, "features")
REPDIR = os.path.join(ROOT, "reports")

GRID = [3.0, 10.0, 30.0, 100.0, 300.0]
CW = "balanced"


def l2(x):
    return normalize(np.asarray(x, dtype=np.float64), norm="l2")


def acc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((a == b).mean()) if len(a) else float("nan")


def make_lr(C):
    return LogisticRegression(C=C, max_iter=3000, class_weight=CW)


def grouped_cv(X, y, groups, C, n_splits=3):
    k = min(n_splits, len(np.unique(groups)))
    if k < 2:
        return float("nan")
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0)
    scores = []
    for tr, va in skf.split(X, y, groups):
        clf = make_lr(C).fit(X[tr], y[tr])
        scores.append((clf.predict(X[va]) == y[va]).mean())
    return float(np.mean(scores))


def proto_predict(Xtr, ytr, Xte, n_class, dim):
    C = np.zeros((n_class, dim), dtype=np.float64)
    for c in np.unique(ytr):
        C[c] = Xtr[ytr == c].mean(axis=0)
    C = normalize(C, norm="l2")
    return np.argmax(Xte @ C.T, axis=1)


def nn_top1(Xtr, ytr, Xte, yte):
    S = Xte @ Xtr.T
    pred = ytr[np.argmax(S, axis=1)]
    return acc(yte, pred)


def mlp_probe(Xtr, ytr, Xte, yte, hidden=512, alpha=1e-3):
    """一层隐层的小 MLP。特征是 L2 归一化的,先按 train 统计做标准化。"""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=alpha,
                        max_iter=400, early_stopping=True,
                        n_iter_no_change=20, random_state=0)
    clf.fit(sc.transform(Xtr), ytr)
    return acc(yte, clf.predict(sc.transform(Xte)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--cw", default="balanced", choices=["balanced", "none"])
    ap.add_argument("--mlp", action="store_true", help="额外跑一个 MLP 探针(较慢)")
    args = ap.parse_args()

    global CW
    CW = None if args.cw == "none" else args.cw

    os.makedirs(REPDIR, exist_ok=True)
    p = args.feats
    if not os.path.isabs(p) and not os.path.exists(p):
        p = os.path.join(FEATDIR, p)
    d = np.load(p, allow_pickle=True)
    tag = os.path.splitext(os.path.basename(p))[0]
    print(f"loaded {p}")

    split = d["split"].astype(str)
    y_sp = d["sp_idx"].astype(int)
    y_gen = d["g_idx"].astype(int)
    sp_uniq = [str(x) for x in d["sp_uniq"]]
    g_uniq = [str(x) for x in d["g_uniq"]]
    dmg = d["is_damaged"].astype(int)
    N_SP, N_G = len(sp_uniq), len(g_uniq)

    if "base_idx" in d.files:
        base = d["base_idx"].astype(int)
        aug = d["aug_id"].astype(int)
    else:
        base = np.arange(len(split))
        aug = np.zeros(len(split), dtype=int)

    tr = split == "train"
    te = (split == "test") & (aug == 0)
    print(f"  train reps {tr.sum()} (from {len(np.unique(base[tr]))} base imgs)")
    print(f"  test       {te.sum()}  damaged {dmg[te].sum()}/{te.sum()}")
    print(f"  class_weight = {CW}")

    pools = {"cls": d["cls"], "patch": d["patchmean"]}
    pools["cls+patch"] = np.concatenate([d["cls"], d["patchmean"]], axis=1)

    results = {}
    for pname, X in pools.items():
        X = l2(X)
        r = {"dim": int(X.shape[1])}

        ps = proto_predict(X[tr], y_sp[tr], X[te], N_SP, X.shape[1])
        pg = proto_predict(X[tr], y_gen[tr], X[te], N_G, X.shape[1])
        r["prototype"] = {
            "species": acc(y_sp[te], ps),
            "genus": acc(y_gen[te], pg),
            "damaged": acc(y_sp[te][dmg[te] == 1], ps[dmg[te] == 1])
            if (dmg[te] == 1).any() else None,
        }

        groups = base[tr]
        cvs = {c: grouped_cv(X[tr], y_sp[tr], groups, c) for c in GRID}
        bestc = max(cvs, key=lambda c: (cvs[c] if not np.isnan(cvs[c]) else -1))
        clf_sp = make_lr(bestc).fit(X[tr], y_sp[tr])
        clf_gen = make_lr(bestc).fit(X[tr], y_gen[tr])
        ps2 = clf_sp.predict(X[te])
        pg2 = clf_gen.predict(X[te])
        r["linear"] = {
            "best_C": bestc, "cv": cvs,
            "species": acc(y_sp[te], ps2),
            "genus": acc(y_gen[te], pg2),
            "damaged": acc(y_sp[te][dmg[te] == 1], ps2[dmg[te] == 1])
            if (dmg[te] == 1).any() else None,
        }

        if args.mlp:
            print(f"  [{pname}] MLP probe ...", flush=True)
            r["mlp"] = {
                "species": mlp_probe(X[tr], y_sp[tr], X[te], y_sp[te]),
                "genus": mlp_probe(X[tr], y_gen[tr], X[te], y_gen[te]),
            }

        r["nn_species"] = nn_top1(X[tr], y_sp[tr], X[te], y_sp[te])
        r["nn_genus"] = nn_top1(X[tr], y_gen[tr], X[te], y_gen[te])

        # 层级:oracle 属路由 + 真实端到端
        oracle = np.full(int(te.sum()), -1)
        for g in np.unique(y_gen[tr]):
            mtr = y_gen[tr] == g
            mte = y_gen[te] == g
            if mte.sum() == 0:
                continue
            cls_in = np.unique(y_sp[tr][mtr])
            if len(cls_in) == 1:
                pred = np.full(int(mte.sum()), cls_in[0])
            else:
                clf = make_lr(max(bestc, 1.0)).fit(X[tr][mtr], y_sp[tr][mtr])
                pred = clf.predict(X[te][mte])
            oracle[mte] = pred
        sp_te = y_sp[te]
        r["hier_oracle_genus"] = acc(sp_te, oracle)
        r["hier_end_to_end"] = acc(sp_te, np.where(pg2 == y_gen[te], oracle, -1))

        per = {}
        for g in np.unique(y_gen[te]):
            m = y_gen[te] == g
            per[g_uniq[g]] = {"n": int(m.sum()), "acc": acc(sp_te[m], oracle[m])}
        r["per_genus"] = per
        results[pname] = r

    print()
    print(f"{'pooling':10s} {'probe':10s} {'sp':>7s} {'gen':>7s} {'dmg':>7s} "
          f"{'hier_or':>8s} {'hier_e2e':>9s} {'nn1':>6s} {'nnG':>6s}")
    print("-" * 84)
    for pname, r in results.items():
        for probe in ("prototype", "linear", "mlp"):
            if probe not in r:
                continue
            m = r[probe]
            dv = m.get("damaged")
            print(f"{pname:10s} {probe:10s} "
                  f"{m['species']*100:6.1f}% {m['genus']*100:6.1f}% "
                  f"{(dv*100 if dv is not None else float('nan')):6.1f}% "
                  f"{r['hier_oracle_genus']*100:7.1f}% {r['hier_end_to_end']*100:8.1f}% "
                  f"{r['nn_species']*100:5.1f}% {r['nn_genus']*100:5.1f}%")

    best = max(results, key=lambda k: results[k]["linear"]["species"])
    bl = results[best]["linear"]
    print(f"\nbest pooling: {best}   species {bl['species']*100:.1f}%  "
          f"genus {bl['genus']*100:.1f}%  C={bl['best_C']}")
    print("  CV grid:", {k: round(v * 100, 1) for k, v in bl["cv"].items()})
    if bl["best_C"] == max(GRID):
        print("  ⚠ best_C 撞网格上界,建议继续扩大")

    print("\nper-genus (oracle-genus routing):")
    for g, v in sorted(results[best]["per_genus"].items(), key=lambda x: x[1]["acc"]):
        print(f"  {g:16s} n={v['n']:3d}  {v['acc']*100:6.1f}%")

    suffix = "" if CW == "balanced" else "_noweight"
    out = args.out or os.path.join(REPDIR, f"probe_{tag}{suffix}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"feats": p, "tag": tag, "cw": CW, "results": results}, fh,
                  ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
