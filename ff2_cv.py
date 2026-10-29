#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FF2 —— 5 折分组交叉验证(和 HISNET-FF 的评测口径对齐)

要点:
  - 以【标本】为单位划分折,同一标本的增强副本不会跨折(无泄漏)
  - 每折内部再用 2 折选 C,测试折不参与任何选择
  - 报告 mean ± std,以及逐属汇总(全折累计)

用法:
    python ff2_cv.py --feats feats_aug_dinov2l_518_x10_all.npz --pool cls+patch
"""
import argparse
import collections
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

ROOT = os.environ.get("FF2_ROOT", "/mnt/nfs_share/EC_DATABASE_UGREEN/yanhaiqiao/FF2")
FEATDIR = os.path.join(ROOT, "features")
REPDIR = os.path.join(ROOT, "reports")


def l2(x):
    return normalize(np.asarray(x, dtype=np.float64), norm="l2")


def acc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((a == b).mean()) if len(a) else float("nan")


def make_lr(C, cw):
    return LogisticRegression(C=C, max_iter=2000, class_weight=cw)


def pick_C(X, y, groups, cgrid, cw, inner=2):
    """在训练折内部选 C(以标本分组,增强副本不跨折)。"""
    uniq = np.unique(groups)
    if len(uniq) < inner * 2:
        return cgrid[len(cgrid) // 2], {}
    # 把 base 样本折叠成 group 级
    base_y = {}
    for g in uniq:
        base_y[g] = y[groups == g][0]
    skf = StratifiedKFold(n_splits=inner, shuffle=True, random_state=0)
    gkeys = np.array(sorted(uniq))
    gy = np.array([base_y[g] for g in gkeys])
    scores = {c: [] for c in cgrid}
    for tr_g, va_g in skf.split(gkeys, gy):
        tr_set = set(gkeys[tr_g])
        va_set = set(gkeys[va_g])
        mtr = np.array([g in tr_set for g in groups])
        mva = np.array([g in va_set for g in groups])
        for c in cgrid:
            clf = make_lr(c, cw).fit(X[mtr], y[mtr])
            scores[c].append(acc(y[mva], clf.predict(X[mva])))
    means = {c: float(np.mean(v)) for c, v in scores.items()}
    best = max(means, key=lambda c: means[c])
    return best, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--pool", default="cls+patch",
                    choices=["cls", "patch", "cls+patch"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cw", default="balanced", choices=["balanced", "none"])
    ap.add_argument("--c-grid", default="30,100,300")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cw = None if args.cw == "none" else args.cw
    cgrid = [float(x) for x in args.c_grid.split(",")]

    p = args.feats
    if not os.path.isabs(p) and not os.path.exists(p):
        p = os.path.join(FEATDIR, p)
    d = np.load(p, allow_pickle=True)
    tag = os.path.splitext(os.path.basename(p))[0]
    print(f"loaded {p}")

    y_sp = d["sp_idx"].astype(int)
    y_gen = d["g_idx"].astype(int)
    dmg = d["is_damaged"].astype(int)
    sp_uniq = [str(x) for x in d["sp_uniq"]]
    g_uniq = [str(x) for x in d["g_uniq"]]
    n_sp, n_g = len(sp_uniq), len(g_uniq)

    if "base_idx" in d.files:
        base = d["base_idx"].astype(int)
        aug = d["aug_id"].astype(int)
    else:
        base = np.arange(len(y_sp))
        aug = np.zeros(len(y_sp), dtype=int)

    if args.pool == "cls":
        X = d["cls"]
    elif args.pool == "patch":
        X = d["patchmean"]
    else:
        X = np.concatenate([d["cls"], d["patchmean"]], axis=1)
    X = l2(X)
    print(f"  pool={args.pool}  dim={X.shape[1]}  rows={X.shape[0]}")

    # 每个 base 一条标签
    base_ids = np.unique(base)
    b_sp = np.array([y_sp[base == b][0] for b in base_ids])
    b_gen = np.array([y_gen[base == b][0] for b in base_ids])
    b_dmg = np.array([dmg[base == b][0] for b in base_ids])
    print(f"  base specimens: {len(base_ids)}  "
          f"species {len(np.unique(b_sp))}  genera {len(np.unique(b_gen))}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_metrics = []
    genus_tally = collections.defaultdict(lambda: [0, 0])   # name -> [correct, total]
    chosen_c = []

    for f, (tr_b, te_b) in enumerate(skf.split(base_ids, b_sp)):
        tr_set = set(base_ids[tr_b])
        te_set = set(base_ids[te_b])
        mtr = np.array([b in tr_set for b in base])
        mte = np.array([b in te_set for b in base]) & (aug == 0)

        C, cmeans = pick_C(X[mtr], y_sp[mtr], base[mtr], cgrid, cw)
        chosen_c.append(C)

        clf_sp = make_lr(C, cw).fit(X[mtr], y_sp[mtr])
        clf_gen = make_lr(C, cw).fit(X[mtr], y_gen[mtr])
        ps = clf_sp.predict(X[mte])
        pg = clf_gen.predict(X[mte])

        # 层级(oracle 属路由)
        oracle = np.full(int(mte.sum()), -1)
        for g in np.unique(y_gen[mtr]):
            a_tr = y_gen[mtr] == g
            a_te = y_gen[mte] == g
            if a_te.sum() == 0:
                continue
            cls_in = np.unique(y_sp[mtr][a_tr])
            if len(cls_in) == 1:
                pred = np.full(int(a_te.sum()), cls_in[0])
            else:
                clf = make_lr(C, cw).fit(X[mtr][a_tr], y_sp[mtr][a_tr])
                pred = clf.predict(X[mte][a_te])
            oracle[a_te] = pred

        ysp_te, ygen_te, dmg_te = y_sp[mte], y_gen[mte], dmg[mte]
        m = {
            "fold": f, "C": C, "n_test": int(mte.sum()),
            "species": acc(ysp_te, ps),
            "genus": acc(ygen_te, pg),
            "damaged": acc(ysp_te[dmg_te == 1], ps[dmg_te == 1]) if (dmg_te == 1).any() else None,
            "hier_oracle": acc(ysp_te, oracle),
            "hier_e2e": acc(ysp_te, np.where(pg == ygen_te, oracle, -1)),
        }
        fold_metrics.append(m)

        for g in np.unique(ygen_te):
            k = g_uniq[g]
            sel = ygen_te == g
            genus_tally[k][0] += int((ysp_te[sel] == oracle[sel]).sum())
            genus_tally[k][1] += int(sel.sum())

        print(f"  fold {f}: n={m['n_test']:3d} C={C:<5g} "
              f"sp={m['species']*100:5.1f}% gen={m['genus']*100:5.1f}% "
              f"dmg={(m['damaged']*100 if m['damaged'] is not None else float('nan')):5.1f}% "
              f"h_or={m['hier_oracle']*100:5.1f}% h_e2e={m['hier_e2e']*100:5.1f}%")

    # ---------- 汇总 ----------
    print()
    print("=" * 62)
    print(f"5-fold CV summary  ({args.folds} folds, seed={args.seed}, pool={args.pool})")
    print("=" * 62)
    keys = ["species", "genus", "hier_oracle", "hier_e2e", "damaged"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if m[k] is not None]
        if not vals:
            continue
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        print(f"  {k:14s} {np.mean(vals)*100:6.2f}% ± {np.std(vals)*100:5.2f}")

    print()
    print("  per-genus (oracle routing, all folds pooled):")
    for g, (c, t) in sorted(genus_tally.items(), key=lambda x: x[1][0] / max(x[1][1], 1)):
        print(f"    {g:16s} {c:4d}/{t:<4d} {c/t*100:6.1f}%")

    out = args.out or os.path.join(REPDIR, f"cv_{tag}_{args.pool.replace('+','_')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"feats": p, "pool": args.pool, "folds": args.folds,
                   "seed": args.seed, "cw": cw, "c_grid": cgrid,
                   "chosen_C": chosen_c, "per_fold": fold_metrics,
                   "summary": summary,
                   "per_genus": {k: {"correct": v[0], "total": v[1]}
                                 for k, v in genus_tally.items()}},
                  fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
