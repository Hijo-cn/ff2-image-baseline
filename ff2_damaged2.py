#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
破损样本的【属内对照】评测

为什么需要属内对照:
    破损样本高度集中在 Euroscaptor / Scaptonyx / Uropsilus 等本来就难的属
    (Scaptonyx 破损率 37%,是平均值的 3.4 倍)。
    直接比较"破损 vs 完好"会把"属的难度"混进来。

做法:
    采用 oracle 属路由 —— 在每个属内部单独训练种分类器,
    再在该属内部比较 破损 与 完好 样本的精度。
    这样属的难度被控制住,只剩"破损"这一个变量。

同时输出:
    A. 标签一致性核对
    B. 属内对照表(逐属:完好精度 / 破损精度 / 差值)
    C. 全体 pooled 结果

用法:
    python ff2_damaged2.py --feats feats_aug_dinov3l_512_x10_all.npz --outdir ~/ff2_reports
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

ROOT = os.environ.get("FF2_ROOT", "")
FEATDIR = os.path.join(ROOT, "features")
DATA = os.path.join(ROOT, "datasets", "yanshu_600_4cranium_1teeth_2measurement")

POOL = {"dinov3l": "cls+patch", "dinov2l": "cls+patch",
        "in21k": "cls+patch", "siglip2l": "cls+patch",
        "bioclip2": "cls", "convnextl": "cls", "efficientnetb7": "cls"}

C_INNER = 10          # 属内分类器的正则强度
N_SPLITS = 5
SEED = 0


def check_labels():
    print("=" * 72)
    print("[A] 标签一致性:文件夹路径 vs metadata.csv")
    print("=" * 72)
    rows = list(csv.DictReader(open(os.path.join(DATA, "metadata.csv"),
                                   encoding="utf-8-sig", newline="")))
    bad_g, bad_s, total = [], [], 0
    for r in rows:
        parts = r["image_path"].replace("\\", "/").split("/")
        if len(parts) < 6:
            continue
        total += 1
        if parts[2] != r["genus"]:
            bad_g.append((parts[2], r["genus"]))
        if parts[3] != f'{r["genus"]} {r["species"]}':
            bad_s.append((parts[3], f'{r["genus"]} {r["species"]}'))
    print(f"  总记录 {total}")
    print(f"  属不一致 {len(bad_g)}    种不一致 {len(bad_s)}")
    kinds = defaultdict(int)
    for folder, meta in bad_s:
        if folder.startswith(meta) and folder[len(meta):].startswith("_"):
            kinds["仅馆号后缀差异(非错标)"] += 1
        else:
            kinds["实质不一致"] += 1
    for k, v in kinds.items():
        print(f"    {k}: {v}")
    return {"total": total, "genus_mismatch": len(bad_g),
            "species_mismatch": len(bad_s), "kinds": dict(kinds)}


def eval_within_genus(feats):
    fp = feats if os.path.isabs(feats) else os.path.join(FEATDIR, feats)
    tag = os.path.basename(fp).replace("feats_aug_", "").replace("_x10_all", "").replace(".npz", "")
    pool = POOL.get(tag.split("_")[0], "cls+patch")

    d = np.load(fp, allow_pickle=True)
    X = {"cls": d["cls"], "patch": d["patchmean"]}.get(
        pool, np.concatenate([d["cls"], d["patchmean"]], axis=1))
    X = normalize(np.asarray(X, dtype=np.float64), norm="l2")

    y_sp = d["sp_idx"].astype(int)
    y_gen = d["g_idx"].astype(int)
    dmg = d["is_damaged"].astype(int)
    base = d["base_idx"].astype(int) if "base_idx" in d.files else np.arange(len(y_sp))
    aug = d["aug_id"].astype(int) if "aug_id" in d.files else np.zeros(len(y_sp), dtype=int)
    g_names = [str(x) for x in d["g_uniq"]]
    sp_names = [str(x) for x in d["sp_uniq"]]

    base_ids = np.unique(base)
    b_sp = np.array([y_sp[base == b][0] for b in base_ids])
    b_gen = np.array([y_gen[base == b][0] for b in base_ids])
    b_dmg = np.array([dmg[base == b][0] for b in base_ids])
    bidx = {b: i for i, b in enumerate(base_ids)}

    ok = np.full(len(base_ids), -1, dtype=int)   # -1 未预测

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for tr_b, te_b in skf.split(base_ids, b_sp):
        tr_set, te_set = set(base_ids[tr_b]), set(base_ids[te_b])
        mtr_all = np.array([b in tr_set for b in base])
        mte_all = np.array([b in te_set for b in base]) & (aug == 0)
        for g in np.unique(y_gen[mte_all]):
            mtr = mtr_all & (y_gen == g)
            mte = mte_all & (y_gen == g)
            if mte.sum() == 0 or mtr.sum() == 0:
                continue
            cls_in = np.unique(y_sp[mtr])
            if len(cls_in) == 1:
                pred = np.full(int(mte.sum()), cls_in[0])
            else:
                clf = LogisticRegression(C=C_INNER, max_iter=3000,
                                         class_weight="balanced").fit(X[mtr], y_sp[mtr])
                pred = clf.predict(X[mte])
            for b, p in zip(base[mte], pred):
                ok[bidx[b]] = int(p == y_sp[base == b][0])

    valid = ok >= 0
    print()
    print("=" * 72)
    print(f"[B] 属内对照(破损 vs 完好)   feats={tag}   C={C_INNER}")
    print("=" * 72)
    print(f"  {'genus':16s} {'完好':>12s} {'破损':>12s} {'差值':>9s}")
    print("  " + "-" * 54)

    per_genus = {}
    for g in np.unique(b_gen):
        m = valid & (b_gen == g)
        mi = m & (b_dmg == 0)
        md = m & (b_dmg == 1)
        ai = float(ok[mi].mean()) if mi.sum() else None
        ad = float(ok[md].mean()) if md.sum() else None
        per_genus[g_names[int(g)]] = {
            "n_intact": int(mi.sum()), "acc_intact": ai,
            "n_damaged": int(md.sum()), "acc_damaged": ad,
        }
        if md.sum() == 0:
            continue
        line = f"  {g_names[int(g)]:16s}"
        line += f"{(f'{ai*100:5.1f}% ({mi.sum()})' if ai is not None else '—'):>12s}"
        line += f"{f'{ad*100:5.1f}% ({md.sum()})':>12s}"
        if ai is not None:
            line += f"{(ad - ai) * 100:+8.1f}"
        print(line)

    print()
    print("=" * 72)
    print("[C] 全体 pooled(oracle 属路由)")
    print("=" * 72)
    overall = {}
    for name, m in (("all", valid), ("intact", valid & (b_dmg == 0)),
                    ("damaged", valid & (b_dmg == 1))):
        a = float(ok[m].mean()) if m.sum() else None
        overall[name] = {"n": int(m.sum()), "acc": a}
        print(f"  {name:10s} n={int(m.sum()):4d}   species {a*100:5.1f}%")

    # 属内加权:只看同时含完好与破损的属,按属等权平均差值
    deltas = []
    for k, v in per_genus.items():
        if v["n_damaged"] >= 3 and v["acc_intact"] is not None and v["acc_damaged"] is not None:
            deltas.append(v["acc_damaged"] - v["acc_intact"])
    if deltas:
        print()
        print(f"  只看破损样本 ≥3 的属(n={len(deltas)}):")
        print(f"    属内平均差值 {np.mean(deltas)*100:+.1f} 点 "
              f"(min {min(deltas)*100:+.1f}, max {max(deltas)*100:+.1f})")

    return {"per_genus": per_genus, "overall": overall,
            "within_genus_delta_mean": float(np.mean(deltas)) if deltas else None,
            "delta_n_genus": len(deltas)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="feats_aug_dinov3l_512_x10_all.npz")
    ap.add_argument("--outdir", default=os.path.expanduser("~/ff2_reports"))
    args = ap.parse_args()
    outdir = os.path.abspath(os.path.expanduser(args.outdir))
    os.makedirs(outdir, exist_ok=True)

    lab = check_labels()
    dmg = eval_within_genus(args.feats)

    out = os.path.join(outdir, "damaged_within_genus.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"label_check": lab, "damaged": dmg}, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
