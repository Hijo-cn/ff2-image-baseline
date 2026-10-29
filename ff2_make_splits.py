#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FF2 基线 —— 固定数据划分生成器

作用:
    生成一份【全组共用】的 splits.json。所有人做训练/评测必须读这个文件,
    不允许在运行时用 seed 现场生成(避免 sklearn 版本差异导致划分不同)。

核心原则:
    1. 以【标本】为单位划分 —— 同一标本的所有视角/增强副本必须落在同一侧
    2. 按【物种】分层 —— 保证每折的类别分布接近
    3. 生成后带 sha256,写进训练配置,便于追溯

用法:
    # 默认:对全部 746 个标本做 5 折
    python ff2_make_splits.py --out splits/splits_v1.json --report splits/splits_v1_report.md

    # 或用数据集自带的 593/153 划分(单折)
    python ff2_make_splits.py --use fixed --out splits/splits_fixed.json

输出:
    splits/splits_v1.json          划分文件(含 sha256)
    splits/splits_v1_report.md     划分统计报表(给师兄过目用)
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = os.environ.get("FF2_ROOT", "/mnt/nfs_share/EC_DATABASE_UGREEN/yanhaiqiao/FF2")
DATA = os.path.join(ROOT, "datasets", "yanshu_600_4cranium_1teeth_2measurement")


def load_specimens(view="skull_ventral"):
    """从 metadata.csv 读标本级信息(每个标本一条)。"""
    csvp = os.path.join(DATA, "metadata.csv")
    rows = [r for r in csv.DictReader(open(csvp, encoding="utf-8-sig", newline=""))
            if r["image_type"] == view]

    specs = {}
    for r in rows:
        fn = os.path.splitext(r["image_name"])[0]
        damaged = 1 if fn.split("#")[-1].lower() == "p" else 0
        sid = r["specimen"]
        if sid in specs:
            # 同一标本多张图时,任一破损即记为破损
            specs[sid]["is_damaged"] = max(specs[sid]["is_damaged"], damaged)
            continue
        parts = r["image_path"].replace("\\", "/").split("/")
        specs[sid] = {
            "id": sid,
            "genus": r["genus"],
            "species": f'{r["genus"]} {r["species"]}',
            "split": parts[1] if len(parts) > 1 else "",
            "is_damaged": damaged,
        }
    return sorted(specs.values(), key=lambda x: x["id"])


def build_folds(specs, n_folds=5, seed=0, mode="all"):
    species = [s["species"] for s in specs]
    y = np.array([sorted(set(species)).index(s) for s in species])

    if mode == "fixed":
        test_idx = [i for i, s in enumerate(specs) if s["split"] == "test"]
        train_idx = [i for i, s in enumerate(specs) if s["split"] != "test"]
        return [test_idx], [train_idx]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds_test, folds_train = [], []
    for tr, te in skf.split(np.zeros(len(specs)), y):
        folds_test.append(sorted(int(i) for i in te))
        folds_train.append(sorted(int(i) for i in tr))
    return folds_test, folds_train


def diagnose(specs, folds_test, folds_train):
    """统计每折的类别覆盖、破损分布,并找出『训练集缺席的类』。"""
    sp_names = [s["species"] for s in specs]
    gn_names = [s["genus"] for s in specs]
    uniq_sp = sorted(set(sp_names))
    uniq_gn = sorted(set(gn_names))

    per_fold = []
    guaranteed = []
    for k, (te, tr) in enumerate(zip(folds_test, folds_train)):
        tr_sp = Counter(sp_names[i] for i in tr)
        absent = [c for c in uniq_sp if tr_sp.get(c, 0) == 0]
        n_guaranteed = sum(1 for i in te if sp_names[i] in absent)
        per_fold.append({
            "fold": k,
            "n_test": len(te),
            "n_train": len(tr),
            "n_species_test": len({sp_names[i] for i in te}),
            "n_genus_test": len({gn_names[i] for i in te}),
            "n_damaged_test": sum(specs[i]["is_damaged"] for i in te),
            "classes_absent_from_train": absent,
            "n_test_samples_guaranteed_wrong": n_guaranteed,
        })
        guaranteed.append((k, absent, n_guaranteed))
    return per_fold, uniq_sp, uniq_gn


def write_report(path, specs, folds_test, folds_train, per_fold, uniq_sp, uniq_gn,
                 mode, seed, sha):
    L = []
    A = L.append
    A("# 数据划分统计报表\n")
    A(f"- 数据集:`{os.path.basename(DATA)}`")
    A(f"- 标本总数:**{len(specs)}**")
    A(f"- 折数:**{len(folds_test)}**")
    A(f"- 模式:`{mode}`(all = 全部标本 5 折;fixed = 数据集自带 593/153)")
    if mode == "all":
        A(f"- 随机种子:`{seed}`")
    A(f"- 分层依据:物种")
    A(f"- 划分单位:**标本**(同一标本的所有视角/增强副本必在同一侧)")
    A(f"- `splits` 文件 sha256:`{sha}`\n")

    A("## 一、总体覆盖\n")
    A(f"- 属:**{len(uniq_gn)}** 个")
    A(f"- 物种:**{len(uniq_sp)}** 个")
    A(f"- 破损标本:**{sum(s['is_damaged'] for s in specs)}** 个"
      f"({sum(s['is_damaged'] for s in specs) / len(specs) * 100:.1f}%)\n")

    A("## 二、每折概况\n")
    A("| 折 | 测试标本 | 训练标本 | 测试含属 | 测试含种 | 测试破损 | 训练缺席的类 | 必错样本 |")
    A("|---|---|---|---|---|---|---|---|")
    for f in per_fold:
        A(f"| {f['fold']} | {f['n_test']} | {f['n_train']} | {f['n_genus_test']} | "
          f"{f['n_species_test']} | {f['n_damaged_test']} | "
          f"{len(f['classes_absent_from_train'])} | {f['n_test_samples_guaranteed_wrong']} |")
    A("")

    A("## 三、稀有物种问题(重要)\n")
    A("物种样本数分布:\n")
    cnt = Counter(s["species"] for s in specs)
    dist = Counter(cnt.values())
    A("| 每物种标本数 | 物种个数 |")
    A("|---|---|")
    for n in sorted(dist):
        A(f"| {n} | {dist[n]} |")
    A("")
    rare = [c for c, n in cnt.items() if n < len(folds_test)]
    if rare:
        A(f"**有 {len(rare)} 个物种的标本数少于折数({len(folds_test)}),"
          f"必然在某些折的训练集中完全缺席:**\n")
        for c in sorted(rare, key=lambda x: cnt[x]):
            A(f"- `{c}`:{cnt[c]} 个标本")
        A("")
        tot = sum(f["n_test_samples_guaranteed_wrong"] for f in per_fold)
        A(f"**全折累计有 {tot} 个测试样本因『该类在训练集缺席』而必然判错。**"
          f"这是评测协议的结构性问题,不是模型问题,报告里应明确说明。\n")

    A("## 四、逐类分布(全部标本)\n")
    A("| 物种 | 标本数 | 破损数 |")
    A("|---|---|---|")
    dmg = defaultdict(int)
    for s in specs:
        dmg[s["species"]] += s["is_damaged"]
    for c in sorted(cnt, key=lambda x: (-cnt[x], x)):
        A(f"| {c} | {cnt[c]} | {dmg[c]} |")
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="splits/splits_v1.json")
    ap.add_argument("--report", default="")
    ap.add_argument("--use", choices=["all", "fixed"], default="all")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", default="skull_ventral")
    args = ap.parse_args()

    specs = load_specimens(args.view)
    if not specs:
        sys.exit(f"没读到标本,检查 {DATA}/metadata.csv")
    print(f"标本数: {len(specs)}")

    folds_test, folds_train = build_folds(specs, args.folds, args.seed, args.use)
    per_fold, uniq_sp, uniq_gn = diagnose(specs, folds_test, folds_train)

    payload = {
        "version": 1,
        "dataset": os.path.basename(DATA),
        "view": args.view,
        "mode": args.use,
        "seed": args.seed if args.use == "all" else None,
        "n_folds": len(folds_test),
        "n_specimens": len(specs),
        "unit": "specimen",
        "stratify_by": "species",
        "class_names": uniq_sp,
        "genus_names": uniq_gn,
        "specimens": specs,
        "folds": [{"test": te, "train": tr}
                  for te, tr in zip(folds_test, folds_train)],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    sha = hashlib.sha256(blob).hexdigest()
    payload["sha256"] = sha

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")
    print(f"sha256: {sha}")

    rep = args.report or os.path.splitext(args.out)[0] + "_report.md"
    write_report(rep, specs, folds_test, folds_train, per_fold, uniq_sp, uniq_gn,
                 args.use, args.seed, sha)
    print(f"wrote {rep}")

    print()
    print(f"{'fold':>4s} {'test':>6s} {'train':>6s} {'absent_cls':>10s} {'guaranteed_wrong':>16s}")
    for f in per_fold:
        print(f"{f['fold']:4d} {f['n_test']:6d} {f['n_train']:6d} "
              f"{len(f['classes_absent_from_train']):10d} "
              f"{f['n_test_samples_guaranteed_wrong']:16d}")


if __name__ == "__main__":
    main()
