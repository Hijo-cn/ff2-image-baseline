#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FF2 统一评测脚本

设计原则
--------
1. **只吃概率,不吃模型** —— 输入是一个 (N, C) 的概率矩阵。这样微调模型、冻结特征、
   多视图融合、乃至"专家人工判定"的结果,都能接进同一套评测。
2. **训练与评测彻底解耦** —— 训练脚本只负责吐出 preds/*.npz,指标随时可改、可重算。
3. **自动校验** —— 重复标本、类别名对不上、概率未归一化,都会报出来。

输入文件格式(preds/*.npz)
--------------------------
必填:
    probs        (N, C) float   概率;若传的是 logits,加 --logits 会自动 softmax
    labels       (N,)   int     真实类别索引
    class_names  (C,)   str     类别名(顺序必须和 probs 的列一致)

强烈建议填:
    specimen     (N,)   str     标本 ID —— 用于查重、按折/按标本聚合
    fold         (N,)   int     折号
    genus        (N,)   str     每个样本真实属名(用于层级口径)
    is_damaged   (N,)   int     破损标记
    meta         json 字符串    骨干、尺寸、增强、种子、git commit 等

用法
----
    # 单人评测
    python evaluation.py --pred preds/dinov3_ft_fold0.npz --out reports/

    # 汇总排行榜(把所有人的结果放一张表里)
    python evaluation.py --leaderboard "preds/*.npz" --out reports/leaderboard.md

    # 只看属级
    python evaluation.py --pred xxx.npz --level genus
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

try:
    from sklearn.metrics import f1_score, confusion_matrix
except Exception:  # pragma: no cover
    sys.exit("需要 scikit-learn: pip install scikit-learn")


# ---------------------------------------------------------------- 基础工具

def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def topk_acc(probs, labels, k):
    k = min(k, probs.shape[1])
    idx = np.argpartition(-probs, k - 1, axis=1)[:, :k]
    return float(np.mean([labels[i] in idx[i] for i in range(len(labels))]))


def ece_score(probs, labels, bins=15):
    """期望校准误差(简版)。"""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def load_pred(path, use_logits=False):
    d = np.load(path, allow_pickle=True)
    if "logits" in d.files and "probs" not in d.files:
        probs = softmax(d["logits"])
    else:
        probs = np.asarray(d["probs"], dtype=np.float64)
        if use_logits:
            probs = softmax(probs)
    out = {
        "probs": probs,
        "labels": np.asarray(d["labels"]).astype(int),
        "class_names": [str(x) for x in d["class_names"]],
        "path": path,
    }
    for k in ("specimen", "fold", "genus", "is_damaged"):
        if k in d.files:
            out[k] = np.asarray(d[k])
    if "meta" in d.files:
        try:
            out["meta"] = json.loads(str(d["meta"]))
        except Exception:
            out["meta"] = {"raw": str(d["meta"])}
    return out


def validate(p):
    """返回 (警告列表, 错误列表)。有错误就不该采信这个结果。"""
    warns, errs = [], []
    probs, labels = p["probs"], p["labels"]
    C = probs.shape[1]

    if len(p["class_names"]) != C:
        errs.append(f"class_names 长度 {len(p['class_names'])} != probs 列数 {C}")
    if labels.max() >= C or labels.min() < 0:
        errs.append(f"labels 越界: [{labels.min()}, {labels.max()}],C={C}")

    s = probs.sum(axis=1)
    if not np.allclose(s, 1.0, atol=1e-3):
        warns.append(f"概率行和不是 1(范围 {s.min():.3f}~{s.max():.3f}),已按行归一化")
        p["probs"] = probs / s[:, None]

    if np.isnan(probs).any():
        errs.append("probs 中有 NaN")

    if "specimen" in p:
        dup = [k for k, v in Counter(p["specimen"]).items() if v > 1]
        if dup:
            errs.append(f"有 {len(dup)} 个标本被重复预测(例如 {dup[:3]}),"
                        f"这通常意味着把增强副本也当成测试样本了")
    else:
        warns.append("未提供 specimen 字段 —— 无法查重、无法按折/按属聚合")

    return warns, errs


# ---------------------------------------------------------------- 指标计算

def species_to_genus_mapping(class_names):
    """'Euroscaptor longirostris' -> 'Euroscaptor'"""
    return {c: c.split(" ", 1)[0] for c in class_names}


def aggregate_to_genus(p):
    """把种级概率按属求和,得到属级概率。"""
    mapping = species_to_genus_mapping(p["class_names"])
    genera = sorted(set(mapping.values()))
    gidx = {g: i for i, g in enumerate(genera)}
    C = p["probs"].shape[1]
    M = np.zeros((C, len(genera)))
    for j, c in enumerate(p["class_names"]):
        M[j, gidx[mapping[c]]] = 1
    gprobs = p["probs"] @ M
    glabels = np.array([gidx[mapping[p["class_names"][l]]] for l in p["labels"]])
    return gprobs, glabels, genera


def compute(p, level="species"):
    probs = p["probs"]
    if level == "genus":
        probs, labels, names = aggregate_to_genus(p)
    else:
        labels, names = p["labels"], p["class_names"]
    pred = probs.argmax(axis=1)
    C = probs.shape[1]

    m = {
        "n": int(len(labels)),
        "n_classes": int(C),
        "top1": float((pred == labels).mean()),
        "top5": topk_acc(probs, labels, 5) if C >= 5 else None,
        "macro_f1": float(f1_score(labels, pred, average="macro",
                                   labels=list(range(C)), zero_division=0)),
        "weighted_f1": float(f1_score(labels, pred, average="weighted", zero_division=0)),
        "balanced_acc": float(np.mean([
            (pred[labels == c] == c).mean() for c in range(C) if (labels == c).any()
        ])),
        "ece": ece_score(probs, labels),
    }

    # 逐类
    per_class = {}
    for c in range(C):
        sel = labels == c
        if not sel.any():
            continue
        tp = int(((pred == c) & sel).sum())
        rec = tp / int(sel.sum())
        prec = tp / max(int((pred == c).sum()), 1)
        per_class[names[c]] = {
            "support": int(sel.sum()),
            "recall": float(rec),
            "precision": float(prec),
            "f1": float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0,
        }
    m["per_class"] = per_class
    m["confusion"] = confusion_matrix(labels, pred, labels=list(range(C))).tolist()

    # 按折
    if "fold" in p:
        folds = defaultdict(list)
        for i, f in enumerate(np.asarray(p["fold"]).astype(int)):
            folds[f].append(i)
        by_fold = {}
        for f, idx in sorted(folds.items()):
            idx = np.array(idx)
            by_fold[int(f)] = {
                "n": int(len(idx)),
                "top1": float((pred[idx] == labels[idx]).mean()),
                "macro_f1": float(f1_score(labels[idx], pred[idx], average="macro",
                                           labels=list(range(C)), zero_division=0)),
            }
        t1 = [v["top1"] for v in by_fold.values()]
        m["by_fold"] = by_fold
        m["fold_top1_mean"] = float(np.mean(t1))
        if len(t1) >= 2:                      # 单折时不给"折间标准差",否则会误导
            m["fold_top1_std"] = float(np.std(t1))
            m["fold_macro_f1_std"] = float(np.std(
                [v["macro_f1"] for v in by_fold.values()]))

    # 破损 / 完好
    if "is_damaged" in p:
        d = np.asarray(p["is_damaged"]).astype(int)
        for name, mask in (("intact", d == 0), ("damaged", d == 1)):
            if mask.any():
                m.setdefault("subsets", {})[name] = {
                    "n": int(mask.sum()),
                    "top1": float((pred[mask] == labels[mask]).mean()),
                }

    # 逐属
    if level == "species":
        mapping = species_to_genus_mapping(names)
        g_true = np.array([mapping[names[l]] for l in labels])
        per_genus = {}
        for g in sorted(set(g_true)):
            sel = g_true == g
            per_genus[g] = {
                "n": int(sel.sum()),
                "top1": float((pred[sel] == labels[sel]).mean()),
            }
        m["per_genus"] = per_genus

    m["_pred"] = pred
    m["_labels"] = labels
    m["_names"] = names
    return m


def strip_private(m):
    return {k: v for k, v in m.items() if not k.startswith("_")}


# ---------------------------------------------------------------- 输出

def fmt_pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def write_report(path, title, m, p, warns, errs, level):
    L = []
    A = L.append
    A(f"# {title}\n")
    meta = p.get("meta", {})
    if meta:
        A("| 项目 | 值 |")
        A("|---|---|")
        for k, v in meta.items():
            A(f"| {k} | {v} |")
        A("")
    A(f"- 预测来源:`{p['path']}`")
    A(f"- 样本数:**{m['n']}**")
    A(f"- 类别数:**{m['n_classes']}**(口径:{level})")
    A(f"- 划分说明:按标本分组,同一标本不跨折\n")

    if errs:
        A("## ❌ 校验未通过\n")
        for e in errs:
            A(f"- {e}")
        A("")
        A("**该结果不应被采信。**\n")
    if warns:
        A("## ⚠️ 警告\n")
        for w in warns:
            A(f"- {w}")
        A("")

    A("## 主指标\n")
    A("| 指标 | 值 |")
    A("|---|---|")
    A(f"| **Top-1** | **{fmt_pct(m['top1'])}** |")
    A(f"| Top-5 | {fmt_pct(m['top5'])} |")
    A(f"| **Macro-F1** | **{m['macro_f1']:.4f}** |")
    A(f"| Weighted-F1 | {m['weighted_f1']:.4f} |")
    A(f"| Balanced Acc | {fmt_pct(m['balanced_acc'])} |")
    A(f"| ECE(校准) | {m['ece']:.4f} |")
    if "fold_top1_mean" in m:
        if "fold_top1_std" in m:
            A(f"| 折间 Top-1 | {m['fold_top1_mean']*100:.2f}% ± {m['fold_top1_std']*100:.2f} |")
        else:
            A(f"| 折间 Top-1 | {m['fold_top1_mean']*100:.2f}%(仅 {len(m['by_fold'])} 折) |")
    A("")

    if "by_fold" in m:
        A("## 逐折\n")
        A("| 折 | n | Top-1 | Macro-F1 |")
        A("|---|---|---|---|")
        for f, v in m["by_fold"].items():
            A(f"| {f} | {v['n']} | {fmt_pct(v['top1'])} | {v['macro_f1']:.4f} |")
        A("")

    if "subsets" in m:
        A("## 完好 / 破损\n")
        A("| 子集 | n | Top-1 |")
        A("|---|---|---|")
        for k, v in m["subsets"].items():
            A(f"| {k} | {v['n']} | {fmt_pct(v['top1'])} |")
        A("")

    if "per_genus" in m:
        A("## 逐属(最容易看出短板)\n")
        A("| 属 | n | Top-1 |")
        A("|---|---|---|")
        for g, v in sorted(m["per_genus"].items(), key=lambda x: x[1]["top1"]):
            A(f"| {g} | {v['n']} | {fmt_pct(v['top1'])} |")
        A("")

    worst = sorted(m["per_class"].items(), key=lambda x: x[1]["recall"])[:10]
    A("## 最差 10 类召回(长尾诊断)\n")
    A("| 类别 | support | recall | precision |")
    A("|---|---|---|---|")
    for c, v in worst:
        A(f"| {c} | {v['support']} | {fmt_pct(v['recall'])} | {fmt_pct(v['precision'])} |")
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def write_leaderboard(path, rows, sort_key):
    """双指标榜:top-1 和 macro-F1 各排一次名,再按平均名次排序。"""
    n = len(rows)
    for key in ("top1", "macro_f1"):
        order = sorted(range(n), key=lambda i: rows[i]["m"][key], reverse=True)
        for rank, i in enumerate(order, 1):
            rows[i].setdefault("_rank", {})[key] = rank
    for r in rows:
        r["_avg_rank"] = (r["_rank"]["top1"] + r["_rank"]["macro_f1"]) / 2
    rows = sorted(rows, key=lambda r: r["_avg_rank"])

    L = []
    A = L.append
    A("# 排行榜\n")
    A("**双指标:top-1 与 macro-F1 各排一次名,按平均名次排序。**")
    A("折间波动约 ±3 个点,**不要用单折结果比较优劣**。\n")
    A("| # | 提交 | 骨干 | 口径 | Top-1 | 名次 | Macro-F1 | 名次 | 均名次 | 折间± | 破损 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        meta = r["p"].get("meta", {})
        name = meta.get("name") or os.path.basename(r["p"]["path"])
        bb = meta.get("backbone", "—")
        s = r["m"]
        fold = (f"{s['fold_top1_std']*100:.2f}" if "fold_top1_std" in s else "—")
        dmg = s.get("subsets", {}).get("damaged", {}).get("top1")
        A(f"| {i} | {name} | {bb} | {r['level']} | {fmt_pct(s['top1'])} | "
          f"{r['_rank']['top1']} | {s['macro_f1']:.4f} | {r['_rank']['macro_f1']} | "
          f"{r['_avg_rank']:.1f} | {fold} | {fmt_pct(dmg)} |")
    A("")
    if n == 1:
        A("> 只有一条结果,排行榜暂无比较意义。\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


# ---------------------------------------------------------------- main

def process(path, args):
    p = load_pred(path, use_logits=args.logits)
    warns, errs = validate(p)
    m = compute(p, level=args.level)
    tag = os.path.splitext(os.path.basename(path))[0]
    print(f"[{tag}] top1={m['top1']*100:.2f}%  macroF1={m['macro_f1']:.4f}"
          + (f"  折间±{m['fold_top1_std']*100:.2f}" if "fold_top1_std" in m else ""))
    for e in errs:
        print(f"    ❌ {e}")
    for w in warns:
        print(f"    ⚠️  {w}")
    return p, m, warns, errs, tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="")
    ap.add_argument("--leaderboard", default="", help="glob 模式,可匹配多个文件")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--level", default="species", choices=["species", "genus"])
    ap.add_argument("--logits", action="store_true", help="输入是 logits,自动 softmax")
    ap.add_argument("--sort", default="macro_f1",
                    choices=["top1", "macro_f1", "balanced_acc"])
    args = ap.parse_args()

    if not args.pred and not args.leaderboard:
        ap.error("需要 --pred 或 --leaderboard")

    # 排行榜模式
    if args.leaderboard:
        files = sorted(glob.glob(args.leaderboard))
        if not files:
            sys.exit(f"没有匹配到文件: {args.leaderboard}")
        rows = []
        for f in files:
            p, m, w, e, tag = process(f, args)
            if e:
                print(f"    → 已跳过 {tag}(校验未通过)")
                continue
            rows.append({"p": p, "m": m, "level": args.level, "tag": tag})
        os.makedirs(args.out, exist_ok=True)
        lb = os.path.join(args.out, "leaderboard.md")
        write_leaderboard(lb, rows, args.sort)
        print(f"\nwrote {lb}({len(rows)} 条)")
        return

    # 单人模式
    p, m, w, e, tag = process(args.pred, args)
    os.makedirs(args.out, exist_ok=True)
    jp = os.path.join(args.out, f"eval_{tag}_{args.level}.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump({"meta": p.get("meta", {}), "level": args.level,
                   "metrics": strip_private(m), "warnings": w, "errors": e},
                  fh, ensure_ascii=False, indent=2)
    rp = os.path.join(args.out, f"eval_{tag}_{args.level}.md")
    write_report(rp, f"评测报告 · {tag}", m, p, w, e, args.level)
    print(f"\nwrote {jp}\nwrote {rp}")


if __name__ == "__main__":
    main()
