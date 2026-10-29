#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验状态监控

一次看清:
    - 每个训练任务跑到第几个 epoch、还剩多久
    - 有没有跑完、有没有早停、有没有报错
    - 已经有预测文件的,顺便把 top-1 / macro-F1 打出来(若已评测)

用法:
    python -u status.py
    python -u status.py --watch 30      # 每 30 秒刷新一次
"""
import argparse
import glob
import json
import os
import re
import time


def parse_log(path):
    """返回 dict: ep, total_ep, last_line, done, early, error, sec, val_best"""
    info = {"ep": 0, "total": None, "last": "", "done": False,
            "early": False, "error": False, "sec": None, "val_best": None}
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        return info

    info["error"] = ("Traceback" in txt and "OSError: [Errno 16]" not in txt)
    info["early"] = "early stop" in txt
    # 完成判定:存了 .pth 或 .npz,或者打出了最后一行 peak VRAM(--no-save-weights 时没有 .pth)
    info["done"] = bool(re.search(r"^wrote .*\.(pth|npz)", txt, re.M)) \
        or bool(re.search(r"^peak VRAM", txt, re.M))

    eps = re.findall(r"^ep\s+(\d+)/(\d+)\s+train_loss\s+([\d.]+)\s+train_acc\s+([\d.]+)%"
                     r"(?:\s+val_acc\s+([\d.]+)%)?\s+(\d+)s", txt, re.M)
    if eps:
        e = eps[-1]
        info["ep"], info["total"] = int(e[0]), int(e[1])
        info["loss"], info["acc"] = float(e[2]), float(e[3])
        info["val_last"] = float(e[4]) if e[4] else None
        info["sec"] = int(e[5])
        info["last"] = f"ep {info['ep']}/{info['total']}"
        vals = [float(m) for m in re.findall(r"val_acc\s+([\d.]+)%", txt)]
        info["val_best"] = max(vals) if vals else None

    m = re.search(r"peak VRAM ([\d.]+) GB", txt)
    if m:
        info["vram"] = float(m.group(1))
    return info


def load_leaderboard(reports):
    """从 reports/leaderboard.md 里读回每个 tag 的指标(排行榜模式不写单文件 json)。"""
    p = os.path.join(reports, "leaderboard.md")
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 8 or not cols[0].isdigit():
            continue
        name, t1, mf = cols[1], cols[4], cols[6]
        try:
            out[name] = {"top1": float(t1.rstrip("%")) / 100.0,
                         "macro_f1": float(mf), "n": None}
        except Exception:
            continue
    return out


def parse_eval(reports, tag):
    """优先读单文件 json;没有再回落到排行榜。"""
    p = os.path.join(reports, f"eval_{tag}_species.json")
    if os.path.exists(p):
        try:
            d = json.load(open(p, encoding="utf-8"))
            m = d["metrics"]
            return {"top1": m.get("top1"), "macro_f1": m.get("macro_f1"),
                    "n": m.get("n")}
        except Exception:
            pass
    return None


def scan(logs, preds, reports):
    lb = load_leaderboard(reports)
    rows = []
    for lp in sorted(glob.glob(os.path.join(logs, "*.log"))):
        tag = os.path.splitext(os.path.basename(lp))[0]
        info = parse_log(lp)
        ev = parse_eval(reports, tag) or lb.get(tag)
        has_pred = os.path.exists(os.path.join(preds, tag + ".npz"))
        rows.append({"tag": tag, "info": info, "eval": ev, "has_pred": has_pred})
    return rows


def fmt_eta(info):
    if not info["total"] or not info["sec"] or info["ep"] == 0:
        return "—"
    per = info["sec"] / info["ep"]
    left = (info["total"] - info["ep"]) * per
    if info["done"]:
        return "done"
    return f"{left/60:.0f}min"


def render(rows):
    W = 20
    print(f"{'tag':<{W}} {'状态':<8} {'epoch':>9} {'train_acc':>10} "
          f"{'val_best':>9} {'剩余':>7} {'top1':>7} {'macroF1':>8}")
    print("-" * (W + 62))
    for r in rows:
        i, ev = r["info"], r["eval"]
        if i["error"]:
            st = "❌报错"
        elif i["done"]:
            st = "✅完成"
        elif i["early"]:
            st = "⏹早停"
        else:
            st = "⏳跑中"
        ep = f'{i["ep"]}/{i["total"]}' if i["total"] else "—"
        acc = f'{i["acc"]:.1f}%' if "acc" in i else "—"
        vb = f'{i["val_best"]:.1f}%' if i["val_best"] else "—"
        t1 = f'{ev["top1"]*100:.1f}%' if ev and ev["top1"] is not None else "—"
        mf = f'{ev["macro_f1"]:.4f}' if ev and ev["macro_f1"] is not None else "—"
        print(f'{r["tag"]:<{W}} {st:<8} {ep:>9} {acc:>10} {vb:>9} '
              f'{fmt_eta(i):>7} {t1:>7} {mf:>8}')
    print()
    done = sum(1 for r in rows if r["info"]["done"])
    run = sum(1 for r in rows if not r["info"]["done"] and not r["info"]["error"])
    err = sum(1 for r in rows if r["info"]["error"])
    print(f"共 {len(rows)} 个任务:✅{done}  ⏳{run}  ❌{err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--preds", default="preds")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--watch", type=int, default=0, help="每隔 N 秒刷新;0=只看一次")
    args = ap.parse_args()

    while True:
        os.system("clear" if os.name != "nt" else "cls")
        print(f"FF2 实验状态   {time.strftime('%H:%M:%S')}\n")
        rows = scan(args.logs, args.preds, args.reports)
        if not rows:
            print(f"（{args.logs}/ 下没有 .log 文件）")
        else:
            render(rows)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
