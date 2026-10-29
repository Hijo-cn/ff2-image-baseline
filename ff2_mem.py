#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FF2 显存与算力基准

对每个候选骨干测量:
  1. 参数量(总 / 可训练)
  2. 推理峰值显存(前向)
  3. 训练峰值显存(前向 + 反向 + 优化器步)—— 用 --train 开启
  4. 推理吞吐(images/s)

每个骨干在多个 batch size 下测,遇到 OOM 记为 OOM 并继续。

用法:
    python ff2_mem.py                        # 只测推理
    python ff2_mem.py --train                # 额外测训练峰值
    python ff2_mem.py --batch 1,8,16,32      # 指定 batch
    python ff2_mem.py --only dinov3l,bioclip2
"""
import argparse
import gc
import os
import time

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ff2_features import build_model, snap_size, pool_features, BACKBONES  # noqa: E402

ROOT = os.environ.get("FF2_ROOT", "/mnt/nfs_share/EC_DATABASE_UGREEN/yanhaiqiao/FF2")
REPDIR = os.path.join(ROOT, "reports")


def human(nbytes):
    if nbytes is None:
        return "OOM"
    return f"{nbytes / 1024 ** 3:.2f} GB"


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


def measure(model, size, bs, device, train_mode=False, iters=3):
    """返回 (peak_alloc_bytes, peak_reserved_bytes, images_per_sec) 或 None(OOM)。"""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(bs, 3, size, size, device=device)

        # 预热一次,避免首次分配把峰值抬高
        with torch.no_grad():
            pool_features(model.forward_features(x))
        del x
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        optim = None
        if train_mode:
            model.train()
            optim = torch.optim.SGD(model.parameters(), lr=0.0, momentum=0.9)

        x = torch.randn(bs, 3, size, size, device=device)
        out = None
        t0 = time.time()
        if train_mode:
            for _ in range(iters):
                optim.zero_grad(set_to_none=True)
                out = model.forward_features(x)
                cls, pm = pool_features(out)
                loss = (cls.float() ** 2).mean() + (pm.float() ** 2).mean()
                loss.backward()
                optim.step()
        else:
            with torch.no_grad():
                for _ in range(iters):
                    out = model.forward_features(x)
                    pool_features(out)
        torch.cuda.synchronize()
        dt = time.time() - t0

        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_res = torch.cuda.max_memory_reserved(device)
        ips = bs * iters / dt

        del x
        if out is not None:
            del out
        if optim is not None:
            del optim
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        return peak_alloc, peak_res, ips
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            gc.collect()
            torch.cuda.empty_cache()
            return None
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔的骨干代号")
    ap.add_argument("--batch", default="1,8,16,32")
    ap.add_argument("--train", action="store_true", help="额外测训练峰值(前向+反向)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    batches = [int(b) for b in args.batch.split(",")]
    names = args.only.split(",") if args.only else list(BACKBONES.keys())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    outdir = args.outdir or REPDIR
    os.makedirs(outdir, exist_ok=True)

    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"total VRAM: {torch.cuda.get_device_properties(device).total_memory/1024**3:.2f} GB")
    print()

    rows = []
    for name in names:
        if name not in BACKBONES:
            print(f"  跳过未知骨干 {name}")
            continue
        try:
            model, mean, std, dsize = build_model(name)
        except Exception as e:
            print(f"[{name}] 加载失败: {type(e).__name__}: {e}")
            continue
        size = snap_size(model, dsize)
        model = model.to(device).eval()
        total_p, train_p = count_params(model)
        print(f"[{name}]  {BACKBONES[name][0]}  size={size}  "
              f"params={total_p/1e6:.1f}M")

        for bs in batches:
            r_inf = measure(model, size, bs, device, train_mode=False)
            if r_inf is None:
                print(f"     bs={bs:<3d} 推理: OOM")
                rows.append({"backbone": name, "size": size, "params_M": total_p/1e6,
                             "batch": bs, "infer_peak": None, "infer_ips": None,
                             "train_peak": None})
                continue
            pa, pr, ips = r_inf
            line = f"     bs={bs:<3d} 推理峰值 {human(pa):>9s}  {ips:7.1f} img/s"
            train_peak = None
            if args.train:
                r_tr = measure(model, size, bs, device, train_mode=True)
                if r_tr is None:
                    line += "   |  训练: OOM"
                else:
                    train_peak = r_tr[0]
                    line += f"  |  训练峰值 {human(train_peak):>9s}"
            print(line)
            rows.append({"backbone": name, "size": size, "params_M": round(total_p/1e6, 1),
                         "batch": bs, "infer_peak": pa, "infer_ips": round(ips, 1),
                         "train_peak": train_peak})
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 写 CSV
    cp = os.path.join(outdir, "mem_bench.csv")
    with open(cp, "w", encoding="utf-8") as fh:
        fh.write("backbone,size,params_M,batch,infer_peak_GB,infer_img_per_s,train_peak_GB\n")
        for r in rows:
            ip = "" if r["infer_peak"] is None else f"{r['infer_peak']/1024**3:.3f}"
            tp = "" if r["train_peak"] is None else f"{r['train_peak']/1024**3:.3f}"
            ips = "" if r["infer_ips"] is None else r["infer_ips"]
            fh.write(f"{r['backbone']},{r['size']},{r['params_M']},{r['batch']},{ip},{ips},{tp}\n")
    print(f"\nwrote {cp}")

    # 打印汇总(取每个骨干最大的可跑 batch)
    print()
    print(f"{'backbone':16s} {'size':>5s} {'params':>8s} {'bs':>4s} "
          f"{'推理峰值':>10s} {'吞吐':>12s} {'训练峰值':>10s}")
    print("-" * 74)
    for name in names:
        rs = [r for r in rows if r["backbone"] == name and r["infer_peak"] is not None]
        if not rs:
            continue
        best = max(rs, key=lambda r: r["batch"])
        tp = "—" if best["train_peak"] is None else human(best["train_peak"])
        print(f"{name:16s} {best['size']:5d} {best['params_M']:7.1f}M {best['batch']:4d} "
              f"{human(best['infer_peak']):>10s} {best['infer_ips']:9.1f} img/s {tp:>10s}")


if __name__ == "__main__":
    main()
