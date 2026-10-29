#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
给 ff2_features.py 的 BACKBONES 表加一条 DINOv3-H+ (840M)

用法(在 ~/projects/ff2 下):
    python patch_dinov3h.py
    python -c "from ff2_features import BACKBONES; print(BACKBONES['dinov3h'])"
"""
import os
import re
import sys

P = "ff2_features.py"
if not os.path.exists(P):
    sys.exit(f"找不到 {P},请在 ~/projects/ff2 下运行")

s = open(P, encoding="utf-8").read()

if '"dinov3h"' in s or "'dinov3h'" in s:
    print("已经有 dinov3h 了,跳过")
    sys.exit(0)

m = re.search(r"^BACKBONES\s*=\s*\{\s*\n", s, re.M)
if not m:
    sys.exit("没找到 BACKBONES = { ,请把该段贴出来")

entry = '    "dinov3h": ("vit_huge_plus_patch16_dinov3.lvd1689m", 512),\n'
s = s[:m.end()] + entry + s[m.end():]

open(P, "w", encoding="utf-8").write(s)
print("已加入:")
print(entry.rstrip())
print("\n备份提示:如需回滚,删掉这一行即可")
