#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
给 ff2_features_aug.py 加 --view 参数(支持多视角特征提取)

改动:
1. 新增 --view 参数(默认 skull_ventral)
2. load_index("skull_ventral") → load_index(args.view)
3. 输出文件名自动带视角后缀(腹面不带,保持兼容)

用法(在 ~/projects/ff2 下):
    python patch_views.py
    python -u ff2_features_aug.py --help
"""
import os
import sys

P = "ff2_features_aug.py"

if not os.path.exists(P):
    sys.exit(f"找不到 {P},请在 ~/projects/ff2 目录下运行")

s = open(P, encoding="utf-8").read()

if '"--view"' in s:
    print("已经打过补丁,无需重复")
    sys.exit(0)

# 1) load_index 调用改为读参数
old = 'rows = load_index("skull_ventral")'
if old not in s:
    sys.exit("没找到 load_index 调用,请贴出该行内容")
s = s.replace(old, "rows = load_index(args.view)")

# 2) 在 --backbone 之前插入 --view
anchor = 'ap.add_argument("--backbone"'
if anchor not in s:
    sys.exit("没找到 --backbone 参数定义")
i = s.index(anchor)
ins = ('ap.add_argument("--view", default="skull_ventral",\n'
       '                    help="metadata.csv 里的 image_type,例如 skull_ventral / '
       'skull_dorsal / skull_lateral / mandible_lateral / teeth")\n    ')
s = s[:i] + ins + s[i:]

# 3) 输出文件名加视角后缀
old_out = 'out = args.out or f"feats_aug_{args.backbone}_{size}_x{args.aug_train}"'
if old_out not in s:
    sys.exit("没找到 out 定义")
vt = ('vt = "" if args.view == "skull_ventral" else '
      '"_" + args.view.replace("skull_", "").replace("mandible_lateral", "mand")\n    ')
s = s.replace(old_out, vt + old_out)

if '{sfx}.npz"' not in s:
    sys.exit("没找到输出文件名后缀")
s = s.replace('{sfx}.npz"', '{sfx}{vt}.npz"')

open(P, "w", encoding="utf-8").write(s)
print("补丁完成:")
print('  - 新增 --view')
print('  - load_index(args.view)')
print('  - 输出文件名带视角后缀(腹面保持原样)')
