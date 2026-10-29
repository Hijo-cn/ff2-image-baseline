# FF2 图像基线

FF2 项目(鼹科 Talpidae 标本识别)的图像侧基线代码。包含固定划分、统一评测、冻结特征探针、多视角融合与微调框架。

---

## 1. 环境

```
Python 3.10
torch >= 2.1 (CUDA)
timm >= 1.0
open_clip_torch          # 仅 BioCLIP 系列需要
scikit-learn, numpy, pillow, pandas
umap-learn, matplotlib   # 仅表征分析需要
```

安装:

```bash
pip install -r requirements.txt
```

---

## 2. 数据准备

所有路径通过环境变量 `FF2_ROOT` 指定:

```bash
export FF2_ROOT=/path/to/FF2
```

目录结构要求:

```
$FF2_ROOT/
├── datasets/yanshu_600_4cranium_1teeth_2measurement/
│   ├── metadata.csv              # 每行一张图
│   ├── cranium/{train,test}/<属>/<种>/<标本号>/<图>.JPG
│   └── teeth/{train,test}/...
├── features/                     # 特征缓存(脚本自动创建)
└── weights/hf_cache/             # 预训练权重缓存
```

`metadata.csv` 需要包含以下列:

| 列名 | 说明 |
|---|---|
| `image_path` | 相对 `datasets/.../` 的路径,形如 `cranium/train/属/种/标本号/图.JPG` |
| `image_name` | 文件名 |
| `image_type` | 视角类型,取值:`skull_ventral` / `skull_dorsal` / `skull_lateral` / `mandible_lateral` / `teeth_skull_ventral` |
| `specimen` | 标本号(评测时按此分组) |
| `genus` / `species` | 分类标签 |

> 图片文件名格式为 `<属>#<种>#<馆号>#<编号>#s#v.JPG`,末位 `#p` 表示破损标本。

---

## 3. 快速开始

### 3.1 生成固定划分

```bash
python ff2_make_splits.py --out splits/splits_v1.json \
                          --report splits/splits_v1_report.md
```

生成 746 个标本的 5 折划分(按标本分组、按物种分层)。**后续所有实验必须读取同一个 `splits_v1.json`,不要在运行时随机生成。**

### 3.2 提取冻结特征

```bash
python ff2_features_aug.py --backbone dinov3l --view skull_ventral \
    --aug-train 10 --aug-all \
    --out feats_aug_dinov3l_512_x10_all.npz
```

- `--backbone`:见 `ff2_features.py` 的 `BACKBONES` 表
- `--view`:视角
- `--aug-train 10 --aug-all`:每张图生成 10 个增强副本(任何标本都可能进训练折,所以要对全部样本增强)

多视角依次提取即可,输出文件名建议带视角后缀。

### 3.3 冻结特征 + 线性探针

```bash
python ff2_frozen_to_preds.py --feats feats_aug_dinov3l_512_x10_all.npz \
    --splits splits/splits_v1.json --out preds/frozen_oof.npz

python evaluation.py --pred preds/frozen_oof.npz --out reports/
```

### 3.4 微调(每视角一个模型)

```bash
python train.py --backbone dinov3l --fold 0 \
    --splits splits/splits_v1.json --views ventral \
    --dual-head --lpft 10 --epochs 10 --lr 5e-5 --head-lr 1e-3 \
    --warmup-epochs 2 --llrd 0.75 --batch 12 --no-save-weights \
    --tag dinov3l_ft_ventral
```

### 3.5 多视角融合

```bash
python ff2_fuse_views.py --splits splits/splits_v1.json --fold 0 \
    --out preds/fused_fold0.npz \
    preds/dinov3l_ft_ventral_fold0.npz \
    preds/dinov3l_ft_dorsal_fold0.npz \
    preds/dinov3l_ft_lateral_fold0.npz \
    preds/dinov3l_ft_mand_fold0.npz \
    preds/dinov3l_ft_teeth_fold0.npz

python ff2_hier_eval.py preds/fused_oof.npz
```

---

## 4. 脚本说明

### 划分与评测

| 脚本 | 作用 |
|---|---|
| `ff2_make_splits.py` | 生成固定 5 折划分 + 统计报表 |
| `evaluation.py` | 统一评测(输入概率矩阵),输出 top-1 / macro-F1 / 逐类 / 逐属 / 逐折 / 破损子集 |
| `ff2_hier_eval.py` | 层级四口径:flat / genus / hier_oracle / hier_e2e |

### 特征

| 脚本 | 作用 |
|---|---|
| `ff2_features.py` | 冻结特征提取(单视角,无增强) |
| `ff2_features_aug.py` | 冻结特征提取(带增强),支持 `--view` |
| `ff2_frozen_to_preds.py` | 冻结特征 → 线性探针 → 统一格式预测 |
| `ff2_mem.py` | 显存 / 吞吐基准 |

### 微调

| 脚本 | 作用 |
|---|---|
| `train.py` | 微调主脚本。支持多视角、`--dual-head`(2048 维头)、`--lpft`、`--peft lora/bitfit`、`--llrd`、`--warmup-epochs`、`--opt sgd`、`--aug none/weak/current` |

### 融合

| 脚本 | 作用 |
|---|---|
| `ff2_multiview.py` | 多种融合配置的横向对比(池化 / 晚期融合 / 加权) |
| `ff2_fuse_views.py` | 按标本平均各视角概率 |
| `ff2_merge_folds.py` | 5 折预测合并为 OOF |
| `ff2_merge_genus.py` | 按属分解模型的合并 + oracle 属路由 |
| `ff2_stacking.py` | 多视角 stacking(嵌套 CV) |
| `ff2_feat_concat.py` | 多视角特征拼接 |

### 分类器对照

| 脚本 | 作用 |
|---|---|
| `ff2_crt.py` | 长尾再平衡(cRT / τ-normalization) |
| `ff2_genus_probe.py` | 专用 18 类属分类器 |
| `ff2_pergenus_probe.py` | 专用属内分类器 |

### 工具

| 脚本 | 作用 |
|---|---|
| `status.py` | 训练进度监控 |
| `run_e40_queue.sh` | 多任务自动排队(flock 分配 GPU) |
| `patch_views.py` / `patch_dinov3h.py` | 给旧脚本追加视角 / 骨干支持的补丁 |

---

## 5. 评测口径

所有结果基于 `splits_v1.json` 的 5 折,**按标本分组**(同一标本的多个视角不跨折)。

| 指标 | 定义 |
|---|---|
| `flat` | 51 类直接分类 |
| `genus` | 18 类属级 |
| `hier_oracle` | 已知正确属,属内分种 |
| `hier_e2e` | 先用预测属,再属内分种 |

**折间标准差约 ±2.9 个点,单折结果不足以支撑结论。**

---

## 6. 已知注意事项

1. **单折波动大**:折间 ±2.9,出现过多次"单折提升、5 折回落"。报告结论应使用 5 折均值。
2. **`--record-epoch-test` 仅用于诊断**:该选项每轮在测试集上评估,会产生测试集信息泄露,不能用于正式实验。
3. **NAS 权限**:部分环境下列表目录下新建文件会变成 `000` 权限,脚本失败时先 `chmod -R u+rwX $FF2_ROOT/features`。
4. **NFS 临时文件**:DataLoader worker 在 NFS 上可能报 `OSError: [Errno 16]`,设置 `TMPDIR=/tmp` 可消除。
5. **显存**:ViT-L 微调在 24 GB 卡上 batch ≤ 12;EfficientNet-B7 @600 只能 batch 4。

---

## 7. 复现的基线数值

5 折,n = 746:

| 配置 | flat | genus | hier_oracle | hier_e2e |
|---|---|---|---|---|
| 单视角冻结(DINOv3-L @512) | 84.18 | 94.15 | 88.08 | 84.30 |
| 5 视角冻结 + 概率平均 | 89.54 | 96.55 | 92.19 | 89.83 |
| 5 视角微调融合(@512) | 90.21 | 97.18 | 92.76 | 90.75 |
| 5 视角 × 2 分辨率融合(10 模型) | 90.88 | 97.59 | 92.96 | 91.29 |

---

## 8. 引用

如使用本代码,请引用:

```
待补充
```
