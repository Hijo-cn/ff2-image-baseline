#!/bin/bash
# FF2:5 视角 × 5 折,40 epoch 微调 —— 自动排队,3 卡并发
#
# 原理:25 个任务一次性全部提交,每个任务用 flock 抢占自己那条 GPU 队列。
#       只有 3 个能同时拿到锁(对应 GPU 0/1/2),其余在锁上排队,
#       前一个结束、释放锁,下一个立刻接上。全程不需要人工干预。
#
# 用法:
#   bash run_e40_queue.sh              # 正式跑
#   nohup bash run_e40_queue.sh > logs/e40_queue.log 2>&1 &   # 后台跑
#
# 进度:
#   tail -f logs/e40_queue.log

set -u
cd "$(dirname "$0")"

export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS=(0 1 2)          # 只占 3 张卡,留 1 张给别人
VIEWS=(ventral dorsal lateral mand teeth)
FOLDS=(0 1 2 3 4)
EPOCHS=40
TAGPREFIX="dino_e40"

mkdir -p logs preds

echo "=============================================="
echo "开始时间: $(date '+%F %T')"
echo "任务数: $(( ${#VIEWS[@]} * ${#FOLDS[@]} ))  并发: ${#GPUS[@]}"
echo "=============================================="

i=0
for v in "${VIEWS[@]}"; do
  for k in "${FOLDS[@]}"; do
    g=${GPUS[$(( i % ${#GPUS[@]} ))]}
    tag="${TAGPREFIX}_${v}"
    log="logs/e40_${v}_f${k}.log"
    i=$((i+1))

    # flock 持有 locks/gpuN.lock 期间独占该卡;前一个结束才轮到下一个
    (
      flock -w 86400 "/tmp/ff2_gpu_${g}.lock" \
        -c "CUDA_VISIBLE_DEVICES=${g} python -u train.py \
              --backbone dinov3l --fold ${k} \
              --splits splits/splits_v1.json --views ${v} \
              --dual-head --lpft 10 --epochs ${EPOCHS} \
              --lr 5e-5 --head-lr 1e-3 \
              --warmup-epochs 2 --llrd 0.75 \
              --batch 12 --no-save-weights \
              --tag ${tag} > ${log} 2>&1"
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "[$(date '+%H:%M:%S')] ❌ ${v} fold${k} (GPU${g}) 退出码 ${rc}"
      else
        echo "[$(date '+%H:%M:%S')] ✅ ${v} fold${k} (GPU${g})"
      fi
    ) &
    sleep 3
  done
done

wait
echo "=============================================="
echo "全部结束: $(date '+%F %T')"
echo "=============================================="
echo "下一步:融合 + 评测"
echo "  for k in 0 1 2 3 4; do"
echo "    python -u ff2_fuse_views.py --splits splits/splits_v1.json --fold \$k \\"
echo "      --out preds/dino_e40_fused_fold\${k}.npz \\"
echo "      preds/dino_e40_ventral_fold\${k}.npz preds/dino_e40_dorsal_fold\${k}.npz \\"
echo "      preds/dino_e40_lateral_fold\${k}.npz preds/dino_e40_mand_fold\${k}.npz \\"
echo "      preds/dino_e40_teeth_fold\${k}.npz | tail -2"
echo "  done"
