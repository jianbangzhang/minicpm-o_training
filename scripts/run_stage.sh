#!/bin/bash
# ==========================================================
# Stage 1: 模态对齐预训练
# ==========================================================

set -e
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

echo "=========================================="
echo "Stage 1: 模态对齐预训练"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "节点数: 1 | GPU数: 8"
echo "=========================================="

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=29500 \
    trainer_api.py \
    --stage 1 \
    --config configs/training_config.yaml

echo "Stage 1 完成！"


# ==========================================================
# Stage 2: 统一多模态预训练
# ==========================================================
# 多节点启动示例（在主节点执行）:
#
# torchrun \
#     --nproc_per_node=8 \
#     --nnodes=4 \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=29500 \
#     --node_rank=${NODE_RANK} \
#     trainer_api.py \
#     --stage 2 \
#     --config configs/training_config.yaml
