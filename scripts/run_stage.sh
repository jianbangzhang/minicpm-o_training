#!/bin/bash
# ==========================================================
# Stage 1: 模态对齐预训练
# ==========================================================

set -e
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

#echo "=========================================="
#echo "Stage 1: 模态对齐预训练"
#echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
#echo "=========================================="

#torchrun \
#    --nproc_per_node=1 \
#    --nnodes=1 \
#    --master_addr=localhost \
#    --master_port=29500 \
#    trainer_api.py \
#    --stage 1 \
#    --config configs/training_config.yaml

#echo "Stage 1 完成！"


# ==========================================================
# Stage 2: 统一多模态预训练
# ==========================================================
# 多节点启动示例（在主节点执行）:
#
#CUDA_VISIBLE_DEVICES=5 torchrun \
#     --nproc_per_node=1 \
#     --nnodes=1 \
#     --master_addr=localhost \
#     --master_port=29501 \
#     --node_rank=0 \
#     trainer_api.py \
#     --stage 2 \
#     --config configs/training_config.yaml

# ==========================================================
# Stage 3: 指令微调训练
# ==========================================================
# 多节点启动示例（在主节点执行）:

#CUDA_VISIBLE_DEVICES=5 torchrun \
#     --nproc_per_node=1 \
#     --nnodes=1 \
#     --master_addr=localhost \
#     --master_port=29501 \
#     --node_rank=0 \
#     trainer_api.py \
#     --stage 3 \
#     --config configs/training_config.yaml


# ==========================================================
# Stage 4: 偏好对齐训练
# ==========================================================
# 多节点启动示例（在主节点执行）:

CUDA_VISIBLE_DEVICES=5 torchrun \
     --nproc_per_node=1 \
     --nnodes=1 \
     --master_addr=localhost \
     --master_port=29501 \
     --node_rank=0 \
     trainer_api.py \
     --stage 4 \
     --config configs/training_config.yaml

