# MiniCPM-o 4.5 完整训练框架

## 项目结构

```
minicpmo_training/
├── configs/                    # 各阶段训练配置
│   ├── stage1_align.yaml       # Stage 1: 模态对齐预训练
│   ├── stage2_pretrain.yaml    # Stage 2: 多模态统一预训练
│   ├── stage3_sft.yaml         # Stage 3: 监督微调
│   └── stage4_rl.yaml          # Stage 4: 强化学习后训练
├── data/
│   ├── dataset_stage1.py       # Stage 1 数据集 (图文对 + 语音文本对)
│   ├── dataset_stage2.py       # Stage 2 数据集 (OCR噪声掩码 + 视频 + 多模态)
│   ├── dataset_stage3.py       # Stage 3 SFT 数据集 (两阶段指令数据)
│   ├── dataset_stage4.py       # Stage 4 RL 数据集 (偏好数据 + 可验证推理)
│   └── data_utils.py           # 数据处理工具函数
├── models/
│   ├── modeling_minicpmo.py    # 完整模型架构 (对齐官方结构)
│   └── projector.py            # 视觉/语音 Projector 模块
├── trainers/
│   ├── trainer_stage1.py       # Stage 1 Projector对齐训练器
│   ├── trainer_stage2.py       # Stage 2 全参数预训练器
│   ├── trainer_stage3.py       # Stage 3 SFT训练器 (两阶段数据调度)
│   └── trainer_stage4.py       # Stage 4 GRPO + DPO 训练器
├── utils/
│   ├── ocr_noise.py            # 动态OCR噪声掩码工具
│   ├── video_processor.py      # 视频帧压缩 (6帧→64tokens)
│   ├── audio_processor.py      # 音频预处理与特征提取
│   └── reward_functions.py     # RL奖励函数 (RLPR + RLAIF-V)
└── scripts/
    ├── run_stage1.sh
    ├── run_stage2.sh
    ├── run_stage3.sh
    └── run_stage4.sh
```

## 快速启动

```bash
# Stage 1: 模态对齐 (8×A100, ~5天)
bash scripts/run_stage.sh

# Stage 2: 多模态预训练 (32×A100, ~3周)
bash scripts/run_stage2.sh

# Stage 3: 监督微调 (16×A100, ~2周)
bash scripts/run_stage3.sh

# Stage 4: 强化学习后训练 (16×A100, ~7天)
bash scripts/run_stage4.sh
```

## 架构概述

| 组件 | 基础模型 | 参数量       |
|------|---------|-----------|
| 视觉编码器 | SigLip2-400M | 400M      |
| 语音编码器 | Whisper-medium | 300M      |
| 语言模型 | Qwen3-8B | 8B/0.6B可选 |
| 语音解码器 | CosyVoice2 | ~500M     |
| **总计** | **MiniCPM-o 4.5** | **~9B**   |
