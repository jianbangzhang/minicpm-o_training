# MiniCPM-o 4.5 完整训练框架

## 项目结构

```
minicpmo_training/
├── README.md
├── build_data
│   ├── audio_text
│   │   ├── G00126S1001.txt
│   │   ├── G00126S1001.wav
│   │   └── audio_text.json
│   ├── audio_text.py
│   ├── chat_stage3
│   │   └── chat.json
│   ├── chat_stage4
│   │   ├── conversation.json
│   │   └── data.json
│   ├── create_stage3_dataset.py
│   ├── create_stage4_dataset.py
│   ├── image_text
│   │   ├── Train_GCC-training_truncated.tsv
│   │   ├── coco_dataset
│   │   │   ├── coco_download.py
│   │   │   └── download.py
│   │   ├── data.txt
│   │   ├── image_text.json
│   │   └── images
│   │       ├── 00000000.jpg
│   │       └── 00000018.jpg
│   ├── image_text.py
│   ├── ocr_data
│   │   ├── Challenge4_Test_Task1_GT
│   │   │   ├── gt_img_1.txt
│   │   │   └── 00001.jpg
│   │   └── videos
│   └── visual_text.py
├── checkpoints
│   ├── stage1
│   ├── stage2
│   ├── stage3
│   └── stage4
├── configs
│   ├── __init__.py
│   ├── training_config.yaml
│   └── training_config2.yaml
├── data
│   ├── __init__.py
│   ├── dataset_stage1_2.py
│   ├── dataset_stage3.py
│   └── dataset_stage4.py
├── doc
│   ├── MiniCPM-o_4.5_训练方案设计.docx
│   └── MiniCPM_o_4_5_训练数据方案.docx
├── models
│   ├── __init__.py
│   └── modeling_minicpmo.py
├── raw
│   ├── datasets
│   │   ├── stage1
│   │   │   ├── example.json
│   │   │   └── train_data.json
│   │   ├── stage2
│   │   │   ├── example.json
│   │   │   └── train_data.json
│   │   ├── stage3
│   │   │   └── example.json
│   │   └── stage4
│   │       └── example.json
│   └── models
│       ├── download_cosyvoice2.sh
│       ├── download_qwen3-0.6b.sh
│       ├── download_siglip2-so400m-patch14-384.sh
│       └── download_whisper_medium.sh
├── requirements.txt
├── scripts
│   ├── __init__.py
│   └── run_stage.sh
├── trainer_api.py
├── trainers
│   ├── __init__.py
│   ├── trainer_stage1_2.py
│   ├── trainer_stage3.py
│   └── trainer_stage4.py
└── utils
    ├── __init__.py
    ├── ocr_noise.py
    └── video_audio_processor.py
```

## 快速启动

```bash
# Stage 1: 模态对齐
bash scripts/run_stage.sh

# Stage 2: 多模态预训练
bash scripts/run_stage.sh

# Stage 3: 监督微调
bash scripts/run_stage.sh

# Stage 4: 强化学习后训练
bash scripts/run_stage.sh
```

## 架构概述

| 组件 | 基础模型 | 参数量       |
|------|---------|-----------|
| 视觉编码器 | SigLip2-400M | 400M      |
| 语音编码器 | Whisper-medium | 300M      |
| 语言模型 | Qwen3-8B | 8B/0.6B可选 |
| 语音解码器 | CosyVoice2 | ~500M     |
| **总计** | **MiniCPM-o 4.5** | **~9B**   |
