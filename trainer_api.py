#!/usr/bin/env python3
"""
MiniCPM-o 4.5 四阶段训练启动脚本
用法: torchrun --nproc_per_node=8 trainer_api.py --stage 1
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

# 项目根目录
sys.path.insert(0, str(Path(__file__).parent))

from models.modeling_minicpmo import MiniCPMOConfig, MiniCPMOModel
from utils.ocr_noise import DynamicOCRNoise
from utils.video_audio_processor import AudioProcessor, VideoFrameProcessor


def setup_distributed():
    """初始化分布式训练环境"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return local_rank, world_size


def load_config(config_path: str, stage: int) -> dict:
    with open(config_path, "r") as f:
        all_config = yaml.safe_load(f)
    stage_key = f"stage{stage}"
    return all_config[stage_key]


# ============================================================
# Stage 1 入口
# ============================================================

def run_stage1(cfg: dict, local_rank: int, world_size: int):
    from data.dataset_stage1_2 import Stage1AlignmentDataset
    from trainers.trainer_stage1_2 import Stage1AlignmentTrainer
    from transformers import AutoTokenizer

    device = torch.device(f"cuda:{local_rank}")
    print(f"[Stage1] Rank {local_rank}/{world_size} | Device: {device}")

    # 构建配置
    model_config = MiniCPMOConfig(
        vision_encoder_name=cfg["vision_encoder"],
        audio_encoder_name=cfg["audio_encoder"],
        llm_name=cfg["llm"],
    )

    # 初始化模型
    model = MiniCPMOModel.from_pretrained_components(
        model_config, init_vision=True, init_audio=True
    ).to(device)

    # 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["llm"])
    # 添加特殊 token
    special_tokens = ["<image>", "</image>", "<audio>", "</audio>",
                      "<video>", "</video>", "<think>", "</think>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.llm.resize_token_embeddings(len(tokenizer))

    # 设置特殊 token id
    model._im_start_id = tokenizer.convert_tokens_to_ids("<image>")
    model._im_end_id = tokenizer.convert_tokens_to_ids("</image>")
    model._audio_start_id = tokenizer.convert_tokens_to_ids("<audio>")
    model._audio_end_id = tokenizer.convert_tokens_to_ids("</audio>")

    # 初始化处理器
    audio_processor = AudioProcessor(train_mode=True)

    # 数据集
    data_cfg = cfg["data"]
    all_paths = data_cfg.get("image_text_paths", []) + data_cfg.get("audio_text_paths", [])
    # 过滤存在的路径
    valid_paths = [p for p in all_paths if Path(p).exists()]
    if not valid_paths:
        print("[Warning] 未找到数据文件，使用示例路径（请替换为真实数据）")
        valid_paths = ["/data/stage1/example.jsonl"]

    dataset = Stage1AlignmentDataset(
        data_paths=valid_paths,
        tokenizer=tokenizer,
        audio_processor=audio_processor,
    )

    # 训练器
    trainer = Stage1AlignmentTrainer(
        model=model,
        train_dataset=dataset,
        output_dir=cfg["output_dir"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        max_steps=cfg["max_steps"],
        batch_size_per_gpu=cfg["batch_size_per_gpu"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        save_steps=cfg["save_steps"],
        log_steps=cfg["log_steps"],
        local_rank=local_rank,
        world_size=world_size,
    )

    trainer.train()


# ============================================================
# Stage 2 入口
# ============================================================

def run_stage2(cfg: dict, local_rank: int, world_size: int):
    from data.dataset_stage1_2 import Stage2MultimodalDataset
    from trainers.trainer_stage1_2 import Stage2PretrainTrainer
    from transformers import AutoTokenizer

    device = torch.device(f"cuda:{local_rank}")
    print(f"[Stage2] Rank {local_rank}/{world_size}")

    model_config = MiniCPMOConfig()
    model = MiniCPMOModel.from_pretrained_components(model_config).to(device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.get("llm", "Qwen/Qwen3-8B"))
    special_tokens = ["<image>", "</image>", "<audio>", "</audio>",
                      "<video>", "</video>", "<think>", "</think>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    audio_processor = AudioProcessor(train_mode=True)
    video_processor = VideoFrameProcessor(train_mode=True)
    ocr_noiser = DynamicOCRNoise(
        noise_level_range=tuple(cfg["ocr_noise"]["noise_level_range"]),
        apply_prob=cfg["ocr_noise"]["apply_prob"],
    )

    # 构建数据路径字典
    data_paths = {}
    for data_type, paths in cfg["data"]["paths"].items():
        valid = [p for p in paths if Path(p).exists()]
        if valid:
            data_paths[data_type] = valid

    if not data_paths:
        print("[Warning] 未找到 Stage2 数据，使用示例结构")
        data_paths = {"interleaved": ["/data/stage2/example.jsonl"]}

    dataset = Stage2MultimodalDataset(
        data_paths=data_paths,
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        video_processor=video_processor,
        ocr_noiser=ocr_noiser,
        max_seq_len=cfg["max_seq_len"],
    )

    trainer = Stage2PretrainTrainer(
        model=model,
        train_dataset=dataset,
        output_dir=cfg["output_dir"],
        encoder_lr=cfg["encoder_lr"],
        projector_lr=cfg["projector_lr"],
        llm_lr=cfg["llm_lr"],
        warmup_steps=cfg["warmup_steps"],
        max_steps=cfg["max_steps"],
        batch_size_per_gpu=cfg["batch_size_per_gpu"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        local_rank=local_rank,
        world_size=world_size,
        projector_checkpoint=cfg.get("projector_checkpoint"),
    )

    trainer.train()


# ============================================================
# Stage 3 入口
# ============================================================

def run_stage3(cfg: dict, local_rank: int, world_size: int):
    from data.dataset_stage3 import SFTDataset
    from trainers.trainer_stage3 import Stage3SFTTrainer
    from transformers import AutoTokenizer

    device = torch.device(f"cuda:{local_rank}")
    model_config = MiniCPMOConfig()
    model = MiniCPMOModel.from_pretrained_components(model_config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    special_tokens = ["<image>", "</image>", "<audio>", "</audio>",
                      "<video>", "</video>", "<think>", "</think>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    audio_processor = AudioProcessor(train_mode=True)
    video_processor = VideoFrameProcessor(train_mode=True)

    def make_dataset(data_paths, phase):
        valid_paths = [p for p in data_paths if Path(p).exists()]
        return SFTDataset(
            data_paths=valid_paths if valid_paths else ["/data/sft/example.jsonl"],
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            video_processor=video_processor,
            max_seq_len=8192,
            phase=phase,
            fast_think_ratio=cfg["fast_think_ratio"],
        )

    data_cfg = cfg["data"]
    part1_dataset = make_dataset(data_cfg["part1"], "part1")
    part2_dataset = make_dataset(data_cfg["part2"], "part2")
    final_dataset = make_dataset(data_cfg["final"], "final")

    trainer = Stage3SFTTrainer(
        model=model,
        part1_dataset=part1_dataset,
        part2_dataset=part2_dataset,
        final_dataset=final_dataset,
        output_dir=cfg["output_dir"],
        pretrained_checkpoint=cfg.get("pretrained_checkpoint"),
        learning_rate=cfg["learning_rate"],
        total_steps=cfg["total_steps"],
        batch_size_per_gpu=cfg["batch_size_per_gpu"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        use_lora=cfg["use_lora"],
        lora_rank=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        local_rank=local_rank,
        world_size=world_size,
    )

    trainer.train(batch_size_per_gpu=cfg["batch_size_per_gpu"])


# ============================================================
# Stage 4 入口
# ============================================================

def run_stage4(cfg: dict, local_rank: int, world_size: int):
    from data.dataset_stage4 import RLPRDataset, RLAIFVDataset
    from trainers.trainer_stage4 import Stage4RLTrainer
    from transformers import AutoTokenizer
    import copy

    device = torch.device(f"cuda:{local_rank}")
    model_config = MiniCPMOConfig()

    # 策略模型（继续训练）
    policy_model = MiniCPMOModel.from_pretrained_components(model_config).to(device)
    sft_state = torch.load(cfg["sft_checkpoint"], map_location=device)
    policy_model.load_state_dict(sft_state, strict=False)

    # 参考模型（冻结，保持 SFT 权重）
    ref_model = MiniCPMOModel.from_pretrained_components(model_config).to(device)
    ref_model.load_state_dict(sft_state, strict=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    special_tokens = ["<image>", "</image>", "<audio>", "</audio>",
                      "<video>", "</video>", "<think>", "</think>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # RL 数据集
    rlpr_paths = [p for p in cfg["data"]["rlpr"] if Path(p).exists()]
    rlaifv_paths = [p for p in cfg["data"]["rlaifv"] if Path(p).exists()]

    rlpr_dataset = RLPRDataset(
        data_paths=rlpr_paths if rlpr_paths else ["/data/rl/example_rlpr.jsonl"],
        tokenizer=tokenizer,
    )
    rlaifv_dataset = RLAIFVDataset(
        data_paths=rlaifv_paths if rlaifv_paths else ["/data/rl/example_rlaifv.jsonl"],
        tokenizer=tokenizer,
    )

    trainer = Stage4RLTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        rlpr_dataset=rlpr_dataset,
        rlaifv_dataset=rlaifv_dataset,
        output_dir=cfg["output_dir"],
        grpo_lr=cfg["grpo_lr"],
        dpo_lr=cfg["dpo_lr"],
        dpo_beta=cfg["dpo_beta"],
        kl_coeff=cfg["kl_coeff"],
        num_samples_per_prompt=cfg["num_samples_per_prompt"],
        total_rl_steps=cfg["total_rl_steps"],
        grpo_dpo_ratio=cfg["grpo_dpo_ratio"],
        grpo_batch_size=cfg["grpo_batch_size"],
        dpo_batch_size=cfg["dpo_batch_size"],
        local_rank=local_rank,
        world_size=world_size,
    )

    trainer.train()


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MiniCPM-o 4.5 训练脚本")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4],
                        help="训练阶段 (1/2/3/4)")
    parser.add_argument("--config", type=str,
                        default="configs/training_config.yaml",
                        help="配置文件路径")
    args = parser.parse_args()

    local_rank, world_size = setup_distributed()

    # 设置随机种子
    torch.manual_seed(42 + local_rank)
    torch.cuda.manual_seed(42 + local_rank)

    cfg = load_config(args.config, args.stage)

    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"MiniCPM-o 4.5 训练 — Stage {args.stage}")
        print(f"GPU: {world_size} × {torch.cuda.get_device_name(0)}")
        print(f"{'='*60}\n")

    stage_runners = {
        1: run_stage1,
        2: run_stage2,
        3: run_stage3,
        4: run_stage4,
    }

    stage_runners[args.stage](cfg, local_rank, world_size)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
