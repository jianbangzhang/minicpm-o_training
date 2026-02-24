"""
Stage 1: 模态对齐预训练 Trainer
Stage 2: 统一多模态预训练 Trainer

Stage 1: 冻结 LLM 和 Encoder，只训练 Projector
Stage 2: 全参数训练，分差异化学习率
"""

import os
import time
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..models.modeling_minicpmo import MiniCPMOConfig, MiniCPMOModel
from ..data.dataset_stage1_2 import Stage1AlignmentDataset, Stage2MultimodalDataset


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """余弦退火学习率调度（含 warmup）"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


class TrainingMetrics:
    """训练指标追踪"""
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_dir / "train_log.jsonl", "a")
        self.step_losses = []

    def log(self, step: int, metrics: Dict):
        import json
        metrics["step"] = step
        metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        self.log_file.flush()

    def __del__(self):
        if hasattr(self, "log_file"):
            self.log_file.close()


def collate_fn_stage1(batch: List[Dict]) -> Dict:
    """Stage 1 collate function，处理变长序列"""
    from torch.nn.utils.rnn import pad_sequence

    keys_to_pad = ["input_ids", "attention_mask", "labels"]
    result = {}

    for key in keys_to_pad:
        tensors = [b[key] for b in batch if key in b]
        if tensors:
            result[key] = pad_sequence(tensors, batch_first=True,
                                       padding_value=-100 if key == "labels" else 0)

    # 图像处理：收集所有图像样本
    image_batch = [b for b in batch if "pixel_values" in b]
    if image_batch:
        result["pixel_values"] = torch.cat([b["pixel_values"] for b in image_batch], dim=0)
        result["image_sample_indices"] = [
            i for i, b in enumerate(batch) if "pixel_values" in b
        ]

    # 语音处理
    audio_batch = [b for b in batch if "audio_mel" in b]
    if audio_batch:
        result["audio_mel"] = torch.stack([b["audio_mel"] for b in audio_batch])
        result["audio_mask"] = torch.stack([b["audio_mask"] for b in audio_batch])
        result["audio_sample_indices"] = [
            i for i, b in enumerate(batch) if "audio_mel" in b
        ]

    return result


# ============================================================
# Stage 1 Trainer
# ============================================================

class Stage1AlignmentTrainer:
    """
    Stage 1 模态对齐预训练 Trainer。

    训练目标：冻结 LLM 和 Encoder，只训练 Projector 对齐层。
    数据：大规模图文对 + 语音文本对 (各 ~50M 条)

    关键超参数：
      lr = 2e-3 (Projector 学习率偏高，因其从头训练)
      warmup = 1000 步
      batch_size = 4096 (有效)
      max_steps ≈ 50000
    """

    def __init__(
        self,
        model: MiniCPMOModel,
        train_dataset: Stage1AlignmentDataset,
        output_dir: str,
        # 超参数
        learning_rate: float = 2e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 50_000,
        batch_size_per_gpu: int = 32,
        gradient_accumulation_steps: int = 16,
        max_grad_norm: float = 1.0,
        save_steps: int = 2000,
        log_steps: int = 50,
        # 分布式
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.local_rank = local_rank
        self.world_size = world_size
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.log_steps = log_steps
        self.is_main_process = local_rank == 0

        # Stage 1: 只训练 Projector
        trainable_params = model.get_trainable_parameters(stage=1)
        num_trainable = sum(p.numel() for p in trainable_params)
        if self.is_main_process:
            print(f"[Stage1] 可训练参数: {num_trainable/1e6:.1f}M")
            total = sum(p.numel() for p in model.parameters())
            print(f"[Stage1] 总参数: {total/1e6:.1f}M, 训练比例: {num_trainable/total*100:.1f}%")

        # 优化器（只优化 Projector 参数）
        self.optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, max_steps
        )

        # 数据加载器
        if world_size > 1:
            sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
        else:
            sampler = None
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size_per_gpu,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=collate_fn_stage1,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        # DDP 包装
        if world_size > 1:
            self.model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

        self.metrics = TrainingMetrics(str(self.output_dir / "logs"))

        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler()
        self.dtype = torch.bfloat16

    def train(self):
        """主训练循环"""
        self.model.train()
        device = next(self.model.parameters()).device

        step = 0
        total_loss = 0.0
        self.optimizer.zero_grad()

        data_iter = iter(self.dataloader)

        while step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            # 移动数据到 GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch.get("pixel_values", None)
            audio_mel = batch.get("audio_mel", None)
            audio_mask = batch.get("audio_mask", None)

            if pixel_values is not None:
                pixel_values = pixel_values.to(device, dtype=self.dtype)
            if audio_mel is not None:
                audio_mel = audio_mel.to(device, dtype=self.dtype)
            if audio_mask is not None:
                audio_mask = audio_mask.to(device)

            # 前向（混合精度）
            with torch.cuda.amp.autocast(dtype=self.dtype):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    audio_mel=audio_mel,
                    audio_mask=audio_mask,
                    labels=labels,
                )
                loss = outputs["loss"] / self.gradient_accumulation_steps

            # 反向
            self.scaler.scale(loss).backward()
            total_loss += loss.item() * self.gradient_accumulation_steps

            # 梯度累积更新
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

                opt_step = (step + 1) // self.gradient_accumulation_steps

                # 日志
                if self.is_main_process and opt_step % self.log_steps == 0:
                    avg_loss = total_loss / self.log_steps
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"[Stage1] Step {opt_step}/{self.max_steps//self.gradient_accumulation_steps}"
                        f" | loss={avg_loss:.4f} | lr={lr:.2e} | grad_norm={grad_norm:.3f}"
                    )
                    self.metrics.log(opt_step, {
                        "stage": 1, "loss": avg_loss, "lr": lr, "grad_norm": float(grad_norm)
                    })
                    total_loss = 0.0

                # 保存
                if self.is_main_process and opt_step % self.save_steps == 0:
                    self._save_checkpoint(opt_step)

            step += 1

        if self.is_main_process:
            self._save_checkpoint("final")
            print("[Stage1] 训练完成！")

    def _save_checkpoint(self, step):
        """保存检查点（只保存 Projector 权重）"""
        save_dir = self.output_dir / f"checkpoint-{step}"
        save_dir.mkdir(exist_ok=True)

        model = self.model.module if hasattr(self.model, "module") else self.model
        # Stage 1 只保存 Projector
        projector_state = {
            "vision_projector": model.vision_projector.state_dict(),
            "video_resampler": model.video_resampler.state_dict(),
            "audio_resampler": model.audio_resampler.state_dict(),
        }
        torch.save(projector_state, save_dir / "projector.pt")
        print(f"[Stage1] 保存 Projector checkpoint → {save_dir}")


# ============================================================
# Stage 2 Trainer
# ============================================================

class Stage2PretrainTrainer:
    """
    Stage 2 统一多模态预训练 Trainer。

    全参数训练，差异化学习率：
      encoder:    1e-5 (已预训练，学习率低)
      projector:  5e-4 (从 Stage1 初始化，继续提升)
      llm:        1e-4 (骨干，中等学习率)

    核心特性：
      - 动态 OCR 噪声掩码（在 Dataset 中处理）
      - WeightedRandomSampler 控制数据配比
      - 长序列训练（max_len=8192）
      - Flash Attention 2 加速
    """

    def __init__(
        self,
        model: MiniCPMOModel,
        train_dataset: Stage2MultimodalDataset,
        output_dir: str,
        # 学习率（按组）
        encoder_lr: float = 1e-5,
        projector_lr: float = 5e-4,
        llm_lr: float = 1e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 200_000,
        batch_size_per_gpu: int = 4,
        gradient_accumulation_steps: int = 64,
        max_grad_norm: float = 1.0,
        save_steps: int = 5000,
        log_steps: int = 100,
        local_rank: int = 0,
        world_size: int = 32,
        projector_checkpoint: Optional[str] = None,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.local_rank = local_rank
        self.world_size = world_size
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.log_steps = log_steps
        self.is_main_process = local_rank == 0

        # 加载 Stage 1 的 Projector 权重
        if projector_checkpoint:
            state = torch.load(projector_checkpoint, map_location="cpu")
            model.vision_projector.load_state_dict(state["vision_projector"])
            model.video_resampler.load_state_dict(state["video_resampler"])
            model.audio_resampler.load_state_dict(state["audio_resampler"])
            if self.is_main_process:
                print(f"[Stage2] 加载 Stage1 Projector: {projector_checkpoint}")

        # 差异化参数组学习率
        param_groups = model.get_trainable_parameters(stage=2)
        # param_groups 是 [{"params": ..., "lr": ...}, ...]
        # 覆盖为配置的学习率
        for group in param_groups:
            if group.get("lr") == 1e-5:
                group["lr"] = encoder_lr
            elif group.get("lr") == 5e-4:
                group["lr"] = projector_lr
            elif group.get("lr") == 1e-4:
                group["lr"] = llm_lr

        self.optimizer = AdamW(
            param_groups,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, max_steps
        )

        # WeightedRandomSampler 实现数据配比
        sample_weights = train_dataset.get_sampler_weights()
        weighted_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )

        self.dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size_per_gpu,
            sampler=weighted_sampler,
            collate_fn=collate_fn_stage1,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        if world_size > 1:
            self.model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

        self.metrics = TrainingMetrics(str(self.output_dir / "logs"))
        self.scaler = torch.cuda.amp.GradScaler()
        self.dtype = torch.bfloat16

    def train(self):
        """主训练循环（与 Stage1 类似，但全参数更新）"""
        self.model.train()
        device = next(self.model.parameters()).device
        step = 0
        total_loss = 0.0
        loss_by_type = {"image_text": 0.0, "ocr_document": 0.0, "video": 0.0, "audio_text": 0.0}
        type_counts = {k: 0 for k in loss_by_type}

        self.optimizer.zero_grad()
        data_iter = iter(self.dataloader)

        while step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            pixel_values = batch.get("pixel_values")
            video_frames = batch.get("video_frames")
            audio_mel = batch.get("audio_mel")
            audio_mask = batch.get("audio_mask")

            if pixel_values is not None:
                pixel_values = pixel_values.to(device, dtype=self.dtype)
            if video_frames is not None:
                video_frames = video_frames.to(device, dtype=self.dtype)
            if audio_mel is not None:
                audio_mel = audio_mel.to(device, dtype=self.dtype)

            with torch.cuda.amp.autocast(dtype=self.dtype):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    video_frames=video_frames,
                    audio_mel=audio_mel,
                    audio_mask=audio_mask.to(device) if audio_mask is not None else None,
                    labels=labels,
                )
                loss = outputs["loss"] / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward()
            total_loss += loss.item() * self.gradient_accumulation_steps

            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

                opt_step = (step + 1) // self.gradient_accumulation_steps

                if self.is_main_process and opt_step % self.log_steps == 0:
                    avg_loss = total_loss / self.log_steps
                    lrs = [g["lr"] for g in self.optimizer.param_groups]
                    print(
                        f"[Stage2] Step {opt_step} | loss={avg_loss:.4f} | "
                        f"lr_enc={lrs[0]:.2e} lr_proj={lrs[1]:.2e} lr_llm={lrs[2]:.2e} | "
                        f"grad_norm={grad_norm:.3f}"
                    )
                    self.metrics.log(opt_step, {
                        "stage": 2, "loss": avg_loss,
                        "lr_encoder": lrs[0], "lr_projector": lrs[1], "lr_llm": lrs[2],
                        "grad_norm": float(grad_norm),
                    })
                    total_loss = 0.0

                if self.is_main_process and opt_step % self.save_steps == 0:
                    self._save_checkpoint(opt_step)

            step += 1

        if self.is_main_process:
            self._save_checkpoint("final")
            print("[Stage2] 预训练完成！")

    def _save_checkpoint(self, step):
        """保存全量检查点（包含所有组件）"""
        save_dir = self.output_dir / f"checkpoint-{step}"
        save_dir.mkdir(exist_ok=True)
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model.state_dict(), save_dir / "model.pt")

        # 单独保存 Projector（方便 Stage3 加载）
        projector_state = {
            "vision_projector": model.vision_projector.state_dict(),
            "video_resampler": model.video_resampler.state_dict(),
            "audio_resampler": model.audio_resampler.state_dict(),
        }
        torch.save(projector_state, save_dir / "projector.pt")
        print(f"[Stage2] 保存全量 checkpoint → {save_dir}")
