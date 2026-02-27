"""
Stage 1: 模态对齐预训练 Trainer
Stage 2: 统一多模态预训练 Trainer
"""

import os
import shutil
import time
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.modeling_minicpmo import MiniCPMOConfig, MiniCPMOModel
from data.dataset_stage1_2 import Stage1AlignmentDataset, Stage2MultimodalDataset


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


class TrainingMetrics:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_dir / "train_log.jsonl", "a")

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
    from torch.nn.utils.rnn import pad_sequence
    keys_to_pad = ["input_ids", "attention_mask", "labels"]
    result = {}
    for key in keys_to_pad:
        tensors = [b[key] for b in batch if key in b]
        if tensors:
            result[key] = pad_sequence(tensors, batch_first=True,
                                       padding_value=-100 if key == "labels" else 0)
    image_batch = [b for b in batch if "pixel_values" in b]
    if image_batch:
        result["pixel_values"] = torch.cat([b["pixel_values"] for b in image_batch], dim=0)
        result["image_sample_indices"] = [i for i, b in enumerate(batch) if "pixel_values" in b]
    audio_batch = [b for b in batch if "audio_mel" in b]
    if audio_batch:
        result["audio_mel"] = torch.stack([b["audio_mel"] for b in audio_batch])
        result["audio_mask"] = torch.stack([b["audio_mask"] for b in audio_batch])
        result["audio_sample_indices"] = [i for i, b in enumerate(batch) if "audio_mel" in b]
    return result


# ============================================================
# 检查点滚动管理器
# ============================================================

class CheckpointManager:
    """
    滚动保存检查点：
    - 最多保留 max_keep 个常规检查点（自动删除最旧的）
    - final 检查点始终保留，不计入 max_keep
    - 维护 checkpoint-latest 软链接指向最新检查点
    """

    def __init__(self, output_dir: Path, max_keep: int = 3):
        self.output_dir = output_dir
        self.max_keep = max_keep
        self.saved: List[Path] = []  # 只追踪常规检查点（不含 final）

    def _cleanup(self):
        """删除超出 max_keep 的最旧检查点"""
        while len(self.saved) > self.max_keep:
            oldest = self.saved.pop(0)
            if oldest.exists():
                shutil.rmtree(oldest)
                print(f"[Checkpoint] 删除旧检查点: {oldest.name}", flush=True)

    def save(self, step, save_fn):
        """
        step    : int 或 "final"
        save_fn : callable(save_dir: Path)，负责实际写文件
        """
        save_dir = self.output_dir / f"checkpoint-{step}"
        save_dir.mkdir(parents=True, exist_ok=True)

        save_fn(save_dir)

        is_final = (step == "final")
        if not is_final:
            self.saved.append(save_dir)
            self._cleanup()

        # 更新 latest 软链接
        latest_link = self.output_dir / "checkpoint-latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(save_dir.name)

        kept = [p.name for p in self.saved]
        print(f"[Checkpoint] 保存 → {save_dir.name}  保留列表: {kept}", flush=True)


# ============================================================
# Stage 1 Trainer
# ============================================================

class Stage1AlignmentTrainer:
    def __init__(
        self,
        model: MiniCPMOModel,
        train_dataset: Stage1AlignmentDataset,
        output_dir: str,
        learning_rate: float = 2e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 10,
        batch_size_per_gpu: int = 2,
        gradient_accumulation_steps: int = 2,
        max_grad_norm: float = 1.0,
        save_steps: int = 1,           # 每隔多少 opt_step 保存一次
        max_keep_checkpoints: int = 3,   # 最多保留几个检查点
        log_steps: int = 1,
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

        # 检查点管理器
        self.ckpt_manager = CheckpointManager(self.output_dir, max_keep=max_keep_checkpoints)

        trainable_params = model.get_trainable_parameters(stage=1)
        num_trainable = sum(p.numel() for p in trainable_params)
        if self.is_main_process:
            print(f"[Stage1] 可训练参数: {num_trainable/1e6:.1f}M")
            total = sum(p.numel() for p in model.parameters())
            print(f"[Stage1] 总参数: {total/1e6:.1f}M, 训练比例: {num_trainable/total*100:.1f}%")

        self.optimizer = AdamW(trainable_params, lr=learning_rate,
                               weight_decay=weight_decay, betas=(0.9, 0.95))
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, max_steps)

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

        if world_size > 1:
            self.model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

        self.metrics = TrainingMetrics(str(self.output_dir / "logs"))
        self.dtype = torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler() if self.dtype == torch.float16 else None

    def train(self):
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

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            pixel_values   = batch.get("pixel_values", None)
            audio_mel      = batch.get("audio_mel", None)
            audio_mask     = batch.get("audio_mask", None)

            if pixel_values is not None:
                pixel_values = pixel_values.to(device, dtype=self.dtype)
            if audio_mel is not None:
                audio_mel = audio_mel.to(device, dtype=self.dtype)
            if audio_mask is not None:
                audio_mask = audio_mask.to(device)

            with torch.amp.autocast('cuda', dtype=self.dtype):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    audio_mel=audio_mel,
                    audio_mask=audio_mask,
                    labels=labels,
                )
                loss = outputs["loss"] / self.gradient_accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.gradient_accumulation_steps
            print(f"[LOSS] step={step}, loss={loss.item():.4f}", flush=True)

            #  修复：所有保存与日志逻辑统一移入梯度累积块内，使用 opt_step 判断
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()

                opt_step = (step + 1) // self.gradient_accumulation_steps

                #  修复：optimizer.step() 之后保存，以 opt_step 为基准
                if self.is_main_process and opt_step % self.save_steps == 0:
                    print("正在保存模型...", flush=True)
                    self._save_checkpoint(opt_step)

                if self.is_main_process and opt_step % self.log_steps == 0:
                    avg_loss = total_loss / self.log_steps
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"[Stage1] Step {opt_step}/{self.max_steps // self.gradient_accumulation_steps}"
                        f" | loss={avg_loss:.4f} | lr={lr:.2e} | grad_norm={grad_norm:.3f}",
                        flush=True
                    )
                    self.metrics.log(opt_step, {
                        "stage": 1, "loss": avg_loss, "lr": lr, "grad_norm": float(grad_norm)
                    })
                    total_loss = 0.0

            step += 1

        if self.is_main_process:
            self._save_checkpoint("final")
            print("[Stage1] 训练完成！", flush=True)

    def _save_checkpoint(self, step):
        model = self.model.module if hasattr(self.model, "module") else self.model

        def _write(save_dir: Path):
            torch.save({
                "vision_projector": model.vision_projector.state_dict(),
                "video_resampler":  model.video_resampler.state_dict(),
                "audio_resampler":  model.audio_resampler.state_dict(),
            }, save_dir / "projector.pt")

        self.ckpt_manager.save(step, _write)


# ============================================================
# Stage 2 Trainer
# ============================================================

class Stage2PretrainTrainer:
    def __init__(
        self,
        model: MiniCPMOModel,
        train_dataset: Stage2MultimodalDataset,
        output_dir: str,
        encoder_lr: float = 1e-5,
        projector_lr: float = 5e-4,
        llm_lr: float = 1e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100,
        batch_size_per_gpu: int = 4,
        gradient_accumulation_steps: int = 64,
        max_grad_norm: float = 1.0,
        save_steps: int = 5,           # 每隔多少 opt_step 保存一次
        max_keep_checkpoints: int = 3,   # 最多保留几个检查点
        log_steps: int = 1,
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

        # 检查点管理器
        self.ckpt_manager = CheckpointManager(self.output_dir, max_keep=max_keep_checkpoints)

        if projector_checkpoint:
            state = torch.load(projector_checkpoint, map_location="cpu")
            model.vision_projector.load_state_dict(state["vision_projector"])
            model.video_resampler.load_state_dict(state["video_resampler"])
            model.audio_resampler.load_state_dict(state["audio_resampler"])
            if self.is_main_process:
                print(f"[Stage2] 加载 Stage1 Projector: {projector_checkpoint}")

        param_groups = model.get_trainable_parameters(stage=2)
        for group in param_groups:
            if group.get("lr") == 1e-5:   group["lr"] = encoder_lr
            elif group.get("lr") == 5e-4: group["lr"] = projector_lr
            elif group.get("lr") == 1e-4: group["lr"] = llm_lr

        self.optimizer = AdamW(param_groups, weight_decay=weight_decay, betas=(0.9, 0.95))
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, max_steps)

        sample_weights = train_dataset.get_sampler_weights()
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size_per_gpu,
            sampler=WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True),
            collate_fn=collate_fn_stage1,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        if world_size > 1:
            self.model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

        self.metrics = TrainingMetrics(str(self.output_dir / "logs"))
        self.dtype = torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler() if self.dtype == torch.float16 else None

    def train(self):
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

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            pixel_values   = batch.get("pixel_values")
            video_frames   = batch.get("video_frames")
            audio_mel      = batch.get("audio_mel")
            audio_mask     = batch.get("audio_mask")

            if pixel_values  is not None: pixel_values  = pixel_values.to(device, dtype=self.dtype)
            if video_frames  is not None: video_frames  = video_frames.to(device, dtype=self.dtype)
            if audio_mel     is not None: audio_mel     = audio_mel.to(device, dtype=self.dtype)

            with torch.amp.autocast('cuda', dtype=self.dtype):
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

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.gradient_accumulation_steps
            print(f"[STEP LOSS] step={step}, loss={loss.item():.4f}", flush=True)
            #  修复：所有保存与日志逻辑统一移入梯度累积块内，使用 opt_step 判断
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()

                opt_step = (step + 1) // self.gradient_accumulation_steps

                #  修复：optimizer.step() 之后保存，以 opt_step 为基准
                if self.is_main_process and opt_step % self.save_steps == 0:
                    print("正在保存模型...", flush=True)
                    self._save_checkpoint(opt_step)

                if self.is_main_process and opt_step % self.log_steps == 0:
                    avg_loss = total_loss / self.log_steps
                    lrs = [g["lr"] for g in self.optimizer.param_groups]
                    print(
                        f"[Stage2] Step {opt_step} | loss={avg_loss:.4f} | "
                        f"lr_enc={lrs[0]:.2e} lr_proj={lrs[1]:.2e} lr_llm={lrs[2]:.2e} | "
                        f"grad_norm={grad_norm:.3f}", flush=True
                    )
                    self.metrics.log(opt_step, {
                        "stage": 2, "loss": avg_loss,
                        "lr_encoder": lrs[0], "lr_projector": lrs[1], "lr_llm": lrs[2],
                        "grad_norm": float(grad_norm),
                    })
                    total_loss = 0.0

            step += 1

        if self.is_main_process:
            self._save_checkpoint("final")
            print("[Stage2] 训练完成！", flush=True)  #  修复：原来错误地写成了 [Stage1]

    def _save_checkpoint(self, step):
        model = self.model.module if hasattr(self.model, "module") else self.model

        def _write(save_dir: Path):
            torch.save(model.state_dict(), save_dir / "model.pt")
            torch.save({
                "vision_projector": model.vision_projector.state_dict(),
                "video_resampler":  model.video_resampler.state_dict(),
                "audio_resampler":  model.audio_resampler.state_dict(),
            }, save_dir / "projector.pt")

        self.ckpt_manager.save(step, _write)
