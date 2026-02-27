"""
Stage 3: 监督微调 (SFT) Trainer

两阶段数据调度 + 混合快/深思考模式训练
支持全参数微调和 LoRA 两种模式
"""

import os
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence

from models.modeling_minicpmo import MiniCPMOModel
from data.dataset_stage3 import SFTDataset, TwoPhaseDataScheduler
import warnings
warnings.filterwarnings("ignore")

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    import math
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


def collate_fn_sft(batch: List[Dict]) -> Dict:
    """SFT collate fn，处理多模态 batch"""
    result = {}

    # Pad text tokens
    for key in ["input_ids", "attention_mask", "labels"]:
        tensors = [b[key] for b in batch]
        result[key] = pad_sequence(
            tensors, batch_first=True,
            padding_value=-100 if key == "labels" else 0
        )

    # 图像（各样本可能有不同数量的切片）
    all_pv = [b["pixel_values"] for b in batch if "pixel_values" in b]
    if all_pv:
        result["pixel_values"] = torch.cat(all_pv, dim=0)
        result["pixel_values_per_sample"] = [
            b["pixel_values"].shape[0] if "pixel_values" in b else 0
            for b in batch
        ]

    # 音频
    all_mel = [b["audio_mel"] for b in batch if "audio_mel" in b]
    if all_mel:
        result["audio_mel"] = torch.stack(all_mel)
        result["audio_mask"] = torch.stack([b["audio_mask"] for b in batch if "audio_mask" in b])

    # 语音输出标签
    all_speech = [b["speech_labels"] for b in batch if "speech_labels" in b]
    if all_speech:
        result["speech_labels"] = pad_sequence(all_speech, batch_first=True, padding_value=-100)

    # 元信息
    result["thinking_modes"] = [b.get("thinking_mode", "fast") for b in batch]
    result["sample_types"] = [b.get("sample_type", "unknown") for b in batch]

    return result


class LoRALinear(nn.Module):
    """
    LoRA 适配器，替换 LLM 中的 Linear 层。
    rank=64, alpha=128
    """
    def __init__(self, linear: nn.Linear, rank: int = 64, alpha: float = 128):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)

        # 初始化：A 用高斯，B 全零（确保初始为恒等映射）
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        # 冻结原始权重
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = self.lora_b(self.lora_a(x)) * self.scaling
        return base_out + lora_out


def apply_lora_to_llm(model: MiniCPMOModel, rank: int = 64, alpha: float = 128):
    """
    对 LLM 的所有 attention Linear 层应用 LoRA。
    目标层：q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    """
    target_modules = {"q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"}
    replaced = 0

    for name, module in model.llm.named_modules():
        parts = name.split(".")
        if parts[-1] in target_modules and isinstance(module, nn.Linear):
            parent = model.llm
            for part in parts[:-1]:
                parent = getattr(parent, part)
            lora = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, parts[-1], lora)
            replaced += 1

    print(f"[LoRA] 替换 {replaced} 个 Linear 层 (rank={rank}, alpha={alpha})")
    return model


class Stage3SFTTrainer:
    """
    Stage 3 SFT Trainer。

    核心功能：
    1. 两阶段数据动态切换（TwoPhaseDataScheduler）
    2. 混合快/深思考模式（同 batch 内混合）
    3. 支持全参数微调 / LoRA 微调
    4. 训练末尾 10% 步数自动切换高质量数据

    关键超参数：
      lr = 1e-5 (全参数)
      batch_size = 512 (有效)
      epoch = 1~2
    """

    def __init__(
        self,
        model: MiniCPMOModel,
        part1_dataset: SFTDataset,
        part2_dataset: SFTDataset,
        final_dataset: SFTDataset,
        output_dir: str,
        pretrained_checkpoint: str,
        # 超参数
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03,
        total_steps: int = 30_000,
        batch_size_per_gpu: int = 2,
        gradient_accumulation_steps: int = 128,
        max_grad_norm: float = 1.0,
        save_steps: int = 1000,
        log_steps: int = 50,
        # 训练模式
        use_lora: bool = False,
        lora_rank: int = 64,
        lora_alpha: float = 128,
        # 分布式
        local_rank: int = 0,
        world_size: int = 16,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.local_rank = local_rank
        self.world_size = world_size
        self.total_steps = total_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.log_steps = log_steps
        self.is_main_process = local_rank == 0

        # 加载 Stage 2 检查点
        if pretrained_checkpoint:
            state = torch.load(pretrained_checkpoint, map_location="cpu")
            model.load_state_dict(state, strict=False)
            if self.is_main_process:
                print(f"[Stage3] 加载 Stage2 权重: {pretrained_checkpoint}")

        # LoRA or 全参数
        if use_lora:
            model = apply_lora_to_llm(model, rank=lora_rank, alpha=lora_alpha)
            # LoRA 模式：只更新 Projector + LoRA + SpeechHead
            for p in model.parameters():
                p.requires_grad_(False)
            for name, p in model.named_parameters():
                if any(k in name for k in ["lora_a", "lora_b", "projector", "resampler", "speech_lm_head"]):
                    p.requires_grad_(True)
        else:
            model.get_trainable_parameters(stage=3)

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if self.is_main_process:
            print(f"[Stage3] 可训练参数: {num_trainable/1e6:.1f}M (LoRA={use_lora})")

        # 两阶段数据调度器
        self.scheduler_data = TwoPhaseDataScheduler(
            part1_dataset=part1_dataset,
            part2_dataset=part2_dataset,
            final_dataset=final_dataset,
            total_steps=total_steps,
        )

        # 当前 dataloader（随数据集切换而重建）
        self._current_phase = "part1"
        self._build_dataloader(part1_dataset, batch_size_per_gpu)

        # 优化器
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        warmup_steps = int(total_steps * warmup_ratio)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        self.model = model
        if world_size > 1:
            self.model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

        # 修复：bfloat16 不需要 GradScaler（动态范围够宽，不会溢出）
        #    float16 才需要 GradScaler；原代码用 bfloat16 却初始化 scaler=None，
        #    导致后续 self.scaler.scale() 崩溃。
        self.dtype = torch.bfloat16
        self.scaler = (
            torch.cuda.amp.GradScaler() if self.dtype == torch.float16 else None
        )
        if self.is_main_process:
            print(f"[Stage3] 训练精度: {self.dtype}, GradScaler: {'启用' if self.scaler else '禁用（bfloat16 不需要）'}")

    def _build_dataloader(self, dataset: SFTDataset, batch_size: int):
        """构建/重建 DataLoader"""
        sampler = DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.local_rank
        ) if self.world_size > 1 else None

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=collate_fn_sft,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        self.data_iter = iter(self.dataloader)

    def _maybe_switch_phase(self, current_step: int, batch_size: int):
        """检查是否需要切换数据集"""
        new_phase = self.scheduler_data.get_phase_name(current_step)
        if new_phase != self._current_phase:
            if self.is_main_process:
                print(f"\n[Stage3] 切换数据集: {self._current_phase} → {new_phase} (step={current_step})")
            new_dataset = self.scheduler_data.get_dataset(current_step)
            self._build_dataloader(new_dataset, batch_size)
            self._current_phase = new_phase

    def train(self, batch_size_per_gpu: int = 2):
        self.model.train()
        device = next(self.model.parameters()).device

        step = 0
        opt_step = 0
        total_loss = 0.0
        fast_losses, deep_losses = [], []

        self.optimizer.zero_grad()

        while opt_step < self.total_steps:
            # 检查数据集切换
            self._maybe_switch_phase(opt_step, batch_size_per_gpu)

            try:
                batch = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataloader)
                batch = next(self.data_iter)

            device_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            pixel_values = device_batch.get("pixel_values")
            audio_mel = device_batch.get("audio_mel")
            audio_mask = device_batch.get("audio_mask")
            speech_labels = device_batch.get("speech_labels")

            if pixel_values is not None:
                pixel_values = pixel_values.to(dtype=self.dtype)
            if audio_mel is not None:
                audio_mel = audio_mel.to(dtype=self.dtype)

            with torch.cuda.amp.autocast(dtype=self.dtype):
                outputs = self.model(
                    input_ids=device_batch["input_ids"],
                    attention_mask=device_batch["attention_mask"],
                    pixel_values=pixel_values,
                    audio_mel=audio_mel,
                    audio_mask=audio_mask,
                    labels=device_batch["labels"],
                    speech_labels=speech_labels,
                )
                loss = outputs["loss"] / self.gradient_accumulation_steps

            # 修复1：backward 根据 scaler 是否存在分支
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.gradient_accumulation_steps

            # 分类型记录 loss
            thinking_modes = batch.get("thinking_modes", [])
            loss_val = loss.item() * self.gradient_accumulation_steps
            for mode in thinking_modes:
                if mode == "fast":
                    fast_losses.append(loss_val)
                else:
                    deep_losses.append(loss_val)

            step += 1
            if step % self.gradient_accumulation_steps == 0:
                # 修复2：unscale_ / clip / step / update 全部条件分支
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm
                )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                opt_step += 1

                if self.is_main_process and opt_step % self.log_steps == 0:
                    avg_loss = total_loss / self.log_steps
                    avg_fast = sum(fast_losses[-50:]) / max(len(fast_losses[-50:]), 1)
                    avg_deep = sum(deep_losses[-50:]) / max(len(deep_losses[-50:]), 1)
                    lr = self.lr_scheduler.get_last_lr()[0]
                    print(
                        f"[Stage3|{self._current_phase}] Step {opt_step}/{self.total_steps}"
                        f" | loss={avg_loss:.4f}"
                        f" | fast={avg_fast:.4f} deep={avg_deep:.4f}"
                        f" | lr={lr:.2e} | grad={grad_norm:.3f}"
                    )
                    total_loss = 0.0

                if self.is_main_process and opt_step % self.save_steps == 0:
                    self._save_checkpoint(opt_step)

        if self.is_main_process:
            self._save_checkpoint("final")
            print("[Stage3] SFT 训练完成！")

    def _save_checkpoint(self, step):
        save_dir = self.output_dir / f"sft-checkpoint-{step}"
        save_dir.mkdir(exist_ok=True)
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model.state_dict(), save_dir / "model.pt")
        print(f"[Stage3] 保存 SFT checkpoint → {save_dir}")
