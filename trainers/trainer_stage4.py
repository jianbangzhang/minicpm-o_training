"""
Stage 4: 强化学习后训练 Trainer

实现两种 RL 算法：
1. GRPO (Group Relative Policy Optimization) → RLPR 可验证推理
2. DPO (Direct Preference Optimization) → RLAIF-V 幻觉抑制

混合快/深思考联合优化：同 batch 同时包含两种模式样本
"""

import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence

from ..models.modeling_minicpmo import MiniCPMOModel
from ..data.dataset_stage4 import RLPRDataset, RLAIFVDataset, RuleBasedRewardFunction


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    from torch.optim.lr_scheduler import LambdaLR
    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return LambdaLR(optimizer, lr_fn)


# ============================================================
# GRPO 核心逻辑（RLPR）
# ============================================================

class GRPOTrainer:
    """
    Group Relative Policy Optimization for RLPR.

    算法步骤（每个 optimization step）：
    1. 对每个 prompt，用当前策略采样 G 个回复 (G=8)
    2. 计算每个回复的奖励 r_i（规则打分）
    3. 归一化奖励为优势估计：A_i = (r_i - mean(r)) / (std(r) + eps)
    4. 计算 GRPO loss（带 KL 惩罚和 clip）：
       L = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] + β * KL

    注意：快/深思考样本混合在同一 batch 内。
    """

    def __init__(
        self,
        policy_model: MiniCPMOModel,
        ref_model: MiniCPMOModel,       # 参考模型（SFT 权重，冻结）
        tokenizer,
        reward_fn: RuleBasedRewardFunction,
        # GRPO 超参
        num_samples_per_prompt: int = 8,   # G
        clip_epsilon: float = 0.2,
        kl_coeff: float = 0.04,            # β，KL 惩罚系数
        # 训练超参
        learning_rate: float = 5e-7,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.G = num_samples_per_prompt
        self.clip_eps = clip_epsilon
        self.kl_coeff = kl_coeff
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # 冻结参考模型
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.ref.eval()

        # 只训练 LLM 参数
        self.policy.get_trainable_parameters(stage=4)
        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=learning_rate, weight_decay=0.01)

    @torch.no_grad()
    def _sample_responses(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
    ) -> List[List[int]]:
        """
        对同一 prompt 采样 G 个回复（nucleus sampling）。
        返回 G 个 token id 序列（list of lists）。
        """
        all_responses = []
        model = self.policy.module if hasattr(self.policy, "module") else self.policy

        # 扩展输入：batch_size=1 → batch_size=G
        G = self.G
        expanded_ids = input_ids.repeat(G, 1)
        expanded_mask = attention_mask.repeat(G, 1)
        expanded_pv = pixel_values.repeat(G, 1, 1, 1) if pixel_values is not None else None

        # 获取 inputs_embeds
        embeds = model.llm.get_input_embeddings()(expanded_ids)
        if expanded_pv is not None:
            visual_tokens = model.encode_images(expanded_pv)
            embeds = model._replace_placeholder(
                embeds, visual_tokens, expanded_ids,
                model._im_start_id, model._im_end_id
            )

        # 生成
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.llm.generate(
                inputs_embeds=embeds,
                attention_mask=expanded_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        for i in range(G):
            response_ids = outputs[i][input_ids.shape[1]:].tolist()
            all_responses.append(response_ids)

        return all_responses

    def _compute_log_probs(
        self,
        model: MiniCPMOModel,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算模型在给定序列上的 token log-probability（仅 response 部分）。
        返回 shape: (B, response_len)
        """
        m = model.module if hasattr(model, "module") else model
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        full_mask = torch.ones_like(full_ids)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = m(
                input_ids=full_ids,
                attention_mask=full_mask,
                pixel_values=pixel_values,
            )

        logits = outputs["lm_logits"]  # (B, L, vocab)
        prompt_len = input_ids.shape[1]

        # 只取 response 部分的 log prob
        response_logits = logits[:, prompt_len - 1:-1, :]  # 右移一位
        log_probs = F.log_softmax(response_logits, dim=-1)
        response_log_probs = log_probs.gather(
            dim=-1,
            index=response_ids.unsqueeze(-1)
        ).squeeze(-1)  # (B, response_len)

        return response_log_probs

    def compute_grpo_loss(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        ground_truths: List[str],
        task_types: List[str],
        thinking_modes: List[str],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        GRPO 损失计算主函数。

        Returns:
            loss: 标量
            metrics: {"reward_mean", "reward_std", "kl", "fast_reward", "deep_reward"}
        """
        device = prompt_input_ids.device
        batch_size = prompt_input_ids.shape[0]

        all_grpo_losses = []
        all_rewards = []
        fast_rewards, deep_rewards = [], []

        for b in range(batch_size):
            # 1. 采样 G 个回复
            pid = prompt_input_ids[b:b+1]
            pmask = prompt_attention_mask[b:b+1]
            pv = pixel_values[b:b+1] if pixel_values is not None else None

            response_token_lists = self._sample_responses(pid, pmask, pv)
            responses_text = [
                self.tokenizer.decode(r, skip_special_tokens=True)
                for r in response_token_lists
            ]

            # 2. 计算奖励
            rewards = []
            for resp_text in responses_text:
                reward_info = self.reward_fn.compute_reward(
                    response=resp_text,
                    ground_truth=ground_truths[b],
                    task_type=task_types[b],
                )
                rewards.append(reward_info["total"])

            rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
            all_rewards.extend(rewards)

            if thinking_modes[b] == "fast":
                fast_rewards.extend(rewards)
            else:
                deep_rewards.extend(rewards)

            # 3. 归一化优势
            advantage = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

            # 4. 计算每个回复的 policy/ref log prob
            max_resp_len = max(len(r) for r in response_token_lists)
            # Pad 所有回复到相同长度
            padded_responses = torch.zeros(self.G, max_resp_len, dtype=torch.long, device=device)
            resp_masks = torch.zeros(self.G, max_resp_len, dtype=torch.bool, device=device)
            for i, r in enumerate(response_token_lists):
                padded_responses[i, :len(r)] = torch.tensor(r, device=device)
                resp_masks[i, :len(r)] = True

            # 扩展 prompt
            expanded_pid = pid.expand(self.G, -1)
            expanded_pmask = pmask.expand(self.G, -1)
            expanded_pv = pv.expand(self.G, -1, -1, -1) if pv is not None else None

            policy_log_probs = self._compute_log_probs(
                self.policy, expanded_pid, padded_responses, expanded_pmask, expanded_pv
            )  # (G, max_resp_len)

            with torch.no_grad():
                ref_log_probs = self._compute_log_probs(
                    self.ref, expanded_pid, padded_responses, expanded_pmask, expanded_pv
                )  # (G, max_resp_len)

            # 5. GRPO loss
            log_ratio = policy_log_probs - ref_log_probs  # (G, L)
            ratio = torch.exp(log_ratio)

            # 只对有效 token 计算 loss
            advantage_expanded = advantage.unsqueeze(1).expand_as(ratio)  # (G, L)

            clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            surrogate = torch.min(ratio * advantage_expanded, clipped_ratio * advantage_expanded)

            # Mask 填充位置
            surrogate = surrogate * resp_masks.float()
            kl = (policy_log_probs - ref_log_probs) * resp_masks.float()

            # 每个 response 的平均 loss
            token_count = resp_masks.float().sum(dim=1).clamp(min=1)
            per_sample_loss = -(surrogate.sum(dim=1) / token_count)
            per_sample_kl = (kl.sum(dim=1) / token_count)

            grpo_loss = (per_sample_loss + self.kl_coeff * per_sample_kl).mean()
            all_grpo_losses.append(grpo_loss)

        total_loss = torch.stack(all_grpo_losses).mean()

        metrics = {
            "reward_mean": sum(all_rewards) / len(all_rewards),
            "reward_std": torch.tensor(all_rewards).std().item(),
            "fast_reward": sum(fast_rewards) / max(len(fast_rewards), 1),
            "deep_reward": sum(deep_rewards) / max(len(deep_rewards), 1),
        }

        return total_loss, metrics


# ============================================================
# DPO 逻辑（RLAIF-V）
# ============================================================

def compute_dpo_loss(
    policy_model: MiniCPMOModel,
    ref_model: MiniCPMOModel,
    batch: Dict,
    beta: float = 0.1,
    device: torch.device = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    DPO 损失计算。

    L_DPO = -E[ log σ( β * (log π(y_w|x) - log π_ref(y_w|x))
                       - β * (log π(y_l|x) - log π_ref(y_l|x)) ) ]

    Args:
        batch: 来自 RLAIFVDataset 的 batch
        beta: DPO 温度参数（控制偏好强度）

    Returns:
        loss: DPO 损失
        metrics: {"chosen_reward", "rejected_reward", "margin"}
    """
    def get_per_token_log_probs(model, input_ids, attention_mask, labels, pv):
        """计算每个 token 的 log prob（只对 response 部分）"""
        m = model.module if hasattr(model, "module") else model
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pv,
                labels=None,
            )
        logits = out["lm_logits"]  # (B, L, V)
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_label_mask = (labels[:, 1:] != -100).float()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (B, L-1)

        # 平均 response log prob
        seq_log_prob = (token_log_probs * shift_label_mask).sum(dim=-1) / \
                       shift_label_mask.sum(dim=-1).clamp(min=1)
        return seq_log_prob

    pixel_values = batch.get("pixel_values")
    if pixel_values is not None and device:
        pixel_values = pixel_values.to(device, dtype=torch.bfloat16)

    # Policy log probs
    policy_chosen_lp = get_per_token_log_probs(
        policy_model,
        batch["chosen_input_ids"].to(device),
        batch["chosen_attention_mask"].to(device),
        batch["chosen_labels"].to(device),
        pixel_values,
    )
    policy_rejected_lp = get_per_token_log_probs(
        policy_model,
        batch["rejected_input_ids"].to(device),
        batch["rejected_attention_mask"].to(device),
        batch["rejected_labels"].to(device),
        pixel_values,
    )

    # Reference log probs（无梯度）
    with torch.no_grad():
        ref_chosen_lp = get_per_token_log_probs(
            ref_model,
            batch["chosen_input_ids"].to(device),
            batch["chosen_attention_mask"].to(device),
            batch["chosen_labels"].to(device),
            pixel_values,
        )
        ref_rejected_lp = get_per_token_log_probs(
            ref_model,
            batch["rejected_input_ids"].to(device),
            batch["rejected_attention_mask"].to(device),
            batch["rejected_labels"].to(device),
            pixel_values,
        )

    # DPO 损失
    logits_w = beta * (policy_chosen_lp - ref_chosen_lp)
    logits_l = beta * (policy_rejected_lp - ref_rejected_lp)
    loss = -F.logsigmoid(logits_w - logits_l).mean()

    # 隐式奖励（用于监控）
    chosen_reward = (policy_chosen_lp - ref_chosen_lp).mean().item()
    rejected_reward = (policy_rejected_lp - ref_rejected_lp).mean().item()
    accuracy = ((logits_w > logits_l).float().mean().item())

    metrics = {
        "chosen_reward": chosen_reward,
        "rejected_reward": rejected_reward,
        "margin": chosen_reward - rejected_reward,
        "accuracy": accuracy,
    }
    return loss, metrics


# ============================================================
# Stage 4 联合 Trainer
# ============================================================

class Stage4RLTrainer:
    """
    Stage 4 联合强化学习 Trainer。

    训练流程（每个 iteration）：
      1. 从 RLPR 数据集采样 → GRPO 更新（推理能力）
      2. 从 RLAIF-V 数据集采样 → DPO 更新（幻觉抑制）
      3. 快/深思考混合在同一 GRPO batch 内

    关键设计：
      - 同一模型同时优化两个目标（共享参数）
      - GRPO:DPO 交替执行（1:1 轮换或加权混合）
      - 参考模型保持 SFT 初始化不更新
      - 监控 PPL 防止奖励 hacking
    """

    def __init__(
        self,
        policy_model: MiniCPMOModel,
        ref_model: MiniCPMOModel,
        tokenizer,
        rlpr_dataset: RLPRDataset,
        rlaifv_dataset: RLAIFVDataset,
        output_dir: str,
        # RL 超参
        grpo_lr: float = 5e-7,
        dpo_lr: float = 5e-7,
        dpo_beta: float = 0.1,
        kl_coeff: float = 0.04,
        num_samples_per_prompt: int = 8,
        # 训练设置
        total_rl_steps: int = 3000,
        grpo_batch_size: int = 4,        # prompts per GRPO step
        dpo_batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        save_steps: int = 500,
        log_steps: int = 20,
        grpo_dpo_ratio: float = 0.5,     # GRPO 占比，0.5=交替
        local_rank: int = 0,
        world_size: int = 16,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main_process = local_rank == 0
        self.total_rl_steps = total_rl_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.log_steps = log_steps
        self.grpo_dpo_ratio = grpo_dpo_ratio

        # 奖励函数
        reward_fn = RuleBasedRewardFunction()

        # GRPO Trainer
        self.grpo = GRPOTrainer(
            policy_model=policy_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            num_samples_per_prompt=num_samples_per_prompt,
            kl_coeff=kl_coeff,
            learning_rate=grpo_lr,
        )

        # 策略模型和参考模型
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dpo_beta = dpo_beta

        # DPO 优化器（共用策略模型参数）
        trainable = [p for p in policy_model.parameters() if p.requires_grad]
        self.dpo_optimizer = AdamW(trainable, lr=dpo_lr, weight_decay=0.01)

        # 统一 LR Scheduler（GRPO 使用自己的 optimizer，DPO 共用）
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.dpo_optimizer, warmup_steps=100, total_steps=total_rl_steps
        )

        # DataLoaders
        self.rlpr_loader = DataLoader(
            rlpr_dataset, batch_size=grpo_batch_size, shuffle=True,
            num_workers=2, drop_last=True,
        )
        self.rlaifv_loader = DataLoader(
            rlaifv_dataset, batch_size=dpo_batch_size, shuffle=True,
            num_workers=2, drop_last=True,
        )

        self.rlpr_iter = iter(self.rlpr_loader)
        self.rlaifv_iter = iter(self.rlaifv_loader)

        if world_size > 1:
            self.policy_model = DDP(policy_model, device_ids=[local_rank],
                                    find_unused_parameters=True)

        self.dtype = torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler()

    def _get_rlpr_batch(self):
        try:
            return next(self.rlpr_iter)
        except StopIteration:
            self.rlpr_iter = iter(self.rlpr_loader)
            return next(self.rlpr_iter)

    def _get_rlaifv_batch(self):
        try:
            return next(self.rlaifv_iter)
        except StopIteration:
            self.rlaifv_iter = iter(self.rlaifv_loader)
            return next(self.rlaifv_iter)

    def _compute_ppl(self, batch) -> float:
        """计算 PPL，用于监控奖励 hacking"""
        model = self.policy_model.module if hasattr(self.policy_model, "module") else self.policy_model
        device = next(model.parameters()).device

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.dtype):
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["input_ids"].to(device),
            )
        return torch.exp(out["loss"]).item() if out["loss"] is not None else 0.0

    def train(self):
        """
        主 RL 训练循环。
        按 grpo_dpo_ratio 交替执行 GRPO 和 DPO 更新。
        """
        device = next(self.policy_model.parameters()).device
        step = 0
        grpo_losses, dpo_losses = [], []
        all_metrics = {}

        print(f"[Stage4] 开始 RL 训练，总步数={self.total_rl_steps}")
        print(f"  GRPO(RLPR): {self.grpo_dpo_ratio*100:.0f}%  |  DPO(RLAIF-V): {(1-self.grpo_dpo_ratio)*100:.0f}%")

        import random
        while step < self.total_rl_steps:
            use_grpo = random.random() < self.grpo_dpo_ratio

            if use_grpo:
                # ── GRPO 更新（可验证推理）──────────────────
                batch = self._get_rlpr_batch()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch.get("pixel_values")
                if pixel_values is not None:
                    pixel_values = pixel_values.to(device, dtype=self.dtype)

                grpo_loss, grpo_metrics = self.grpo.compute_grpo_loss(
                    prompt_input_ids=input_ids,
                    prompt_attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    ground_truths=batch["ground_truth"],
                    task_types=batch["task_type"],
                    thinking_modes=batch["thinking_mode"],
                )

                # 反向传播
                self.grpo.optimizer.zero_grad()
                self.scaler.scale(grpo_loss).backward()
                self.scaler.unscale_(self.grpo.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy_model.parameters() if p.requires_grad],
                    self.max_grad_norm
                )
                self.scaler.step(self.grpo.optimizer)
                self.scaler.update()

                grpo_losses.append(grpo_loss.item())
                all_metrics.update({f"grpo/{k}": v for k, v in grpo_metrics.items()})

            else:
                # ── DPO 更新（幻觉抑制）─────────────────────
                batch = self._get_rlaifv_batch()

                dpo_loss, dpo_metrics = compute_dpo_loss(
                    policy_model=self.policy_model,
                    ref_model=self.ref_model,
                    batch=batch,
                    beta=self.dpo_beta,
                    device=device,
                )

                self.dpo_optimizer.zero_grad()
                self.scaler.scale(dpo_loss).backward()
                self.scaler.unscale_(self.dpo_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy_model.parameters() if p.requires_grad],
                    self.max_grad_norm
                )
                self.scaler.step(self.dpo_optimizer)
                self.scaler.update()
                self.lr_scheduler.step()

                dpo_losses.append(dpo_loss.item())
                all_metrics.update({f"dpo/{k}": v for k, v in dpo_metrics.items()})

            step += 1

            # 日志
            if self.is_main_process and step % self.log_steps == 0:
                avg_grpo = sum(grpo_losses[-self.log_steps:]) / max(len(grpo_losses[-self.log_steps:]), 1)
                avg_dpo = sum(dpo_losses[-self.log_steps:]) / max(len(dpo_losses[-self.log_steps:]), 1)
                fast_r = all_metrics.get("grpo/fast_reward", 0)
                deep_r = all_metrics.get("grpo/deep_reward", 0)
                dpo_acc = all_metrics.get("dpo/accuracy", 0)
                margin = all_metrics.get("dpo/margin", 0)

                print(
                    f"[Stage4] Step {step}/{self.total_rl_steps}"
                    f" | GRPO_loss={avg_grpo:.4f} (fast_r={fast_r:.3f} deep_r={deep_r:.3f})"
                    f" | DPO_loss={avg_dpo:.4f} (acc={dpo_acc:.2%} margin={margin:.3f})"
                )

            # 保存
            if self.is_main_process and step % self.save_steps == 0:
                self._save_checkpoint(step)

        if self.is_main_process:
            self._save_checkpoint("final")
            print("[Stage4] RL 后训练完成！")

    def _save_checkpoint(self, step):
        save_dir = self.output_dir / f"rl-checkpoint-{step}"
        save_dir.mkdir(exist_ok=True)
        model = self.policy_model.module if hasattr(self.policy_model, "module") else self.policy_model
        torch.save(model.state_dict(), save_dir / "model.pt")
        print(f"[Stage4] 保存 RL checkpoint → {save_dir}")
