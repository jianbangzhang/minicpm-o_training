"""
Stage 4: 强化学习后训练数据集与奖励函数

双轨 RL 策略：
  1. RLPR (Rule-based Process Reward)
     - 用于可验证推理：数学、代码、逻辑题
     - 奖励信号：最终答案正确/错误 + 格式奖励
     - 算法：GRPO (Group Relative Policy Optimization)

  2. RLAIF-V (AI Feedback for Vision)
     - 用于减少多模态幻觉
     - 数据：对比偏好对 (chosen vs rejected)
     - 算法：DPO (Direct Preference Optimization)

混合快/深思考联合 RL 优化：
  - 同一 batch 内混合两种模式数据
  - 共享策略网络，通过 system prompt 区分模式
"""
import os.path
import re
import json
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from PIL import Image


# ============================================================
# 奖励函数（RLPR）
# ============================================================

class RuleBasedRewardFunction:
    """
    基于规则的奖励函数，用于可验证推理任务。

    奖励分解：
      r_total = r_answer + r_format + r_length_penalty
    """

    # 答案提取正则
    ANSWER_PATTERNS = [
        r"\\boxed\{([^}]+)\}",            # LaTeX boxed 答案
        r"答案[：:]\s*([^\n]+)",           # 中文答案格式
        r"Answer[：:]\s*([^\n]+)",         # 英文答案格式
        r"The answer is\s+([^\n\.]+)",
        r"Therefore[,，]\s*([^\n]+)",
    ]

    def __init__(
        self,
        answer_correct_reward: float = 1.0,
        answer_wrong_reward: float = -1.0,
        format_reward: float = 0.2,        # 格式规范奖励
        length_penalty_factor: float = 0.001,  # 过长惩罚系数
        max_ideal_length: int = 1024,
    ):
        self.answer_correct_reward = answer_correct_reward
        self.answer_wrong_reward = answer_wrong_reward
        self.format_reward = format_reward
        self.length_penalty_factor = length_penalty_factor
        self.max_ideal_length = max_ideal_length

    def extract_answer(self, response: str) -> Optional[str]:
        """从 LLM 回复中提取最终答案"""
        # 如果包含 <think>，取 </think> 之后的内容
        if "</think>" in response:
            response = response.split("</think>", 1)[1]

        for pattern in self.ANSWER_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def check_math_answer(
        self, predicted: Optional[str], ground_truth: str
    ) -> bool:
        """数学答案验证（支持数值等价、化简等）"""
        if predicted is None:
            return False

        # 规范化
        pred = self._normalize_math(predicted)
        gt = self._normalize_math(ground_truth)

        if pred == gt:
            return True

        # 尝试数值比较
        try:
            return abs(float(pred) - float(gt)) < 1e-6
        except (ValueError, TypeError):
            return False

    def _normalize_math(self, s: str) -> str:
        """数学表达式规范化"""
        s = s.strip().replace(" ", "").replace(",", "")
        # 去除 LaTeX 命令
        s = re.sub(r"\\(frac|cdot|times|div)", "", s)
        return s.lower()

    def check_code_answer(
        self, response: str, test_cases: List[Dict]
    ) -> Tuple[bool, float]:
        """
        代码题答案验证（沙箱执行）。
        返回 (pass_all, pass_ratio)
        """
        import subprocess, tempfile, os

        code_match = re.search(r"```python\n(.+?)\n```", response, re.DOTALL)
        if not code_match:
            return False, 0.0

        code = code_match.group(1)
        passed = 0

        for tc in test_cases:
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(code + "\n")
                    f.write(f"print({tc['call']})\n")
                    fname = f.name

                result = subprocess.run(
                    ["python", fname],
                    capture_output=True, text=True, timeout=5
                )
                os.unlink(fname)

                if result.returncode == 0:
                    output = result.stdout.strip()
                    if output == str(tc["expected"]):
                        passed += 1
            except Exception:
                pass

        pass_ratio = passed / len(test_cases) if test_cases else 0.0
        return pass_ratio == 1.0, pass_ratio

    def compute_reward(
        self,
        response: str,
        ground_truth: str,
        task_type: str = "math",
        test_cases: Optional[List[Dict]] = None,
    ) -> Dict[str, float]:
        """
        计算总奖励。

        Returns:
            {
                "total": float,
                "answer_reward": float,
                "format_reward": float,
                "length_penalty": float,
            }
        """
        # ── 答案奖励 ──
        if task_type == "math":
            predicted = self.extract_answer(response)
            correct = self.check_math_answer(predicted, ground_truth)
            answer_reward = self.answer_correct_reward if correct else self.answer_wrong_reward
        elif task_type == "code":
            _, pass_ratio = self.check_code_answer(response, test_cases or [])
            answer_reward = self.answer_correct_reward * pass_ratio + self.answer_wrong_reward * (1 - pass_ratio)
        elif task_type == "choice":
            # 选择题：直接匹配选项字母
            predicted = self.extract_answer(response) or ""
            correct = predicted.strip().upper() == ground_truth.strip().upper()
            answer_reward = self.answer_correct_reward if correct else self.answer_wrong_reward
        else:
            answer_reward = 0.0

        # ── 格式奖励 ──
        fmt_score = 0.0
        # 检查是否有结构化答案格式
        if "\\boxed{" in response or "答案：" in response or "Answer:" in response:
            fmt_score += self.format_reward * 0.5
        # 检查推理链质量（有步骤分解）
        if re.search(r"(Step \d+|第\d+步|First|Then|Finally)", response):
            fmt_score += self.format_reward * 0.5

        # ── 长度惩罚 ──
        response_len = len(response.split())
        length_penalty = 0.0
        if response_len > self.max_ideal_length:
            length_penalty = -self.length_penalty_factor * (response_len - self.max_ideal_length)

        total = answer_reward + fmt_score + length_penalty

        return {
            "total": total,
            "answer_reward": answer_reward,
            "format_reward": fmt_score,
            "length_penalty": length_penalty,
        }


# ============================================================
# RLPR 数据集（GRPO）
# ============================================================

class RLPRDataset(Dataset):
    """
    GRPO 训练数据集。

    数据格式：
    {
        "id": "math_001",
        "task_type": "math",          # "math" | "code" | "choice"
        "thinking_mode": "deep",      # RLPR 主要用深思考模式
        "conversations": [
            {"role": "user", "content": [
                {"type": "image", "path": "..."},  # 可选
                {"type": "text", "content": "求解..."}
            ]}
        ],
        "ground_truth": "42",          # 可验证答案
        "test_cases": []               # 代码题测试用例（可选）
    }
    """

    def __init__(
        self,
        data_paths: List[str],
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 4096,
        fast_deep_mix_ratio: float = 0.3,  # 快思考样本比例（RL 阶段以深思考为主）
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.fast_deep_mix_ratio = fast_deep_mix_ratio
        self.reward_fn = RuleBasedRewardFunction()

        self.samples = []
        for path in data_paths:
            try:
                if "jsonl" in path:
                    with open(path, "r",encoding="utf-8") as f:
                        for line in f:
                            try:
                                sample = json.loads(line.strip())
                                self.samples.append(sample)
                            except json.JSONDecodeError:
                                continue
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        sample = json.load(f)
                        if isinstance(sample,dict) and "annotations" in sample:
                            sample = sample["annotations"]
                        self.samples.extend(sample)
            except Exception:
                continue
        print(f"[RLPRDataset] 加载 {len(self.samples):,} 条样本")

    def __len__(self):
        return len(self.samples)

    def build_prompt(self, sample: Dict, thinking_mode: str = "deep") -> str:
        """构建 GRPO 采样用的 prompt"""
        system_content = (
            "You are a helpful assistant. Think carefully and show your reasoning."
            if thinking_mode == "deep"
            else "You are a helpful assistant. Answer directly and concisely."
        )

        turns = []
        turns.append(f"<|system|>\n{system_content}")

        for turn in sample["conversations"]:
            if turn["role"] == "user":
                content = turn["content"]
                user_text = ""
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "image":
                            user_text += "<image>X" * 256 + "</image>"
                        elif item["type"] == "text":
                            user_text += item["content"]
                else:
                    user_text = str(content)
                turns.append(f"<|user|>\n{user_text}")

        turns.append("<|assistant|>\n")
        return "\n".join(turns)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 混合快/深思考模式
        thinking_mode = (
            "fast" if random.random() < self.fast_deep_mix_ratio else "deep"
        )

        prompt = self.build_prompt(sample, thinking_mode)
        tokens = self.tokenizer(
            prompt, max_length=self.max_seq_len, truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "prompt": prompt,
            "ground_truth": sample["ground_truth"],
            "task_type": sample.get("task_type", "math"),
            "test_cases": sample.get("test_cases", []),
            "thinking_mode": thinking_mode,
            "sample_id": sample.get("id", str(idx)),
        }


# ============================================================
# RLAIF-V 数据集（DPO）
# ============================================================

class RLAIFVDataset(Dataset):
    """
    RLAIF-V DPO 数据集（减少多模态幻觉）。

    数据格式：
    {
        "id": "hallucination_001",
        "image": "/path/to/image.jpg",
        "question": "Describe the objects in the image.",
        "chosen": "The image shows a red car parked on the street...",  # 准确回复
        "rejected": "The image shows a blue car flying in the sky...", # 含幻觉的回复
        "thinking_mode": "fast"
    }

    偏好数据由更强模型（GPT-4V / Claude 3.5）标注生成。
    """

    def __init__(
        self,
        data_paths: List[str],
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.samples = []
        for path in data_paths:
            try:
                if "jsonl" in path:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                sample = json.loads(line.strip())
                                if os.path.exists(sample["image"]):
                                    self.samples.append(sample)

                            except json.JSONDecodeError:
                                continue
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        sample = json.load(f)
                        if isinstance(sample, dict) and "annotations" in sample:
                            sample = sample["annotations"]
                        sample = [item for item in sample if os.path.exists(item["image"])]
                        self.samples.extend(sample)
            except Exception:
                continue

        valid_samples = []
        for sample in self.samples:
            try:
                with Image.open(sample["image"]) as img:
                    img.verify()  # 只验证，不加载完整图像
                valid_samples.append(sample)
            except Exception:
                continue

        self.samples = valid_samples
        print(f"[RLAIFVDataset] 加载 {len(self.samples):,} 条偏好样本")

    def __len__(self):
        return len(self.samples)

    def _tokenize_response(
        self, prompt: str, response: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize (prompt + response)，返回 (input_ids, attention_mask, labels)。
        Labels: prompt 部分 = -100，response 部分 = token ids
        """
        full_text = prompt + response
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)

        labels = input_ids.clone()
        labels[:min(prompt_len, len(labels))] = -100

        return input_ids, attention_mask, labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        import numpy as np
        from PIL import Image

        # 构建 prompt
        thinking_mode = sample.get("thinking_mode", "fast")
        sys_prompt = (
            "You are a helpful and accurate vision assistant."
            if thinking_mode == "fast"
            else "You are a careful vision assistant. Think step by step."
        )
        prompt = (
            f"<|system|>\n{sys_prompt}\n"
            f"<|user|>\n<image>{'X' * 256}</image>\n{sample['question']}\n"
            f"<|assistant|>\n"
        )

        # Tokenize chosen 和 rejected
        chosen_ids, chosen_mask, chosen_labels = self._tokenize_response(
            prompt, sample["chosen"]
        )
        rejected_ids, rejected_mask, rejected_labels = self._tokenize_response(
            prompt, sample["rejected"]
        )

        # 加载图像
        img = Image.open(sample["image"]).convert("RGB")
        img = img.resize((448, 448), Image.BICUBIC)
        pixel_values = torch.from_numpy(
            (np.array(img).astype(np.float32) / 255.0 - 0.5) / 0.5
        ).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 448, 448)

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
            "rejected_labels": rejected_labels,
            "pixel_values": pixel_values,
            "thinking_mode": thinking_mode,
        }

