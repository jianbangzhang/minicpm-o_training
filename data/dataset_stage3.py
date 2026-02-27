"""
Stage 3: 监督微调 (SFT) 数据集

两阶段数据调度策略（对齐 MiniCPM-V 系列经验）：
  Part-1 (前 60% 步): 基础感知，短回复，任务型 QA
  Part-2 (后 40% 步): 复杂交互，长回复，混合思考模式

混合思考模式：
  - 快思考样本 (70%): 直接回复，无 CoT 链
  - 深思考样本 (30%): <think>...</think> 推理过程 + 最终答案

数据 JSONL 格式（对话格式）：
{
    "id": "sample_001",
    "type": "vision_qa",           # 样本类型
    "thinking_mode": "fast",       # "fast" | "deep"
    "conversations": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "path": "/path/to/img.jpg"},
            {"type": "text", "content": "What is shown in this image?"}
        ]},
        {"role": "assistant", "content": "The image shows..."}
    ]
}
"""

import json
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedTokenizer

from utils.video_audio_processor import AudioProcessor, VideoFrameProcessor
import warnings
warnings.filterwarnings("ignore")

class ThinkingMode(Enum):
    FAST = "fast"
    DEEP = "deep"


class SFTDataset(Dataset):
    """
    Stage 3 SFT 统一数据集。

    支持的模态组合：
      - 纯文本对话
      - 图像 + 文本 QA
      - 多图理解
      - 视频 + 文本 QA
      - 语音输入 + 文本输出
      - 语音输入 + 语音输出（TTS 训练）

    快/深思考模式通过 system prompt + <think> 标记控制。
    """

    # 各类型目标数据量（Part-1 + Part-2）
    DATA_BUDGET = {
        "vision_qa": 500_000,
        "ocr_document": 200_000,
        "math_reasoning": 300_000,
        "video_qa": 200_000,
        "speech_dialog": 300_000,
        "long_response": 100_000,
        "multilingual": 90_000,
    }

    def __init__(
        self,
        data_paths: List[str],
        tokenizer: PreTrainedTokenizer,
        audio_processor: Optional[AudioProcessor] = None,
        video_processor: Optional[VideoFrameProcessor] = None,
        max_seq_len: int = 8192,
        phase: str = "part1",          # "part1" | "part2" | "final"
        fast_think_ratio: float = 0.7,  # 快思考样本比例
        image_max_slices: int = 9,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.video_processor = video_processor
        self.max_seq_len = max_seq_len
        self.phase = phase
        self.fast_think_ratio = fast_think_ratio
        self.image_max_slices = image_max_slices

        # Part-1/Part-2 数据类型过滤
        self.phase_type_filter = {
            "part1": {"vision_qa", "ocr_document", "speech_dialog"},
            "part2": {"math_reasoning", "video_qa", "speech_dialog", "long_response", "multilingual"},
            "final": None,  # 所有类型（最高质量数据）
        }

        self.samples = self._load_and_filter(data_paths)
        print(f"[SFTDataset] Phase={phase}, 加载 {len(self.samples):,} 条样本")
        self._log_type_distribution()

    def _load_and_filter(self, data_paths: List[str]) -> List[Dict]:
        """加载并按 phase 过滤数据"""
        allowed_types = self.phase_type_filter.get(self.phase)
        all_samples = []
        for path in data_paths:
            if "jsonl" in path:
                with open(path, "r",encoding="utf-8") as f:
                    for line in f:
                        try:
                            sample = json.loads(line.strip())
                            if allowed_types is None or sample.get("type") in allowed_types:
                                all_samples.append(sample)
                        except json.JSONDecodeError:
                            continue
            else:
                with open(path, "r", encoding="utf-8") as f:
                    sample = json.load(f)
                    if isinstance(sample,dict) and "annotations" in sample:
                        sample = sample["annotations"]
                    all_samples.extend(sample)
        return all_samples

    def _log_type_distribution(self):
        from collections import Counter
        counts = Counter(s.get("type", "unknown") for s in self.samples)
        print("  数据类型分布：")
        for t, c in sorted(counts.items()):
            print(f"    {t}: {c:,}")

    def __len__(self) -> int:
        return len(self.samples)

    # ── 对话构建 ──────────────────────────────────────────────

    def _build_prompt(
        self,
        conversations: List[Dict],
        thinking_mode: ThinkingMode,
        pixel_values_list: List[torch.Tensor],
        audio_info: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将对话序列转换为 input_ids / attention_mask / labels。

        标签规则：
          - system / user 部分: labels = -100 (不参与 loss)
          - assistant 部分: labels = input_ids (参与 loss)
          - fast 模式: assistant 直接输出答案
          - deep 模式: assistant 输出 <think>推理</think>答案
        """
        # 构建对话 prompt（Qwen 格式）
        full_text_parts = []
        label_mask_parts = []  # True = 计算 loss，False = 不计算

        im_slot = "<image>" + "X" * 256 + "</image>"  # 视觉占位符
        audio_slot = "<audio>" + "A" * 128 + "</audio>"  # 语音占位符

        image_idx = 0

        for turn in conversations:
            role = turn["role"]
            content = turn["content"]

            if role == "system":
                text = f"<|system|>\n{content}\n"
                full_text_parts.append(text)
                label_mask_parts.append(False)

            elif role == "user":
                text = "<|user|>\n"
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "image":
                            text += im_slot
                            image_idx += 1
                        elif item["type"] == "audio":
                            text += audio_slot
                        elif item["type"] == "text":
                            text += item["content"]
                else:
                    text += str(content)
                text += "\n"
                full_text_parts.append(text)
                label_mask_parts.append(False)

            elif role == "assistant":
                response = str(content)

                if thinking_mode == ThinkingMode.DEEP:
                    # 深思考：在 response 前插入 <think> 推理链
                    # 实际训练数据中 think_content 应由 GPT-4/human 提供
                    think_content = turn.get("think", "Let me analyze this step by step.")
                    response = f"<think>\n{think_content}\n</think>\n{response}"

                text = f"<|assistant|>\n{response}\n"
                full_text_parts.append(text)
                label_mask_parts.append(True)  # assistant 回复计算 loss

        full_text = "".join(full_text_parts)
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # 构建 labels（粗略按段落设置，精确实现需要跟踪 offset）
        labels = self._build_labels_by_parts(input_ids, full_text_parts, label_mask_parts)

        return input_ids, attention_mask, labels

    def _build_labels_by_parts(
        self,
        input_ids: torch.Tensor,
        text_parts: List[str],
        mask_parts: List[bool],
    ) -> torch.Tensor:
        """
        按文本段落设置 labels。
        assistant 部分 = input_ids，其余 = -100。
        """
        labels = torch.full_like(input_ids, fill_value=-100)

        # 重新 tokenize 各段落，计算偏移
        offset = 0
        for text, compute_loss in zip(text_parts, mask_parts):
            part_tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            part_len = len(part_tokens)
            end = min(offset + part_len, len(input_ids))
            if compute_loss:
                labels[offset:end] = input_ids[offset:end]
            offset = end
            if offset >= len(input_ids):
                break

        return labels

    def _load_image_slices(self, image_path: str) -> torch.Tensor:
        """加载图像并进行 LLaVA-UHD 切片"""
        import numpy as np
        img = Image.open(image_path).convert("RGB")
        patch_size = 448
        w, h = img.size

        # 计算切片数
        cols = max(1, min(round(w / patch_size), 3))
        rows = max(1, min(round(h / patch_size), 3))
        while rows * cols > self.image_max_slices:
            if rows > cols:
                rows -= 1
            else:
                cols -= 1

        slices = []
        # 添加全局缩略图
        thumb = img.resize((patch_size, patch_size), Image.BICUBIC)
        slices.append(torch.from_numpy(
            (np.array(thumb).astype(np.float32) / 255.0 - 0.5) / 0.5
        ).permute(2, 0, 1))

        # 切片
        sw, sh = w // cols, h // rows
        for r in range(rows):
            for c in range(cols):
                patch = img.crop((c * sw, r * sh, (c + 1) * sw, (r + 1) * sh))
                patch = patch.resize((patch_size, patch_size), Image.BICUBIC)
                slices.append(torch.from_numpy(
                    (np.array(patch).astype(np.float32) / 255.0 - 0.5) / 0.5
                ).permute(2, 0, 1))

        return torch.stack(slices)  # (num_slices, 3, 448, 448)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        try:
            # 决定思考模式
            sample_mode = sample.get("thinking_mode", None)
            if sample_mode is None:
                thinking_mode = (
                    ThinkingMode.FAST
                    if random.random() < self.fast_think_ratio
                    else ThinkingMode.DEEP
                )
            else:
                thinking_mode = ThinkingMode(sample_mode)

            conversations = sample["conversations"]

            # 提取图像
            pixel_values_list = []
            for turn in conversations:
                if turn["role"] != "user":
                    continue
                content = turn["content"]
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "image":
                            pv = self._load_image_slices(item["path"])
                            pixel_values_list.append(pv)
                        elif item["type"] == "video" and self.video_processor:
                            item["_video_key"] = True  # 标记，后续处理

            # 提取音频
            audio_info = None
            for turn in conversations:
                if turn["role"] != "user":
                    continue
                content = turn["content"] if isinstance(turn["content"], list) else []
                for item in content:
                    if item["type"] == "audio" and self.audio_processor:
                        mel, mask = self.audio_processor.process(item["path"])
                        audio_info = (mel, mask)
                        break

            # 构建 token 序列
            input_ids, attention_mask, labels = self._build_prompt(
                conversations, thinking_mode, pixel_values_list, audio_info
            )

            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "thinking_mode": thinking_mode.value,
                "sample_type": sample.get("type", "unknown"),
            }

            if pixel_values_list:
                result["pixel_values"] = torch.cat(pixel_values_list, dim=0)
            if audio_info:
                result["audio_mel"] = audio_info[0]
                result["audio_mask"] = audio_info[1]

            # 语音输出标签（如果 sample 包含 speech_tokens）
            if "speech_tokens" in sample:
                result["speech_labels"] = torch.tensor(sample["speech_tokens"], dtype=torch.long)

            return result

        except Exception as e:
            #print(f"[Warning] SFT 跳过损坏样本 idx={idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))


class TwoPhaseDataScheduler:
    """
    两阶段数据调度器：按训练步数自动切换 Part-1 / Part-2 数据集。

    训练计划：
      0% ~ 60% 步: Part-1 数据集（基础感知）
      60% ~ 90% 步: Part-2 数据集（复杂交互）
      90% ~ 100% 步: Final 数据集（高质量精标，塑造模型风格）
    """

    def __init__(
        self,
        part1_dataset: SFTDataset,
        part2_dataset: SFTDataset,
        final_dataset: SFTDataset,
        total_steps: int,
        part1_ratio: float = 0.60,
        part2_ratio: float = 0.30,
        # final_ratio = 0.10
    ):
        self.datasets = {
            "part1": part1_dataset,
            "part2": part2_dataset,
            "final": final_dataset,
        }
        self.total_steps = total_steps
        self.boundaries = {
            "part1": int(total_steps * part1_ratio),
            "part2": int(total_steps * (part1_ratio + part2_ratio)),
        }

    def get_dataset(self, current_step: int) -> SFTDataset:
        """根据当前步数返回对应数据集"""
        if current_step < self.boundaries["part1"]:
            return self.datasets["part1"]
        elif current_step < self.boundaries["part2"]:
            return self.datasets["part2"]
        else:
            return self.datasets["final"]

    def get_phase_name(self, current_step: int) -> str:
        if current_step < self.boundaries["part1"]:
            return "part1"
        elif current_step < self.boundaries["part2"]:
            return "part2"
        return "final"

