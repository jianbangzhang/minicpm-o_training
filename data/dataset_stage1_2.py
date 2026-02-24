"""
Stage 1 & Stage 2 数据集

Stage 1: 模态对齐预训练数据集
  - 图文对 (LAION/CC/COYO 子集)
  - 语音-文本对 (WenetSpeech/LibriSpeech)

Stage 2: 统一多模态预训练数据集
  - 动态 OCR 噪声掩码文档数据
  - 视频理解数据
  - 交错图文数据
  - 语音对话数据
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import PreTrainedTokenizer

from ..utils.ocr_noise import DynamicOCRNoise, get_noise_system_prompt
from ..utils.video_audio_processor import AudioProcessor, VideoFrameProcessor


# ============================================================
# 数据格式规范
# ============================================================

@dataclass
class DataSchema:
    """
    统一数据格式规范（所有阶段共用 JSONL 格式）

    Stage 1/2 图文样本：
    {
        "type": "image_text",
        "image": "/path/to/image.jpg",
        "text": "图像描述文字"
    }

    Stage 2 OCR 文档样本：
    {
        "type": "ocr_document",
        "image": "/path/to/doc.png",
        "ocr_text": "文档完整文字内容",
        "text_boxes": [[x0,y0,x1,y1], ...],  # 可选，文字区域坐标
        "noise_level": null  # null 表示随机，0~5 表示固定
    }

    Stage 2 视频样本：
    {
        "type": "video",
        "video": "/path/to/video.mp4",
        "caption": "视频描述"
    }

    Stage 1 语音文本对：
    {
        "type": "audio_text",
        "audio": "/path/to/audio.wav",
        "transcript": "语音转录文本"
    }

    Stage 2 交错图文：
    {
        "type": "interleaved",
        "items": [
            {"type": "text", "content": "前置文字"},
            {"type": "image", "path": "/path/img1.jpg"},
            {"type": "text", "content": "图片间文字"},
            {"type": "image", "path": "/path/img2.jpg"},
        ]
    }
    """
    pass


# ============================================================
# Stage 1: 模态对齐数据集
# ============================================================

class Stage1AlignmentDataset(Dataset):
    """
    Stage 1 模态对齐数据集。
    数据配比：图文对 70% + 语音-文本对 30%

    典型数据源：
      图文: LAION-400M 子集, CC3M/CC12M, COCO Caption
      语音: LibriSpeech, WenetSpeech, AISHELL-3

    数据 JSONL 格式：每行一个样本（见 DataSchema）
    """

    def __init__(
        self,
        data_paths: List[str],           # 多个 JSONL 文件路径
        tokenizer: PreTrainedTokenizer,
        audio_processor: AudioProcessor,
        image_size: int = 448,
        max_text_len: int = 512,
        # 特殊 token
        im_start_token: str = "<image>",
        im_end_token: str = "</image>",
        audio_start_token: str = "<audio>",
        audio_end_token: str = "</audio>",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.max_text_len = max_text_len
        self.image_size = image_size

        self.im_start_id = tokenizer.convert_tokens_to_ids(im_start_token)
        self.im_end_id = tokenizer.convert_tokens_to_ids(im_end_token)
        self.audio_start_id = tokenizer.convert_tokens_to_ids(audio_start_token)
        self.audio_end_id = tokenizer.convert_tokens_to_ids(audio_end_token)

        # 加载所有数据元信息（只读路径，不预加载图像）
        self.samples = []
        for path in data_paths:
            with open(path, "r") as f:
                for line in f:
                    sample = json.loads(line.strip())
                    self.samples.append(sample)

        print(f"[Stage1Dataset] 加载 {len(self.samples)} 条样本")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """加载并预处理图像为 (3, H, W) tensor"""
        img = Image.open(image_path).convert("RGB")
        # 简单 resize（Stage 1 不做切片，Stage 2 再引入 LLaVA-UHD）
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        img_np = torch.tensor(
            (torch.tensor(list(img.getdata())).view(self.image_size, self.image_size, 3).float() / 255.0 - 0.5) / 0.5
        ).permute(2, 0, 1)
        return img_np

    def _build_image_text_sample(self, sample: Dict) -> Dict[str, Any]:
        """构建图文对训练样本"""
        # 构造输入序列: [BOS] <image> [视觉占位符×N] </image> [文字描述] [EOS]
        num_visual_slots = (self.image_size // 14) ** 2 // 4  # 切片后 token 数估算

        prompt = f"Describe the image in detail."
        response = sample.get("text", sample.get("caption", ""))

        # Tokenize（视觉部分用占位符，训练时替换）
        visual_placeholder = "X" * num_visual_slots  # 占位，实际由模型替换
        input_text = (
            f"<image>{visual_placeholder}</image>\n"
            f"User: {prompt}\nAssistant: {response}"
        )
        tokens = self.tokenizer(
            input_text,
            max_length=self.max_text_len + num_visual_slots,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # 标签：只对 response 部分计算 loss（其余位置 -100）
        labels = input_ids.clone()
        # 找到 "Assistant:" 之后的位置
        assistant_start = self._find_response_start(input_ids)
        labels[:assistant_start] = -100

        # 加载图像
        pixel_values = self._load_and_preprocess_image(sample["image"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values.unsqueeze(0),  # (1, 3, H, W)
            "sample_type": "image_text",
        }

    def _build_audio_text_sample(self, sample: Dict) -> Dict[str, Any]:
        """构建语音-文本对训练样本"""
        transcript = sample.get("transcript", "")
        mel, mask = self.audio_processor.process(sample["audio"])

        # 输入序列：[BOS] <audio> [音频占位符] </audio> Transcribe: [EOS]
        input_text = f"<audio>{'A' * 128}</audio>\nTranscribe: {transcript}"
        tokens = self.tokenizer(
            input_text,
            max_length=self.max_text_len + 128,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        # 只对转录文本计算 loss
        transcribe_pos = self._find_response_start(input_ids)
        labels[:transcribe_pos] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "audio_mel": mel,         # (80, T)
            "audio_mask": mask,       # (T,)
            "sample_type": "audio_text",
        }

    def _find_response_start(self, input_ids: torch.Tensor) -> int:
        """找到 response 开始位置（简化版，实际按 tokenizer 特殊 token 定位）"""
        # 实际实现：找到 Assistant: 对应的 token 位置
        return len(input_ids) // 2

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        sample_type = sample.get("type", "image_text")

        try:
            if sample_type == "image_text":
                return self._build_image_text_sample(sample)
            elif sample_type == "audio_text":
                return self._build_audio_text_sample(sample)
            else:
                # 默认作为图文处理
                return self._build_image_text_sample(sample)
        except Exception as e:
            # 跳过损坏样本
            print(f"[Warning] 跳过损坏样本 idx={idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))


# ============================================================
# Stage 2: 统一多模态预训练数据集
# ============================================================

class Stage2MultimodalDataset(Dataset):
    """
    Stage 2 统一多模态预训练数据集。

    核心特性：
    1. 动态 OCR 噪声掩码：文档图像随机施加 0~5 级噪声
    2. LLaVA-UHD 高分辨率切片：支持任意宽高比
    3. 视频 6 帧联合压缩

    数据配比（通过 WeightedRandomSampler 控制）：
      图文交错: 35%, OCR文档: 25%, 视频: 20%, 语音: 20%
    """

    # 数据类型权重（按比例采样）
    TYPE_WEIGHTS = {
        "interleaved": 0.35,
        "ocr_document": 0.25,
        "video": 0.20,
        "audio_text": 0.20,
    }

    def __init__(
        self,
        data_paths: Dict[str, List[str]],  # {"type": [paths]}
        tokenizer: PreTrainedTokenizer,
        audio_processor: AudioProcessor,
        video_processor: VideoFrameProcessor,
        ocr_noiser: DynamicOCRNoise,
        max_seq_len: int = 8192,
        image_max_pixels: int = 1_800_000,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.video_processor = video_processor
        self.ocr_noiser = ocr_noiser
        self.max_seq_len = max_seq_len
        self.image_max_pixels = image_max_pixels

        # 加载所有类型数据
        self.samples_by_type: Dict[str, List[Dict]] = {}
        self.all_samples: List[Tuple[str, int]] = []  # (type, index)

        for data_type, paths in data_paths.items():
            samples = []
            for path in paths:
                with open(path, "r") as f:
                    for line in f:
                        try:
                            samples.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
            self.samples_by_type[data_type] = samples
            for i in range(len(samples)):
                self.all_samples.append((data_type, i))

        print(f"[Stage2Dataset] 各类型数据量：")
        for t, s in self.samples_by_type.items():
            print(f"  {t}: {len(s):,} 条")
        print(f"  总计: {len(self.all_samples):,} 条")

    def get_sampler_weights(self) -> List[float]:
        """
        计算每个样本的采样权重（实现数据配比）。
        权重 = type_weight / type_count
        """
        type_counts = {t: len(s) for t, s in self.samples_by_type.items()}
        weights = []
        for data_type, idx in self.all_samples:
            weight = self.TYPE_WEIGHTS.get(data_type, 0.1) / max(type_counts[data_type], 1)
            weights.append(weight)
        return weights

    def __len__(self) -> int:
        return len(self.all_samples)

    # ── OCR 文档样本 ─────────────────────────────────────────

    def _build_ocr_document_sample(self, sample: Dict) -> Dict[str, Any]:
        """
        构建 OCR 文档训练样本（含动态噪声掩码）。

        根据噪声级别决定训练目标：
          - 低噪声(0~2): 精确转录任务
          - 高噪声(3~5): 内容理解/补全任务
        """
        image = Image.open(sample["image"]).convert("RGB")
        text_boxes = sample.get("text_boxes", None)
        fixed_level = sample.get("noise_level", None)

        # 施加动态噪声
        noisy_image, actual_level = self.ocr_noiser.apply(
            image, level=fixed_level, text_boxes=text_boxes
        )

        # 根据噪声级别选择任务类型
        sys_prompt = get_noise_system_prompt(actual_level)
        ocr_text = sample.get("ocr_text", "")

        if actual_level <= 2:
            # 精确 OCR 任务
            task_prompt = "Please transcribe all text in this document."
            response = ocr_text
        elif actual_level <= 4:
            # 部分推理任务
            task_prompt = "Describe the content and key information in this document."
            # 截取部分文字作为 response（模拟部分可见+推理）
            words = ocr_text.split()
            visible_ratio = (5 - actual_level) / 3.0
            visible_words = words[:int(len(words) * visible_ratio)]
            response = " ".join(visible_words) + " [content inferred from context]"
        else:
            # 纯推理任务
            task_prompt = "What type of document is this? What is its main purpose?"
            response = f"Based on the document structure, this appears to be related to: {ocr_text[:100]}"

        # 像素预处理（LLaVA-UHD 切片）
        pixel_values = self._preprocess_image_uhd(noisy_image)

        input_text = (
            f"System: {sys_prompt}\n"
            f"<image>{'X' * pixel_values.shape[0] * 256}</image>\n"
            f"User: {task_prompt}\nAssistant: {response}"
        )
        tokens = self.tokenizer(
            input_text, max_length=self.max_seq_len, truncation=True, return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[:len(input_ids) // 2] = -100  # 简化：后半段计算 loss

        return {
            "input_ids": input_ids,
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels,
            "pixel_values": pixel_values,
            "sample_type": "ocr_document",
            "noise_level": actual_level,
        }

    def _preprocess_image_uhd(self, image: Image.Image) -> torch.Tensor:
        """
        LLaVA-UHD 高分辨率切片预处理。
        将图像切为多个 448×448 切片，返回 (num_slices, 3, 448, 448)
        """
        w, h = image.size
        total_pixels = w * h

        # 降分辨率到最大像素限制
        if total_pixels > self.image_max_pixels:
            scale = (self.image_max_pixels / total_pixels) ** 0.5
            w, h = int(w * scale), int(h * scale)
            image = image.resize((w, h), Image.BICUBIC)

        # 按 448 切片
        patch_size = 448
        cols = max(1, round(w / patch_size))
        rows = max(1, round(h / patch_size))
        # 最多 9 个切片
        while rows * cols > 9:
            if rows > cols:
                rows -= 1
            else:
                cols -= 1

        slice_w = w // cols
        slice_h = h // rows
        slices = []
        for r in range(rows):
            for c in range(cols):
                box = (c * slice_w, r * slice_h, (c + 1) * slice_w, (r + 1) * slice_h)
                patch = image.crop(box).resize((patch_size, patch_size), Image.BICUBIC)
                import numpy as np
                patch_t = torch.from_numpy(
                    (np.array(patch).astype(np.float32) / 255.0 - 0.5) / 0.5
                ).permute(2, 0, 1)
                slices.append(patch_t)

        # 额外加一个全局缩略图
        thumbnail = image.resize((patch_size, patch_size), Image.BICUBIC)
        import numpy as np
        thumb_t = torch.from_numpy(
            (np.array(thumbnail).astype(np.float32) / 255.0 - 0.5) / 0.5
        ).permute(2, 0, 1)
        slices.insert(0, thumb_t)

        return torch.stack(slices, dim=0)  # (num_slices, 3, 448, 448)

    # ── 视频样本 ──────────────────────────────────────────────

    def _build_video_sample(self, sample: Dict) -> Dict[str, Any]:
        """构建视频理解训练样本"""
        import decord
        vr = decord.VideoReader(sample["video"])
        total_frames = len(vr)

        indices = self.video_processor.sample_frames(
            total_frames, strategy="random_uniform"
        )
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)

        frame_tensors = self.video_processor.process_video_frames(list(frames))
        # (T, 3, 448, 448) → (T//6, 6, 3, 448, 448) for 3D resampler
        T = frame_tensors.shape[0]
        chunk_size = self.video_processor.frames_per_chunk
        num_chunks = T // chunk_size
        frame_tensors = frame_tensors[:num_chunks * chunk_size]
        frame_chunks = frame_tensors.view(num_chunks, chunk_size, 3, 448, 448)

        caption = sample.get("caption", sample.get("description", ""))
        num_video_tokens = num_chunks * 64  # 每 chunk 64 tokens

        input_text = (
            f"<video>{'V' * num_video_tokens}</video>\n"
            f"User: Describe what happens in this video.\nAssistant: {caption}"
        )
        tokens = self.tokenizer(
            input_text, max_length=self.max_seq_len, truncation=True, return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[:num_video_tokens + 20] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels,
            "video_frames": frame_chunks,   # (num_chunks, 6, 3, 448, 448)
            "sample_type": "video",
        }

    # ── 交错图文样本 ──────────────────────────────────────────

    def _build_interleaved_sample(self, sample: Dict) -> Dict[str, Any]:
        """构建交错图文预训练样本"""
        items = sample.get("items", [])
        all_pixel_values = []
        text_parts = []

        for item in items:
            if item["type"] == "text":
                text_parts.append(item["content"])
            elif item["type"] == "image":
                try:
                    img = Image.open(item["path"]).convert("RGB")
                    pv = self._preprocess_image_uhd(img)
                    all_pixel_values.append(pv)
                    text_parts.append(f"<image>{'X' * 256}</image>")
                except Exception:
                    text_parts.append("[IMAGE]")

        full_text = " ".join(text_parts)
        tokens = self.tokenizer(
            full_text, max_length=self.max_seq_len, truncation=True, return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze(0)
        # 交错预训练：全序列计算 LM loss
        labels = input_ids.clone()

        result = {
            "input_ids": input_ids,
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels,
            "sample_type": "interleaved",
        }
        if all_pixel_values:
            # 拼接所有图像切片
            result["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        return result

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data_type, sample_idx = self.all_samples[idx]
        sample = self.samples_by_type[data_type][sample_idx]

        try:
            if data_type == "ocr_document":
                return self._build_ocr_document_sample(sample)
            elif data_type == "video":
                return self._build_video_sample(sample)
            elif data_type == "interleaved":
                return self._build_interleaved_sample(sample)
            elif data_type == "audio_text":
                # 复用 Stage1 的逻辑
                mel, mask = self.audio_processor.process(sample["audio"])
                transcript = sample.get("transcript", "")
                input_text = f"<audio>{'A' * 128}</audio>\nTranscribe: {transcript}"
                tokens = self.tokenizer(
                    input_text, max_length=self.max_seq_len, truncation=True, return_tensors="pt"
                )
                input_ids = tokens["input_ids"].squeeze(0)
                labels = input_ids.clone()
                labels[:140] = -100
                return {
                    "input_ids": input_ids,
                    "attention_mask": tokens["attention_mask"].squeeze(0),
                    "labels": labels,
                    "audio_mel": mel,
                    "audio_mask": mask,
                    "sample_type": "audio_text",
                }
        except Exception as e:
            print(f"[Warning] Stage2 跳过损坏样本 idx={idx} type={data_type}: {e}")
            return self.__getitem__((idx + 1) % len(self))
