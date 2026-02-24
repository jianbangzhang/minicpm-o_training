"""
视频帧处理器 & 音频预处理工具
"""

import math
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from PIL import Image


# ============================================================
# 视频帧处理
# ============================================================

class VideoFrameProcessor:
    """
    视频帧采样与预处理。
    MiniCPM-o 4.5: 每 6 帧联合压缩为 64 tokens（96× 压缩率）
    """

    def __init__(
        self,
        frame_size: int = 448,
        frames_per_chunk: int = 6,
        max_frames: int = 96,
        train_mode: bool = True,
    ):
        self.frame_size = frame_size
        self.frames_per_chunk = frames_per_chunk
        self.max_frames = max_frames
        self.train_mode = train_mode

        # ImageNet 归一化（SigLip2 使用 0.5/0.5 标准化）
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def sample_frames(
        self,
        total_frames: int,
        strategy: str = "uniform",
    ) -> List[int]:
        """
        帧采样策略。

        Args:
            total_frames: 视频总帧数
            strategy: "uniform" | "random_uniform" | "keyframe"
        """
        # 训练时随机采样帧数（增强泛化性）
        if self.train_mode:
            num_frames = random.choice(
                [f for f in [12, 24, 48, 96] if f <= total_frames]
            )
            if not num_frames:
                num_frames = min(self.frames_per_chunk, total_frames)
        else:
            num_frames = min(self.max_frames, total_frames)

        # 对齐到 frames_per_chunk 的倍数
        num_frames = (num_frames // self.frames_per_chunk) * self.frames_per_chunk
        num_frames = max(num_frames, self.frames_per_chunk)

        if strategy == "random_uniform":
            # 在均匀分组内随机抖动，增加多样性
            indices = []
            segment = total_frames / num_frames
            for i in range(num_frames):
                base = int(i * segment)
                jitter = random.randint(0, max(0, int(segment) - 1))
                indices.append(min(base + jitter, total_frames - 1))
            return indices
        else:
            # 均匀采样
            return [int(i * total_frames / num_frames) for i in range(num_frames)]

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """单帧预处理：resize + 归一化"""
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)

        # 中心裁剪为正方形后 resize
        w, h = frame.size
        min_dim = min(w, h)
        frame = frame.crop((
            (w - min_dim) // 2, (h - min_dim) // 2,
            (w + min_dim) // 2, (h + min_dim) // 2,
        ))
        frame = frame.resize((self.frame_size, self.frame_size), Image.BICUBIC)
        frame_np = np.array(frame).astype(np.float32) / 255.0

        # 归一化
        mean = np.array(self.mean)
        std = np.array(self.std)
        frame_np = (frame_np - mean) / std

        return torch.from_numpy(frame_np).permute(2, 0, 1).float()  # (3, H, W)

    def process_video_frames(
        self,
        frames: List[np.ndarray],
        frame_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        处理视频帧序列。

        Returns:
            tensor: (T, 3, H, W) — T 帧，已归一化
        """
        if frame_indices is not None:
            frames = [frames[i] for i in frame_indices]

        processed = [self.preprocess_frame(f) for f in frames]
        return torch.stack(processed, dim=0)  # (T, 3, H, W)


# ============================================================
# 音频预处理
# ============================================================

class AudioProcessor:
    """
    音频 mel spectrogram 提取。
    Whisper 使用 80 维 mel，窗口 25ms，帧移 10ms，16kHz。
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        chunk_size_ms: int = 640,   # 全双工流式推理时每帧 640ms
        train_mode: bool = True,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_size_ms = chunk_size_ms
        self.chunk_samples = int(sample_rate * chunk_size_ms / 1000)
        self.train_mode = train_mode

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0.0,
            f_max=8000.0,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)

    def load_audio(self, path: str, max_duration: float = 30.0) -> torch.Tensor:
        """加载并重采样音频"""
        wav, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # 转单声道
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # 截断到最大时长
        max_samples = int(max_duration * self.sample_rate)
        wav = wav[:, :max_samples]

        return wav.squeeze(0)  # (T,)

    def compute_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """
        计算 mel spectrogram，对齐 Whisper 格式。

        Args:
            wav: (T,) 原始波形

        Returns:
            mel: (n_mels, T') 对数 mel 频谱图
        """
        mel = self.mel_transform(wav.unsqueeze(0)).squeeze(0)  # (80, T')
        mel = self.amplitude_to_db(mel)
        # Whisper 归一化：[-1, 1]
        mel = (mel + 40.0) / 40.0
        return mel  # (80, T')

    def pad_or_trim(
        self,
        mel: torch.Tensor,
        target_length: int = 3000,  # 30秒 @ 10ms帧移
        pad_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        填充或裁剪 mel 到固定长度，返回 (mel, mask)。
        mask: 1 表示有效帧，0 表示填充帧
        """
        T_mel = mel.shape[-1]
        if T_mel >= target_length:
            mel = mel[:, :target_length]
            mask = torch.ones(target_length, dtype=torch.bool)
        else:
            pad_len = target_length - T_mel
            mel = torch.cat([mel, torch.full((self.n_mels, pad_len), pad_value)], dim=-1)
            mask = torch.cat([
                torch.ones(T_mel, dtype=torch.bool),
                torch.zeros(pad_len, dtype=torch.bool),
            ])
        return mel, mask

    def augment(self, wav: torch.Tensor) -> torch.Tensor:
        """训练时数据增强"""
        if not self.train_mode:
            return wav

        # 音量扰动
        gain = random.uniform(0.7, 1.3)
        wav = wav * gain

        # 加性高斯噪声（低概率）
        if random.random() < 0.3:
            noise_level = random.uniform(0.001, 0.01)
            wav = wav + torch.randn_like(wav) * noise_level

        # 速度扰动（0.9x ~ 1.1x，通过重采样实现）
        if random.random() < 0.3:
            speed = random.uniform(0.9, 1.1)
            target_sr = int(self.sample_rate * speed)
            wav = torchaudio.functional.resample(wav.unsqueeze(0), target_sr, self.sample_rate).squeeze(0)

        return wav.clamp(-1.0, 1.0)

    def process(
        self,
        audio_path: str,
        max_duration: float = 30.0,
        target_mel_length: int = 3000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完整音频处理流水线。

        Returns:
            mel: (n_mels, target_mel_length)
            mask: (target_mel_length,) — True=有效
        """
        wav = self.load_audio(audio_path, max_duration)
        if self.train_mode:
            wav = self.augment(wav)
        mel = self.compute_mel(wav)
        mel, mask = self.pad_or_trim(mel, target_mel_length)
        return mel, mask

    def chunk_audio_stream(
        self, wav: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        将音频切割为全双工流式推理的 640ms chunks。

        Returns:
            List of (mel_chunk, mask_chunk)
        """
        chunks = []
        for start in range(0, len(wav), self.chunk_samples):
            chunk = wav[start:start + self.chunk_samples]
            if len(chunk) < self.chunk_samples // 4:
                break  # 太短的尾部丢弃
            mel = self.compute_mel(chunk)
            # 每 chunk 对应的 mel 帧数
            chunk_mel_len = int(self.chunk_size_ms / 10)  # 64 帧
            mel, mask = self.pad_or_trim(mel, chunk_mel_len)
            chunks.append((mel, mask))
        return chunks
