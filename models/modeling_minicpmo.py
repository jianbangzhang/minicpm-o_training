"""
MiniCPM-o 4.5 模型架构
对齐官方 modeling_minicpmo.py 的核心结构：
  - SigLip2 视觉编码器 + MLP Projector (LLaVA-UHD 高分辨率切片)
  - Whisper-medium 语音编码器 + Perceiver Resampler
  - Qwen3-8B 语言模型骨干 (Fast/Deep Thinking 双模式)
  - CosyVoice2 语音解码器 (流式 TTS)
referred from:https://github.com/OpenBMB/minicpm-o-4_5-pytorch-simple-demo/blob/main/MiniCPMO45/modeling_minicpmo.py
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
    WhisperModel,
    Qwen2Config,
)
from transformers.modeling_outputs import ModelOutput


# ============================================================
# 配置类
# ============================================================

@dataclass
class MiniCPMOConfig:
    """MiniCPM-o 4.5 统一配置"""
    # 视觉
    vision_encoder_name: str = "raw/models/siglip2-so400m-patch14-384"
    vision_hidden_size: int = 1152          # SigLip2 输出维度
    vision_max_pixels: int = 1_800_000     # 最大 1.8M 像素
    vision_slice_mode: str = "llava_uhd"   # 高分辨率切片策略
    vision_token_compression: int = 4      # 视觉 token 压缩倍率

    # 语音
    audio_encoder_name: str = "raw/models/whisper-medium"
    audio_hidden_size: int = 1024          # Whisper-medium 输出维度
    audio_resampler_tokens: int = 128      # Perceiver 压缩后 token 数
    audio_sample_rate: int = 16000
    audio_chunk_size_ms: int = 640         # 全双工时每帧 640ms

    # 语言模型
    llm_name: str = "raw/models/qwen3"
    llm_hidden_size: int = 4096
    llm_max_length: int = 32768
    think_token: str = "<think>"
    end_think_token: str = "</think>"

    # TTS (CosyVoice2)
    tts_model_name: str = "raw/models/cosyvoice2"
    speech_token_size: int = 6561          # CosyVoice2 speech token vocab

    # 视频
    video_frames_per_chunk: int = 6        # 每 6 帧联合压缩
    video_tokens_per_chunk: int = 64       # 压缩后 64 tokens
    video_max_frames: int = 96

    # Projector
    projector_type: str = "mlp"           # 视觉侧 MLP
    projector_hidden_size: int = 4096

    # 训练
    use_flash_attention: bool = True
    dtype: str = "bfloat16"


# ============================================================
# 视觉模块
# ============================================================

class LLaVAUHDImageProcessor:
    """
    LLaVA-UHD 高分辨率图像切片处理器。
    将任意宽高比的高分辨率图像切分为多个 448x448 切片，
    相比 naive resize 视觉 token 减少 4x。
    """
    def __init__(self, patch_size: int = 448, max_slices: int = 9):
        self.patch_size = patch_size
        self.max_slices = max_slices

    def slice_image(self, image_size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """
        计算最优切片方案，返回 [(x0,y0,x1,y1), ...] 坐标列表。
        """
        W, H = image_size
        best_slices = []
        best_score = float("inf")

        for rows in range(1, self.max_slices + 1):
            for cols in range(1, self.max_slices + 1):
                if rows * cols > self.max_slices:
                    continue
                slice_w = W / cols
                slice_h = H / rows
                # 尽量接近 patch_size x patch_size
                aspect_ratio_diff = abs(slice_w / slice_h - 1.0)
                size_diff = abs(slice_w - self.patch_size) + abs(slice_h - self.patch_size)
                score = aspect_ratio_diff + size_diff / self.patch_size
                if score < best_score:
                    best_score = score
                    best_r, best_c = rows, cols

        slice_w = W // best_c
        slice_h = H // best_r
        for r in range(best_r):
            for c in range(best_c):
                best_slices.append((
                    c * slice_w, r * slice_h,
                    (c + 1) * slice_w, (r + 1) * slice_h
                ))
        return best_slices


class VisionProjector(nn.Module):
    """
    两层 MLP 视觉 Projector：
    SigLip2 特征空间 (1152) → LLM 隐状态空间 (4096)
    """
    def __init__(self, in_dim: int = 1152, out_dim: int = 4096, compress: int = 4):
        super().__init__()
        self.compress = compress  # token 压缩倍率 (2x2 pixel shuffle)
        # pixel shuffle 后维度变为 in_dim * compress
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * compress, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D pixel shuffle 压缩 token 数：
        (B, H*W, C) -> (B, H*W//compress, C*compress)
        """
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        s = int(self.compress ** 0.5)  # stride = 2
        # reshape to 2D grid
        x = x.view(B, H, W, C)
        # 每 s×s 区域合并成一个 token
        x = x.view(B, H // s, s, W // s, s, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, (H // s) * (W // s), s * s * C)
        return x

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        x = self.pixel_shuffle(vision_features)
        return self.mlp(x)


class Video3DResampler(nn.Module):
    """
    视频 3D Resampler：将 6 帧视觉特征联合压缩为 64 tokens。
    基于 MiniCPM-V 4.5 的 unified 3D-Resampler 设计。
    96× 压缩率: 6帧 × (24×24 tokens/帧) → 64 tokens
    """
    def __init__(self, in_dim: int = 1152, out_dim: int = 4096,
                 frames_per_chunk: int = 6, out_tokens: int = 64):
        super().__init__()
        self.frames_per_chunk = frames_per_chunk
        self.out_tokens = out_tokens

        # 时序 cross-attention resampler
        self.query = nn.Parameter(torch.randn(out_tokens, out_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=8, batch_first=True
        )
        self.proj_in = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Linear(out_dim * 4, out_dim),
        )

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        frame_features: (B, T, N, C) — T 帧，每帧 N 个 patch tokens，C=1152
        返回: (B, out_tokens, out_dim)
        """
        B, T, N, C = frame_features.shape
        # flatten 时序+空间维度作为 key/value
        kv = frame_features.view(B, T * N, C)
        kv = self.proj_in(kv)  # (B, T*N, out_dim)

        # learnable queries
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # (B, out_tokens, out_dim)

        out, _ = self.cross_attn(q, kv, kv)
        out = self.norm(out + q)
        out = out + self.ffn(out)
        return out  # (B, out_tokens, out_dim)


# ============================================================
# 语音模块
# ============================================================

class AudioPerceiverResampler(nn.Module):
    """
    语音 Perceiver Resampler：
    Whisper 连续音频特征 → 固定长度 128 tokens → LLM 空间
    """
    def __init__(self, in_dim: int = 1024, out_dim: int = 4096,
                 num_latents: int = 128, num_layers: int = 2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, out_dim))
        self.proj_in = nn.Linear(in_dim, out_dim)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(out_dim, 8, batch_first=True),
                "self_attn": nn.MultiheadAttention(out_dim, 8, batch_first=True),
                "norm1": nn.LayerNorm(out_dim),
                "norm2": nn.LayerNorm(out_dim),
                "ffn": nn.Sequential(
                    nn.Linear(out_dim, out_dim * 4),
                    nn.GELU(),
                    nn.Linear(out_dim * 4, out_dim),
                ),
                "norm3": nn.LayerNorm(out_dim),
            })
            for _ in range(num_layers)
        ])

    def forward(self, audio_features: torch.Tensor,
                audio_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        audio_features: (B, T, 1024) — Whisper 编码器输出
        返回: (B, 128, 4096)
        """
        B = audio_features.shape[0]
        kv = self.proj_in(audio_features)
        x = self.latents.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            # cross-attention: latents attend to audio
            attended, _ = layer["cross_attn"](x, kv, kv, key_padding_mask=audio_mask)
            x = layer["norm1"](x + attended)
            # self-attention: latents attend to each other
            self_out, _ = layer["self_attn"](x, x, x)
            x = layer["norm2"](x + self_out)
            # FFN
            x = layer["norm3"](x + layer["ffn"](x))
        return x


# ============================================================
# 主模型
# ============================================================

class MiniCPMOModel(PreTrainedModel):
    """
    MiniCPM-o 4.5 完整模型。

    前向路径：
      图像 → SigLip2 → VisionProjector → visual_tokens
      视频 → SigLip2(逐帧) → Video3DResampler → video_tokens
      音频 → Whisper → AudioPerceiverResampler → audio_tokens
      [visual_tokens | video_tokens | audio_tokens | text_tokens]
            → Qwen3-8B LLM → 文本 logits + 语音 token logits
    """

    def __init__(self, config: MiniCPMOConfig):
        super().__init__(config)  # 实际使用时传入 PretrainedConfig
        self.config = config

        # —— 视觉编码器（SigLip2，通常从预训练加载后冻结）——
        self.vision_encoder = None  # 通过 from_pretrained 加载
        self.vision_projector = VisionProjector(
            in_dim=config.vision_hidden_size,
            out_dim=config.llm_hidden_size,
            compress=config.vision_token_compression,
        )
        self.video_resampler = Video3DResampler(
            in_dim=config.vision_hidden_size,
            out_dim=config.llm_hidden_size,
            frames_per_chunk=config.video_frames_per_chunk,
            out_tokens=config.video_tokens_per_chunk,
        )
        self.image_processor = LLaVAUHDImageProcessor()

        # —— 语音编码器（Whisper-medium，通常从预训练加载后冻结）——
        self.audio_encoder = None  # 通过 from_pretrained 加载
        self.audio_resampler = AudioPerceiverResampler(
            in_dim=config.audio_hidden_size,
            out_dim=config.llm_hidden_size,
            num_latents=config.audio_resampler_tokens,
        )

        # —— 语言模型骨干（Qwen3-8B）——
        self.llm = None  # 通过 from_pretrained 加载

        # —— 语音输出头（映射到 CosyVoice2 speech tokens）——
        self.speech_lm_head = nn.Linear(
            config.llm_hidden_size, config.speech_token_size, bias=False
        )

        # 特殊 token 嵌入占位（实际从 tokenizer 读取）
        self._im_start_id = None
        self._im_end_id = None
        self._audio_start_id = None
        self._audio_end_id = None
        self._speech_start_id = None

    @classmethod
    def from_pretrained_components(
        cls,
        config: MiniCPMOConfig,
        init_vision: bool = True,
        init_audio: bool = True,
        init_tts: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "MiniCPMOModel":
        """工厂方法：加载所有预训练组件并组装"""
        model = cls(config)

        if init_vision:
            print(f"[Init] 加载视觉编码器: {config.vision_encoder_name}")
            model.vision_encoder = AutoModel.from_pretrained(
                config.vision_encoder_name, torch_dtype=dtype
            )

        if init_audio:
            print(f"[Init] 加载语音编码器: {config.audio_encoder_name}")
            model.audio_encoder = WhisperModel.from_pretrained(
                config.audio_encoder_name, torch_dtype=dtype
            ).encoder

        if init_tts:
            print(f"[Init] 加载语音解码器: {config.audio_encoder_name}")

        print(f"[Init] 加载语言模型: {config.llm_name}")
        model.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_name,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2" if config.use_flash_attention else "sdpa",
        )
        return model.to(dtype)

    def encode_images(self, pixel_values: torch.Tensor,
                      slice_coords: Optional[List] = None) -> torch.Tensor:
        """
        图像编码：SigLip2 + 切片合并 + VisionProjector
        pixel_values: (B, 3, H, W) 或 (B*num_slices, 3, 448, 448)
        返回: (B, num_visual_tokens, llm_hidden_size)
        """
        # SigLip2 编码
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        # 取 patch token 特征 (去掉 CLS)
        features = vision_outputs.last_hidden_state[:, 1:, :]  # (B, N_patch, 1152)

        # 切片合并（如果有多切片）
        if slice_coords is not None:
            features = self._merge_slices(features, slice_coords)

        # Projector: pixel shuffle + MLP
        visual_tokens = self.vision_projector(features)  # (B, N/4, 4096)
        return visual_tokens

    def encode_video(self, frame_pixel_values: torch.Tensor) -> torch.Tensor:
        """
        视频编码：每帧独立 SigLip2 → 3D Resampler 压缩
        frame_pixel_values: (B, T, 3, 448, 448)
        返回: (B, num_chunks * 64, 4096)
        """
        B, T, C, H, W = frame_pixel_values.shape
        chunk_size = self.config.video_frames_per_chunk
        num_chunks = T // chunk_size
        all_tokens = []

        for i in range(num_chunks):
            chunk = frame_pixel_values[:, i * chunk_size:(i + 1) * chunk_size]
            # 批量编码所有帧
            chunk_flat = chunk.view(B * chunk_size, C, H, W)
            vision_out = self.vision_encoder(pixel_values=chunk_flat)
            feats = vision_out.last_hidden_state[:, 1:, :]  # (B*6, N, 1152)
            feats = feats.view(B, chunk_size, feats.shape[1], feats.shape[2])
            # 3D Resampler: 6帧 → 64 tokens
            chunk_tokens = self.video_resampler(feats)  # (B, 64, 4096)
            all_tokens.append(chunk_tokens)

        return torch.cat(all_tokens, dim=1)  # (B, num_chunks*64, 4096)

    def encode_audio(self, audio_features: torch.Tensor,
                     audio_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        语音编码：Whisper encoder → Perceiver Resampler
        audio_features: (B, 80, T) — mel spectrogram
        返回: (B, 128, 4096)
        """
        # Whisper 编码
        whisper_out = self.audio_encoder(audio_features)
        audio_hidden = whisper_out.last_hidden_state  # (B, T/2, 1024)

        # Perceiver 重采样到固定 128 tokens
        audio_tokens = self.audio_resampler(audio_hidden, audio_mask)
        return audio_tokens  # (B, 128, 4096)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        # 可选模态输入
        pixel_values: Optional[torch.Tensor] = None,
        slice_coords: Optional[List] = None,
        video_frames: Optional[torch.Tensor] = None,
        audio_mel: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        # 标签（训练用）
        labels: Optional[torch.LongTensor] = None,
        speech_labels: Optional[torch.LongTensor] = None,
        # 控制参数
        enable_thinking: bool = False,
        return_dict: bool = True,
    ):
        """
        统一前向函数，支持文本/视觉/视频/语音任意组合输入。
        视觉/语音 token 通过特殊占位符嵌入位置替换注入。
        """
        # Step 1: 获取文本 token 嵌入
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # Step 2: 替换视觉占位符
        if pixel_values is not None:
            visual_tokens = self.encode_images(pixel_values, slice_coords)
            inputs_embeds = self._replace_placeholder(
                inputs_embeds, visual_tokens, input_ids, self._im_start_id, self._im_end_id
            )

        if video_frames is not None:
            video_tokens = self.encode_video(video_frames)
            inputs_embeds = self._replace_placeholder(
                inputs_embeds, video_tokens, input_ids, self._im_start_id, self._im_end_id
            )

        # Step 3: 替换语音占位符
        if audio_mel is not None:
            audio_tokens = self.encode_audio(audio_mel, audio_mask)
            inputs_embeds = self._replace_placeholder(
                inputs_embeds, audio_tokens, input_ids, self._audio_start_id, self._audio_end_id
            )

        # Step 4: LLM 前向
        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=None,  # 我们手动计算 loss
            output_hidden_states=True,
            return_dict=True,
        )

        lm_logits = llm_outputs.logits  # (B, L, vocab_size)
        last_hidden = llm_outputs.hidden_states[-1]  # (B, L, 4096)

        # Step 5: 语音输出头
        speech_logits = self.speech_lm_head(last_hidden)  # (B, L, speech_vocab)

        # Step 6: 计算 Loss
        total_loss = None
        if labels is not None:
            # 文本 LM loss
            shift_logits = lm_logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            text_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            total_loss = text_loss

            # 语音预测 loss（如果有标签）
            if speech_labels is not None:
                shift_speech_logits = speech_logits[:, :-1].contiguous()
                shift_speech_labels = speech_labels[:, 1:].contiguous()
                speech_loss = F.cross_entropy(
                    shift_speech_logits.view(-1, shift_speech_logits.size(-1)),
                    shift_speech_labels.view(-1),
                    ignore_index=-100,
                )
                total_loss = total_loss + speech_loss

        if return_dict:
            return {
                "loss": total_loss,
                "lm_logits": lm_logits,
                "speech_logits": speech_logits,
                "last_hidden_state": last_hidden,
            }
        return total_loss, lm_logits, speech_logits

    def _replace_placeholder(
        self, inputs_embeds, modal_tokens, input_ids, start_id, end_id
    ) -> torch.Tensor:
        """
        将 inputs_embeds 中 [start_id ... end_id] 区间的占位符
        替换为实际的多模态 token 嵌入。
        modal_tokens: (B, N_modal, hidden)
        """
        B = inputs_embeds.shape[0]
        new_embeds = inputs_embeds.clone()

        for b in range(B):
            ids = input_ids[b]
            start_pos = (ids == start_id).nonzero(as_tuple=True)[0]
            end_pos = (ids == end_id).nonzero(as_tuple=True)[0]
            if len(start_pos) == 0 or len(end_pos) == 0:
                continue
            # 假设单个模态区间（多区间需要迭代处理）
            s, e = start_pos[0].item() + 1, end_pos[0].item()
            slot_len = e - s
            tok_len = modal_tokens.shape[1]
            # 如果 token 数不匹配，截断或填充
            actual_len = min(slot_len, tok_len)
            new_embeds[b, s:s + actual_len] = modal_tokens[b, :actual_len]
        return new_embeds

    def _merge_slices(self, features, slice_coords):
        """切片特征合并（占位，实际按切片索引拼接）"""
        return features

    def get_trainable_parameters(self, stage: int):
        """根据训练阶段返回需要更新的参数组"""
        if stage == 1:
            # Stage 1: 只训练 Projector
            trainable = list(self.vision_projector.parameters()) + \
                        list(self.video_resampler.parameters()) + \
                        list(self.audio_resampler.parameters())
            # 冻结所有 encoder 和 LLM
            for name, p in self.named_parameters():
                p.requires_grad = False
            for p in trainable:
                p.requires_grad = True
            return trainable

        elif stage == 2:
            # Stage 2: 全参数训练，但 encoder 学习率更低
            for p in self.parameters():
                p.requires_grad = True
            encoder_params = (
                list(self.vision_encoder.parameters()) +
                list(self.audio_encoder.parameters())
            )
            projector_params = (
                list(self.vision_projector.parameters()) +
                list(self.video_resampler.parameters()) +
                list(self.audio_resampler.parameters())
            )
            llm_params = list(self.llm.parameters())
            return [
                {"params": encoder_params, "lr": 1e-5},
                {"params": projector_params, "lr": 5e-4},
                {"params": llm_params, "lr": 1e-4},
            ]

        elif stage == 3:
            # Stage 3: SFT 全参数（或 LoRA）
            for p in self.parameters():
                p.requires_grad = True
            return list(self.parameters())

        elif stage == 4:
            # Stage 4: RL，只训练 LLM（保持 encoder 固定）
            for p in self.parameters():
                p.requires_grad = False
            for p in self.llm.parameters():
                p.requires_grad = True
            for p in self.speech_lm_head.parameters():
                p.requires_grad = True
            return list(self.llm.parameters()) + list(self.speech_lm_head.parameters())
