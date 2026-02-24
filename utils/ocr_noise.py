"""
动态 OCR 噪声掩码工具 (Dynamic Text Corruption)
MiniCPM-o 4.5 核心创新：统一 OCR 能力与文档知识学习

噪声强度级别 0~5：
  0 = 原图（完全可读）
  1 = 轻微模糊
  2 = 中度模糊 + 轻微遮挡
  3 = 强模糊 + 中度遮挡
  4 = 严重扭曲 + 大面积遮挡
  5 = 几乎不可读（依赖上下文推理）
"""

import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class DynamicOCRNoise:
    """
    对文档图像中文字区域施加分级动态噪声。
    训练目标：
      - 低噪声时精确转录文字 → 学习 OCR 能力
      - 高噪声时依据上下文补全 → 学习文档知识与推理
    """

    def __init__(
        self,
        noise_level_range: Tuple[int, int] = (0, 5),
        apply_prob: float = 0.7,          # 施加噪声的概率
        text_region_prob: float = 0.8,    # 只在文字区域施加噪声的概率
    ):
        self.min_level = noise_level_range[0]
        self.max_level = noise_level_range[1]
        self.apply_prob = apply_prob
        self.text_region_prob = text_region_prob

    def _sample_level(self) -> int:
        """均匀采样噪声级别"""
        return random.randint(self.min_level, self.max_level)

    # ── 基础噪声操作 ──────────────────────────────────────────

    def _gaussian_blur(self, img_np: np.ndarray, level: int) -> np.ndarray:
        """高斯模糊，核大小随级别增大"""
        radius = [0, 1, 3, 5, 9, 15][level]
        if radius == 0:
            return img_np
        return cv2.GaussianBlur(img_np, (radius * 2 + 1, radius * 2 + 1), 0)

    def _random_occlusion(self, img_np: np.ndarray, level: int,
                          text_boxes: Optional[List] = None) -> np.ndarray:
        """
        随机矩形遮挡（模拟印刷/扫描瑕疵）。
        level 越高，遮挡面积越大、块数越多。
        """
        if level < 2:
            return img_np
        result = img_np.copy()
        H, W = result.shape[:2]
        num_blocks = [0, 0, 2, 5, 10, 20][level]
        max_size = [0, 0, 0.05, 0.1, 0.2, 0.35][level]

        for _ in range(num_blocks):
            bh = int(H * random.uniform(0.01, max_size))
            bw = int(W * random.uniform(0.02, max_size * 2))
            x = random.randint(0, max(0, W - bw))
            y = random.randint(0, max(0, H - bh))
            # 用随机灰色填充（而非纯黑，更接近真实噪声）
            fill_val = random.randint(200, 255)
            result[y:y + bh, x:x + bw] = fill_val
        return result

    def _motion_blur(self, img_np: np.ndarray, level: int) -> np.ndarray:
        """运动模糊（模拟扫描抖动）"""
        if level < 3:
            return img_np
        size = [0, 0, 0, 7, 15, 25][level]
        kernel = np.zeros((size, size))
        direction = random.choice(["h", "v", "d"])
        if direction == "h":
            kernel[size // 2, :] = 1
        elif direction == "v":
            kernel[:, size // 2] = 1
        else:
            np.fill_diagonal(kernel, 1)
        kernel /= kernel.sum()
        return cv2.filter2D(img_np, -1, kernel)

    def _perspective_distort(self, img_np: np.ndarray, level: int) -> np.ndarray:
        """透视扭曲（模拟拍照角度偏差）"""
        if level < 4:
            return img_np
        H, W = img_np.shape[:2]
        strength = [0, 0, 0, 0, 0.03, 0.07][level]
        jitter = lambda: int(random.uniform(-strength, strength) * min(H, W))
        src = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
        dst = np.float32([
            [jitter(), jitter()], [W + jitter(), jitter()],
            [jitter(), H + jitter()], [W + jitter(), H + jitter()]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img_np, M, (W, H), borderValue=255)

    def _add_noise(self, img_np: np.ndarray, level: int) -> np.ndarray:
        """高斯随机噪点"""
        if level < 2:
            return img_np
        sigma = [0, 0, 5, 15, 30, 50][level]
        noise = np.random.randn(*img_np.shape) * sigma
        return np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # ── 主接口 ────────────────────────────────────────────────

    def apply(
        self,
        image: Image.Image,
        level: Optional[int] = None,
        text_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Tuple[Image.Image, int]:
        """
        对输入 PIL 图像施加 OCR 噪声。

        Args:
            image: 输入图像
            level: 噪声级别 (0~5)，None 时随机采样
            text_boxes: 文字区域坐标列表 [(x0,y0,x1,y1), ...]
                        如果提供，只在文字区域施加噪声

        Returns:
            (corrupted_image, actual_level)
        """
        if random.random() > self.apply_prob:
            return image, 0

        if level is None:
            level = self._sample_level()

        if level == 0:
            return image, 0

        img_np = np.array(image.convert("RGB"))

        # 如果有文字框且决定只处理文字区域
        if text_boxes and random.random() < self.text_region_prob:
            for box in text_boxes:
                x0, y0, x1, y1 = [int(v) for v in box]
                roi = img_np[y0:y1, x0:x1]
                if roi.size == 0:
                    continue
                roi = self._apply_pipeline(roi, level)
                img_np[y0:y1, x0:x1] = roi
        else:
            img_np = self._apply_pipeline(img_np, level)

        return Image.fromarray(img_np), level

    def _apply_pipeline(self, img_np: np.ndarray, level: int) -> np.ndarray:
        """按级别组合噪声操作"""
        img_np = self._gaussian_blur(img_np, level)
        img_np = self._motion_blur(img_np, level)
        img_np = self._add_noise(img_np, level)
        img_np = self._random_occlusion(img_np, level)
        img_np = self._perspective_distort(img_np, level)
        return img_np

    def batch_apply(
        self,
        images: List[Image.Image],
        levels: Optional[List[int]] = None,
    ) -> Tuple[List[Image.Image], List[int]]:
        """批量处理"""
        results, actual_levels = [], []
        for i, img in enumerate(images):
            level = levels[i] if levels is not None else None
            corrupted, actual = self.apply(img, level)
            results.append(corrupted)
            actual_levels.append(actual)
        return results, actual_levels


def get_noise_system_prompt(level: int) -> str:
    """
    根据噪声级别生成对应的 system prompt。
    告知模型图像质量，引导其选择 OCR 或推理策略。
    """
    prompts = {
        0: "The document image is clear. Please accurately transcribe all text.",
        1: "The document image has slight blur. Transcribe text carefully.",
        2: "The document image has moderate degradation. Transcribe visible text and infer unclear parts.",
        3: "The document image is significantly degraded. Use context to infer partially obscured content.",
        4: "The document image is heavily corrupted. Rely on document structure and context for understanding.",
        5: "The document image is severely corrupted. Infer content based on document type and any visible cues.",
    }
    return prompts.get(level, prompts[0])
