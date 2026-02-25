"""
多模态数据集构建脚本 v2
生成两种格式：
  - video 类型：视频 + 字幕
  - interleaved 类型：图文混排

数据来源：
  - 视频：multimodalart/panda-70m (YouTube 片段 via yt-dlp)
  - 图文：HuggingFaceM4/OBELICS (流式)

依赖安装：
  pip install datasets requests huggingface_hub
  pip install yt-dlp
"""

import os
import json
import subprocess
import requests
from pathlib import Path
from datasets import load_dataset

# ============================================================
# 配置
# ============================================================
OUTPUT_DIR = Path("build_data/visual_chat")
VIDEO_DIR  = OUTPUT_DIR / "videos"
IMAGE_DIR  = OUTPUT_DIR / "images"
TARGET_VIDEO_COUNT       = 100
TARGET_INTERLEAVED_COUNT = 100
FINAL_JSONL  = OUTPUT_DIR / "dataset.jsonl"
PREVIEW_JSON = OUTPUT_DIR / "dataset_preview.json"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 工具函数
# ============================================================
def check_ytdlp():
    """检查 yt-dlp 是否已安装"""
    result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "yt-dlp 未安装，请先运行: pip install yt-dlp\n"
            "或: brew install yt-dlp (macOS)"
        )
    print(f"  yt-dlp 版本: {result.stdout.strip()}")


def download_youtube_clip(video_id: str, start: float, end: float, save_path: Path) -> bool:
    """
    用 yt-dlp 下载 YouTube 视频的指定片段
    start/end 单位为秒
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    duration = end - start

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "-o", str(save_path),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]
    try:
        result = subprocess.run(cmd, timeout=120, capture_output=True, text=True)
        # 文件存在且大于 1KB 即视为成功
        if save_path.exists() and save_path.stat().st_size > 1024:
            return True
        # 有些情况文件会被加上额外扩展名
        candidates = list(save_path.parent.glob(save_path.stem + "*"))
        if candidates:
            candidates[0].rename(save_path)
            return True
        return False
    except subprocess.TimeoutExpired:
        print(f"    [WARN] 下载超时: {video_id}")
        return False
    except Exception as e:
        print(f"    [WARN] 下载异常: {e}")
        return False


def parse_timestamp(ts) -> tuple[float, float]:
    """
    解析 Panda-70M 的 timestamp 字段
    格式可能为 "[start, end]" 字符串或列表
    """
    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
        return float(ts[0]), float(ts[1])
    if isinstance(ts, str):
        ts = ts.strip().strip("[]")
        parts = ts.split(",")
        if len(parts) >= 2:
            return float(parts[0].strip()), float(parts[1].strip())
    return 0.0, 30.0  # 默认取前30秒


def download_image(url: str, save_path: Path) -> bool:
    try:
        resp = requests.get(url, timeout=20, stream=True)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return save_path.stat().st_size > 0
    except Exception as e:
        print(f"    [WARN] 图片下载失败: {e}")
        return False


# ============================================================
# Part 1: 视频数据（video 类型）—— Panda-70M + yt-dlp
# ============================================================
def build_video_samples() -> list[dict]:
    print("\n========== [1/2] 构建 video 类型数据 (Panda-70M) ==========")
    check_ytdlp()

    samples = []
    # 用 validation split，只有 2000 条，不需要全量流式
    dataset = load_dataset(
        "multimodalart/panda-70m",
        split="validation",
        streaming=True,
    )

    idx = 0
    skipped = 0
    for row in dataset:
        if idx >= TARGET_VIDEO_COUNT:
            break
        if skipped > TARGET_VIDEO_COUNT * 3:  # 跳过太多就放弃
            print("  [WARN] 跳过条目过多，提前结束")
            break

        video_id  = row.get("video_id", "")
        caption   = row.get("caption", "")
        timestamp = row.get("timestamp", None)

        if not video_id or not caption:
            skipped += 1
            continue

        start, end = parse_timestamp(timestamp) if timestamp else (0.0, 30.0)
        # 最多下载60秒
        end = min(end, start + 60)

        filename  = f"{idx:05d}.mp4"
        save_path = VIDEO_DIR / filename

        print(f"  [{idx+1}/{TARGET_VIDEO_COUNT}] 下载视频片段: {video_id} [{start:.1f}s-{end:.1f}s] -> {filename}")
        success = download_youtube_clip(video_id, start, end, save_path)

        if not success:
            print(f"    跳过（下载失败）")
            skipped += 1
            continue

        samples.append({
            "type":    "video",
            "video":   str(save_path),
            "caption": caption.strip(),
        })
        idx += 1

    print(f"  ✓ 共完成 {len(samples)} 条 video 记录（跳过 {skipped} 条）")
    return samples


# ============================================================
# Part 2: 图文混排数据（interleaved 类型）—— OBELICS
# ============================================================
def build_interleaved_samples() -> list[dict]:
    print("\n========== [2/2] 构建 interleaved 类型数据 (OBELICS) ==========")

    samples = []
    dataset = load_dataset(
        "HuggingFaceM4/OBELICS",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    img_global_idx = 0
    sample_idx = 0

    for row in dataset:
        if sample_idx >= TARGET_INTERLEAVED_COUNT:
            break

        texts  = row.get("texts",  [])
        images = row.get("images", [])

        if not texts:
            continue

        items = []
        for i, text in enumerate(texts):
            if text and text.strip():
                items.append({"type": "text", "content": text.strip()})

            if i < len(images) and images[i]:
                img_url = images[i]
                ext = img_url.split(".")[-1].split("?")[0].lower()
                if ext not in ("jpg", "jpeg", "png", "webp"):
                    ext = "jpg"
                img_filename = f"{img_global_idx:05d}.{ext}"
                img_path = IMAGE_DIR / img_filename

                print(f"  [样本{sample_idx+1}] 下载图片: {img_filename}")
                if download_image(img_url, img_path):
                    items.append({"type": "image", "path": str(img_path)})
                    img_global_idx += 1

        has_text  = any(it["type"] == "text"  for it in items)
        has_image = any(it["type"] == "image" for it in items)
        if not (has_text and has_image):
            continue

        samples.append({"type": "interleaved", "items": items})
        sample_idx += 1

    print(f"  ✓ 共完成 {len(samples)} 条 interleaved 记录")
    return samples


# ============================================================
# Part 3: 保存
# ============================================================
def save_dataset(video_samples: list, interleaved_samples: list):
    all_samples = video_samples + interleaved_samples

    with open(FINAL_JSONL, "w", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(PREVIEW_JSON, "w", encoding="utf-8") as f:
        json.dump(all_samples[:10], f, ensure_ascii=False, indent=2)

    print(f"\n========== 完成 ==========")
    print(f"  总记录数    : {len(all_samples)}")
    print(f"  video       : {len(video_samples)}")
    print(f"  interleaved : {len(interleaved_samples)}")
    print(f"  JSONL 文件  : {FINAL_JSONL}")
    print(f"  预览文件    : {PREVIEW_JSON}")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    video_samples       = build_video_samples()
    interleaved_samples = build_interleaved_samples()
    save_dataset(video_samples, interleaved_samples)