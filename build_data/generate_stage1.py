"""
Stage 1 数据生成器
对标公开数据集：COCO Captions, CC3M, LAION (图文对) + LibriSpeech, WenetSpeech (语音文本)
生成 2000 条样本：图文对 1400 条 + 语音文本对 600 条

图文对内容来自 COCO 常见场景描述风格
语音文本来自 LibriSpeech 风格英文句子 + 中文普通话句子
"""

import json
import random
import math
import struct
import wave
import io
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = Path("raw/datasets/stage1")
IMG_DIR = OUTPUT_DIR / "images"
AUDIO_DIR = OUTPUT_DIR / "audio"
IMG_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# ====================================================
# 图文对数据（对标 COCO Captions 风格）
# ====================================================

SCENE_TEMPLATES = [
    # (背景色, 前景色, 场景描述模板, 图像绘制类型)
    ((135, 206, 235), (34, 139, 34), 
     "A {animal} is {action} in a {location} on a sunny day. The sky is clear and blue.",
     "outdoor_nature"),
    ((255, 255, 240), (139, 69, 19),
     "A {person} is {activity} at a {place}. There are {object} nearby.",
     "indoor_activity"),
    ((200, 220, 200), (70, 130, 180),
     "Several {animals} are {action} near a {water_feature}. The scene is {atmosphere}.",
     "wildlife"),
    ((240, 230, 210), (128, 0, 0),
     "A {food_item} is served on a {surface}. It looks {quality} and {adjective}.",
     "food"),
    ((180, 180, 200), (220, 20, 60),
     "A {vehicle} is parked on a {road_type}. The {weather} makes the scene look {atmosphere}.",
     "urban"),
    ((245, 245, 220), (75, 0, 130),
     "A {person} wearing {clothing} is standing {location}. They appear to be {emotion}.",
     "portrait"),
    ((200, 230, 255), (0, 100, 0),
     "Children are {activity} in a {location}. The {time_of_day} light casts {shadow} shadows.",
     "children"),
    ((255, 240, 230), (150, 75, 0),
     "The {building_type} building has {architecture_feature}. People are {activity} outside.",
     "architecture"),
]

FILL_WORDS = {
    "animal": ["dog", "cat", "bird", "horse", "elephant", "lion", "fox", "rabbit"],
    "animals": ["birds", "ducks", "deer", "horses", "sheep", "cattle"],
    "action": ["running", "resting", "playing", "grazing", "flying", "swimming"],
    "location": ["park", "meadow", "forest", "garden", "hillside", "beach"],
    "person": ["woman", "man", "child", "elderly man", "young girl", "boy"],
    "activity": ["reading", "cooking", "playing music", "painting", "exercising", "shopping"],
    "place": ["kitchen", "library", "café", "living room", "studio", "market"],
    "object": ["books", "flowers", "chairs", "paintings", "musical instruments"],
    "water_feature": ["lake", "river", "pond", "ocean", "waterfall", "stream"],
    "atmosphere": ["peaceful", "serene", "vibrant", "dramatic", "calm", "lively"],
    "food_item": ["salad", "soup", "sandwich", "pasta", "sushi", "cake", "pizza"],
    "surface": ["white plate", "wooden table", "marble counter", "dark tray"],
    "quality": ["fresh", "delicious", "colorful", "appetizing"],
    "adjective": ["well-presented", "mouth-watering", "elegantly plated"],
    "vehicle": ["red car", "blue bus", "yellow taxi", "white truck", "motorcycle"],
    "road_type": ["busy street", "quiet alley", "highway", "cobblestone road"],
    "weather": ["overcast sky", "bright sunshine", "foggy morning"],
    "clothing": ["a blue jacket", "a red dress", "casual clothes", "a business suit"],
    "emotion": ["happy", "contemplative", "excited", "calm", "focused"],
    "time_of_day": ["morning", "afternoon", "evening", "golden hour"],
    "shadow": ["long", "soft", "dramatic", "gentle"],
    "building_type": ["modern", "historic", "colonial", "glass-facade"],
    "architecture_feature": ["large windows", "ornate columns", "a grand entrance"],
    "children": ["children"],
}

def fill_template(template):
    result = template
    for key, values in FILL_WORDS.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result

def generate_scene_image(bg_color, fg_color, scene_type, size=(448, 448)):
    """生成对应场景的示意图像"""
    img = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(img)
    w, h = size

    if scene_type == "outdoor_nature":
        # 天空 + 草地 + 简单动物轮廓
        draw.rectangle([0, 0, w, h//2], fill=bg_color)
        draw.rectangle([0, h//2, w, h], fill=(34, 100, 34))
        # 太阳
        draw.ellipse([w-80, 20, w-20, 80], fill=(255, 215, 0))
        # 树
        draw.rectangle([w//4-10, h//3, w//4+10, h//2+20], fill=(101, 67, 33))
        draw.ellipse([w//4-40, h//5, w//4+40, h//3+20], fill=(0, 128, 0))
        # 动物（简单矩形代替）
        draw.ellipse([w//2-30, h//2-20, w//2+30, h//2+20], fill=fg_color)

    elif scene_type == "food":
        # 白色盘子 + 食物
        draw.ellipse([w//2-120, h//2-120, w//2+120, h//2+120], fill=(255, 255, 255), outline=(200,200,200), width=3)
        # 食物颜色块
        colors = [(255,100,100), (100,200,100), (255,200,50), (200,100,50)]
        for i, c in enumerate(colors):
            angle = i * 90
            cx = w//2 + int(60 * math.cos(math.radians(angle)))
            cy = h//2 + int(60 * math.sin(math.radians(angle)))
            draw.ellipse([cx-25, cy-20, cx+25, cy+20], fill=c)

    elif scene_type == "urban":
        # 街道场景
        draw.rectangle([0, h//2, w, h], fill=(150, 150, 150))  # 路面
        draw.rectangle([0, 0, w, h//2], fill=(180, 210, 240))  # 天空
        # 建筑
        for i in range(3):
            bx = i * (w//3)
            bh = random.randint(h//4, h//2)
            draw.rectangle([bx+5, h//2-bh, bx+w//3-5, h//2], fill=(fg_color[0]+i*20, fg_color[1], fg_color[2]))
            # 窗户
            for wy in range(bh//30):
                for wx in range(2):
                    draw.rectangle([bx+10+wx*30, h//2-bh+10+wy*25, bx+30+wx*30, h//2-bh+20+wy*25],
                                   fill=(255, 255, 150))
        # 车
        draw.rectangle([w//3, h*3//4-15, w//3+60, h*3//4+5], fill=(220, 50, 50))

    else:
        # 通用：渐变背景 + 几何形状
        for y in range(h):
            ratio = y / h
            r = int(bg_color[0] * (1-ratio) + fg_color[0] * ratio * 0.3)
            g = int(bg_color[1] * (1-ratio) + fg_color[1] * ratio * 0.3)
            b = int(bg_color[2] * (1-ratio) + fg_color[2] * ratio * 0.3)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        # 主体对象
        draw.ellipse([w//2-80, h//2-80, w//2+80, h//2+80], fill=fg_color, outline=(255,255,255), width=3)
        draw.rectangle([w//4, h*2//3, w*3//4, h*5//6], fill=fg_color)

    # 添加轻微随机纹理（更真实）
    pixels = np.array(img)
    noise = np.random.randint(-8, 8, pixels.shape, dtype=np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)

def generate_wav_file(text, output_path, duration_sec=2.5):
    """
    生成合成的 WAV 音频文件（模拟语音，用正弦波叠加）。
    真实训练中替换为 TTS 合成音频。
    """
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    
    # 用文字长度决定基频（模拟音调变化）
    base_freq = 150 + (len(text) % 20) * 5  # 150~250 Hz
    
    t = np.linspace(0, duration_sec, num_samples)
    # 叠加多个谐波（更接近语音频谱）
    wave_data = (
        0.4 * np.sin(2 * np.pi * base_freq * t) +
        0.2 * np.sin(2 * np.pi * base_freq * 2 * t) +
        0.1 * np.sin(2 * np.pi * base_freq * 3 * t) +
        0.05 * np.random.randn(num_samples)  # 噪声
    )
    # 振幅包络（语音有起伏）
    envelope = np.ones(num_samples)
    attack = int(0.05 * sample_rate)
    release = int(0.1 * sample_rate)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    wave_data *= envelope * 0.5
    
    # 转为 16-bit PCM
    wave_int = (wave_data * 32767).astype(np.int16)
    
    with wave.open(str(output_path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wave_int.tobytes())

# ====================================================
# 语音文本对（对标 LibriSpeech + WenetSpeech 风格）
# ====================================================

ENGLISH_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverside.",
    "She opened the window and breathed in the fresh morning air.",
    "Scientists discovered a new species of deep-sea fish last year.",
    "The children gathered around the teacher to listen to the story.",
    "He carefully placed the fragile vase on the marble shelf.",
    "Technology has transformed the way people communicate worldwide.",
    "The ancient temple stood at the top of the misty mountain.",
    "Fresh vegetables from the garden make the best salads.",
    "The musician played a beautiful melody on the grand piano.",
    "Astronauts aboard the space station observe Earth from above.",
    "The library contained thousands of books on every subject.",
    "Rain fell gently on the colorful autumn leaves below.",
    "She painted landscapes of the countryside she loved so much.",
    "The chef prepared an elaborate meal for the special occasion.",
    "Children playing in the park filled the air with laughter.",
    "The old lighthouse guided ships safely through the stormy waters.",
    "He worked late into the night to finish the important project.",
    "The market was full of fresh fruits, vegetables, and flowers.",
    "Scientists are studying the effects of climate on animal migration.",
    "The documentary explored the hidden depths of the ocean floor.",
    "Volunteers cleaned up the beach on a warm Saturday morning.",
    "The golden sunset painted the sky in shades of orange and pink.",
    "A gentle breeze rustled through the tall trees in the forest.",
    "The professor explained complex theories in simple, clear language.",
    "Hikers followed the winding trail to reach the mountain summit.",
]

CHINESE_SENTENCES = [
    "今天天气晴朗，阳光明媚，非常适合户外活动。",
    "科学家们发现了一种新型材料，具有超强的导电性能。",
    "孩子们在公园里快乐地玩耍，笑声传遍整个广场。",
    "这座古城已有两千年的历史，保存着丰富的文化遗产。",
    "她仔细地阅读了每一份文件，确保没有遗漏任何细节。",
    "随着技术的进步，人们的生活方式发生了巨大的变化。",
    "厨师用新鲜的食材烹饪出了一桌美味可口的菜肴。",
    "登山队员克服重重困难，终于登上了山顶。",
    "图书馆里静悄悄的，同学们都在认真地学习。",
    "雨后的空气格外清新，远处的山峦也变得更加清晰。",
    "研究人员花了三年时间，终于完成了这项重要的实验。",
    "志愿者们利用周末时间，帮助社区清洁环境。",
    "这首古典音乐旋律优美，令人心旷神怡。",
    "市场上摆满了各种新鲜的蔬菜水果，色彩十分丰富。",
    "他每天坚持锻炼身体，保持健康的生活习惯。",
    "这幅画作描绘了江南水乡的秀美风光，令人陶醉。",
    "工程师们设计了一种新型桥梁，可以抵御强烈地震。",
    "小镇上的人们过着平静而幸福的生活。",
    "这家餐厅以其独特的风味和优质的服务闻名全城。",
    "夕阳西下，金色的余晖洒在宁静的湖面上。",
    "博物馆里陈列着许多珍贵的历史文物和艺术品。",
    "医生耐心地向患者解释治疗方案和注意事项。",
    "新学期开始了，同学们带着新的期待走进校园。",
    "这部纪录片生动地展现了大自然的神奇与美丽。",
    "他努力学习新技术，不断提升自己的专业能力。",
]


def generate_stage1_data():
    print("=" * 60)
    print("生成 Stage 1 数据（模态对齐预训练）")
    print("对标数据集: COCO Captions + CC3M (图文) | LibriSpeech + WenetSpeech (语音)")
    print("=" * 60)

    all_samples = []

    # ── 图文对 (1400 条) ──────────────────────────────────────
    print("\n[1/2] 生成图文对 1400 条...")
    for idx in range(1400):
        scene = SCENE_TEMPLATES[idx % len(SCENE_TEMPLATES)]
        bg_color, fg_color, template, scene_type = scene
        caption = fill_template(template)

        # 生成图像
        img = generate_scene_image(bg_color, fg_color, scene_type)
        img_path = IMG_DIR / f"img_{idx:04d}.jpg"
        img.save(img_path, "JPEG", quality=90)

        sample = {
            "id": f"stage1_img_{idx:04d}",
            "type": "image_text",
            "source": "synthetic_coco_style",  # 对标 COCO Captions
            "image": str(img_path),
            "text": caption,
            "meta": {
                "scene_type": scene_type,
                "image_size": "448x448",
                "language": "en"
            }
        }
        all_samples.append(sample)

        if (idx + 1) % 200 == 0:
            print(f"  图文对: {idx+1}/1400 完成")

    # ── 语音文本对 (600 条) ──────────────────────────────────
    print("\n[2/2] 生成语音文本对 600 条 (英文300 + 中文300)...")

    # 英文语音 (300 条, 对标 LibriSpeech)
    for idx in range(300):
        sentence = ENGLISH_SENTENCES[idx % len(ENGLISH_SENTENCES)]
        # 增加变体
        if idx >= len(ENGLISH_SENTENCES):
            prefixes = ["Yesterday, ", "In the morning, ", "Surprisingly, ", "According to reports, "]
            sentence = random.choice(prefixes) + sentence.lower()

        audio_path = AUDIO_DIR / f"en_{idx:04d}.wav"
        duration = 1.5 + len(sentence) * 0.05
        generate_wav_file(sentence, audio_path, duration_sec=min(duration, 8.0))

        sample = {
            "id": f"stage1_audio_en_{idx:04d}",
            "type": "audio_text",
            "source": "synthetic_librispeech_style",  # 对标 LibriSpeech
            "audio": str(audio_path),
            "transcript": sentence,
            "meta": {
                "language": "en",
                "duration_sec": round(min(duration, 8.0), 2),
                "speaker_id": f"spk_{idx % 50:03d}",
                "sample_rate": 16000
            }
        }
        all_samples.append(sample)

    # 中文语音 (300 条, 对标 WenetSpeech/AISHELL)
    for idx in range(300):
        sentence = CHINESE_SENTENCES[idx % len(CHINESE_SENTENCES)]
        if idx >= len(CHINESE_SENTENCES):
            sentence = random.choice(["据悉，", "研究显示，", "专家表示，"]) + sentence

        audio_path = AUDIO_DIR / f"zh_{idx:04d}.wav"
        duration = 1.5 + len(sentence) * 0.08
        generate_wav_file(sentence, audio_path, duration_sec=min(duration, 8.0))

        sample = {
            "id": f"stage1_audio_zh_{idx:04d}",
            "type": "audio_text",
            "source": "synthetic_wenetspeech_style",  # 对标 WenetSpeech
            "audio": str(audio_path),
            "transcript": sentence,
            "meta": {
                "language": "zh",
                "duration_sec": round(min(duration, 8.0), 2),
                "speaker_id": f"spk_zh_{idx % 30:03d}",
                "sample_rate": 16000
            }
        }
        all_samples.append(sample)

    print(f"  语音文本对: 600/600 完成")

    # ── 保存 JSONL ───────────────────────────────────────────
    random.shuffle(all_samples)
    output_path = OUTPUT_DIR / "stage1_train.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nStage 1 数据生成完成")
    print(f"   总样本: {len(all_samples)} 条")
    print(f"   图文对: 1400 条 (70%)")
    print(f"   语音文本对: 600 条 (30%: 英文300 + 中文300)")
    print(f"   保存至: {output_path}")

    # 保存数据集统计信息
    stats = {
        "dataset_name": "MiniCPM-o Stage1 Alignment Dataset",
        "reference_datasets": ["COCO Captions", "CC3M", "LibriSpeech", "WenetSpeech", "AISHELL-3"],
        "total_samples": len(all_samples),
        "split": {"image_text": 1400, "audio_en": 300, "audio_zh": 300},
        "image_size": "448x448",
        "audio_sample_rate": 16000,
        "note": "合成数据，用于验证训练流程。生产环境请替换为真实公开数据集。"
    }
    with open(OUTPUT_DIR / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return all_samples


if __name__ == "__main__":
    generate_stage1_data()
