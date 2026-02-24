"""
Stage 2 数据生成器 - 统一多模态预训练
对标数据集：
  - OCR文档: OmniDocBench, ArXiv截图, 合成文档
  - 视频理解: WebVid-10M, ActivityNet, Ego4D (视频帧)
  - 交错图文: MINT-1T, OmniCorpus
  - 语音对话: VGGSound, AudioSet

生成 2000 条样本：
  OCR文档 500条 (25%) | 交错图文 700条 (35%) | 视频帧 400条 (20%) | 语音对话 400条 (20%)
"""

import json
import random
import math
import wave
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

random.seed(123)
np.random.seed(123)

OUTPUT_DIR = Path("raw/datasets/stage2")
IMG_DIR = OUTPUT_DIR / "images"
DOC_DIR = OUTPUT_DIR / "images" / "documents"
VID_DIR = OUTPUT_DIR / "images" / "video_frames"
AUDIO_DIR = OUTPUT_DIR / "audio"

for d in [IMG_DIR, DOC_DIR, VID_DIR, AUDIO_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# OCR 文档数据（对标 OmniDocBench / ArXiv 样式）
# ============================================================

DOCUMENT_TEMPLATES = [
    {
        "doc_type": "academic_paper",
        "title": "Deep Learning Approaches for Multimodal Understanding",
        "sections": [
            ("Abstract", "This paper presents a comprehensive study of multimodal learning frameworks. We propose a novel architecture that integrates visual and linguistic representations. Experimental results demonstrate significant improvements over baseline methods on standard benchmarks."),
            ("1. Introduction", "The field of artificial intelligence has witnessed remarkable advances in recent years. Multimodal learning, which combines information from multiple modalities such as images, text, and audio, has emerged as a promising research direction."),
            ("2. Methodology", "Our approach builds upon the transformer architecture. The model consists of three components: (1) a visual encoder, (2) a language model, and (3) a cross-modal fusion module. We train the system end-to-end using a combination of contrastive and generative objectives."),
            ("3. Results", "We evaluate our method on five standard benchmarks. Table 1 shows the quantitative results. Our model achieves state-of-the-art performance on four out of five benchmarks, with an average improvement of 3.2% over the previous best method."),
        ]
    },
    {
        "doc_type": "financial_report",
        "title": "Q3 2024 Financial Performance Summary",
        "sections": [
            ("Executive Summary", "Total revenue for Q3 2024 reached $2.4 billion, representing a 12.3% year-over-year increase. Operating income improved to $480 million. Net profit margin stood at 18.2%, up from 15.7% in Q3 2023."),
            ("Revenue Breakdown", "Product Sales: $1.6B (+8.5% YoY)\nService Revenue: $0.6B (+22.1% YoY)\nLicensing Fees: $0.2B (+5.3% YoY)\nTotal: $2.4B"),
            ("Key Metrics", "EBITDA: $620M | EPS: $3.42 | P/E Ratio: 28.5x | Debt-to-Equity: 0.45 | Return on Equity: 22.3% | Current Ratio: 1.87"),
            ("Outlook", "For Q4 2024, we project revenue between $2.6B and $2.8B. We expect continued growth driven by expansion in emerging markets and new product launches scheduled for November 2024."),
        ]
    },
    {
        "doc_type": "medical_record",
        "title": "Patient Assessment Report",
        "sections": [
            ("Patient Information", "Age: 45 | Gender: Male | Weight: 78kg | Height: 175cm | BMI: 25.5 | Blood Type: A+"),
            ("Chief Complaint", "Patient presents with persistent headaches for 3 weeks, rated 6/10 severity. Headaches are predominantly frontal, worse in morning. No fever, nausea, or visual disturbances reported."),
            ("Laboratory Results", "CBC: WBC 7.2, RBC 4.8, Hgb 14.2, Hct 43%, Plt 215K\nMetabolic Panel: Na 138, K 4.1, Cl 102, CO2 24, BUN 18, Creat 0.9\nLipid Panel: Total Chol 195, LDL 120, HDL 48, TG 135"),
            ("Treatment Plan", "1. Start Ibuprofen 400mg TID for pain management\n2. Referral to neurology for MRI brain\n3. Lifestyle modifications: stress reduction, adequate sleep\n4. Follow-up in 2 weeks"),
        ]
    },
    {
        "doc_type": "invoice",
        "title": "INVOICE #INV-2024-08-1547",
        "sections": [
            ("Billing Information", "From: TechSolutions Inc.\n123 Innovation Drive, San Francisco, CA 94105\nTo: Global Enterprises Ltd.\n456 Business Ave, New York, NY 10001"),
            ("Item Details", "1. Software License (Annual) - $2,400.00\n2. Implementation Services (40hrs @ $150/hr) - $6,000.00\n3. Training Sessions (2 days) - $1,500.00\n4. Support Package (12 months) - $1,200.00"),
            ("Payment Summary", "Subtotal: $11,100.00\nTax (8.5%): $943.50\nDiscount (5%): -$555.00\nTotal Due: $11,488.50\nPayment Due: September 15, 2024"),
        ]
    },
    {
        "doc_type": "recipe_book",
        "title": "Classic Italian Recipes Collection",
        "sections": [
            ("Spaghetti Carbonara", "Ingredients:\n- 400g spaghetti\n- 200g pancetta\n- 4 eggs\n- 100g Parmesan\n- Salt, pepper, garlic"),
            ("Instructions", "1. Cook pasta in salted water until al dente.\n2. Fry pancetta until crispy.\n3. Mix eggs with Parmesan and pepper.\n4. Combine hot pasta with pancetta.\n5. Remove from heat, add egg mixture. Stir quickly."),
            ("Chef's Tips", "Never add cream to authentic carbonara. The key is to add the egg mixture off heat to prevent scrambling. Use high-quality Pecorino Romano for authentic flavor."),
        ]
    }
]

# 中文文档模板
CHINESE_DOC_TEMPLATES = [
    {
        "doc_type": "research_report",
        "title": "人工智能在医疗领域的应用研究报告",
        "sections": [
            ("执行摘要", "本报告对人工智能技术在医疗领域的最新应用进行了系统性综述。研究涵盖医学影像诊断、药物研发、病历分析等核心领域，分析了技术成熟度、市场规模及发展趋势。"),
            ("市场规模", "2023年全球AI医疗市场规模达到156亿美元，同比增长35.8%。中国市场规模为28亿美元，预计2025年将突破100亿美元。医学影像AI占比最高，达到42%。"),
            ("技术分析", "深度学习在医学影像诊断中的准确率已超过专科医生平均水平。自然语言处理技术能够快速分析海量医学文献，辅助药物研发。机器人辅助手术精度提升约30%。"),
            ("政策环境", "国家出台了《医疗器械监督管理条例》等相关法规，为AI医疗产品的注册审批提供了清晰路径。多个省市推出专项扶持政策，加速产业落地应用。"),
        ]
    },
    {
        "doc_type": "news_article",
        "title": "量子计算突破：实现千量子比特稳定运行",
        "sections": [
            ("新闻导语", "据最新报道，某科技公司的研究团队成功实现了1024个量子比特的稳定运行，相干时间突破100微秒，创下新的世界纪录，标志着量子计算进入新的发展阶段。"),
            ("技术突破", "研究人员采用了新型超导量子比特设计，通过优化量子误差纠正算法，将量子比特的出错率降低至0.1%以下。这一成果为实现量子优势铺平了道路。"),
            ("专家评论", "中科院量子信息重点实验室主任表示：这一突破具有重要的里程碑意义，为未来量子计算机的实用化奠定了坚实基础。实用级量子计算机或将在10年内问世。"),
        ]
    }
]

def generate_document_image(doc_data, size=(600, 800), noise_level=0):
    """生成文档样式图像（对标 OmniDocBench）"""
    img = Image.new("RGB", size, color=(252, 252, 252))
    draw = ImageDraw.Draw(img)
    w, h = size

    # 页面边框
    draw.rectangle([20, 20, w-20, h-20], outline=(200, 200, 200), width=1)

    # 标题区域
    draw.rectangle([20, 20, w-20, 80], fill=(240, 240, 250))
    title_text = doc_data["title"]
    # 用文字占位线表示（PIL 默认字体）
    draw.text((30, 35), title_text[:50], fill=(30, 30, 120))
    draw.text((30, 55), doc_data["doc_type"].replace("_", " ").upper(), fill=(100, 100, 150))

    # 各段落
    y_pos = 90
    for section_title, content in doc_data["sections"][:3]:
        if y_pos > h - 100:
            break
        # 段落标题
        draw.rectangle([25, y_pos, w-25, y_pos+20], fill=(230, 230, 245))
        draw.text((28, y_pos+3), section_title, fill=(50, 50, 150))
        y_pos += 28

        # 内容行（模拟文字行）
        lines = content[:200].split("\n")
        for line in lines[:4]:
            if y_pos > h - 50:
                break
            if line.strip():
                # 绘制文字行（用灰色矩形模拟文字密度）
                text_width = min(len(line) * 5, w - 60)
                draw.rectangle([28, y_pos+2, 28+text_width, y_pos+10], fill=(80, 80, 80))
                draw.text((28, y_pos), line[:80], fill=(60, 60, 60))
            y_pos += 18

        y_pos += 10

    # 页脚
    draw.line([(25, h-40), (w-25, h-40)], fill=(180, 180, 180), width=1)
    draw.text((30, h-30), "Page 1 of 1", fill=(150, 150, 150))

    # 施加噪声（OCR噪声掩码）
    if noise_level > 0:
        pixels = np.array(img)
        if noise_level >= 2:
            # 高斯模糊效果（用随机噪声模拟）
            sigma = noise_level * 3
            noise = np.random.normal(0, sigma, pixels.shape)
            pixels = np.clip(pixels.astype(float) + noise, 0, 255).astype(np.uint8)
        if noise_level >= 3:
            # 随机遮挡块
            num_blocks = noise_level * 2
            for _ in range(num_blocks):
                bh = random.randint(5, 20)
                bw = random.randint(20, 80)
                bx = random.randint(0, w-bw)
                by = random.randint(80, h-bh)
                pixels[by:by+bh, bx:bx+bw] = 240
        img = Image.fromarray(pixels)

    return img

def generate_video_frame_sequence(scene_name, num_frames=6, size=(448, 448)):
    """生成视频帧序列（模拟连续视频帧，对标 WebVid-10M）"""
    frames = []
    # 场景参数
    scenes = {
        "cooking": {"bg": (250, 240, 220), "fg": (200, 100, 50), "objects": "kitchen"},
        "sports": {"bg": (100, 180, 100), "fg": (255, 100, 0), "objects": "field"},
        "traffic": {"bg": (150, 150, 170), "fg": (200, 50, 50), "objects": "road"},
        "nature": {"bg": (100, 180, 100), "fg": (50, 150, 200), "objects": "outdoor"},
        "classroom": {"bg": (240, 240, 220), "fg": (50, 50, 150), "objects": "indoor"},
    }
    scene = scenes.get(scene_name, scenes["nature"])

    for frame_idx in range(num_frames):
        # 每帧略有变化（模拟运动）
        t = frame_idx / num_frames
        img = Image.new("RGB", size, color=scene["bg"])
        draw = ImageDraw.Draw(img)
        w, h = size

        # 移动的主体（模拟连续动作）
        x_offset = int(50 * math.sin(t * math.pi))
        y_offset = int(20 * math.cos(t * math.pi * 2))

        # 背景元素
        draw.rectangle([0, h//2, w, h], fill=(max(0, scene["bg"][0]-30),
                                               max(0, scene["bg"][1]-30),
                                               max(0, scene["bg"][2]-30)))

        # 主体
        cx, cy = w//2 + x_offset, h//2 + y_offset
        draw.ellipse([cx-50, cy-40, cx+50, cy+40], fill=scene["fg"])

        # 时间戳叠加（真实视频常见）
        timestamp = f"{frame_idx//25:02d}:{frame_idx%25:02d}"
        draw.text((10, 10), timestamp, fill=(255, 255, 255))
        draw.text((10, 30), scene_name, fill=(255, 255, 0))

        # 添加轻微噪声
        pixels = np.array(img)
        noise = np.random.randint(-5, 5, pixels.shape, dtype=np.int16)
        pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(pixels))

    return frames

def generate_wav_audio(duration_sec, base_freq=180):
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples)
    wave_data = (
        0.3 * np.sin(2 * np.pi * base_freq * t) +
        0.15 * np.sin(2 * np.pi * base_freq * 2 * t) +
        0.05 * np.random.randn(num_samples)
    )
    envelope = np.ones(num_samples)
    attack = min(int(0.05 * sample_rate), num_samples // 4)
    release = min(int(0.1 * sample_rate), num_samples // 4)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    wave_int = (wave_data * envelope * 0.5 * 32767).astype(np.int16)
    return wave_int, sample_rate

def save_wav(wave_int, sample_rate, path):
    with wave.open(str(path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wave_int.tobytes())


# ============================================================
# 交错图文数据（对标 MINT-1T / OmniCorpus）
# ============================================================

INTERLEAVED_TEMPLATES = [
    {
        "topic": "climate_change",
        "items": [
            {"type": "text", "content": "Global temperatures have risen significantly over the past century, largely due to human activities. Scientists have documented this warming trend through various measurement methods."},
            {"type": "image_desc", "content": "A graph showing global temperature anomalies from 1880 to 2023, with a clear upward trend especially after 1980."},
            {"type": "text", "content": "The consequences of climate change are wide-ranging, affecting ecosystems, weather patterns, and sea levels. Polar ice caps are melting at an unprecedented rate, threatening coastal communities worldwide."},
            {"type": "image_desc", "content": "Satellite images comparing Arctic ice coverage between 1979 and 2023, showing dramatic reduction in ice area."},
            {"type": "text", "content": "International efforts to address climate change include the Paris Agreement, which aims to limit global warming to 1.5°C above pre-industrial levels."},
        ]
    },
    {
        "topic": "space_exploration",
        "items": [
            {"type": "text", "content": "Human space exploration has reached new milestones in the 21st century. The Artemis program aims to return humans to the Moon by 2026."},
            {"type": "image_desc", "content": "A rocket launch at night, with the exhaust plume illuminating the surrounding area in brilliant orange and white light."},
            {"type": "text", "content": "Mars exploration has been led by robotic missions such as Perseverance rover, which landed in Jezero Crater in February 2021 and has been collecting rock samples ever since."},
            {"type": "image_desc", "content": "The Martian surface captured by the Perseverance rover, showing reddish-brown rocky terrain with a thin dusty atmosphere on the horizon."},
        ]
    },
    {
        "topic": "urban_architecture",
        "items": [
            {"type": "text", "content": "Modern urban architecture balances aesthetic appeal with functional requirements. Sustainable design principles have become increasingly important in contemporary construction."},
            {"type": "image_desc", "content": "A glass-and-steel skyscraper with a distinctive curved facade, surrounded by smaller historic buildings in a city center."},
            {"type": "text", "content": "Green roofs and vertical gardens are becoming popular features in urban buildings, helping to reduce heat islands and improve air quality in densely populated areas."},
        ]
    },
    {
        "topic": "traditional_cuisine",
        "items": [
            {"type": "text", "content": "中华饮食文化有着数千年的历史传承，八大菜系各具特色，形成了丰富多样的烹饪艺术。"},
            {"type": "image_desc", "content": "一桌丰盛的中式宴席，包括红烧肉、清蒸鱼、炒蔬菜、汤羹等多道菜肴，色彩鲜艳，摆盘精美。"},
            {"type": "text", "content": "四川菜以麻辣著称，广东菜注重原汁原味，苏州菜讲究精致清淡。不同地区的饮食习惯与当地气候、地理环境密切相关。"},
        ]
    },
    {
        "topic": "ocean_biology",
        "items": [
            {"type": "text", "content": "The deep ocean remains one of Earth's least explored frontiers. Scientists estimate that over 80% of the world's oceans remain unmapped and unexplored."},
            {"type": "image_desc", "content": "A deep-sea creature photographed at 3,000 meters depth, featuring bioluminescent patterns glowing in the absolute darkness of the abyss."},
            {"type": "text", "content": "Coral reefs support approximately 25% of all marine species despite covering less than 1% of the ocean floor. These ecosystems are under severe threat from rising ocean temperatures."},
            {"type": "image_desc", "content": "A vibrant coral reef teeming with colorful fish of various species, with healthy coral formations in shades of orange, purple, and green."},
        ]
    },
]

def generate_interleaved_image(desc_text, idx, size=(448, 448)):
    """根据描述生成对应图像"""
    img = Image.new("RGB", size, color=(240, 248, 255))
    draw = ImageDraw.Draw(img)
    w, h = size

    # 根据描述关键词选择风格
    keywords = desc_text.lower()
    if "graph" in keywords or "chart" in keywords or "temperature" in keywords:
        # 图表样式
        draw.rectangle([30, 30, w-30, h-30], fill=(255, 255, 255), outline=(100,100,100), width=2)
        # X/Y 轴
        draw.line([(50, h-60), (w-40, h-60)], fill=(0,0,0), width=2)
        draw.line([(50, 50), (50, h-60)], fill=(0,0,0), width=2)
        # 数据折线（模拟上升趋势）
        points = [(50 + i*(w-90)//20, h-60 - int(i**1.5 * (h-120)/20**1.5)) for i in range(21)]
        for i in range(len(points)-1):
            draw.line([points[i], points[i+1]], fill=(200,50,50), width=2)
        draw.text((w//2-40, h-25), "Year →", fill=(0,0,0))
    elif "satellite" in keywords or "aerial" in keywords or "map" in keywords:
        # 俯视/卫星图样式
        colors = [(50,120,50), (100,160,100), (200,200,150), (100,100,200), (180,200,220)]
        for y in range(0, h, h//8):
            for x in range(0, w, w//8):
                c = random.choice(colors)
                draw.rectangle([x, y, x+w//8, y+h//8], fill=c)
                # 添加网格线
                draw.rectangle([x, y, x+w//8, y+h//8], outline=(150,150,150), width=1)
    elif "rocket" in keywords or "launch" in keywords or "space" in keywords:
        # 夜晚发射场景
        img = Image.new("RGB", size, color=(10, 10, 30))
        draw = ImageDraw.Draw(img)
        # 火箭
        draw.rectangle([w//2-15, h//4, w//2+15, h*3//4], fill=(200,200,210))
        draw.polygon([(w//2-15, h//4), (w//2+15, h//4), (w//2, h//10)], fill=(180,50,50))
        # 排气
        for i in range(5):
            x_off = random.randint(-20, 20)
            draw.ellipse([w//2-20+x_off, h*3//4, w//2+20+x_off, h*3//4+random.randint(30,60)],
                         fill=(255, 150, 50))
        # 星空
        for _ in range(50):
            sx, sy = random.randint(0, w), random.randint(0, h//2)
            draw.point((sx, sy), fill=(255,255,200))
    elif "coral" in keywords or "reef" in keywords or "ocean" in keywords:
        # 海洋场景
        img = Image.new("RGB", size, color=(30, 80, 150))
        draw = ImageDraw.Draw(img)
        # 珊瑚
        coral_colors = [(255,100,50), (200,50,150), (50,200,100), (255,200,50)]
        for _ in range(8):
            cx = random.randint(30, w-30)
            cy = random.randint(h//2, h-30)
            cc = random.choice(coral_colors)
            draw.ellipse([cx-20, cy-30, cx+20, cy+30], fill=cc)
            for _ in range(3):
                draw.line([(cx, cy), (cx+random.randint(-30,30), cy+random.randint(-50,0))],
                         fill=cc, width=3)
        # 鱼
        for _ in range(6):
            fx, fy = random.randint(0, w), random.randint(h//4, h//2)
            fc = random.choice(coral_colors)
            draw.ellipse([fx-15, fy-8, fx+15, fy+8], fill=fc)
    else:
        # 通用场景图
        bg_colors = [(200, 230, 200), (220, 200, 230), (200, 220, 240)]
        bg = random.choice(bg_colors)
        img = Image.new("RGB", size, bg)
        draw = ImageDraw.Draw(img)
        # 主体
        main_color = (random.randint(50,200), random.randint(50,200), random.randint(50,200))
        draw.ellipse([w//2-80, h//2-80, w//2+80, h//2+80], fill=main_color)
        # 背景元素
        for _ in range(5):
            ex, ey = random.randint(0,w), random.randint(0,h)
            ec = (random.randint(100,255), random.randint(100,255), random.randint(100,255))
            draw.ellipse([ex-20, ey-20, ex+20, ey+20], fill=ec, outline=(255,255,255), width=1)

    # 添加图说文字
    if draw and len(desc_text) > 20:
        caption = desc_text[:60] + "..."
        draw.rectangle([0, h-25, w, h], fill=(0,0,0,128))
        draw.text((5, h-22), caption, fill=(255,255,255))

    return img


def generate_stage2_data():
    print("=" * 60)
    print("生成 Stage 2 数据（统一多模态预训练）")
    print("对标数据集: OmniDocBench + ArXiv + WebVid-10M + MINT-1T")
    print("=" * 60)

    all_samples = []

    # ── OCR 文档 500 条 (25%) ──────────────────────────────────
    print("\n[1/4] 生成 OCR 文档数据 500 条...")

    all_docs = DOCUMENT_TEMPLATES * 50 + CHINESE_DOC_TEMPLATES * 75
    random.shuffle(all_docs)

    for idx in range(500):
        doc_data = all_docs[idx % len(all_docs)]
        noise_level = idx % 6  # 0~5 循环，均匀覆盖各噪声级别

        img = generate_document_image(doc_data, noise_level=noise_level)
        img_path = DOC_DIR / f"doc_{idx:04d}_noise{noise_level}.jpg"
        img.save(img_path, "JPEG", quality=85)

        # OCR 文本（完整文档文字）
        full_text = doc_data["title"] + "\n\n"
        for sec_title, content in doc_data["sections"]:
            full_text += f"[{sec_title}]\n{content}\n\n"

        # 文字区域坐标（简化版）
        text_boxes = [
            [25, 20, 580, 80],   # 标题区域
            [25, 90, 580, 200],  # 第一段
            [25, 210, 580, 320], # 第二段
        ]

        # 根据噪声级别决定任务类型
        if noise_level <= 2:
            question = "Please transcribe all text visible in this document."
            answer = full_text.strip()
        elif noise_level <= 4:
            question = "Describe the main content and key information in this document."
            answer = f"This is a {doc_data['doc_type'].replace('_', ' ')}. Title: {doc_data['title']}. " + \
                     f"Main content covers: {doc_data['sections'][0][1][:100]}..."
        else:
            question = "What type of document is this and what is its main purpose?"
            answer = f"This appears to be a {doc_data['doc_type'].replace('_', ' ')} document. " + \
                     f"Based on the visible structure, it contains {len(doc_data['sections'])} sections."

        sample = {
            "id": f"stage2_ocr_{idx:04d}",
            "type": "ocr_document",
            "source": "synthetic_omnidocbench_style",
            "image": str(img_path),
            "ocr_text": full_text.strip(),
            "text_boxes": text_boxes,
            "noise_level": noise_level,
            "conversations": [
                {"role": "user", "content": [
                    {"type": "image", "path": str(img_path)},
                    {"type": "text", "content": question}
                ]},
                {"role": "assistant", "content": answer}
            ],
            "meta": {
                "doc_type": doc_data["doc_type"],
                "noise_level": noise_level,
                "noise_description": ["clear", "slight blur", "moderate blur",
                                      "heavy blur", "severe corruption", "unreadable"][noise_level]
            }
        }
        all_samples.append(sample)

        if (idx + 1) % 100 == 0:
            print(f"  OCR文档: {idx+1}/500 完成")

    # ── 交错图文 700 条 (35%) ─────────────────────────────────
    print("\n[2/4] 生成交错图文数据 700 条...")

    img_counter = 0
    for idx in range(700):
        template = INTERLEAVED_TEMPLATES[idx % len(INTERLEAVED_TEMPLATES)]

        items_data = []
        for item in template["items"]:
            if item["type"] == "text":
                items_data.append({"type": "text", "content": item["content"]})
            elif item["type"] == "image_desc":
                # 生成对应图像
                img = generate_interleaved_image(item["content"], img_counter)
                img_path = IMG_DIR / f"interleaved_{img_counter:04d}.jpg"
                img.save(img_path, "JPEG", quality=88)
                items_data.append({"type": "image", "path": str(img_path),
                                   "description": item["content"]})
                img_counter += 1

        # 构建 caption（所有文本合并）
        caption = " ".join(item.get("content", "") for item in items_data if item["type"] == "text")

        sample = {
            "id": f"stage2_interleaved_{idx:04d}",
            "type": "interleaved",
            "source": "synthetic_mint1t_style",
            "topic": template["topic"],
            "items": items_data,
            "full_caption": caption,
            "meta": {"num_images": sum(1 for i in items_data if i["type"] == "image"),
                     "num_text_segments": sum(1 for i in items_data if i["type"] == "text")}
        }
        all_samples.append(sample)

        if (idx + 1) % 100 == 0:
            print(f"  交错图文: {idx+1}/700 完成")

    # ── 视频帧序列 400 条 (20%) ───────────────────────────────
    print("\n[3/4] 生成视频帧数据 400 条...")

    VIDEO_SCENES = {
        "cooking": {
            "captions": [
                "A chef is carefully chopping vegetables on a wooden cutting board in a professional kitchen.",
                "The cook stirs a pot of soup, adding spices and herbs to enhance the flavor.",
                "A person is kneading dough on a floured surface, preparing bread for baking.",
            ],
            "qa": [
                ("What activity is being shown?", "The video shows someone cooking or food preparation in a kitchen setting."),
                ("Describe the action in this video clip.", "A person is engaged in cooking activities, handling food ingredients in a kitchen environment."),
            ]
        },
        "sports": {
            "captions": [
                "Athletes are competing in a track and field event at a large stadium under bright lights.",
                "A soccer player dribbles the ball across the field toward the goal.",
                "The tennis player serves the ball with a powerful overhead stroke.",
            ],
            "qa": [
                ("What sport is being played?", "This video shows an athletic competition or sports activity taking place."),
                ("Describe the scene in the video.", "Athletes are engaged in competitive sports activities in an outdoor or indoor sports venue."),
            ]
        },
        "traffic": {
            "captions": [
                "Heavy traffic flows along a busy urban highway during rush hour.",
                "Cars, buses, and motorcycles navigate through an intersection in a city center.",
                "A long line of vehicles waits at a traffic light on a congested road.",
            ],
            "qa": [
                ("What is happening in this video?", "The video shows urban traffic, with various vehicles moving through city streets."),
                ("Describe the traffic situation shown.", "There are multiple vehicles including cars and buses navigating through busy city roads."),
            ]
        },
        "nature": {
            "captions": [
                "Waves crash against rocky cliffs along a rugged coastline under dramatic clouds.",
                "A flock of birds takes flight from a tree at sunrise in a misty forest.",
                "Snow falls gently on a mountain landscape, covering the pine trees in white.",
            ],
            "qa": [
                ("What natural scene is depicted?", "The video shows a beautiful natural landscape with dynamic environmental elements."),
                ("Describe the natural environment shown.", "The video captures outdoor natural scenery with notable weather and environmental features."),
            ]
        },
        "classroom": {
            "captions": [
                "A teacher writes equations on a whiteboard while students take notes.",
                "Students collaborate in small groups, discussing their project materials.",
                "A professor presents a slideshow to a lecture hall full of attentive students.",
            ],
            "qa": [
                ("What is happening in this scene?", "The video shows an educational setting with teaching and learning activities."),
                ("Describe the activity shown.", "People are engaged in an educational activity in a classroom or lecture hall setting."),
            ]
        },
    }

    scene_names = list(VIDEO_SCENES.keys())
    frame_counter = 0

    for idx in range(400):
        scene_name = scene_names[idx % len(scene_names)]
        scene_data = VIDEO_SCENES[scene_name]

        # 生成 6 帧（对标模型的 6帧→64token 压缩）
        frames = generate_video_frame_sequence(scene_name, num_frames=6)
        frame_paths = []
        for fi, frame in enumerate(frames):
            fp = VID_DIR / f"video_{frame_counter:04d}_frame{fi}.jpg"
            frame.save(fp, "JPEG", quality=88)
            frame_paths.append(str(fp))
        frame_counter += 1

        caption = random.choice(scene_data["captions"])
        qa_pair = random.choice(scene_data["qa"])

        sample = {
            "id": f"stage2_video_{idx:04d}",
            "type": "video",
            "source": "synthetic_webvid_style",
            "frames": frame_paths,
            "num_frames": 6,
            "caption": caption,
            "conversations": [
                {"role": "user", "content": [
                    {"type": "video_frames", "paths": frame_paths},
                    {"type": "text", "content": qa_pair[0]}
                ]},
                {"role": "assistant", "content": qa_pair[1]}
            ],
            "meta": {
                "scene": scene_name,
                "fps_simulated": 25,
                "duration_sec": 6.0 / 25,
                "tokens_after_compression": 64
            }
        }
        all_samples.append(sample)

        if (idx + 1) % 100 == 0:
            print(f"  视频帧: {idx+1}/400 完成")

    # ── 语音对话 400 条 (20%) ─────────────────────────────────
    print("\n[4/4] 生成语音对话数据 400 条...")

    DIALOG_PAIRS = [
        ("What time does the next train leave for downtown?", "The next train to downtown leaves in about fifteen minutes, at three forty-five. You should head to platform two."),
        ("Can you recommend a good restaurant nearby?", "There's an excellent Italian restaurant just two blocks away called La Bella Roma. They have great pasta and the service is wonderful."),
        ("How do I get to the museum from here?", "Take the number seven bus from the stop across the street, ride it for four stops, then walk two blocks north. It takes about twenty minutes."),
        ("What's the weather going to be like tomorrow?", "Tomorrow will be partly cloudy with a high of twenty-three degrees. There's a thirty percent chance of afternoon showers, so you might want to bring an umbrella."),
        ("Could you explain how photosynthesis works?", "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. The chlorophyll in plant cells captures light energy to drive this chemical reaction."),
        ("今天有什么新鲜事吗？", "今天科技界有个重大新闻，一家科技公司宣布成功研发了新一代量子芯片，运算速度比传统芯片快了一千倍。"),
        ("你能帮我翻译这段话吗？", "当然可以，请把需要翻译的内容告诉我，我会尽力帮您准确翻译。您需要翻译成哪种语言呢？"),
        ("最近学习压力很大，有什么建议吗？", "学习压力大是很常见的，建议您合理安排时间，每学习45分钟就休息10分钟。另外，保持规律作息和适量运动也非常重要。"),
    ]

    for idx in range(400):
        dialog = DIALOG_PAIRS[idx % len(DIALOG_PAIRS)]
        user_text, assistant_text = dialog

        # 生成用户语音
        user_audio_path = AUDIO_DIR / f"dialog_{idx:04d}_user.wav"
        duration_u = 1.5 + len(user_text) * 0.06
        wave_data, sr = generate_wav_audio(min(duration_u, 8.0), base_freq=160 + (idx % 40) * 3)
        save_wav(wave_data, sr, user_audio_path)

        # 生成助手语音回复
        asst_audio_path = AUDIO_DIR / f"dialog_{idx:04d}_assistant.wav"
        duration_a = 2.0 + len(assistant_text) * 0.06
        wave_data_a, sr_a = generate_wav_audio(min(duration_a, 12.0), base_freq=140 + (idx % 30) * 3)
        save_wav(wave_data_a, sr_a, asst_audio_path)

        lang = "zh" if any(ord(c) > 127 for c in user_text) else "en"

        sample = {
            "id": f"stage2_audio_{idx:04d}",
            "type": "audio_text",
            "source": "synthetic_vggsound_style",
            "audio": str(user_audio_path),
            "transcript": user_text,
            "response": assistant_text,
            "response_audio": str(asst_audio_path),
            "conversations": [
                {"role": "user", "content": [
                    {"type": "audio", "path": str(user_audio_path)},
                ]},
                {"role": "assistant", "content": assistant_text}
            ],
            "meta": {
                "language": lang,
                "user_duration_sec": round(min(duration_u, 8.0), 2),
                "assistant_duration_sec": round(min(duration_a, 12.0), 2),
                "is_bilingual": True
            }
        }
        all_samples.append(sample)

        if (idx + 1) % 100 == 0:
            print(f"  语音对话: {idx+1}/400 完成")

    # ── 保存 ─────────────────────────────────────────────────
    random.shuffle(all_samples)
    output_path = OUTPUT_DIR / "stage2_train.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 各类型分文件（供 WeightedRandomSampler 使用）
    for dtype in ["ocr_document", "interleaved", "video", "audio_text"]:
        subset = [s for s in all_samples if s["type"] == dtype]
        sub_path = OUTPUT_DIR / f"stage2_{dtype}.jsonl"
        with open(sub_path, "w", encoding="utf-8") as f:
            for s in subset:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nStage 2 数据生成完成")
    print(f"   总样本: {len(all_samples)} 条")
    type_counts = {}
    for s in all_samples:
        type_counts[s["type"]] = type_counts.get(s["type"], 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"   {t}: {c} 条 ({c/len(all_samples)*100:.0f}%)")
    print(f"   保存至: {output_path}")

    stats = {
        "dataset_name": "MiniCPM-o Stage2 Multimodal Pretrain Dataset",
        "reference_datasets": ["OmniDocBench", "ArXiv", "MINT-1T", "WebVid-10M", "VGGSound"],
        "total_samples": len(all_samples),
        "type_distribution": type_counts,
        "ocr_noise_levels": "均匀分布 0~5 级",
        "video_frames_per_sample": 6,
        "video_token_compression": "6帧→64tokens (96x)",
    }
    with open(OUTPUT_DIR / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    generate_stage2_data()
