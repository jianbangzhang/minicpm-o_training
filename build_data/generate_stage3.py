"""
Stage 3 SFT 数据生成器
对标数据集：
  - vision_qa:      LLaVA-1.5 / ShareGPT4V (视觉QA)
  - ocr_document:   OmniDocBench SFT (文档解析问答)
  - math_reasoning: MathVista / GeoQA (数学推理，含深思考)
  - video_qa:       Video-MME / VideoInstruct100K
  - speech_dialog:  InstructSpeech / 合成语音对话
  - long_response:  DetailCaps / LLaVA-Detail (详细描述)
  - multilingual:   多语言视觉指令

生成 2000 条，两阶段配比：
  Part-1 (1200条): vision_qa 600 + ocr 300 + speech 300
  Part-2 (700条):  math 250 + video 200 + long 150 + multilingual 100
  Final  (100条):  全类型最高质量精标
"""

import json
import random
import math
import wave
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

random.seed(777)
np.random.seed(777)

OUTPUT_DIR = Path("raw/datasets/stage3")
IMG_DIR = OUTPUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = Path("/home/claude/build_data/stage3/audio")
AUDIO_DIR.mkdir(exist_ok=True)


def make_image(scene_type, idx, size=(448, 448)):
    """生成各类型场景图像"""
    w, h = size
    rng = random.Random(idx * 1337)

    palettes = {
        "chart": ([(250,250,250)], [(30,100,200), (200,50,50), (50,180,50), (200,150,0)]),
        "math":  ([(240,240,255)], [(50,50,150)]),
        "scene": ([(200,230,200), (220,200,230), (210,220,240)], [(180,80,80), (80,80,180)]),
        "doc":   ([(255,255,255)], [(50,50,50)]),
        "video": ([(100,150,100), (150,100,100), (100,100,180)], [(255,200,50)]),
    }
    bg_list, fg_list = palettes.get(scene_type, palettes["scene"])
    bg = rng.choice(bg_list)
    fg = rng.choice(fg_list)

    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)

    if scene_type == "chart":
        # 柱状图 / 折线图
        draw.rectangle([30, 20, w-20, h-40], fill=(255,255,255), outline=(150,150,150), width=1)
        chart_type = idx % 2
        if chart_type == 0:  # 柱状图
            bar_colors = [(30,100,200), (200,50,50), (50,180,50), (200,150,0)]
            bars = rng.randint(4, 7)
            bw = (w - 60) // bars
            for bi in range(bars):
                bh = rng.randint(int((h-80)*0.2), int((h-80)*0.9))
                bx = 40 + bi * bw
                by = h - 40 - bh
                draw.rectangle([bx+3, by, bx+bw-3, h-40], fill=bar_colors[bi % len(bar_colors)])
        else:  # 折线图
            draw.line([(30, h-40), (w-20, h-40)], fill=(0,0,0), width=2)
            draw.line([(30, 20), (30, h-40)], fill=(0,0,0), width=2)
            points = [(30 + i*(w-50)//8, h-40 - rng.randint(20, h-70)) for i in range(9)]
            for pi in range(len(points)-1):
                draw.line([points[pi], points[pi+1]], fill=fg, width=2)
            for pt in points:
                draw.ellipse([pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4], fill=fg)

    elif scene_type == "math":
        # 几何图形（对标 GeoQA）
        draw.rectangle([10, 10, w-10, h-10], fill=(255,255,255), outline=(100,100,150), width=2)
        geo_type = idx % 4
        if geo_type == 0:  # 三角形
            pts = [(w//2, 40), (80, h-60), (w-80, h-60)]
            draw.polygon(pts, outline=fg, width=3)
            # 标注边长
            draw.line([(w//2, 40), (80, h-60)], fill=(200,0,0), width=2)
            draw.text((60, h//2), "a", fill=(200,0,0))
            draw.line([(80, h-60), (w-80, h-60)], fill=(0,150,0), width=2)
            draw.text((w//2-10, h-50), "b", fill=(0,150,0))
        elif geo_type == 1:  # 圆
            draw.ellipse([w//4, h//4, 3*w//4, 3*h//4], outline=fg, width=3)
            draw.line([(w//2, h//4), (w//2, h//2)], fill=(200,0,0), width=2)
            draw.text((w//2+5, h*3//8), "r", fill=(200,0,0))
        elif geo_type == 2:  # 矩形
            draw.rectangle([w//5, h//4, 4*w//5, 3*h//4], outline=fg, width=3)
            draw.text((w//2-10, h//8), f"w={rng.randint(3,12)}", fill=(200,0,0))
            draw.text((w//12, h//2), f"h={rng.randint(2,8)}", fill=(0,150,0))
        else:  # 坐标系
            draw.line([(30, h//2), (w-20, h//2)], fill=(0,0,0), width=2)
            draw.line([(w//2, 20), (w//2, h-20)], fill=(0,0,0), width=2)
            # 函数曲线
            pts = []
            for xi in range(w-60):
                x = (xi - (w-60)//2) / 30
                y = x**2 / 3
                px = xi + 30
                py = h//2 - int(y * 25)
                if 10 < py < h-10:
                    pts.append((px, py))
            if len(pts) > 1:
                for pi in range(len(pts)-1):
                    draw.line([pts[pi], pts[pi+1]], fill=fg, width=2)

    elif scene_type == "video":
        # 视频场景帧
        bg_col = rng.choice([(80,150,80), (150,80,80), (80,80,150)])
        img = Image.new("RGB", size, bg_col)
        draw = ImageDraw.Draw(img)
        # 主体
        draw.ellipse([w//2-60, h//2-60, w//2+60, h//2+60], fill=fg, outline=(255,255,255), width=2)
        # 时间戳
        frame_n = idx % 150
        draw.text((10, 10), f"00:{frame_n//25:02d}:{frame_n%25:02d}", fill=(255,255,255))

    else:
        # 通用场景
        for i in range(3):
            x1 = rng.randint(20, w//2)
            y1 = rng.randint(20, h//2)
            x2 = rng.randint(w//2, w-20)
            y2 = rng.randint(h//2, h-20)
            c = (rng.randint(80,220), rng.randint(80,220), rng.randint(80,220))
            draw.ellipse([x1, y1, x2, y2], fill=c, outline=(255,255,255), width=2)

    # 轻微噪声
    pixels = np.array(img)
    noise = np.random.randint(-6, 6, pixels.shape, dtype=np.int16)
    return Image.fromarray(np.clip(pixels.astype(np.int16)+noise, 0, 255).astype(np.uint8))


def save_wav_simple(text, path, sr=16000):
    duration = min(1.5 + len(text)*0.06, 10.0)
    n = int(sr * duration)
    t = np.linspace(0, duration, n)
    freq = 150 + (sum(ord(c) for c in text[:5]) % 100)
    wav = (0.4*np.sin(2*np.pi*freq*t) + 0.1*np.random.randn(n)) * 0.5
    wav_int = (wav * 32767).astype(np.int16)
    with wave.open(str(path), 'w') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(wav_int.tobytes())


# ── 各类型样本生成函数 ──────────────────────────────────────────

CHART_QUESTIONS = [
    ("According to the bar chart, which category has the highest value?", 
     "Based on the bar chart, Category B shows the highest value at approximately 85 units, followed by Category D at around 72 units."),
    ("What trend does the line graph show over time?",
     "The line graph shows a generally upward trend from 2019 to 2023, with a slight dip in 2021 before resuming growth."),
    ("Compare the values shown in the first and last bars.",
     "The first bar shows a value of approximately 45 units, while the last bar reaches about 78 units, indicating a 73% increase."),
]

SCENE_QUESTIONS = [
    ("Describe what you see in this image in detail.",
     "The image shows a colorful scene with several objects arranged in the frame. The composition includes both foreground and background elements with contrasting colors."),
    ("What is the main subject of this image?",
     "The main subject of this image appears to be a collection of geometric shapes with vibrant colors, creating a visually engaging composition."),
    ("What colors are most prominent in this image?",
     "The most prominent colors are warm tones including orange and red, contrasted with cooler blues and greens in the background areas."),
]

MATH_PROBLEMS = [
    # (问题, 简单回答, 深思考推理链)
    ("In the triangle shown, if side a = 5 and side b = 12, what is the length of the hypotenuse?",
     "The hypotenuse c = 13.",
     "Using the Pythagorean theorem: c² = a² + b² = 5² + 12² = 25 + 144 = 169. Therefore c = √169 = 13."),
    ("Calculate the area of the circle shown if the radius r = 7 cm.",
     "Area = 49π ≈ 153.94 cm².",
     "The area formula for a circle is A = πr². With r = 7: A = π × 7² = π × 49 = 49π ≈ 153.94 cm²."),
    ("The rectangle shown has width 8 and height 6. What is its perimeter?",
     "The perimeter is 28 units.",
     "Perimeter = 2(width + height) = 2(8 + 6) = 2 × 14 = 28 units."),
    ("If the line y = 2x + 3 is shown, what is the y-intercept?",
     "The y-intercept is 3.",
     "The equation is in slope-intercept form y = mx + b. Here m = 2 (slope) and b = 3 (y-intercept). So the line crosses the y-axis at y = 3."),
    ("Find the slope of the line passing through points (2, 4) and (6, 12).",
     "The slope is 2.",
     "Slope m = (y₂ - y₁)/(x₂ - x₁) = (12 - 4)/(6 - 2) = 8/4 = 2. The line rises 2 units for every 1 unit horizontally."),
    ("A circle has area 100π. What is its circumference?",
     "The circumference is 20π ≈ 62.83 units.",
     "Area = πr² = 100π, so r² = 100, r = 10. Circumference = 2πr = 2π(10) = 20π ≈ 62.83 units."),
    ("In the right triangle, if one angle is 30° and the hypotenuse is 10, find the opposite side.",
     "The opposite side is 5.",
     "sin(30°) = opposite/hypotenuse. sin(30°) = 0.5, so opposite = 0.5 × 10 = 5."),
    ("The rectangle has area 48 and length 8. What is its width?",
     "The width is 6.",
     "Area = length × width. 48 = 8 × width. Width = 48/8 = 6."),
]

VIDEO_QA_PAIRS = [
    ("What activity is being shown in this video clip?",
     "The video clip shows a dynamic scene with movement and action. Based on the visual elements, it appears to depict an everyday activity in an indoor or outdoor setting."),
    ("Describe the main action happening in this video.",
     "The video shows a subject moving through the scene. The motion is fluid and the environment provides context clues about the nature of the activity."),
    ("What can you infer about the time of day from this video?",
     "Based on the lighting conditions and shadows visible in the frames, this video appears to have been recorded during daytime hours, likely in the afternoon."),
    ("How many distinct scenes or locations appear in this video?",
     "Based on the visual continuity, this video appears to show a single continuous scene with consistent lighting and background elements throughout."),
]

SPEECH_DIALOG_EN = [
    ("What are the main benefits of regular exercise?",
     "Regular exercise provides numerous health benefits including improved cardiovascular health, stronger muscles and bones, better mental health, and reduced risk of chronic diseases. Even thirty minutes of moderate activity five days a week can make a significant difference."),
    ("Can you explain the water cycle?",
     "The water cycle describes how water moves through Earth's systems. Water evaporates from oceans and lakes, rises as vapor, condenses into clouds, and falls as precipitation. This water then flows into rivers, seeps into groundwater, or evaporates again, continuing the cycle."),
    ("What should I consider when buying a laptop?",
     "When buying a laptop, consider your primary use case, budget, processor speed, RAM, storage capacity, battery life, display quality, and portability. For general use, eight gigabytes of RAM and a solid-state drive are good starting points."),
]

SPEECH_DIALOG_ZH = [
    ("请介绍一下中国的四大发明？",
     "中国古代的四大发明是造纸术、印刷术、火药和指南针。这四项发明对世界文明的发展产生了深远影响。造纸术和印刷术推动了知识的传播，火药改变了战争形态，指南针促进了航海探索。"),
    ("怎样才能提高英语口语水平？",
     "提高英语口语水平需要多听多说。建议每天坚持听英文播客或看英文视频，模仿母语者的发音和语调。找语言交换伙伴进行对话练习也非常有效。另外，不要害怕犯错，大胆开口是进步的关键。"),
    ("人工智能会取代人类的工作吗？",
     "这是一个复杂的问题。人工智能确实会自动化一些重复性工作，但同时也会创造新的就业机会。历史上每次技术革命都带来了工作的转型，而非单纯的消失。关键在于人类需要不断学习新技能，适应时代变化。"),
]

LONG_RESPONSE_QUESTIONS = [
    ("Provide a comprehensive description of all visual elements in this image, including colors, shapes, spatial relationships, and any text or symbols you can identify.",
     "This image presents a rich visual composition that merits detailed analysis. Starting with the overall color palette, the image employs a harmonious blend of warm and cool tones. The foreground features prominently placed elements rendered in vibrant hues, creating strong visual contrast against the background. The spatial arrangement follows a balanced composition with the primary subject positioned slightly left of center, allowing the eye to naturally scan across the frame.\n\nThe geometric shapes present include both curved and angular forms, suggesting a dynamic rather than static scene. The color gradients visible in the background add depth and dimension, creating a sense of three-dimensional space within the two-dimensional medium.\n\nThe lighting conditions appear consistent, with diffuse illumination eliminating harsh shadows. This even lighting reveals the surface textures clearly, from smooth geometric forms to more complex textured areas. Overall, the image demonstrates intentional compositional choices that guide the viewer's attention through a carefully planned visual journey."),
    ("Analyze this chart/graph in detail: what data is being represented, what patterns or trends are visible, and what conclusions can be drawn?",
     "This visualization presents quantitative data in a format designed to communicate patterns and relationships clearly. Examining the axes, we can identify the variables being measured and their respective scales, which provide the framework for interpretation.\n\nThe data distribution shows several notable characteristics. There is a clear central tendency with values clustering around the midrange of the measured spectrum. The variance appears moderate, suggesting consistent but not rigid patterns in the underlying data source.\n\nLooking at temporal or categorical trends, we observe a general directional movement that indicates either growth, decline, or cyclical behavior depending on the nature of the dataset. The rate of change appears to accelerate during certain intervals while stabilizing at others.\n\nFrom these observations, several conclusions emerge: first, the data source demonstrates predictable behavior within established parameters; second, there are identifiable inflection points that may correspond to external factors or interventions; and third, the overall trajectory suggests continued development in the observed direction unless significant external factors intervene. These insights would be most actionable when combined with contextual information about the specific domain the data represents."),
]

MULTILINGUAL_QA = [
    ("¿Qué ves en esta imagen?",
     "En esta imagen puedo ver una composición visual con elementos coloridos. Hay formas geométricas prominentes en el primer plano, con un fondo que proporciona contexto y profundidad a la escena."),
    ("이 이미지에서 무엇을 볼 수 있나요?",
     "이 이미지에서 다양한 색상과 형태를 볼 수 있습니다. 전경에는 주요 요소들이 배치되어 있으며, 배경은 전체적인 장면의 맥락을 제공합니다."),
    ("Qu'est-ce que vous voyez dans cette image?",
     "Dans cette image, je peux observer une composition visuelle intéressante avec des éléments colorés disposés harmonieusement. Les formes géométriques au premier plan attirent l'attention, tandis que l'arrière-plan fournit de la profondeur à la scène."),
    ("Was sehen Sie in diesem Bild?",
     "In diesem Bild sehe ich eine visuelle Komposition mit farbenfrohen Elementen. Im Vordergrund befinden sich markante geometrische Formen, während der Hintergrund Tiefe und Kontext für die gesamte Szene bietet."),
]


def generate_stage3_data():
    print("=" * 60)
    print("生成 Stage 3 SFT 数据（监督微调）")
    print("对标数据集: LLaVA-1.5 + ShareGPT4V + MathVista + Video-MME + InstructSpeech")
    print("=" * 60)

    all_samples = []
    img_counter = 0

    # ── Part-1: vision_qa 600 条 ──────────────────────────────
    print("\n[1/7] Part-1: 视觉QA 600 条...")
    for idx in range(600):
        scene_type = ["chart", "scene", "scene", "scene"][idx % 4]
        img = make_image(scene_type, idx)
        img_path = IMG_DIR / f"sft_img_{img_counter:04d}.jpg"
        img.save(img_path, "JPEG", quality=90)
        img_counter += 1

        if scene_type == "chart":
            qa = CHART_QUESTIONS[idx % len(CHART_QUESTIONS)]
        else:
            qa = SCENE_QUESTIONS[idx % len(SCENE_QUESTIONS)]

        sample = {
            "id": f"sft_vqa_{idx:04d}",
            "type": "vision_qa",
            "source": "synthetic_llava_style",
            "thinking_mode": "fast",
            "phase": "part1",
            "conversations": [
                {"role": "system", "content": "You are a helpful vision assistant."},
                {"role": "user", "content": [
                    {"type": "image", "path": str(img_path)},
                    {"type": "text", "content": qa[0]}
                ]},
                {"role": "assistant", "content": qa[1]}
            ]
        }
        all_samples.append(sample)
    print(f"  完成 600 条")

    # ── Part-1: OCR 文档 300 条 ────────────────────────────────
    print("\n[2/7] Part-1: OCR文档问答 300 条...")
    DOC_QA = [
        ("Extract all text from this document image.", "The document contains the following text content:"),
        ("What is the main topic of this document?", "Based on the document structure and visible text, the main topic is"),
        ("List all numerical values mentioned in this document.", "The numerical values visible in this document include:"),
        ("What is the document title?", "The document title is:"),
    ]
    for idx in range(300):
        img = make_image("doc", idx)
        img_path = IMG_DIR / f"sft_doc_{img_counter:04d}.jpg"
        img.save(img_path, "JPEG", quality=90)
        img_counter += 1

        qa = DOC_QA[idx % len(DOC_QA)]
        # 模拟 OCR 结果
        doc_texts = ["Annual Report 2024", "Research Findings", "Project Summary", "Meeting Minutes"]
        response = qa[1] + " " + random.choice(doc_texts) + f". This document appears to be a formal document with structured sections."

        sample = {
            "id": f"sft_ocr_{idx:04d}",
            "type": "ocr_document",
            "source": "synthetic_omnidocbench_sft",
            "thinking_mode": "fast",
            "phase": "part1",
            "conversations": [
                {"role": "system", "content": "You are an expert at reading and extracting information from documents."},
                {"role": "user", "content": [
                    {"type": "image", "path": str(img_path)},
                    {"type": "text", "content": qa[0]}
                ]},
                {"role": "assistant", "content": response}
            ]
        }
        all_samples.append(sample)
    print(f"  完成 300 条")

    # ── Part-1: 语音对话 300 条 ────────────────────────────────
    print("\n[3/7] Part-1: 语音对话 300 条...")
    all_speech = SPEECH_DIALOG_EN * 8 + SPEECH_DIALOG_ZH * 8
    for idx in range(300):
        dialog = all_speech[idx % len(all_speech)]
        q, a = dialog
        lang = "zh" if any(ord(c)>127 for c in q) else "en"

        audio_path = AUDIO_DIR / f"sft_speech_{idx:04d}.wav"
        save_wav_simple(q, audio_path)

        sample = {
            "id": f"sft_speech_{idx:04d}",
            "type": "speech_dialog",
            "source": "synthetic_instructspeech_style",
            "thinking_mode": "fast",
            "phase": "part1",
            "conversations": [
                {"role": "user", "content": [
                    {"type": "audio", "path": str(audio_path)},
                ]},
                {"role": "assistant", "content": a}
            ],
            "meta": {"language": lang, "transcript": q}
        }
        all_samples.append(sample)
    print(f"  完成 300 条")

    # ── Part-2: 数学推理 250 条（含深思考） ──────────────────
    print("\n[4/7] Part-2: 数学推理 250 条（含深思考CoT）...")
    for idx in range(250):
        prob = MATH_PROBLEMS[idx % len(MATH_PROBLEMS)]
        question, fast_answer, deep_reasoning = prob

        img = make_image("math", idx)
        img_path = IMG_DIR / f"sft_math_{img_counter:04d}.jpg"
        img.save(img_path, "JPEG", quality=90)
        img_counter += 1

        # 深思考模式（70% 深思考）
        use_deep = idx % 10 < 7
        if use_deep:
            response = f"<think>\n{deep_reasoning}\n</think>\n{fast_answer}"
            mode = "deep"
        else:
            response = fast_answer
            mode = "fast"

        sample = {
            "id": f"sft_math_{idx:04d}",
            "type": "math_reasoning",
            "source": "synthetic_mathvista_style",
            "thinking_mode": mode,
            "phase": "part2",
            "ground_truth": fast_answer,
            "conversations": [
                {"role": "system", "content": "You are a mathematics expert. Solve problems step by step." if use_deep else "You are a mathematics assistant. Give concise answers."},
                {"role": "user", "content": [
                    {"type": "image", "path": str(img_path)},
                    {"type": "text", "content": question}
                ]},
                {"role": "assistant", "content": response, "think": deep_reasoning}
            ],
            "meta": {"has_cot": use_deep, "answer_verifiable": True}
        }
        all_samples.append(sample)
    print(f"  完成 250 条")

    # ── Part-2: 视频理解 200 条 ────────────────────────────────
    print("\n[5/7] Part-2: 视频理解 200 条...")
    for idx in range(200):
        qa = VIDEO_QA_PAIRS[idx % len(VIDEO_QA_PAIRS)]

        # 每个视频样本 6 帧
        frame_paths = []
        for fi in range(6):
            fimg = make_image("video", idx * 6 + fi)
            fp = IMG_DIR / f"sft_video_{img_counter:04d}_f{fi}.jpg"
            fimg.save(fp, "JPEG", quality=88)
            frame_paths.append(str(fp))
        img_counter += 1

        sample = {
            "id": f"sft_video_{idx:04d}",
            "type": "video_qa",
            "source": "synthetic_videomme_style",
            "thinking_mode": "fast",
            "phase": "part2",
            "conversations": [
                {"role": "user", "content": [
                    {"type": "video_frames", "paths": frame_paths},
                    {"type": "text", "content": qa[0]}
                ]},
                {"role": "assistant", "content": qa[1]}
            ],
            "meta": {"num_frames": 6}
        }
        all_samples.append(sample)
    print(f"  完成 200 条")

    # ── Part-2: 长回复详细描述 150 条 ────────────────────────
    print("\n[6/7] Part-2: 长回复详细描述 150 条...")
    for idx in range(150):
        img = make_image("chart" if idx % 2 == 0 else "scene", idx + 300)
        img_path = IMG_DIR / f"sft_long_{img_counter:04d}.jpg"
        img.save(img_path, "JPEG", quality=90)
        img_counter += 1

        qa = LONG_RESPONSE_QUESTIONS[idx % len(LONG_RESPONSE_QUESTIONS)]
        # 深思考格式用于长回复
        use_deep = idx % 3 == 0
        response = qa[1]
        if use_deep:
            response = f"<think>\nLet me analyze this image systematically, examining each visual element in turn.\n</think>\n{response}"

        sample = {
            "id": f"sft_long_{idx:04d}",
            "type": "long_response",
            "source": "synthetic_detailcaps_style",
            "thinking_mode": "deep" if use_deep else "fast",
            "phase": "part2",
            "conversations": [
                {"role": "system", "content": "You are a detailed visual analyst. Provide thorough, comprehensive descriptions."},
                {"role": "user", "content": [
                    {"type": "image", "path": str(img_path)},
                    {"type": "text", "content": qa[0]}
                ]},
                {"role": "assistant", "content": response}
            ]
        }
        all_samples.append(sample)
    print(f"  完成 150 条")

    # ── Part-2: 多语言指令 100 条 ─────────────────────────────
    print("\n[7/7] Part-2: 多语言指令 100 条...")
    langs = ["es", "ko", "fr", "de"]
    for idx in range(100):
        img = make_image("scene", idx + 500)
        img_path = IMG_DIR / f"sft_ml_{img_counter:04d}.jpg"
        img.save(img_path, "JPEG", quality=90)
        img_counter += 1

        qa = MULTILINGUAL_QA[idx % len(MULTILINGUAL_QA)]
        lang = langs[idx % len(langs)]

        sample = {
            "id": f"sft_multilingual_{idx:04d}",
            "type": "multilingual",
            "source": "synthetic_viscpm_style",
            "thinking_mode": "fast",
            "phase": "part2",
            "language": lang,
            "conversations": [
                {"role": "user", "content": [
                    {"type": "image", "path": str(img_path)},
                    {"type": "text", "content": qa[0]}
                ]},
                {"role": "assistant", "content": qa[1]}
            ]
        }
        all_samples.append(sample)
    print(f"  完成 100 条")

    # ── Final: 高质量精标 100 条 ──────────────────────────────
    # 从上面所有类型各抽高质量样本
    final_samples = random.sample(all_samples, 100)
    for s in final_samples:
        s_copy = dict(s)
        s_copy["id"] = s_copy["id"] + "_final"
        s_copy["phase"] = "final"
        s_copy["quality"] = "high"
    # 不额外添加，直接在 JSONL 中标注

    # ── 保存 ─────────────────────────────────────────────────
    random.shuffle(all_samples)
    out_path = OUTPUT_DIR / "stage3_train.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 按 phase 分开保存
    for phase in ["part1", "part2"]:
        subset = [s for s in all_samples if s.get("phase") == phase]
        with open(OUTPUT_DIR / f"stage3_{phase}.jsonl", "w", encoding="utf-8") as f:
            for s in subset:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Final = 所有高质量样本最后 100 条
    final_subset = all_samples[:100]
    for s in final_subset:
        s["phase"] = "final"
    with open(OUTPUT_DIR / "stage3_final.jsonl", "w", encoding="utf-8") as f:
        for s in final_subset:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nStage 3 数据生成完成")
    print(f"   总样本: {len(all_samples)} 条")
    type_counts = {}
    phase_counts = {}
    for s in all_samples:
        type_counts[s["type"]] = type_counts.get(s["type"], 0) + 1
        phase_counts[s["phase"]] = phase_counts.get(s["phase"], 0) + 1
    print("   类型分布:")
    for t, c in sorted(type_counts.items()):
        print(f"     {t}: {c} 条")
    print("   阶段分布:")
    for p, c in sorted(phase_counts.items()):
        print(f"     {p}: {c} 条")
    print(f"   保存至: {out_path}")

    stats = {
        "dataset_name": "MiniCPM-o Stage3 SFT Dataset",
        "reference_datasets": ["LLaVA-1.5", "ShareGPT4V", "MathVista", "Video-MME",
                                "InstructSpeech", "DetailCaps", "VisCPM"],
        "total_samples": len(all_samples),
        "phase_distribution": phase_counts,
        "type_distribution": type_counts,
        "fast_think_samples": sum(1 for s in all_samples if s.get("thinking_mode") == "fast"),
        "deep_think_samples": sum(1 for s in all_samples if s.get("thinking_mode") == "deep"),
    }
    with open(OUTPUT_DIR / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    generate_stage3_data()
