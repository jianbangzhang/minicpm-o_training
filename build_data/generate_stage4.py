"""
Stage 4 强化学习数据生成器
对标数据集：
  - RLPR: MathVista + GeoQA + CLEVR-Math (可验证数学推理)
  - RLAIF-V: 对比偏好数据（准确回复 vs 含幻觉回复）

生成 2000 条：RLPR 1200 条 (60%) + RLAIF-V 800 条 (40%)
快/深思考混合：RLPR 中深思考 70%，快思考 30%
"""

import json
import random
import math
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

random.seed(2024)
np.random.seed(2024)

OUTPUT_DIR = Path("raw/datasets/stage4")
IMG_DIR = OUTPUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def make_math_image(problem_type, idx, size=(448, 448)):
    """生成数学题配套图像"""
    img = Image.new("RGB", size, (248, 248, 255))
    draw = ImageDraw.Draw(img)
    w, h = size
    rng = random.Random(idx * 999)

    if problem_type == "geometry_triangle":
        # 三角形带标注
        a_len = rng.randint(3, 12)
        b_len = rng.randint(3, 12)
        pts = [(w//2, 60), (80, h-80), (w-80, h-80)]
        draw.polygon(pts, outline=(50,50,180), width=3)
        # 边长标注
        draw.text((50, h//2), f"a={a_len}", fill=(200,0,0))
        draw.text((w//2-15, h-55), f"b={b_len}", fill=(0,150,0))
        # 直角符号
        draw.rectangle([w-80, h-80, w-70, h-70], outline=(0,0,0), width=2)

    elif problem_type == "geometry_circle":
        radius = rng.randint(4, 12)
        cx, cy = w//2, h//2
        r_px = min(w, h)//3
        draw.ellipse([cx-r_px, cy-r_px, cx+r_px, cy+r_px], outline=(50,50,180), width=3)
        draw.line([(cx, cy), (cx+r_px, cy)], fill=(200,0,0), width=2)
        draw.text((cx+r_px//2-10, cy-20), f"r={radius}", fill=(200,0,0))
        draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill=(0,0,0))

    elif problem_type == "bar_chart":
        # 含数据的柱状图（可计算平均值/最大值等）
        values = [rng.randint(10, 90) for _ in range(5)]
        labels = ["A", "B", "C", "D", "E"]
        bar_colors = [(200,50,50), (50,150,50), (50,50,200), (200,150,0), (150,50,200)]
        draw.line([(50, h-50), (w-20, h-50)], fill=(0,0,0), width=2)
        draw.line([(50, 30), (50, h-50)], fill=(0,0,0), width=2)
        bw = (w-80) // 5
        for bi, (val, lbl, col) in enumerate(zip(values, labels, bar_colors)):
            bh = int((h-100) * val / 100)
            bx = 60 + bi * bw
            draw.rectangle([bx, h-50-bh, bx+bw-10, h-50], fill=col)
            draw.text((bx+5, h-45), lbl, fill=(0,0,0))
            draw.text((bx+2, h-55-bh), str(val), fill=(50,50,50))

    elif problem_type == "coordinate":
        # 坐标系中的点和线
        draw.line([(30, h//2), (w-20, h//2)], fill=(0,0,0), width=2)
        draw.line([(w//2, 20), (w//2, h-20)], fill=(0,0,0), width=2)
        # 标注轴
        draw.text((w-20, h//2-15), "x", fill=(0,0,0))
        draw.text((w//2+5, 15), "y", fill=(0,0,0))
        # 几个点
        points_data = [(rng.randint(-4,4), rng.randint(-4,4)) for _ in range(4)]
        scale = 35
        for px_val, py_val in points_data:
            screen_x = w//2 + px_val * scale
            screen_y = h//2 - py_val * scale
            draw.ellipse([screen_x-5, screen_y-5, screen_x+5, screen_y+5], fill=(200,0,0))

    else:
        # 通用数学图形
        draw.rectangle([20, 20, w-20, h-20], fill=(255,255,255), outline=(150,150,200), width=2)
        formula = rng.choice(["y = 2x + 3", "A = πr²", "c² = a² + b²", "f(x) = x²"])
        draw.text((30, h//2-10), formula, fill=(50,50,180))

    return img


def make_scene_image_for_hallucination(scene_id, include_object=True, size=(448, 448)):
    """
    生成用于幻觉测试的场景图像。
    include_object=True: 图中确实包含某物体
    include_object=False: 图中不包含该物体（用于构造幻觉负例）
    """
    img = Image.new("RGB", size, (200, 220, 200))
    draw = ImageDraw.Draw(img)
    w, h = size
    rng = random.Random(scene_id)

    # 背景（户外/室内场景）
    if scene_id % 2 == 0:
        # 户外
        draw.rectangle([0, 0, w, h//2], fill=(135, 200, 235))
        draw.rectangle([0, h//2, w, h], fill=(80, 150, 80))
        # 太阳
        draw.ellipse([w-70, 20, w-20, 70], fill=(255, 220, 50))
    else:
        # 室内
        draw.rectangle([0, 0, w, h], fill=(240, 230, 210))
        # 墙
        draw.rectangle([0, h*2//3, w, h], fill=(180, 160, 140))
        # 窗户
        draw.rectangle([w//4, h//5, 3*w//4, h//2], fill=(180, 220, 250), outline=(120,100,80), width=3)

    # 主体对象（猫/狗/椅子/桌子等）
    OBJECTS = {
        "cat": ((200, 150, 100), "oval"),
        "dog": ((150, 100, 80), "rect_oval"),
        "chair": ((139, 90, 43), "geometric"),
        "car": ((220, 50, 50), "rect"),
        "bird": ((50, 100, 200), "small_oval"),
    }

    obj_names = list(OBJECTS.keys())
    obj_name = obj_names[scene_id % len(obj_names)]
    obj_color, obj_type = OBJECTS[obj_name]

    if include_object:
        ox, oy = w//2 + rng.randint(-60, 60), h//2 + rng.randint(-30, 30)
        if obj_type in ("oval", "small_oval"):
            r = 40 if obj_type == "oval" else 20
            draw.ellipse([ox-r, oy-r//2, ox+r, oy+r//2], fill=obj_color)
            draw.ellipse([ox-r//3, oy-r, ox+r//3, oy-r//2+10], fill=obj_color)
        elif obj_type == "rect":
            draw.rectangle([ox-50, oy-20, ox+50, oy+20], fill=obj_color)
            draw.ellipse([ox-40, oy+15, ox-20, oy+30], fill=(50,50,50))
            draw.ellipse([ox+20, oy+15, ox+40, oy+30], fill=(50,50,50))
        else:
            draw.rectangle([ox-25, oy-40, ox+25, oy+40], fill=obj_color)

    # 其他场景元素
    for _ in range(3):
        ex, ey = rng.randint(0, w), rng.randint(h//2, h)
        ec = (rng.randint(100,200), rng.randint(100,200), rng.randint(100,200))
        draw.ellipse([ex-15, ey-10, ex+15, ey+10], fill=ec)

    return img, obj_name


# ============================================================
# RLPR 数学推理样本（含可验证答案）
# ============================================================

VERIFIABLE_MATH_PROBLEMS = [
    # geometry
    {
        "type": "geometry_triangle",
        "question": "In the right triangle shown, legs a and b are labeled. If a=3 and b=4, what is the hypotenuse? Give your answer using \\boxed{}.",
        "ground_truth": "5",
        "deep_reasoning": "Using the Pythagorean theorem: c² = a² + b² = 3² + 4² = 9 + 16 = 25. Therefore c = √25 = 5.",
        "task_type": "math"
    },
    {
        "type": "geometry_triangle",
        "question": "A right triangle has legs of length 5 and 12. What is the length of the hypotenuse? Use \\boxed{} for your answer.",
        "ground_truth": "13",
        "deep_reasoning": "c² = 5² + 12² = 25 + 144 = 169. c = √169 = 13.",
        "task_type": "math"
    },
    {
        "type": "geometry_circle",
        "question": "The circle shown has radius r=7. What is the area? (Use π≈3.14159, round to 2 decimal places) Answer with \\boxed{}.",
        "ground_truth": "153.94",
        "deep_reasoning": "Area = πr² = 3.14159 × 7² = 3.14159 × 49 ≈ 153.94.",
        "task_type": "math"
    },
    {
        "type": "geometry_circle",
        "question": "If the circle's radius is r=5, what is its circumference? (Use π≈3.14159) Answer with \\boxed{}.",
        "ground_truth": "31.42",
        "deep_reasoning": "Circumference = 2πr = 2 × 3.14159 × 5 = 31.4159 ≈ 31.42.",
        "task_type": "math"
    },
    {
        "type": "bar_chart",
        "question": "Looking at the bar chart, what is the maximum value shown? Answer with \\boxed{}.",
        "ground_truth": "varies",  # 动态答案，由图像决定
        "deep_reasoning": "I need to identify the tallest bar and read its value from the y-axis.",
        "task_type": "chart_reading"
    },
    {
        "type": "coordinate",
        "question": "What is the slope of a line passing through origin and point (4, 8)? Answer with \\boxed{}.",
        "ground_truth": "2",
        "deep_reasoning": "Slope m = (y₂-y₁)/(x₂-x₁) = (8-0)/(4-0) = 8/4 = 2.",
        "task_type": "math"
    },
    {
        "type": "geometry_triangle",
        "question": "A triangle has base 10 and height 6. What is its area? Answer with \\boxed{}.",
        "ground_truth": "30",
        "deep_reasoning": "Area = (1/2) × base × height = (1/2) × 10 × 6 = 30.",
        "task_type": "math"
    },
    {
        "type": "geometry_circle",
        "question": "A circle has diameter 14. What is its area? (Use π≈3.14159) Answer with \\boxed{}.",
        "ground_truth": "153.94",
        "deep_reasoning": "radius = diameter/2 = 7. Area = πr² = 3.14159 × 49 ≈ 153.94.",
        "task_type": "math"
    },
    {
        "type": "coordinate",
        "question": "What is 15% of 240? Answer with \\boxed{}.",
        "ground_truth": "36",
        "deep_reasoning": "15% of 240 = 0.15 × 240 = 36.",
        "task_type": "math"
    },
    {
        "type": "geometry_triangle",
        "question": "Two angles of a triangle are 60° and 45°. What is the third angle? Answer with \\boxed{}.",
        "ground_truth": "75",
        "deep_reasoning": "Sum of angles in a triangle = 180°. Third angle = 180° - 60° - 45° = 75°.",
        "task_type": "math"
    },
    {
        "type": "geometry_circle",
        "question": "A semicircle has radius r=6. What is its perimeter? (include the diameter, use π≈3.14) Answer with \\boxed{}.",
        "ground_truth": "30.84",
        "deep_reasoning": "Perimeter = πr + 2r = 3.14×6 + 12 = 18.84 + 12 = 30.84.",
        "task_type": "math"
    },
    {
        "type": "bar_chart",
        "question": "In a dataset with values 12, 18, 24, 15, 21, what is the average? Answer with \\boxed{}.",
        "ground_truth": "18",
        "deep_reasoning": "Average = (12+18+24+15+21)/5 = 90/5 = 18.",
        "task_type": "math"
    },
]

# ============================================================
# RLAIF-V 偏好数据（准确 vs 幻觉回复）
# ============================================================

HALLUCINATION_SCENARIOS = [
    {
        "question": "What animals can you see in this image?",
        "chosen_prefix": "Looking at the image carefully, I can see",
        "rejected_prefix": "I can clearly see a cat, a dog, and a bird in this image, along with",
        "hallucination_type": "object_fabrication"
    },
    {
        "question": "How many people are visible in this image?",
        "chosen_with_obj": "There is 1 person visible in the image.",
        "chosen_without_obj": "I don't see any people in this image. The scene shows an outdoor environment with natural elements.",
        "rejected": "There are 3 people visible in the image: a man on the left, a woman in the center, and a child on the right.",
        "hallucination_type": "count_error"
    },
    {
        "question": "What text or writing is visible in this image?",
        "chosen": "I don't see any text or writing in this image.",
        "rejected": "The image contains text that reads 'Welcome to the Park' at the top, along with some smaller text below describing the rules.",
        "hallucination_type": "text_fabrication"
    },
    {
        "question": "What color is the main object in the image?",
        "chosen_prefix": "The main object appears to be",
        "rejected_prefix": "The main object is clearly bright red with golden accents and",
        "hallucination_type": "attribute_error"
    },
    {
        "question": "Describe the weather conditions shown in this image.",
        "chosen": "Based on the visual cues in the image, the lighting suggests it may be daytime, though specific weather conditions are not clearly determinable from the image alone.",
        "rejected": "The image shows heavy rain and strong winds. You can see the trees bending in the storm and puddles forming on the ground.",
        "hallucination_type": "context_fabrication"
    },
    {
        "question": "What brand or logo is visible in the image?",
        "chosen": "I cannot identify any brand names or logos in this image.",
        "rejected": "I can see the Nike logo in the top right corner and what appears to be an Apple logo on the device shown.",
        "hallucination_type": "brand_fabrication"
    },
    {
        "question": "What time of day does this image appear to have been taken?",
        "chosen": "The lighting in the image suggests daytime, possibly afternoon based on the light direction and intensity.",
        "rejected": "This image was clearly taken at exactly 3:47 PM based on the sun position and shadow angles, which is consistent with a winter afternoon.",
        "hallucination_type": "overconfident_detail"
    },
    {
        "question": "Are there any vehicles visible in this image?",
        "chosen_with_obj": "Yes, there appears to be a vehicle in the image.",
        "chosen_without_obj": "I don't see any vehicles in this image.",
        "rejected": "Yes, I can see a red sports car parked on the left, a blue bicycle leaning against the wall, and what appears to be a motorcycle in the background.",
        "hallucination_type": "object_multiplication"
    },
]


def generate_stage4_data():
    print("=" * 60)
    print("生成 Stage 4 RL 数据（强化学习后训练）")
    print("对标数据集: MathVista + GeoQA (RLPR) | RLAIF-V 偏好数据")
    print("=" * 60)

    all_rlpr = []
    all_rlaifv = []

    # ── RLPR: 1200 条可验证推理数据 ───────────────────────────
    print("\n[1/2] 生成 RLPR 数据 1200 条...")
    problems = VERIFIABLE_MATH_PROBLEMS
    thinking_modes = ["deep"] * 7 + ["fast"] * 3  # 70% 深思考，30% 快思考

    for idx in range(1200):
        prob = problems[idx % len(problems)]
        mode = thinking_modes[idx % len(thinking_modes)]

        img = make_math_image(prob["type"], idx)
        img_path = IMG_DIR / f"rl_math_{idx:04d}.jpg"
        img.save(img_path, "JPEG", quality=90)

        # 动态调整某些题目的答案（bar_chart 类型）
        gt = prob["ground_truth"]
        if gt == "varies":
            gt = str(random.randint(60, 95))

        # 构建 prompt（用于 GRPO 采样）
        sys_prompt = (
            "You are a mathematics expert. Think carefully step by step before answering."
            if mode == "deep"
            else "You are a mathematics assistant. Give a concise direct answer."
        )

        sample = {
            "id": f"rlpr_{idx:04d}",
            "type": "rlpr",
            "source": "synthetic_mathvista_geoqa_style",
            "thinking_mode": mode,
            "task_type": prob["task_type"],
            "ground_truth": gt,
            "deep_reasoning": prob["deep_reasoning"],
            "conversations": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [
                    {"type": "image", "path": str(img_path)},
                    {"type": "text", "content": prob["question"]}
                ]}
            ],
            "meta": {
                "problem_type": prob["type"],
                "verifiable": prob["task_type"] == "math",
                "expected_format": "\\boxed{answer}",
                "fast_answer_example": f"\\boxed{{{gt}}}",
                "deep_answer_example": f"<think>\n{prob['deep_reasoning']}\n</think>\n\\boxed{{{gt}}}"
            }
        }
        all_rlpr.append(sample)

        if (idx + 1) % 200 == 0:
            print(f"  RLPR: {idx+1}/1200 完成")

    # ── RLAIF-V: 800 条偏好对比数据 ──────────────────────────
    print("\n[2/2] 生成 RLAIF-V 偏好数据 800 条...")

    for idx in range(800):
        scenario = HALLUCINATION_SCENARIOS[idx % len(HALLUCINATION_SCENARIOS)]
        scene_id = idx
        thinking_mode = "fast" if idx % 3 != 0 else "deep"

        # 图像：随机决定是否包含主体物体
        include_obj = (idx % 3 != 2)
        img, obj_name = make_scene_image_for_hallucination(scene_id, include_obj)
        img_path = IMG_DIR / f"rl_scene_{idx:04d}.jpg"
        img.save(img_path, "JPEG", quality=90)

        question = scenario["question"]

        # 构建准确（chosen）回复
        if "chosen_with_obj" in scenario:
            if include_obj:
                chosen = scenario["chosen_with_obj"].replace("a person", f"a {obj_name}")
            else:
                chosen = scenario.get("chosen_without_obj", "I cannot determine this from the image.")
        elif "chosen_prefix" in scenario:
            if include_obj:
                chosen = scenario["chosen_prefix"] + f" {obj_name} in the image, positioned in the center of the frame."
            else:
                chosen = "I don't see any specific objects that match the description in this image."
        else:
            chosen = scenario.get("chosen", "I can analyze the image, but I want to be accurate about what I observe.")

        # 构建含幻觉（rejected）回复
        rejected = scenario.get("rejected",
                                scenario.get("rejected_prefix", "I can clearly see") + " multiple objects including " +
                                ", ".join([f"a {random.choice(['red', 'blue', 'green'])} {random.choice(['car', 'chair', 'table'])}"]
                                         * 3))

        # Deep thinking 版本 chosen
        if thinking_mode == "deep":
            chosen = f"<think>\nLet me carefully examine what is actually visible in this image before responding.\n</think>\n{chosen}"

        sample = {
            "id": f"rlaifv_{idx:04d}",
            "type": "rlaifv",
            "source": "synthetic_rlaifv_style",
            "thinking_mode": thinking_mode,
            "hallucination_type": scenario["hallucination_type"],
            "image": str(img_path),
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "meta": {
                "object_present": include_obj,
                "object_name": obj_name,
                "preference_reason": "chosen is accurate; rejected contains hallucinated details"
            }
        }
        all_rlaifv.append(sample)

        if (idx + 1) % 200 == 0:
            print(f"  RLAIF-V: {idx+1}/800 完成")

    # ── 保存 ─────────────────────────────────────────────────
    # RLPR
    rlpr_path = OUTPUT_DIR / "stage4_rlpr.jsonl"
    with open(rlpr_path, "w", encoding="utf-8") as f:
        for s in all_rlpr:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # RLAIF-V
    rlaifv_path = OUTPUT_DIR / "stage4_rlaifv.jsonl"
    with open(rlaifv_path, "w", encoding="utf-8") as f:
        for s in all_rlaifv:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 统一文件
    all_samples = all_rlpr + all_rlaifv
    random.shuffle(all_samples)
    all_path = OUTPUT_DIR / "stage4_train.jsonl"
    with open(all_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nStage 4 数据生成完成")
    print(f"   总样本: {len(all_samples)} 条")
    print(f"   RLPR (GRPO):   {len(all_rlpr)} 条 (60%)")
    print(f"     快思考:       {sum(1 for s in all_rlpr if s['thinking_mode']=='fast')} 条")
    print(f"     深思考:       {sum(1 for s in all_rlpr if s['thinking_mode']=='deep')} 条")
    print(f"   RLAIF-V (DPO): {len(all_rlaifv)} 条 (40%)")
    htype_counts = {}
    for s in all_rlaifv:
        ht = s["hallucination_type"]
        htype_counts[ht] = htype_counts.get(ht, 0) + 1
    for ht, c in sorted(htype_counts.items()):
        print(f"     {ht}: {c} 条")

    stats = {
        "dataset_name": "MiniCPM-o Stage4 RL Dataset",
        "reference_datasets": ["MathVista", "GeoQA", "CLEVR-Math", "RLAIF-V"],
        "total_samples": len(all_samples),
        "rlpr_samples": len(all_rlpr),
        "rlaifv_samples": len(all_rlaifv),
        "rlpr_thinking_modes": {
            "fast": sum(1 for s in all_rlpr if s["thinking_mode"] == "fast"),
            "deep": sum(1 for s in all_rlpr if s["thinking_mode"] == "deep"),
        },
        "rlaif_hallucination_types": htype_counts,
        "algorithm_mapping": {
            "rlpr": "GRPO (Group Relative Policy Optimization)",
            "rlaifv": "DPO (Direct Preference Optimization)"
        }
    }
    with open(OUTPUT_DIR / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    generate_stage4_data()
