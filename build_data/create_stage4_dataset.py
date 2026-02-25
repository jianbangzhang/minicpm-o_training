import os
import json


"""
[
  {
        "id": "math_001",
        "task_type": "math  # \"math\" | \"code\" | \"choice\"",
        "thinking_mode": "deep  # RLPR 主要用深思考模式",
        "conversations": [
            {"role": "user", "content": [
                {"type": "image", "path": "..."},
                {"type": "text", "content": "求解..."}
            ]}
        ],
        "ground_truth": "42,# 可验证答案",
        "test_cases": ["# 代码题测试用例（可选）"]
  },
  {
        "id": "hallucination_001",
        "image": "/path/to/image.jpg",
        "question": "Describe the objects in the image.",
        "chosen": "The image shows a red car parked on the street...# 准确回复",
        "rejected": "The image shows a blue car flying in the sky... # 含幻觉的回复",
        "thinking_mode": "fast"
    }
]
"""

def get_data(path):
    total_data=[]
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
        content_list = content["annotations"]
        for data in content_list:
            one_data={}
            image_id = "hallucination_"+data["image_id"]
            image_path =data["image"]
            caption=data["caption"]
            conversations = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "content": "What is shown in this image?"}
                ]},
                {"role": "assistant", "content":caption }
            ]
            one_data["id"]=image_id
            one_data["image"]=image_path
            one_data["question"]="Describe the objects in the image."
            one_data["chosen"] = caption
            one_data["rejected"]=caption[3:]
            total_data.append(one_data)
    return total_data


def write_json(path,data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)



path1=r"build_data/image_text/image_text.json"

image_chat=get_data(path1)
write_json("build_data/chat_stage4/data.json",image_chat)
