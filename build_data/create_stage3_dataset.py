import os
import json


"""
[
  {
    "id": "sample_001",
    "type": "vision_qa",
    "thinking_mode": "fast",
    "conversations": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "path": "/path/to/img.jpg"},
            {"type": "text", "content": "What is shown in this image?"}
        ]},
        {"role": "assistant", "content": "The image shows..."}
    ]
  },
  {
    "id": "sample_002",
    "type": "vision_qa",
    "thinking_mode": "deep",
    "conversations": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "path": "/path/to/img.jpg"},
            {"type": "text", "content": "What is shown in this image?"}
        ]},
        {"role": "assistant", "content": "The image shows...","think": "xxxx"}
    ]
  },
  {
    "id": "sample_003",
    "type": "speech_dialog",
    "thinking_mode": "deep",
    "conversations": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "audio", "path": "/path/to/audio"},
            {"type": "text", "content": "What is shown in this audio?"}
        ]},
        {"role": "assistant", "content": "The audio shows..."}
    ]
  }
]
"""

def get_data(path,is_image=True):
    total_data=[]
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
        content_list = content["annotations"]
        for data in content_list:
            if is_image:
                one_data={}
                image_id = data["image_id"]
                type = "vision_qa"
                thinking_mode = "fast"
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
                one_data["type"]=type
                one_data["thinking_mode"] = thinking_mode
                one_data["conversations"]=conversations
                total_data.append(one_data)
            else:
                one_data = {}
                audio_id = data["audio_id"]
                type = "speech_dialog"
                thinking_mode = "deep"
                audio_path = data["audio"]
                text = data["transcript"]
                conversations = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        {"type": "image", "path": audio_path},
                        {"type": "text", "content": "What is shown in this audio?"}
                    ]},
                    {"role": "assistant", "content": text}
                ]
                one_data["id"] = audio_id
                one_data["type"] = type
                one_data["thinking_mode"] = thinking_mode
                one_data["conversations"] = conversations
                total_data.append(one_data)
    return total_data


def write_json(path,data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)



path1=r"build_data/image_text/image_text.json"
path2=r"build_data/audio_text/audio_text.json"

image_chat=get_data(path1,is_image=True)
audio_chat=get_data(path2,is_image=False)

total_data=image_chat+audio_chat
write_json("build_data/chat_stage3/chat.json",total_data)
