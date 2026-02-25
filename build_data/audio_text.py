import os
import json


def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        content=f.read().strip()
        return content


def write_json(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)





if __name__ == '__main__':
    """
    "annotations": [
    {"audio":"/to/audio/path","transcript":"text description"}]
    """
    file_list=[f[:-4] for f in os.listdir('build_data/audio_text') if f.endswith('.txt')]
    print(file_list)
    total_audio_data={"annotations":[]}
    for file in file_list:
        one_data={}
        file_path = os.path.join('build_data/audio_text', file+".txt")
        content = read_txt(file_path)
        audio_path=os.path.join('build_data/audio_text', file+".wav")
        one_data["audio_id"]=file
        one_data["type"]="audio_text"
        one_data["audio"]=audio_path
        one_data["transcript"]=content
        total_audio_data["annotations"].append(one_data)
    print(total_audio_data)
    output_json_path=os.path.join('build_data/audio_text', 'audio_text.json')
    write_json(output_json_path, total_audio_data)


