import os
import json





def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        content=f.read().strip()
        return content


def write_json(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)


def is_remove(text):
    for line in text.split('\n'):
        text = line.split(',')[-1]
        if text.strip() != '###':
            return False
    return True



def get_text(text):
    words=""
    for line in text.split('\n'):
        text = line.split(',')[-1]
        if text.strip() != '###':
            words+=" "+text.strip()
        else:
            continue
    return words.strip()



def bbox_points(coords):
    coords = [int(p) for p in coords]
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

    # 分离 x 和 y
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    # 计算 bbox
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    return [x1, y1, x2, y2]



def get_bbox(text):
    total_bbox=[]
    for line in text.split('\n'):
        text = line.split(',')[-1]
        if text.strip() != '###':
            num_list = line.split(',')[:-1]
            bbox = bbox_points(num_list)
            total_bbox.append(bbox)
        else:
            continue
    return total_bbox



if __name__ == '__main__':
    """
    "annotations": [
      {
        "type": "ocr_document",
        "image": "/path/to/doc.png",
        "ocr_text": "文档完整文字内容",
        "text_boxes": [["x0","y0","x1","y1"]],
        "noise_level": null
  }]
    """
    file_list=[f[:-4].replace("gt_","").strip() for f in os.listdir('build_data/ocr_data/Challenge4_Test_Task1_GT') if f.endswith('.txt')]
    print(file_list)
    total_audio_data={"annotations":[]}
    for file in file_list:
        one_data={}
        text_path = os.path.join('build_data/ocr_data/Challenge4_Test_Task1_GT', 'gt_'+file+".txt")
        content = read_txt(text_path)
        if is_remove(content):
            continue
        try:
            text= get_text(content)
            bbox = get_bbox(content)
        except:
            continue
        image_path=os.path.join('build_data/ocr_data/ch4_test_images', file+".jpg")
        one_data["image_id"]=file
        one_data["type"]="ocr_document"
        one_data["image"]=image_path
        one_data["ocr_text"]=text
        one_data["text_boxes"]=bbox
        one_data["noise_level"]="null"
        total_audio_data["annotations"].append(one_data)
    print(total_audio_data)
    output_json_path=os.path.join('build_data/ocr_data', 'ocr_text.json')
    write_json(output_json_path, total_audio_data)


