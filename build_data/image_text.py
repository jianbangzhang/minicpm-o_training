from datasets import load_dataset
from PIL import Image
import json
import requests
import os
from tqdm import tqdm


"""
https://modelscope.cn/datasets/purshow/cc3m/files
"""



def load_data():
    path = r"build_data/image_text/data.txt"
    data = []
    with open(path, 'r',encoding='utf-8') as file:
        rows_data = file.readlines()
        for row in rows_data:
            url=row.split("\t")[-1]
            text=row.replace(url,"").strip()
            one_data={"image":url,"caption":text}
            data.append(one_data)
    return data




def download_file_with_progress(url, save_path=None):
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        if save_path is None:
            content_disposition = response.headers.get('content-disposition')
            if content_disposition and 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].strip('"\'')
            else:
                filename = url.split('/')[-1].split('?')[0]
            save_path = filename

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=save_path) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"\n下载完成: {save_path}")
        return 1

    except Exception as e:
        print(f"下载失败: {e}")
        return 0





def download_cc3m(output_dir):
    """下载CC3M数据集"""
    dataset = load_data()

    os.makedirs(f"{output_dir}/images", exist_ok=True)
    annotations = []
    for idx, sample in enumerate(dataset):
        if idx >= 1000:  # 限制数量（演示用）
            break

        try:
            # 下载图像
            image_url = sample['image']
            image_path = f"{output_dir}/images/{idx:08d}.jpg"
            sucess=download_file_with_progress(image_url,image_path)
            if sucess==1:
                annotations.append({
                    'image_id': f"{idx:08d}",
                    "type":"image_text",
                    'image': image_path,
                    'caption': sample['caption']
                })
            else:
                continue

        except Exception as e:
            print(f"Error downloading image {idx}: {e}")

    # 保存annotations
    with open(f"{output_dir}/image_text.json", 'w') as f:
        json.dump({'annotations': annotations}, f, indent=2)

    print(f"Total downloaded: {len(annotations)} images")


if __name__ == "__main__":
    download_cc3m("build_data/image_text")
