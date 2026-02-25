from datasets import load_dataset

if __name__ == "__main__":
    # 设置自定义下载路径
    custom_path = "/Users/zhangjianbang/Downloads/minicpmo_training/build_data/dataset"
    # 修改脚本中的路径常量
    globals()["CUSTOM_DOWNLOAD_PATH"] = custom_path

    # 加载数据集
    dataset = load_dataset(
        "/build_data/stage1/coco_dataset/coco_download.py",  # 替换为你的脚本文件路径
        name="2014",
        split="train",
        # 也可以通过 dl_manager 配置传递路径
        download_mode="force_redownload"  # 如果需要重新下载，使用这个参数
    )

    # 验证结果
    print(f"数据集样本数量: {len(dataset)}")
    print(f"数据文件下载路径: {custom_path}")
    print(f"第一个样本的图片路径: {dataset[0]['image'].filename}")