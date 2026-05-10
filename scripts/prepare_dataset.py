import os
import shutil
import random
from pathlib import Path

def prepare_dataset():
    """将标注好的数据划分为训练集和验证集"""

    # 源文件路径
    images_dir = Path("data/images")
    labels_dir = Path("data/labels")

    # 目标路径
    train_images = Path("datasets/train/images")
    train_labels = Path("datasets/train/labels")
    val_images = Path("datasets/val/images")
    val_labels = Path("datasets/val/labels")

    # 确保目录存在
    for path in [train_images, train_labels, val_images, val_labels]:
        path.mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    if not image_files:
        print("错误：data/images目录中没有找到图片文件！")
        return

    # 随机打乱
    random.shuffle(image_files)

    # 8:2划分
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # 复制文件
    def copy_files(file_list, img_dest, label_dest):
        for img_file in file_list:
            # 复制图片
            shutil.copy(img_file, img_dest / img_file.name)

            # 复制对应的标注文件
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy(label_file, label_dest / label_file.name)
            else:
                print(f"警告：未找到标注文件 {label_file}")

    copy_files(train_files, train_images, train_labels)
    copy_files(val_files, val_images, val_labels)

    print(f"数据集划分完成:")
    print(f"训练集: {len(train_files)} 张图片")
    print(f"验证集: {len(val_files)} 张图片")

if __name__ == "__main__":
    prepare_dataset()