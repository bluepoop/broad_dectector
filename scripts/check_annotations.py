import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def check_annotations():
    """检查标注质量并可视化"""

    images_dir = Path("data/images")
    labels_dir = Path("data/labels")

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    if not image_files:
        print("错误：data/images目录中没有找到图片文件！")
        return

    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"

        if not label_file.exists():
            print(f"警告：图片 {img_file.name} 没有对应的标注文件")
            continue

        # 读取图片
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 读取标注
        with open(label_file, 'r') as f:
            lines = f.readlines()

        well_count = 0
        # 绘制边界框
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, box_w, box_h = map(float, parts)

                # 转换为像素坐标
                x_center *= w
                y_center *= h
                box_w *= w
                box_h *= h

                x1 = int(x_center - box_w/2)
                y1 = int(y_center - box_h/2)
                x2 = int(x_center + box_w/2)
                y2 = int(y_center + box_h/2)

                # 绘制矩形框
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, f'well_{well_count}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                well_count += 1

        print(f"{img_file.name}: 检测到 {well_count} 个孔洞")

        # 显示图片
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.title(f"{img_file.name} - {well_count} wells")
        plt.axis('off')
        plt.show()

        # 等待用户确认
        response = input("按回车查看下一张图片，输入'q'退出: ")
        if response.lower() == 'q':
            break

if __name__ == "__main__":
    check_annotations()