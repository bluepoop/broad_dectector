"""
数据增强脚本 — 针对颜色多样性不足的问题
对 data/images + data/labels 中的标注图做颜色增强，生成更多训练样本。

增强策略（全部保留原始标注框，只改变颜色/亮度）：
  1. HSV 色调旋转（全色谱覆盖）
  2. 饱和度缩放
  3. 亮度缩放
  4. 颜色通道 shuffle（模拟不同染料颜色）
  5. 灰度化（模拟空孔/低浓度）
  6. 组合增强

输出到 data/images_aug 和 data/labels_aug，
然后可以把这两个目录的内容合并进 data/images 和 data/labels 再跑 prepare_dataset.py。
"""

import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import random


def augment_hsv_hue(image, hue_shift):
    """色调旋转"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_hsv_sat(image, sat_scale):
    """饱和度缩放"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_brightness(image, val_scale):
    """亮度缩放"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_channel_shuffle(image, order):
    """通道重排（BGR顺序变换，模拟不同颜色染料）"""
    return image[:, :, order]


def augment_grayscale(image):
    """灰度化（模拟空孔）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def augment_color_tint(image, tint_bgr, strength=0.3):
    """叠加颜色色调（模拟不同染料）"""
    tint = np.full_like(image, tint_bgr, dtype=np.float32)
    result = cv2.addWeighted(image.astype(np.float32), 1 - strength,
                              tint, strength, 0)
    return np.clip(result, 0, 255).astype(np.uint8)


def get_augmentations():
    """返回所有增强变换列表，每项为 (suffix, transform_fn)"""
    augmentations = []

    # 色调旋转：每30度一个，覆盖全色谱（0-180 in HSV）
    for hue in range(15, 180, 15):
        augmentations.append((
            f'hue{hue:03d}',
            lambda img, h=hue: augment_hsv_hue(img, h)
        ))

    # 饱和度变化
    for sat in [0.3, 0.5, 0.7, 1.3, 1.6, 2.0]:
        augmentations.append((
            f'sat{int(sat*10):02d}',
            lambda img, s=sat: augment_hsv_sat(img, s)
        ))

    # 亮度变化
    for val in [0.5, 0.7, 1.3, 1.5]:
        augmentations.append((
            f'val{int(val*10):02d}',
            lambda img, v=val: augment_brightness(img, v)
        ))

    # 通道重排（6种排列中取有意义的几种）
    channel_orders = [
        ([2, 1, 0], 'ch_rgb'),   # BGR->RGB
        ([1, 0, 2], 'ch_grb'),   # GRB
        ([0, 2, 1], 'ch_brg'),   # BRG
        ([2, 0, 1], 'ch_rbg'),   # RBG
        ([1, 2, 0], 'ch_gbr'),   # GBR
    ]
    for order, name in channel_orders:
        augmentations.append((
            name,
            lambda img, o=order: augment_channel_shuffle(img, o)
        ))

    # 颜色叠加（模拟不同染料颜色）
    tints = [
        ([255, 0, 0], 'tint_blue'),
        ([0, 255, 0], 'tint_green'),
        ([0, 0, 255], 'tint_red'),
        ([255, 255, 0], 'tint_cyan'),
        ([0, 255, 255], 'tint_yellow'),
        ([255, 0, 255], 'tint_magenta'),
        ([128, 0, 128], 'tint_purple'),
        ([0, 128, 255], 'tint_orange'),
    ]
    for bgr, name in tints:
        for strength in [0.25, 0.5]:
            augmentations.append((
                f'{name}_s{int(strength*10)}',
                lambda img, b=bgr, s=strength: augment_color_tint(img, b, s)
            ))

    # 灰度
    augmentations.append(('gray', augment_grayscale))

    # 组合：色调旋转 + 亮度
    for hue in [30, 90, 150]:
        for val in [0.7, 1.3]:
            augmentations.append((
                f'hue{hue:03d}_val{int(val*10)}',
                lambda img, h=hue, v=val: augment_brightness(augment_hsv_hue(img, h), v)
            ))

    return augmentations


def process_dataset(src_images_dir, src_labels_dir, dst_images_dir, dst_labels_dir):
    src_images_dir = Path(src_images_dir)
    src_labels_dir = Path(src_labels_dir)
    dst_images_dir = Path(dst_images_dir)
    dst_labels_dir = Path(dst_labels_dir)

    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(src_images_dir.glob('*.jpg')) + list(src_images_dir.glob('*.png'))
    if not image_files:
        print(f"错误：{src_images_dir} 中没有图片")
        return

    augmentations = get_augmentations()
    print(f"共 {len(augmentations)} 种增强变换")
    print(f"原始图片数: {len(image_files)}")
    print(f"预计生成: {len(image_files) * len(augmentations)} 张增强图片")

    total_generated = 0

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"跳过无法读取的图片: {img_path}")
            continue

        label_path = src_labels_dir / f'{img_path.stem}.txt'
        if not label_path.exists():
            print(f"警告：未找到标注文件 {label_path}，跳过")
            continue

        label_content = label_path.read_text()

        for suffix, transform_fn in augmentations:
            try:
                aug_image = transform_fn(image)
            except Exception as e:
                print(f"增强 {suffix} 失败 ({img_path.name}): {e}")
                continue

            out_name = f'{img_path.stem}_aug_{suffix}'
            out_img_path = dst_images_dir / f'{out_name}.jpg'
            out_label_path = dst_labels_dir / f'{out_name}.txt'

            cv2.imwrite(str(out_img_path), aug_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            out_label_path.write_text(label_content)
            total_generated += 1

    print(f"\n增强完成！生成了 {total_generated} 张图片")
    print(f"图片保存到: {dst_images_dir}")
    print(f"标注保存到: {dst_labels_dir}")


def merge_into_dataset(aug_images_dir, aug_labels_dir, dst_images_dir, dst_labels_dir):
    """将增强数据合并到目标目录"""
    aug_images_dir = Path(aug_images_dir)
    aug_labels_dir = Path(aug_labels_dir)
    dst_images_dir = Path(dst_images_dir)
    dst_labels_dir = Path(dst_labels_dir)

    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_file in aug_images_dir.glob('*.jpg'):
        shutil.copy(img_file, dst_images_dir / img_file.name)
        label_file = aug_labels_dir / f'{img_file.stem}.txt'
        if label_file.exists():
            shutil.copy(label_file, dst_labels_dir / label_file.name)
        count += 1

    print(f"合并了 {count} 张增强图片到 {dst_images_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='96孔板数据颜色增强')
    parser.add_argument('--src-images', default='data/images', help='原始图片目录')
    parser.add_argument('--src-labels', default='data/labels', help='原始标注目录')
    parser.add_argument('--dst-images', default='data/images_aug', help='增强图片输出目录')
    parser.add_argument('--dst-labels', default='data/labels_aug', help='增强标注输出目录')
    parser.add_argument('--merge', action='store_true',
                        help='增强后自动合并到 data/images 和 data/labels')
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)

    process_dataset(args.src_images, args.src_labels, args.dst_images, args.dst_labels)

    if args.merge:
        print("\n合并增强数据到原始目录...")
        merge_into_dataset(args.dst_images, args.dst_labels,
                           args.src_images, args.src_labels)
        print("合并完成，可以运行 prepare_dataset.py 重新划分数据集")
