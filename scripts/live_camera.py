"""
实时摄像头孔板检测脚本

从摄像头读取视频流，实时检测96孔板并可视化结果。
每隔 N 帧做一次完整检测（YOLO + 网格拟合），其余帧直接叠加上次结果。

操作：
  s     — 保存当前帧的可视化图片到 results/snapshot_<时间戳>.jpg
  q/ESC — 退出
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime

# 把项目根目录加入 path，方便直接运行
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.process_wells import WellProcessor


def draw_wells_on_frame(frame, results):
    """把检测结果叠加到帧上，返回新图像（不修改原帧）"""
    vis = frame.copy()
    img_h, img_w = vis.shape[:2]

    for well in results:
        bbox = well['bbox']
        well_id = well.get('well_id', '?')
        rgb = well.get('rgb', {})
        inferred = well.get('inferred', False)

        x1, y1, x2, y2 = bbox
        x1c = max(0, min(x1, img_w - 1))
        y1c = max(0, min(y1, img_h - 1))
        x2c = max(0, min(x2, img_w - 1))
        y2c = max(0, min(y2, img_h - 1))

        color = (0, 165, 255) if inferred else (0, 255, 0)
        cv2.rectangle(vis, (x1c, y1c), (x2c, y2c), color, 1)

        text_y = max(y1c - 3, 10)
        cv2.putText(vis, well_id, (x1c, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        r, g, b = rgb.get('r'), rgb.get('g'), rgb.get('b')
        if r is not None:
            rgb_text = f"R{r}G{g}B{b}"
            cv2.putText(vis, rgb_text, (x1c, min(y2c + 10, img_h - 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 80, 0), 1, cv2.LINE_AA)

    return vis


def add_status_bar(frame, text, color=(200, 200, 200)):
    """在图像底部加一行状态文字"""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 22), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, text, (6, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='96孔板实时摄像头检测')
    parser.add_argument('--model', default='results/well_detection/weights/best.pt',
                        help='模型路径')
    parser.add_argument('--camera', type=int, default=0,
                        help='摄像头索引（默认0）')
    parser.add_argument('--interval', type=int, default=10,
                        help='每隔多少帧做一次完整检测（默认10）')
    parser.add_argument('--rows', type=int, default=8, help='孔板行数')
    parser.add_argument('--cols', type=int, default=12, help='孔板列数')
    parser.add_argument('--width', type=int, default=0,
                        help='摄像头分辨率宽（0=不设置）')
    parser.add_argument('--height', type=int, default=0,
                        help='摄像头分辨率高（0=不设置）')
    args = parser.parse_args()

    # 切换到项目根目录
    os.chdir(Path(__file__).parent.parent)

    print("加载模型...")
    try:
        processor = WellProcessor(model_path=args.model,
                                  rows=args.rows, cols=args.cols)
    except FileNotFoundError:
        print(f"找不到模型文件: {args.model}")
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.camera}")
        return

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"摄像头已打开，分辨率: "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("按 s 保存快照，按 q 或 ESC 退出")

    os.makedirs('results', exist_ok=True)

    last_results = []
    frame_count = 0
    last_detect_time = ''
    last_well_count = 0
    last_inferred_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取帧失败，退出")
            break

        # 每隔 interval 帧做一次完整检测
        if frame_count % args.interval == 0:
            normalized = processor.normalize_lighting(frame)
            raw = processor.detect_wells_raw(normalized)
            deduped = processor._nms_detections(raw, iou_thresh=0.4)
            grid_params = processor._estimate_grid_params(deduped, frame.shape)

            if grid_params is not None:
                grid_wells = processor._build_grid(grid_params, deduped, frame.shape)
                # 提取颜色
                last_results = []
                for well in grid_wells:
                    color_data = processor.extract_well_color(normalized, well['bbox'])
                    if color_data is None:
                        color_data = {'r': None, 'g': None, 'b': None,
                                      'h': None, 's': None, 'l': None}
                    last_results.append({
                        'well_id': well.get('well_id', '?'),
                        'row': well.get('row', -1),
                        'col': well.get('col', -1),
                        'rgb': {'r': color_data['r'], 'g': color_data['g'], 'b': color_data['b']},
                        'hsl': {'h': color_data['h'], 's': color_data['s'], 'l': color_data['l']},
                        'confidence': well['confidence'],
                        'inferred': well.get('inferred', False),
                        'bbox': well['bbox'],
                    })
                last_well_count = len(last_results)
                last_inferred_count = sum(1 for w in last_results if w['inferred'])
            elif deduped:
                # 检测点太少，只显示原始检测
                last_results = [{
                    'well_id': f'#{i}',
                    'row': -1, 'col': -1,
                    'rgb': {'r': None, 'g': None, 'b': None},
                    'hsl': {'h': None, 's': None, 'l': None},
                    'confidence': d['confidence'],
                    'inferred': False,
                    'bbox': d['bbox'],
                } for i, d in enumerate(deduped)]
                last_well_count = len(last_results)
                last_inferred_count = 0

            last_detect_time = datetime.now().strftime('%H:%M:%S')

        # 叠加可视化
        display = draw_wells_on_frame(frame, last_results)

        # 状态栏
        status = (f"[{last_detect_time}]  孔位: {last_well_count}  "
                  f"推算: {last_inferred_count}  "
                  f"检测间隔: {args.interval}帧  |  s=保存  q=退出")
        add_status_bar(display, status)

        cv2.imshow('96-Well Plate Detector', display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q 或 ESC
            break
        elif key == ord('s'):
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f'results/snapshot_{ts}.jpg'
            cv2.imwrite(save_path, display)
            print(f"快照已保存: {save_path}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("退出")


if __name__ == '__main__':
    main()
