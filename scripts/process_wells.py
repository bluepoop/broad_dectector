import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
import json
import colorsys
import matplotlib.pyplot as plt
from datetime import datetime


class WellProcessor:
    def __init__(self, model_path='models/well_detector.pt',
                 rows=8, cols=12):
        """
        初始化孔洞处理器

        rows: 96孔板行数（默认8，对应A-H）
        cols: 96孔板列数（默认12，对应1-12）
        """
        self.model_path = model_path
        self.rows = rows
        self.cols = cols
        self.model = None
        self.load_model()

    def load_model(self):
        """加载训练好的YOLO模型"""
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            print(f"模型加载成功: {self.model_path}")
        else:
            print(f"错误：模型文件 {self.model_path} 不存在！")
            raise FileNotFoundError

    def normalize_lighting(self, image):
        """简单的光照标准化"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_normalized = clahe.apply(l)
        lab_normalized = cv2.merge([l_normalized, a, b])
        return cv2.cvtColor(lab_normalized, cv2.COLOR_Lab2BGR)

    def detect_wells_raw(self, image):
        """使用YOLO检测孔洞，返回原始检测结果（中心点+尺寸）"""
        results = self.model(image, conf=0.15, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    detections.append({
                        'cx': cx, 'cy': cy,
                        'w': w, 'h': h,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf
                    })
        return detections

    # ------------------------------------------------------------------
    # 网格拟合核心逻辑
    # ------------------------------------------------------------------

    def _nms_detections(self, detections, iou_thresh=0.4):
        """对原始检测做NMS去重"""
        if not detections:
            return []
        boxes = np.array([[d['bbox'][0], d['bbox'][1],
                           d['bbox'][2], d['bbox'][3]] for d in detections], dtype=np.float32)
        scores = np.array([d['confidence'] for d in detections], dtype=np.float32)

        # 手动NMS
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou < iou_thresh]
        return [detections[k] for k in keep]

    def _estimate_grid_params(self, detections, image_shape):
        """
        从检测到的孔位估算网格参数。
        用自相关法找周期性间距，再用圆形均值法估算原点。
        """
        if len(detections) < 2:
            return None

        cxs = np.array([d['cx'] for d in detections])
        cys = np.array([d['cy'] for d in detections])
        ws = np.array([d['w'] for d in detections])
        hs = np.array([d['h'] for d in detections])

        well_w = float(np.median(ws))
        well_h = float(np.median(hs))

        img_h, img_w = image_shape[:2]

        # 用自相关法估算间距，最小间距取孔径的 0.8 倍
        min_dx = max(well_w * 0.8, 20.0)
        min_dy = max(well_h * 0.8, 20.0)
        dx = self._estimate_spacing_autocorr(cxs, img_w, self.cols, min_spacing=min_dx)
        dy = self._estimate_spacing_autocorr(cys, img_h, self.rows, min_spacing=min_dy)

        x0 = self._estimate_origin(cxs, dx, self.cols)
        y0 = self._estimate_origin(cys, dy, self.rows)

        return {
            'x0': x0, 'y0': y0,
            'dx': dx, 'dy': dy,
            'well_w': well_w, 'well_h': well_h,
        }

    def _estimate_spacing_autocorr(self, coords, img_size, n_slots, min_spacing=50.0):
        """
        用自相关找周期性间距。
        同时检查是否有更小的谐波（1/2, 1/3）也是显著峰，优先选更小的间距。
        """
        if len(coords) < 3:
            return img_size / n_slots

        bin_size = max(1, int(img_size / 2000))  # 自适应精度
        n_bins = int(img_size / bin_size) + 2
        hist = np.zeros(n_bins)
        for c in coords:
            idx = int(c / bin_size)
            if 0 <= idx < n_bins:
                hist[idx] += 1

        autocorr = np.correlate(hist, hist, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr[0] = 0

        min_lag = max(1, int(min_spacing / bin_size))
        max_lag = min(int(img_size / n_slots * 2.5 / bin_size), len(autocorr) - 1)

        if min_lag >= max_lag:
            return img_size / n_slots

        search = autocorr[min_lag:max_lag]
        best_lag = int(np.argmax(search)) + min_lag
        best_val = autocorr[best_lag]

        # 检查谐波：如果 best_lag/2 或 best_lag/3 附近也有显著峰，选更小的
        for divisor in [2, 3]:
            candidate_lag = best_lag // divisor
            if candidate_lag < min_lag:
                continue
            window = max(1, candidate_lag // 10)
            lo = max(min_lag, candidate_lag - window)
            hi = min(len(autocorr) - 1, candidate_lag + window)
            local_max = float(autocorr[lo:hi + 1].max())
            if local_max > best_val * 0.5:
                best_lag = lo + int(np.argmax(autocorr[lo:hi + 1]))
                best_val = local_max
                break

        return float(best_lag * bin_size)

    def _cluster_1d(self, coords, gap_thresh):
        """对一维坐标做简单的间隙聚类，返回每个簇的中心列表。"""
        if len(coords) == 0:
            return []
        sorted_c = np.sort(coords)
        clusters = [[sorted_c[0]]]
        for c in sorted_c[1:]:
            if c - clusters[-1][-1] > gap_thresh:
                clusters.append([c])
            else:
                clusters[-1].append(c)
        return [float(np.mean(cl)) for cl in clusters]

    def _estimate_origin(self, coords, spacing, n_slots):
        """
        给定一维坐标和间距，估算网格起点（第0个槽的坐标）。

        策略：用圆形均值法得到相位，再枚举候选原点（相差一个间距的几个），
        选择让最多检测点对齐到网格的那个。
        """
        phases = coords % spacing
        angles = phases / spacing * 2 * np.pi
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        mean_angle = np.arctan2(sin_mean, cos_mean)
        if mean_angle < 0:
            mean_angle += 2 * np.pi
        phase = mean_angle / (2 * np.pi) * spacing

        c_min = float(coords.min())
        offset = (c_min - phase) % spacing
        x0_base = c_min - offset

        # 枚举 x0_base ± 2个间距的候选，但限制原点不能比最小检测点偏左超过一个间距
        snap_thresh = spacing * 0.4
        best_x0 = x0_base
        best_score = -1

        for shift in [-1, 0, 1, 2]:
            x0_cand = x0_base + shift * spacing
            # 原点不能比最小检测点偏左超过一个间距
            if x0_cand < c_min - spacing:
                continue
            score = 0
            for c in coords:
                k = round((c - x0_cand) / spacing)
                if 0 <= k < n_slots:
                    dist = abs(c - (x0_cand + k * spacing))
                    if dist < snap_thresh:
                        score += 1
            if score > best_score:
                best_score = score
                best_x0 = x0_cand

        return float(best_x0)

    def _build_grid(self, grid_params, detections, image_shape):
        """
        根据网格参数构建完整的 rows×cols 方阵。
        对每个网格位置：
          - 如果附近有检测框，使用检测框的位置和置信度
          - 否则用网格推算位置，置信度=0（空孔/漏检）
        返回 list of dict，每项包含 well_id, bbox, confidence, row, col
        """
        x0 = grid_params['x0']
        y0 = grid_params['y0']
        dx = grid_params['dx']
        dy = grid_params['dy']
        well_w = grid_params['well_w']
        well_h = grid_params['well_h']

        img_h, img_w = image_shape[:2]

        # 为每个检测框建立索引（按最近网格位置）
        det_used = [False] * len(detections)

        row_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        results = []

        for row_idx in range(self.rows):
            for col_idx in range(self.cols):
                # 该网格位置的理论中心
                cx_theory = x0 + col_idx * dx
                cy_theory = y0 + row_idx * dy

                # 不过滤超出边界的孔（用户要求输出完整方阵）
                # 超出边界的孔会在颜色提取时被裁剪处理

                # 找最近的未使用检测框
                best_idx = None
                best_dist = float('inf')
                snap_thresh = min(dx, dy) * 0.5  # 吸附半径

                for k, det in enumerate(detections):
                    if det_used[k]:
                        continue
                    dist = np.sqrt((det['cx'] - cx_theory) ** 2 +
                                   (det['cy'] - cy_theory) ** 2)
                    if dist < snap_thresh and dist < best_dist:
                        best_dist = dist
                        best_idx = k

                if best_idx is not None:
                    det = detections[best_idx]
                    det_used[best_idx] = True
                    cx, cy = det['cx'], det['cy']
                    bw, bh = det['w'], det['h']
                    conf = det['confidence']
                else:
                    # 漏检：用理论位置
                    cx, cy = cx_theory, cy_theory
                    bw, bh = well_w, well_h
                    conf = 0.0

                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)
                x2 = int(cx + bw / 2)
                y2 = int(cy + bh / 2)

                well_id = f"{row_letters[row_idx]}{col_idx + 1}"
                results.append({
                    'well_id': well_id,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'row': row_idx,
                    'col': col_idx,
                    'inferred': best_idx is None,  # True表示是推算出来的
                })

        return results

    def detect_wells(self, image):
        """完整检测流程：YOLO → NMS → 网格拟合 → 补全方阵"""
        raw = self.detect_wells_raw(image)
        print(f"  YOLO原始检测: {len(raw)} 个")

        deduped = self._nms_detections(raw, iou_thresh=0.4)
        print(f"  NMS后: {len(deduped)} 个")

        grid_params = self._estimate_grid_params(deduped, image.shape)

        if grid_params is None:
            print("  警告：检测点太少，无法拟合网格，使用原始检测结果")
            # 退化：直接返回NMS结果
            return [{'bbox': d['bbox'], 'confidence': d['confidence'],
                     'inferred': False} for d in deduped]

        print(f"  网格参数: dx={grid_params['dx']:.1f}, dy={grid_params['dy']:.1f}, "
              f"origin=({grid_params['x0']:.1f}, {grid_params['y0']:.1f})")

        grid_wells = self._build_grid(grid_params, deduped, image.shape)
        inferred = sum(1 for w in grid_wells if w['inferred'])
        print(f"  网格补全: {len(grid_wells)} 个孔 (其中推算补全 {inferred} 个)")

        return grid_wells

    def extract_well_color(self, image, bbox):
        """提取单个孔洞的RGB和HSL值（取中心圆形区域）"""
        x1, y1, x2, y2 = bbox
        # 边界裁剪
        h_img, w_img = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)

        well_roi = image[y1:y2, x1:x2]
        if well_roi.size == 0:
            return None

        h, w = well_roi.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 3

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

        mask_area = np.count_nonzero(mask)
        if mask_area == 0:
            return None

        mask_f = mask.astype(np.float32) / 255.0
        b_mean = float(np.sum(well_roi[:, :, 0] * mask_f) / mask_area)
        g_mean = float(np.sum(well_roi[:, :, 1] * mask_f) / mask_area)
        r_mean = float(np.sum(well_roi[:, :, 2] * mask_f) / mask_area)

        # 转换为HSL
        r_n, g_n, b_n = r_mean / 255.0, g_mean / 255.0, b_mean / 255.0
        h_hsl, l_hsl, s_hsl = colorsys.rgb_to_hls(r_n, g_n, b_n)

        return {
            'r': int(round(r_mean)),
            'g': int(round(g_mean)),
            'b': int(round(b_mean)),
            'h': round(h_hsl * 360, 1),
            's': round(s_hsl * 100, 1),
            'l': round(l_hsl * 100, 1),
        }

    def process_image(self, image_path, save_results=True):
        """处理单张图片，返回所有孔洞的颜色数据"""
        print(f"处理图片: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"错误：无法读取图片 {image_path}")
            return None

        normalized_image = self.normalize_lighting(image)

        well_coords = self.detect_wells(normalized_image)
        print(f"最终孔位数: {len(well_coords)}")

        if not well_coords:
            print("未检测到任何孔洞")
            return None

        results = []
        for well in well_coords:
            color_data = self.extract_well_color(normalized_image, well['bbox'])
            if color_data is None:
                # bbox 完全在图像外，仍然保留孔位，颜色置为 None
                color_data = {'r': None, 'g': None, 'b': None,
                              'h': None, 's': None, 'l': None}

            well_result = {
                'well_id': well.get('well_id', '?'),
                'row': well.get('row', -1),
                'col': well.get('col', -1),
                'rgb': {'r': color_data['r'], 'g': color_data['g'], 'b': color_data['b']},
                'hsl': {'h': color_data['h'], 's': color_data['s'], 'l': color_data['l']},
                'confidence': well['confidence'],
                'inferred': well.get('inferred', False),
                'bbox': well['bbox'],
            }
            results.append(well_result)

        if save_results:
            self.save_results(image_path, results, normalized_image)

        return results

    def save_results(self, image_path, results, processed_image):
        """保存处理结果"""
        os.makedirs('results', exist_ok=True)

        image_name = Path(image_path).stem
        json_path = f"results/{image_name}_results.json"

        result_data = {
            'image_path': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'total_wells': len(results),
            'wells': results,
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"结果已保存到: {json_path}")
        self.save_visualization(processed_image, results,
                                f"results/{image_name}_visualization.jpg")
        self.save_excel(results, f"results/{image_name}_results.xlsx", image_name)

    def save_excel(self, results, output_path, image_name=''):
        """保存 Excel 表格：8行×12列的孔板布局，每个孔显示 RGB 和 HSL 值"""
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        wb = openpyxl.Workbook()

        # --- Sheet 1: 孔板布局（RGB 颜色填充） ---
        ws_layout = wb.active
        ws_layout.title = 'RGB布局'

        row_letters = 'ABCDEFGH'
        thin = Side(style='thin', color='AAAAAA')
        border = Border(left=thin, right=thin, top=thin, bottom=thin)

        # 标题行
        ws_layout.cell(1, 1, image_name).font = Font(bold=True, size=12)
        ws_layout.merge_cells(start_row=1, start_column=1, end_row=1, end_column=13)

        # 列标题 1-12
        for col in range(1, 13):
            c = ws_layout.cell(2, col + 1, col)
            c.font = Font(bold=True)
            c.alignment = Alignment(horizontal='center')
            ws_layout.column_dimensions[get_column_letter(col + 1)].width = 12

        ws_layout.column_dimensions['A'].width = 4

        # 按 well_id 建索引
        well_map = {w['well_id']: w for w in results}

        for r_idx, r_letter in enumerate(row_letters):
            row_excel = r_idx + 3  # 从第3行开始
            ws_layout.cell(row_excel, 1, r_letter).font = Font(bold=True)
            ws_layout.cell(row_excel, 1).alignment = Alignment(horizontal='center')
            ws_layout.row_dimensions[row_excel].height = 40

            for col in range(1, 13):
                well_id = f"{r_letter}{col}"
                cell = ws_layout.cell(row_excel, col + 1)
                cell.border = border
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

                well = well_map.get(well_id)
                if well and well['rgb']['r'] is not None:
                    r = well['rgb']['r']
                    g = well['rgb']['g']
                    b = well['rgb']['b']
                    h = well['hsl']['h']
                    s = well['hsl']['s']
                    l = well['hsl']['l']
                    inferred = well.get('inferred', False)

                    # 用孔的实际颜色填充单元格背景
                    hex_color = f'{r:02X}{g:02X}{b:02X}'
                    cell.fill = PatternFill(fill_type='solid', fgColor=hex_color)

                    # 文字颜色：亮色背景用黑字，暗色背景用白字
                    luminance = 0.299 * r + 0.587 * g + 0.114 * b
                    font_color = '000000' if luminance > 128 else 'FFFFFF'
                    cell.value = f"R{r} G{g} B{b}\nH{h:.0f} S{s:.0f} L{l:.0f}"
                    cell.font = Font(size=7, color=font_color,
                                     italic=inferred)  # 推算孔用斜体
                else:
                    cell.value = well_id if well else '—'
                    cell.font = Font(size=8, color='999999')

        # --- Sheet 2: 数据表 ---
        ws_data = wb.create_sheet('数据表')
        headers = ['孔位', '行', '列', 'R', 'G', 'B', 'H(°)', 'S(%)', 'L(%)',
                   '置信度', '是否推算']
        for col, h in enumerate(headers, 1):
            c = ws_data.cell(1, col, h)
            c.font = Font(bold=True)
            c.fill = PatternFill(fill_type='solid', fgColor='4472C4')
            c.font = Font(bold=True, color='FFFFFF')
            c.alignment = Alignment(horizontal='center')

        for row_idx, well in enumerate(sorted(results, key=lambda x: (x['row'], x['col'])), 2):
            rgb = well['rgb']
            hsl = well['hsl']
            row_data = [
                well['well_id'],
                well['row'],
                well['col'],
                rgb['r'], rgb['g'], rgb['b'],
                hsl['h'], hsl['s'], hsl['l'],
                round(well['confidence'], 3),
                '是' if well.get('inferred') else '否',
            ]
            for col, val in enumerate(row_data, 1):
                c = ws_data.cell(row_idx, col, val)
                c.alignment = Alignment(horizontal='center')
                # 给有颜色数据的行填充背景色
                if rgb['r'] is not None and col in (4, 5, 6):
                    r, g, b = rgb['r'], rgb['g'], rgb['b']
                    hex_color = f'{r:02X}{g:02X}{b:02X}'
                    c.fill = PatternFill(fill_type='solid', fgColor=hex_color)
                    luminance = 0.299 * r + 0.587 * g + 0.114 * b
                    font_color = '000000' if luminance > 128 else 'FFFFFF'
                    c.font = Font(color=font_color)

        # 自动列宽
        for col in ws_data.columns:
            max_len = max((len(str(c.value)) for c in col if c.value), default=8)
            ws_data.column_dimensions[get_column_letter(col[0].column)].width = max_len + 2

        wb.save(output_path)
        print(f"Excel 已保存到: {output_path}")

    def save_visualization(self, image, results, output_path):
        """保存可视化结果"""
        vis_image = image.copy()

        for well in results:
            bbox = well['bbox']
            well_id = well['well_id']
            rgb = well['rgb']
            inferred = well.get('inferred', False)

            x1, y1, x2, y2 = bbox
            img_h_v, img_w_v = vis_image.shape[:2]
            # 裁剪到图像范围内再绘制
            x1c = max(0, min(x1, img_w_v - 1))
            y1c = max(0, min(y1, img_h_v - 1))
            x2c = max(0, min(x2, img_w_v - 1))
            y2c = max(0, min(y2, img_h_v - 1))

            # 推算补全的孔用橙色框，正常检测用绿色框
            color = (0, 165, 255) if inferred else (0, 255, 0)
            cv2.rectangle(vis_image, (x1c, y1c), (x2c, y2c), color, 2)

            text_y = max(y1c - 4, 12)
            cv2.putText(vis_image, well_id, (x1c, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            rgb_text = f"R{rgb['r']}G{rgb['g']}B{rgb['b']}" if rgb['r'] is not None else "N/A"
            cv2.putText(vis_image, rgb_text, (x1c, min(y2c + 12, img_h_v - 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 80, 0), 1)

        cv2.imwrite(output_path, vis_image)
        print(f"可视化图片已保存到: {output_path}")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='96孔板颜色提取')
    parser.add_argument('--model', default='results/well_detection/weights/best.pt',
                        help='模型路径')
    parser.add_argument('--input', default='test_img', help='输入图片目录')
    parser.add_argument('--rows', type=int, default=8, help='孔板行数')
    parser.add_argument('--cols', type=int, default=12, help='孔板列数')
    args = parser.parse_args()

    processor = WellProcessor(model_path=args.model, rows=args.rows, cols=args.cols)

    test_dir = Path(args.input)
    if not test_dir.exists():
        print(f"错误：{args.input} 目录不存在")
        return

    image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
    if not image_files:
        print(f"{args.input} 目录中没有找到图片文件")
        return

    all_results = []
    for img_file in sorted(image_files):
        try:
            results = processor.process_image(img_file)
            if results:
                all_results.extend(results)
        except Exception as e:
            print(f"处理图片 {img_file} 时出错: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n处理完成！共处理 {len(image_files)} 张图片，检测 {len(all_results)} 个孔洞")

    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'total_images': len(image_files),
        'total_wells': len(all_results),
        'all_results': all_results,
    }
    os.makedirs('results', exist_ok=True)
    with open('results/summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print("汇总结果已保存到 results/summary.json")


if __name__ == '__main__':
    main()
