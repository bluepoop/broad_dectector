# 孔洞标注指南

## 1. 安装标注工具
推荐使用 **LabelImg** 或 **YOLOv8自带的标注工具**：
```bash
pip install labelImg
# 或者使用在线工具：https://roboflow.com
```

## 2. 标注方法

### 使用LabelImg:
1. 启动LabelImg：`labelImg`
2. 点击 "Open Dir" 选择 `data/images` 文件夹
3. 点击 "Change Save Dir" 选择 `data/labels` 文件夹
4. 选择 "YOLO" 格式
5. 对每个孔洞画矩形框（包括空孔）
6. 标签选择 "well"
7. 保存（快捷键Ctrl+S）

### 标注要点：
- **所有圆形孔洞都要标记**（不管是否有颜色）
- 矩形框要尽可能贴合孔洞边缘
- 空孔也必须标记
- 确保不要漏掉边缘的孔洞

### 文件格式：
- 图片放在 `data/images/`
- 标注文件会自动保存到 `data/labels/`
- 每张图片对应一个 `.txt` 文件

## 3. 检查标注质量
运行以下脚本检查标注：
```bash
python scripts/check_annotations.py
```

## 4. 数据集划分
标注完成后，运行：
```bash
python scripts/prepare_dataset.py
```
这会自动将数据按8:2划分到train和val文件夹。