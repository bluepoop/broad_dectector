# 阵列板孔洞RGB检测系统

## 项目概述
使用YOLO检测96孔板中的所有孔洞并提取RGB颜色值，用于荧光染料浓度分析。

## 完整流程指南

### 1. 环境搭建

#### 当前电脑操作：
```bash
# 安装Python依赖
pip install -r requirements.txt

# 验证安装
python -c "import cv2, ultralytics; print('环境安装成功')"
```

#### 跨电脑使用（U盘传输）：
1. **准备U盘文件夹结构：**
```
well_detection_project/
├── requirements.txt
├── data_config.yaml
├── scripts/
├── 当前文件夹的所有内容
└── README.md
```

2. **在训练电脑上：**
```bash
# 1. 插入U盘并进入项目目录
cd /path/to/usb/well_detection_project

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证安装
python -c "import cv2, ultralytics; print('训练环境准备完成')"
```

### 2. 数据准备和标注

#### 复制图片到data目录：
```bash
# 将test_img中的图片复制到data/images
cp test_img/*.jpg data/images/
```

#### 标注孔洞：
1. **安装LabelImg：**
```bash
pip install labelImg
```

2. **启动标注工具：**
```bash
labelImg
```

3. **标注步骤：**
   - Open Dir → 选择 `data/images`
   - Change Save Dir → 选择 `data/labels`
   - 选择 "YOLO" 格式
   - 对每个孔洞（包括空孔）画矩形框
   - 标签选择 "well"
   - 快捷键：w(画框), d(下一张), a(上一张), Ctrl+S(保存)

4. **检查标注质量：**
```bash
python scripts/check_annotations.py
```

5. **准备训练数据集：**
```bash
python scripts/prepare_dataset.py
```

### 3. 模型训练

#### CPU训练（适用于所有电脑）：
```bash
python scripts/train_yolo.py
```

**训练参数说明：**
- 训练轮数：100轮（可根据需要调整）
- 图片尺寸：640x640
- 批量大小：自动调整
- 设备：CPU
- 早停机制：20轮无改善停止

**预期训练时间：**
- 6张图片约需2-4小时（CPU）
- 训练过程会自动保存检查点

**训练完成标志：**
```
Training complete!
最佳模型保存在: runs/well_detection/weights/best.pt
```

### 4. 模型使用

#### 处理图片提取RGB：
```bash
python scripts/process_wells.py
```

**输出结果：**
- `results/` 目录包含所有结果
- `*_results.json`: 每张图片的详细数据
- `*_visualization.jpg`: 可视化结果
- `summary.json`: 汇总数据

#### 结果分析：
```bash
python scripts/utils.py
```

**生成文件：**
- `rgb_analysis.xlsx`: Excel分析报告
- `rgb_heatmap.png`: RGB热图

### 5. 跨电脑操作详细步骤

#### Step 1: 准备U盘
```bash
# 在当前电脑复制整个项目到U盘
cp -r cv2_test /path/to/usb/well_detection_project

# 确保包含：
# - 所有脚本文件
# - requirements.txt
# - 已标注的数据（data/images和data/labels）
# - 配置文件
```

#### Step 2: 训练电脑操作
```bash
# 1. 插入U盘，进入项目目录
cd /path/to/usb/well_detection_project

# 2. 安装依赖（只需要执行一次）
pip install -r requirements.txt

# 3. 检查数据
ls data/images  # 应该看到jpg文件
ls data/labels  # 应该看到txt标注文件

# 4. 准备数据集
python scripts/prepare_dataset.py

# 5. 开始训练
python scripts/train_yolo.py

# 6. 训练完成后，检查结果
ls runs/well_detection/weights/  # 应该有best.pt和last.pt
ls models/  # 应该有well_detector.pt
```

#### Step 3: 带回原电脑
训练完成后，确保复制以下文件到U盘：
- `models/well_detector.pt` (重要！)
- `runs/well_detection/` (训练日志)
- 更新的scripts文件

#### Step 4: 在原电脑使用
```bash
# 1. 复制训练好的模型
cp /path/to/usb/models/well_detector.pt models/

# 2. 处理图片
python scripts/process_wells.py

# 3. 分析结果
python scripts/utils.py
```

### 6. 故障排除

#### 常见问题：

1. **模型文件不存在**
```bash
# 检查模型文件
ls models/well_detector.pt
# 如果不存在，检查训练是否完成
ls runs/well_detection/weights/best.pt
```

2. **依赖包问题**
```bash
# 重新安装
pip install -r requirements.txt --force-reinstall
```

3. **图片读取错误**
```bash
# 检查图片格式
file test_img/*.jpg
# 确保路径正确
ls -la test_img/
```

4. **标注文件格式错误**
```bash
# 检查标注文件内容
head data/labels/*.txt
# 格式应该是：class_id x_center y_center width height
```

5. **训练中断**
```bash
# 从中断处继续训练
python scripts/train_yolo.py  # 会自动从最新检查点恢复
```

### 7. 数据格式说明

#### JSON结果格式：
```json
{
  "well_id": "A1",
  "row": 0,
  "col": 0,
  "rgb": {"r": 255, "g": 128, "b": 64},
  "confidence": 0.95,
  "bbox": [100, 150, 200, 250]
}
```

#### 坐标系统：
- `well_id`: 阵列板坐标 (A1, A2, B1, B2...)
- `row/col`: 数字索引 (从0开始)
- `rgb`: 0-255的RGB值
- `bbox`: [x1, y1, x2, y2] 像素坐标

### 8. 性能优化建议

#### 提升检测精度：
1. **增加训练数据**：标注更多不同角度的图片
2. **数据增强**：旋转、翻转、亮度调整
3. **调整参数**：降低confidence阈值到0.1-0.2
4. **更大模型**：使用yolov8s或yolov8m

#### 提升处理速度：
1. **批量处理**：修改脚本支持批量输入
2. **多进程**：使用multiprocessing并行处理
3. **GPU加速**：在有GPU的电脑上训练和推理

### 9. 扩展功能

后续可以添加：
- 自动Excel报表生成
- 染料浓度计算算法
- 重金属含量换算
- Web界面
- 实时处理

## 项目结构
```
cv2_test/
├── requirements.txt          # Python依赖
├── data_config.yaml         # YOLO配置
├── annotate_guide.md        # 标注指南
├── README.md               # 本文件
├── data/                   # 原始数据
│   ├── images/            # 原始图片
│   └── labels/            # YOLO标注文件
├── datasets/              # 训练数据
│   ├── train/            # 训练集
│   └── val/              # 验证集
├── scripts/              # 核心脚本
│   ├── train_yolo.py     # 训练脚本
│   ├── process_wells.py  # 主处理脚本
│   ├── prepare_dataset.py # 数据准备
│   ├── check_annotations.py # 标注检查
│   └── utils.py          # 工具函数
├── models/               # 训练好的模型
├── runs/                 # 训练日志
├── results/              # 处理结果
└── test_img/             # 测试图片
```