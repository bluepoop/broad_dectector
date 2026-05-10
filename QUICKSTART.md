# 快速开始指南

## ✅ 已完成
- Python项目结构已建立
- 所有脚本已准备完成
- 图片已复制到 `data/images/` 目录
- 完整系统框架已搭建

## 🎯 接下来的步骤

### 1. 安装Python环境（在有pip的电脑上）
```bash
pip install -r requirements.txt
```

### 2. 安装标注工具
```bash
pip install labelImg
```

### 3. 开始标注孔洞
```bash
labelImg
```
**标注要点：**
- Open Dir → 选择 `data/images`
- Change Save Dir → 选择 `data/labels`
- 选择 "YOLO" 格式
- 每个孔洞（包括空的）都要画矩形框
- 标签选择 "well"

### 4. 检查标注质量
```bash
python scripts/check_annotations.py
```

### 5. 准备训练数据
```bash
python scripts/prepare_dataset.py
```

### 6. U盘传输到训练电脑
复制整个项目文件夹到U盘，在训练电脑上：
```bash
cd /path/to/usb/project
pip install -r requirements.txt
python scripts/train_yolo.py
```

### 7. 训练完成后处理图片
```bash
python scripts/process_wells.py
python scripts/utils.py
```

## 📁 重要文件
- `annotate_guide.md` - 详细标注指南
- `README.md` - 完整使用说明
- `requirements.txt` - Python依赖
- `scripts/train_yolo.py` - 训练脚本
- `scripts/process_wells.py` - 主处理脚本

## 🔧 当前状态
- 6张图片已准备好标注
- 所有代码脚本已就绪
- 项目结构完整
- 等待开始标注数据

**下一步：开始标注孔洞数据！**