from ultralytics import YOLO
import os
import yaml

def train_model():
    """训练YOLO模型检测孔洞"""

    # 检查数据集是否准备好
    required_dirs = [
        "datasets/train/images",
        "datasets/train/labels",
        "datasets/val/images",
        "datasets/val/labels"
    ]

    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"错误：目录 {dir_path} 不存在！")
            print("请先运行 python scripts/prepare_dataset.py")
            return
        if not os.listdir(dir_path):
            print(f"错误：目录 {dir_path} 为空！")
            return

    # 加载预训练的YOLOv8n模型
    model = YOLO('yolov8n.pt')

    # 训练参数
    train_args = {
        'data': 'data_config.yaml',
        'epochs': 100,  # 根据数据量调整
        'imgsz': 640,
        'batch': -1,  # 自动批量大小
        'device': 'cpu',  # 使用CPU
        'workers': 0,  # CPU环境下建议设为0
        'patience': 20,  # 早停耐心
        'save_period': 10,  # 每10轮保存一次
        'project': 'runs',
        'name': 'well_detection',
        'cache': False,  # CPU环境下不使用缓存
        'amp': False,  # CPU不支持混合精度
    }

    print("开始训练YOLO模型...")
    print(f"训练参数: {train_args}")

    try:
        # 开始训练
        results = model.train(**train_args)

        print("训练完成！")
        print(f"最佳模型保存在: runs/well_detection/weights/best.pt")
        print(f"最新模型保存在: runs/well_detection/weights/last.pt")

        # 复制最佳模型到models目录
        import shutil
        os.makedirs('models', exist_ok=True)
        shutil.copy('runs/well_detection/weights/best.pt', 'models/well_detector.pt')
        print("最佳模型已复制到 models/well_detector.pt")

    except Exception as e:
        print(f"训练过程中出错: {e}")
        print("建议检查数据集和标注格式")

    # 验证模型
    print("\n验证模型性能...")
    metrics = model.val(data='data_config.yaml')
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

if __name__ == "__main__":
    train_model()