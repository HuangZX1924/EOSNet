from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.yaml")                            # 从头训练
# model = YOLO("pre_models/yolov8s.pt")                 # 加载预训练模型训练    
# model = YOLO('runs/detect/train5/weights/last.pt')     # 从断点训练，resume设为True，默认为False

# Use the model
# model.train(data="./ultralytics/datasets/antiuav310.yaml", batch=32, lr0=0.02, epochs=20, device=[0,1], patience=20, interval=20, resume=True, cube=True, event=True, dwa=True, max_det=1, add=True, save_period=1, save_json=True, save_txt=True, save_conf=True, val=False)

# model.train(data="./ultralytics/datasets/antiuav310.yaml", batch=128, lr0=0.04, epochs=100, device=[0,1], patience=20, interval=20, resume=False, cube=True, event=True, dwa=True, max_det=1, add=True, save_period=1, save_json=True, save_txt=True, save_conf=True, val=True)

model.train(
    data="./ultralytics/datasets/antiuav310.yaml",
    batch=64,
    lr0=5e-4,               # 🔺调低学习率，SNN脉冲网络容易震荡，0.001偏大
    lrf=0.01,               # 🔺余弦退火最终学习率系数
    epochs=100,
    device=[0, 1],
    patience=25,            # 稍微放大早停，无人机小目标，有时候后期还涨点
    interval=1,
    amp=False,              # 先保持False，稳定优先；收敛好之后再尝试True提速省显存
    cos_lr=True,            # 🔺开启余弦学习率退火，比默认线性更好
    resume=False,
    cube=True,
    event=True,
    dwa=True,
    max_det=300,
    add=True,
    save_period=5,          # 🔺不要每epoch保存，减少磁盘占用，只保留best/last
    save_json=True,
    save_txt=True,
    save_conf=True,
    val=True,
    optimizer="AdamW",
    warmup_bias_lr=0.0,
    warmup_epochs=2,        # 🔺warmup拉长，保护SNN脉冲层，避免前期step爆炸
    weight_decay=0.0005     # 🔺调低weight_decay，SNN膜电位参数不宜强正则
)

# grep -r "/data/users/qinhaolin01/OSFormer/datasets/AntiUAV310" /shenn --include="*.py"