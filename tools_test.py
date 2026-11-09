from ultralytics import YOLO

# Load a model
model = YOLO("/huangzx/Code-jian-v3.12-sk/runs/detect/train/weights/best.pt")  # build a new model from scratch

# Use the model
# metrics = model.val(data="./ultralytics/datasets/antiuav310.yaml", batch=128, device=[0,1,2,3], save_json=True, plots=True, cube=True, event=True, dwa=True, max_det=1)  # evaluate model performance on the validation set
metrics = model.val(data="./ultralytics/datasets/antiuav310.yaml", batch=2, device=[0, 1], save_json=True, save_txt=True, plots=True, cube=True, event=True, dwa=True, max_det=1, add=True)