from ultralytics import YOLO

# Load a model
# model = YOLO("result-models/OSFormer_antiuav310.pt")

# results = model('/data/users/qinhaolin01/OSFormer/datasets/AntiUAV310/images/train/20190926_143632_1_6/000017.jpg', save=True)

#-------------------------------#
model = YOLO("yolov8s.yaml")

results = model.predict("/hy-tmp/Anti-UAV310/images/train/20190926_143632_1_6/000017.jpg")