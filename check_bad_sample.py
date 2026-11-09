import numpy as np
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.yolo.utils import DEFAULT_CFG

import numpy as np
import cv2

fp = "/hy-tmp/Anti-UAV310/images/train/20190925_101846_1_1/000001.jpg" #脚本输出的坏文件路径
# 这里使用你项目7通道事件图真实读取逻辑
im = cv2.imread(fp)
# im = np.load(...)

print(f"raw file nan: {np.count_nonzero(np.isnan(im))}")
print(f"raw file inf: {np.count_nonzero(np.isinf(im))}")

def check_dataset_nan():
    overrides = dict(
        data="./ultralytics/datasets/antiuav310.yaml",
        batch=128,
        cube=True,
        event=True,
        add=True,
        mode="train",
        device="cpu"
    )
    # 实例化训练器，和tools_train.py流程完全一致
    trainer = DetectionTrainer(overrides=overrides)
    # 只构建dataset，不跑训练
    dataset = trainer.build_dataset(
        img_path=trainer.data["train"],
        mode="train",
        batch=overrides["batch"]
    )

    bad_list = []
    total = len(dataset)
    print(f"Total samples = {total}")

    print("total: ", total)

    for idx in range(total):
        print("idx: ", idx)
        sample = dataset[idx]
        img_np = sample["img"]
        nan_cnt = np.count_nonzero(np.isnan(img_np))
        inf_cnt = np.count_nonzero(np.isinf(img_np))
        if nan_cnt > 0 or inf_cnt > 0:
            file_path = dataset.im_files[idx]
            print(f"🔴 BAD idx={idx:5d} | file={file_path} | NaN={nan_cnt}")
            bad_list.append((idx, file_path))

    print("\n======= Bad samples summary =======")
    for idx, fp in bad_list:
        print(f"{idx:6d} | {fp}")
    if len(bad_list) == 0:
        print("✅ All samples NO NaN / Inf")

# if __name__ == "__main__":
#     check_dataset_nan()
