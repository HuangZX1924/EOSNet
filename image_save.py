import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2

# 将特征图的宽高放大8倍
upscaled_feature = F.interpolate(feature, scale_factor=8, mode='nearest')

upscaled_feature = TF.center_crop(upscaled_feature, (512, 640))
# 对所有通道求平均
average_feature = torch.mean(upscaled_feature, dim=1)[0]

# 将张量转换为 NumPy 数组
numpy_feature = average_feature.cpu().numpy()[112:,:]
numpy_feature = np.maximum(numpy_feature, 0)

# 归一化到0-255范围
# normalized_feature = 1 / (1 + np.exp(-numpy_feature))*255
normalized_feature = (numpy_feature - numpy_feature.min()) / (numpy_feature.max() - numpy_feature.min()) * 255
# normalized_feature = (numpy_feature - numpy_feature.mean()) / numpy_feature.std()

normalized_feature = normalized_feature.astype(np.uint8)

score_feature = cv2.applyColorMap(normalized_feature, cv2.COLORMAP_JET)



# 将特征图的宽高放大8倍
upscaled_out = F.interpolate(out, scale_factor=8, mode='nearest')

upscaled_out = TF.center_crop(upscaled_out, (512, 640))
# 对所有通道求平均
average_out = torch.mean(upscaled_out, dim=1)[0]

# 将张量转换为 NumPy 数组
numpy_out = average_out.cpu().numpy()[112:,:]
numpy_out = np.maximum(numpy_out, 0)

# 归一化到0-255范围
# normalized_out = 1 / (1 + np.exp(-numpy_out))*255
normalized_out = (numpy_out - numpy_out.min()) / (numpy_out.max() - numpy_out.min()) * 255
# normalized_out = (numpy_out - numpy_out.mean()) / numpy_out.std()

normalized_out = normalized_out.astype(np.uint8)

score_out = cv2.applyColorMap(normalized_out, cv2.COLORMAP_JET)



ori_path = '/home1/users/tangyuan01/YOLOC/datasets/AntiUAV310/images/test/20190925_111757_1_2/000695.jpg'

frame_ori_feature = cv2.imread(ori_path)[112:,:]
frame_ori_out = cv2.imread(ori_path)[112:,:]

score_map_feature = cv2.addWeighted(frame_ori_feature, 1, score_feature, 0.4, 0)

score_map_out = cv2.addWeighted(frame_ori_out, 1, score_out, 0.4, 0)

cv2.imwrite( 'score_map_feature.jpg', score_map_feature)
cv2.imwrite( 'score_map_out.jpg', score_map_out)


import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2

# 将特征图的宽高放大8倍
upscaled_feature = F.interpolate(feature, scale_factor=8, mode='nearest')

upscaled_feature = TF.center_crop(upscaled_feature, (512, 640))
# 对所有通道求平均
average_feature = torch.mean(upscaled_feature, dim=1)[0]

# 将张量转换为 NumPy 数组
numpy_feature = average_feature.cpu().numpy()[112:,:]

# 归一化到0-255范围
# normalized_feature = 1 / (1 + np.exp(-numpy_feature))*255
normalized_feature = (numpy_feature - numpy_feature.min()) / (numpy_feature.max() - numpy_feature.min()) * 255
# normalized_feature = (numpy_feature - numpy_feature.mean()) / numpy_feature.std()
normalized_feature = normalized_feature.astype(np.uint8)

score_feature = cv2.applyColorMap(normalized_feature, cv2.COLORMAP_JET)



# 将特征图的宽高放大8倍
upscaled_out = F.interpolate(out, scale_factor=8, mode='nearest')

upscaled_out = TF.center_crop(upscaled_out, (512, 640))
# 对所有通道求平均
average_out = torch.mean(upscaled_out, dim=1)[0]

# 将张量转换为 NumPy 数组
numpy_out = average_out.cpu().numpy()[112:,:]


# 归一化到0-255范围
# normalized_out = 1 / (1 + np.exp(-numpy_out))*255
normalized_out = 255 - (numpy_out - numpy_out.min()) / (numpy_out.max() - numpy_out.min()) * 255 -80
normalized_out = np.maximum(normalized_out, 0)
normalized_out = (normalized_out - normalized_out.min()) / (normalized_out.max() - normalized_out.min()) * 255
# normalized_out = (numpy_out - numpy_out.mean()) / numpy_out.std()

normalized_out = normalized_out.astype(np.uint8)

score_out = cv2.applyColorMap(normalized_out, cv2.COLORMAP_JET)



ori_path = '/home1/users/tangyuan01/YOLOC/datasets/AntiUAV310/images/test/20190925_111757_1_2/000699.jpg'

frame_ori_feature = cv2.imread(ori_path)[112:,:]
frame_ori_out = cv2.imread(ori_path)[112:,:]

score_map_feature = cv2.addWeighted(frame_ori_feature, 1, score_feature, 0.4, 0)

score_map_out = cv2.addWeighted(frame_ori_out, 1, score_out, 0.4, 0)

cv2.imwrite( 'score_map_feature.jpg', score_map_feature)
cv2.imwrite( 'score_map_out.jpg', score_map_out)


import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2

# 将特征图的宽高放大8倍
upscaled_feature = F.interpolate(feature, size=(544, 672), mode='bilinear', align_corners=False)

upscaled_feature = TF.center_crop(upscaled_feature, (512, 640))
# 对所有通道求平均
average_feature = torch.mean(upscaled_feature, dim=1)[0]

# 将张量转换为 NumPy 数组
numpy_feature = average_feature.cpu().numpy()[112:,:]

# 归一化到0-255范围
# normalized_feature = 1 / (1 + np.exp(-numpy_feature))*255
normalized_feature = (numpy_feature - numpy_feature.min()) / (numpy_feature.max() - numpy_feature.min()) * 255
# normalized_feature = (numpy_feature - numpy_feature.mean()) / numpy_feature.std()

normalized_feature = normalized_feature.astype(np.uint8)

score_feature = cv2.applyColorMap(normalized_feature, cv2.COLORMAP_JET)



# 将特征图的宽高放大8倍
upscaled_out = F.interpolate(out, size=(544, 672), mode='bilinear', align_corners=False)

upscaled_out = TF.center_crop(upscaled_out, (512, 640))
# 对所有通道求平均
average_out = torch.mean(upscaled_out, dim=1)[0]

# 将张量转换为 NumPy 数组
numpy_out = average_out.cpu().numpy()[112:,:]

# 归一化到0-255范围
# normalized_out = 1 / (1 + np.exp(-numpy_out))*255
normalized_out = 255 - (numpy_out - numpy_out.min()) / (numpy_out.max() - numpy_out.min()) * 255 -80
normalized_out = np.maximum(normalized_out, 0)
normalized_out = (normalized_out - normalized_out.min()) / (normalized_out.max() - normalized_out.min()) * 255
# normalized_out = (numpy_out - numpy_out.mean()) / numpy_out.std()

normalized_out = normalized_out.astype(np.uint8)

score_out = cv2.applyColorMap(normalized_out, cv2.COLORMAP_JET)



ori_path = '/home1/users/tangyuan01/YOLOC/datasets/AntiUAV310/images/test/20190925_111757_1_5/000069.jpg'

frame_ori_feature = cv2.imread(ori_path)[112:,:]
frame_ori_out = cv2.imread(ori_path)[112:,:]

score_map_feature = cv2.addWeighted(frame_ori_feature, 1, score_feature, 0.4, 0)

score_map_out = cv2.addWeighted(frame_ori_out, 1, score_out, 0.4, 0)

cv2.imwrite( 'score_map_feature.jpg', score_map_feature)
cv2.imwrite( 'score_map_out.jpg', score_map_out)