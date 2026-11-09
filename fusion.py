import torch
import cv2
from matplotlib import cm

# 创建一个大小为（1，128，68，84）的随机张量
feature = torch.randn(1, 128, 68, 84)

# 将张量传输到CUDA设备上
feature_cuda = feature.to('cuda')

out = torch.randn(1, 128, 68, 84)

# 将张量传输到CUDA设备上
out = out.to('cuda')
# 选择要提取的通道

import torch
import cv2
from matplotlib import cm
selected_channels = [5, 19, 25]

# 提取选定通道的 feature
selected_feature = feature[:, selected_channels, :, :]

# 将三个通道合并为一个三维张量
three_channel_feature = torch.cat([selected_feature[:, i:i+1, :, :] for i in range(len(selected_channels))], dim=1).squeeze(dim=0).permute(1,2,0)

# 计算三维张量的最小和最大值
min_val = torch.min(three_channel_feature)
max_val = torch.max(three_channel_feature)

# 缩放到 [0, 1] 范围
three_channel_feature = (three_channel_feature - min_val) / (max_val - min_val)

# 选择一个颜色映射，例如 'viridis'
cmap = cm.get_cmap('viridis')

# # 将三维张量映射为伪彩色图
# pseudo_color_image = (cmap(three_channel_feature.cpu().numpy()) * 255).astype('uint8').squeeze()
pseudo_color_image = ((three_channel_feature.cpu().numpy()) * 255).astype('uint8').squeeze()


# 放大图像
enlarged_pseudo_color_image = cv2.resize(pseudo_color_image, (pseudo_color_image.shape[1] * 8, pseudo_color_image.shape[0] * 8))

# 保存伪彩色图像
cv2.imwrite('./feature_pseudo_color_image.jpg', enlarged_pseudo_color_image)

# 提取选定通道的 feature
selected_feature_out = out[:, selected_channels, :, :]

# 将三个通道合并为一个三维张量
three_channel_feature_out = torch.cat([selected_feature_out[:, i:i+1, :, :] for i in range(len(selected_channels))], dim=1).squeeze(dim=0).permute(1,2,0)

# 计算三维张量的最小和最大值
min_val_out = torch.min(three_channel_feature_out)
max_val_out = torch.max(three_channel_feature_out)

# 缩放到 [0, 1] 范围
three_channel_feature_out = (three_channel_feature_out - min_val_out) / (max_val_out - min_val_out)

# 将三维张量映射为伪彩色图
#pseudo_color_image_out = (cmap(three_channel_feature_out.squeeze().cpu().numpy()) * 255).astype('uint8').squeeze(dim=0)
pseudo_color_image_out = ((three_channel_feature_out.cpu().numpy()) * 255).astype('uint8').squeeze()

# 放大图像
enlarged_pseudo_color_image_out = cv2.resize(pseudo_color_image_out, (pseudo_color_image_out.shape[1] * 8, pseudo_color_image_out.shape[0] * 8))

# 保存伪彩色图像
cv2.imwrite('./out_pseudo_color_image.jpg', enlarged_pseudo_color_image_out)