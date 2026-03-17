import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# 模型配置
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
checkpoint = "model/sam2.1_hiera_large.pt"

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
sam2 = build_sam2(model_cfg, checkpoint, device=device)

# 自动mask生成器
mask_generator = SAM2AutomaticMaskGenerator(sam2)
dirs = "image/"
file = "keyframe_00349"
# 读取图片
image = cv2.imread(dirs + file + ".jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 生成mask
masks = mask_generator.generate(image)

print("检测到mask数量:", len(masks))

# 可视化
output = image.copy()

for mask in masks:
    seg = mask['segmentation']
    color = np.random.randint(0,255,(3,))
    output[seg] = color

# 保存结果
cv2.imwrite(dirs + file + ".png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
print("分割结果已保存 result.png")