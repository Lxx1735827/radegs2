import os
import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# =========================
# 1. 模型配置
# =========================
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
checkpoint = "model/sam2.1_hiera_large.pt"

# =========================
# 2. 路径配置
# =========================
dirs = "image/"
file = "keyframe_00349"

input_path = os.path.join(dirs, file + ".jpg")
output_fill_path = os.path.join(dirs, file + "_mask_eroded_fill.png")
output_edge_path = os.path.join(dirs, file + "_mask_eroded_edge.png")
output_edge_only_path = os.path.join(dirs, file + "_mask_eroded_edge_only.png")

# =========================
# 3. 加载模型
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
sam2 = build_sam2(model_cfg, checkpoint, device=device)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

# =========================
# 4. 读取图像
# =========================
image_bgr = cv2.imread(input_path)
if image_bgr is None:
    raise FileNotFoundError(f"无法读取图像: {input_path}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# =========================
# 5. 生成 masks
# =========================
masks = mask_generator.generate(image_rgb)
masks = sorted(masks, key=lambda x: x["area"], reverse=True)

# =========================
# 6. 可视化底图
# =========================
output_fill = image_rgb.copy()
output_edge = image_rgb.copy()
edge_only = np.zeros_like(image_rgb, dtype=np.uint8)

rng = np.random.default_rng(42)

# =========================
# 7. 参数
# =========================
MEDIAN_KSIZE = 5
CLOSE_KERNEL = 5
OPEN_KERNEL = 3

# 腐蚀参数：往里缩
ERODE_KERNEL = 5
ERODE_ITER = 1

# 边缘加宽参数
EDGE_DILATE_KERNEL = 3
EDGE_DILATE_ITER = 1

# 轮廓线宽
CONTOUR_THICKNESS = 1


# =========================
# 8. mask 滤波
# =========================
def filter_mask(seg_bool):
    seg_u8 = (seg_bool.astype(np.uint8)) * 255

    if MEDIAN_KSIZE >= 3 and MEDIAN_KSIZE % 2 == 1:
        seg_u8 = cv2.medianBlur(seg_u8, MEDIAN_KSIZE)

    if CLOSE_KERNEL > 0:
        kernel_close = np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)
        seg_u8 = cv2.morphologyEx(seg_u8, cv2.MORPH_CLOSE, kernel_close)

    if OPEN_KERNEL > 0:
        kernel_open = np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8)
        seg_u8 = cv2.morphologyEx(seg_u8, cv2.MORPH_OPEN, kernel_open)

    return seg_u8 > 127


# =========================
# 9. 腐蚀 mask
# =========================
def erode_mask(seg_bool, erode_kernel=5, erode_iter=1):
    seg_u8 = (seg_bool.astype(np.uint8)) * 255
    kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
    seg_eroded = cv2.erode(seg_u8, kernel, iterations=erode_iter)
    return seg_eroded > 127


# =========================
# 10. 从“腐蚀后的 mask”提取边缘
# =========================
def extract_edge_from_eroded_mask(seg_eroded_bool,
                                  contour_thickness=1,
                                  edge_dilate_kernel=3,
                                  edge_dilate_iter=1):
    seg_u8 = (seg_eroded_bool.astype(np.uint8)) * 255

    contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edge = np.zeros_like(seg_u8)
    cv2.drawContours(edge, contours, -1, 255, contour_thickness)

    if edge_dilate_kernel > 0 and edge_dilate_iter > 0:
        kernel = np.ones((edge_dilate_kernel, edge_dilate_kernel), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=edge_dilate_iter)

    return edge > 0


# =========================
# 11. 逐个 mask 处理
# =========================
for mask in masks:
    seg = mask["segmentation"]

    # 先滤波
    seg_filtered = filter_mask(seg)

    # 再腐蚀
    seg_eroded = erode_mask(
        seg_filtered,
        erode_kernel=ERODE_KERNEL,
        erode_iter=ERODE_ITER
    )

    # 从“腐蚀后的 mask”提边缘
    edge_bool = extract_edge_from_eroded_mask(
        seg_eroded,
        contour_thickness=CONTOUR_THICKNESS,
        edge_dilate_kernel=EDGE_DILATE_KERNEL,
        edge_dilate_iter=EDGE_DILATE_ITER
    )

    color = rng.integers(0, 255, size=(3,), dtype=np.uint8)

    # 腐蚀后的 mask 填充
    output_fill[seg_eroded] = color

    # 腐蚀后 mask 的边缘
    output_edge[edge_bool] = color
    edge_only[edge_bool] = color

# =========================
# 12. 保存结果
# =========================
cv2.imwrite(output_fill_path, cv2.cvtColor(output_fill, cv2.COLOR_RGB2BGR))
cv2.imwrite(output_edge_path, cv2.cvtColor(output_edge, cv2.COLOR_RGB2BGR))
cv2.imwrite(output_edge_only_path, cv2.cvtColor(edge_only, cv2.COLOR_RGB2BGR))