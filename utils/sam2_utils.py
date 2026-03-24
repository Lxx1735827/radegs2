import os
import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def filter_mask(seg_bool):
    """
    对单个 bool mask 做后处理滤波
    输入:
        seg_bool: HxW, bool
    输出:
        HxW, bool
    """
    seg_u8 = (seg_bool.astype(np.uint8)) * 255

    # 中值滤波：平滑边缘小毛刺
    seg_u8 = cv2.medianBlur(seg_u8, 5)

    # 闭运算：填补小孔洞
    kernel_close = np.ones((5, 5), np.uint8)
    seg_u8 = cv2.morphologyEx(seg_u8, cv2.MORPH_CLOSE, kernel_close)

    # 开运算：去小噪点
    kernel_open = np.ones((3, 3), np.uint8)
    seg_u8 = cv2.morphologyEx(seg_u8, cv2.MORPH_OPEN, kernel_open)

    return seg_u8 > 127


def save_dir_segmentations(image_dir, save_dir, mode="stack"):
    """
    输入图片目录，批量生成 segmentation，并保存“滤波后的 mask”

    参数：
        image_dir: 图片文件夹
        save_dir: mask 保存文件夹
        mode: "stack"（推荐）或 "list"

    保存结果：
        stack 模式: shape = [N, H, W], dtype=bool
        list  模式: 长度为 N 的 mask 列表，每个元素 shape = [H, W], dtype=bool
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint = "model/sam2.1_hiera_large.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2 = build_sam2(model_cfg, checkpoint, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    for fname in os.listdir(image_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(image_dir, fname)
        base_name = os.path.splitext(fname)[0]
        save_path = os.path.join(save_dir, base_name + ".npy")

        if os.path.exists(save_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)

        seg_list = []
        for m in masks:
            seg = m["segmentation"]          # 原始 bool mask
            seg_filtered = filter_mask(seg)  # 滤波后 bool mask
            seg_list.append(seg_filtered)

        if mode == "stack":
            if len(seg_list) == 0:
                h, w = image.shape[:2]
                seg_array = np.zeros((0, h, w), dtype=bool)
            else:
                seg_array = np.stack(seg_list, axis=0).astype(bool)
            np.save(save_path, seg_array)

        elif mode == "list":
            np.save(save_path, np.array(seg_list, dtype=object))

        else:
            raise ValueError("mode 必须是 'stack' 或 'list'")