import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def save_image_segmentations(image_path, save_path, mode="stack"):
    """
    输入图片路径，自动生成 segmentation 并保存

    参数：
        image_path: 图片路径
        save_path: 保存 .npy 路径
        mode: "stack"（推荐）或 "list"
    """

    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint = "model/sam2.1_hiera_large.pt"

    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2 = build_sam2(model_cfg, checkpoint, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    # 读取图片
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 生成 masks
    masks = mask_generator.generate(image)
    print("检测到mask数量:", len(masks))

    # 提取 segmentation
    seg_list = [m['segmentation'] for m in masks]

    # 保存
    if mode == "stack":
        seg_array = np.stack(seg_list, axis=0)  # (N, H, W)
        np.save(save_path, seg_array)
    elif mode == "list":
        np.save(save_path, seg_list)
    else:
        raise ValueError("mode 必须是 'stack' 或 'list'")


def load_image_segmentations(load_path):
    """
    读取 segmentation（支持 stack / list）

    返回：
        segs:
            - 如果是 stack → ndarray (N, H, W)
            - 如果是 list → list[(H, W)]
    """
    segs = np.load(load_path, allow_pickle=True)

    return segs