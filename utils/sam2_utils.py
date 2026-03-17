import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def save_dir_segmentations(image_dir, save_dir, mode="stack"):
    """
    输入图片目录，批量生成 segmentation 并保存

    参数：
        image_dir: 图片文件夹
        save_dir: mask保存文件夹
        mode: "stack"（推荐）或 "list"
    """

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint = "model/sam2.1_hiera_large.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2 = build_sam2(model_cfg, checkpoint, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    # 遍历图片
    for fname in os.listdir(image_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(image_dir, fname)
        base_name = os.path.splitext(fname)[0]
        save_path = os.path.join(save_dir, base_name + ".npy")

        # ✅ 已存在就跳过（推荐）
        if os.path.exists(save_path):
            print(f"跳过已存在: {base_name}")
            continue
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        print(f"{base_name} -> mask数量: {len(masks)}")
        seg_list = [m['segmentation'] for m in masks]
        if mode == "stack":
            seg_array = np.stack(seg_list, axis=0)
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