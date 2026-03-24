import os
import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def erode_mask(seg_bool, erode_kernel=5, erode_iter=1):
    """
    对原始 mask 做腐蚀，让边界往里缩
    输入:
        seg_bool: HxW, bool
    输出:
        HxW, bool
    """
    seg_u8 = (seg_bool.astype(np.uint8)) * 255

    if erode_kernel > 0 and erode_iter > 0:
        kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
        seg_u8 = cv2.erode(seg_u8, kernel, iterations=erode_iter)

    return seg_u8 > 127


def filter_mask(seg_bool, median_ksize=5, close_kernel=5, open_kernel=3):
    """
    对腐蚀后的 mask 做滤波
    输入:
        seg_bool: HxW, bool
    输出:
        HxW, bool
    """
    seg_u8 = (seg_bool.astype(np.uint8)) * 255

    if median_ksize >= 3 and median_ksize % 2 == 1:
        seg_u8 = cv2.medianBlur(seg_u8, median_ksize)

    if close_kernel > 0:
        kernel_close = np.ones((close_kernel, close_kernel), np.uint8)
        seg_u8 = cv2.morphologyEx(seg_u8, cv2.MORPH_CLOSE, kernel_close)

    if open_kernel > 0:
        kernel_open = np.ones((open_kernel, open_kernel), np.uint8)
        seg_u8 = cv2.morphologyEx(seg_u8, cv2.MORPH_OPEN, kernel_open)

    return seg_u8 > 127


def extract_edge_from_mask(seg_bool, edge_kernel=3, edge_dilate_kernel=0, edge_dilate_iter=0):
    """
    从 mask 提取边缘，可选对边缘加宽
    输入:
        seg_bool: HxW, bool
    输出:
        HxW, bool
    """
    seg_u8 = (seg_bool.astype(np.uint8)) * 255

    kernel = np.ones((edge_kernel, edge_kernel), np.uint8)
    edge = cv2.morphologyEx(seg_u8, cv2.MORPH_GRADIENT, kernel)

    if edge_dilate_kernel > 0 and edge_dilate_iter > 0:
        kernel_dilate = np.ones((edge_dilate_kernel, edge_dilate_kernel), np.uint8)
        edge = cv2.dilate(edge, kernel_dilate, iterations=edge_dilate_iter)

    return edge > 0


def save_dir_segmentations(
    image_dir,
    save_dir,
    mode="stack",
    sort_by_area=True,
    erode_kernel=9,
    erode_iter=1,
    median_ksize=5,
    close_kernel=5,
    open_kernel=3,
    edge_kernel=5,
    edge_dilate_kernel=0,
    edge_dilate_iter=0
):
    """
    输入图片目录，批量生成 segmentation，并按以下流程保存：
    原始 mask -> 腐蚀 -> 滤波 -> 提取边缘 -> 保存边缘

    参数：
        image_dir: 图片文件夹
        save_dir: 保存 .npy 的文件夹
        mode: "stack"（推荐）或 "list"
        sort_by_area: 是否按面积从大到小排序
        erode_kernel: 腐蚀核大小
        erode_iter: 腐蚀次数
        median_ksize: 中值滤波核大小
        close_kernel: 闭运算核大小
        open_kernel: 开运算核大小
        edge_kernel: 边缘提取核大小
        edge_dilate_kernel: 边缘加宽核大小，0 表示不加宽
        edge_dilate_iter: 边缘加宽次数

    保存结果：
        stack 模式: shape = [N, H, W], dtype=bool
        list 模式: object array，每个元素是 [H, W] 的 bool edge mask
    """
    os.makedirs(save_dir, exist_ok=True)

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

        if sort_by_area:
            masks = sorted(masks, key=lambda x: x["area"], reverse=True)

        seg_list = []
        for m in masks:
            seg = m["segmentation"]

            # 1. 原始 mask -> 腐蚀
            seg_eroded = erode_mask(
                seg_bool=seg,
                erode_kernel=erode_kernel,
                erode_iter=erode_iter
            )

            # 2. 腐蚀后的 mask -> 滤波
            seg_filtered = filter_mask(
                seg_bool=seg_eroded,
                median_ksize=median_ksize,
                close_kernel=close_kernel,
                open_kernel=open_kernel
            )

            # 3. 从腐蚀+滤波后的 mask 提取边缘
            seg_edge = extract_edge_from_mask(
                seg_bool=seg_filtered,
                edge_kernel=edge_kernel,
                edge_dilate_kernel=edge_dilate_kernel,
                edge_dilate_iter=edge_dilate_iter
            )

            seg_list.append(seg_edge)

        if mode == "stack":
            h, w = image.shape[:2]
            if len(seg_list) == 0:
                seg_array = np.zeros((0, h, w), dtype=bool)
            else:
                seg_array = np.stack(seg_list, axis=0).astype(bool)
            np.save(save_path, seg_array)

        elif mode == "list":
            np.save(save_path, np.array(seg_list, dtype=object))

        else:
            raise ValueError("mode 必须是 'stack' 或 'list'")


if __name__ == "__main__":
    save_dir_segmentations(
        image_dir="image/",
        save_dir="mask_edge/",
        mode="stack",
        sort_by_area=True,
        erode_kernel=5,
        erode_iter=1,
        median_ksize=5,
        close_kernel=5,
        open_kernel=3,
        edge_kernel=3,
        edge_dilate_kernel=0,
        edge_dilate_iter=0
    )