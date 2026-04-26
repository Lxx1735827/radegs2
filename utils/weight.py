import os
import math
import numpy as np
import torch
import torch.nn.functional as F


# =========================================================
# 1. 工具函数：从 FoV 计算焦距
# =========================================================
def fov2focal(fov, pixels):
    return pixels / (2.0 * math.tan(fov / 2.0))


# =========================================================
# 2. 统一 normal 形状
# 支持:
#   H W 3
#   3 H W
# =========================================================
def prepare_normal_tensor(normal_np, device="cuda", normal_encoded_01=False):
    normal = torch.tensor(normal_np, dtype=torch.float32, device=device)

    if normal.ndim != 3:
        raise ValueError(f"normal 维度错误，期望 3 维，实际是 {normal.shape}")

    # 如果是 3 x H x W，转成 H x W x 3
    if normal.shape[0] == 3 and normal.shape[-1] != 3:
        normal = normal.permute(1, 2, 0).contiguous()

    if normal.shape[-1] != 3:
        raise ValueError(f"normal 最后一维应该是 3，实际是 {normal.shape}")

    # 如果你的 normal 是可视化图，范围是 [0, 1]，需要转回 [-1, 1]
    # 如果你的 normal 本身已经是 [-1, 1]，保持 False
    if normal_encoded_01:
        normal = normal * 2.0 - 1.0

    normal_norm = torch.norm(normal, dim=-1, keepdim=True)
    valid_normal = torch.isfinite(normal).all(dim=-1) & (normal_norm.squeeze(-1) > 1e-8)

    normal = normal / torch.clamp(normal_norm, min=1e-8)

    return normal, valid_normal


# =========================================================
# 3. 获取相机内参 fx, fy, cx, cy
# 优先使用 viewpoint_cam.fx / fy / cx / cy
# 如果没有，则用 FoVx / FoVy 估计
# =========================================================
def get_intrinsics_from_viewpoint(viewpoint_cam, H, W, device="cuda"):
    if hasattr(viewpoint_cam, "Fx"):
        fx = float(viewpoint_cam.fx)
    else:
        fx = fov2focal(float(viewpoint_cam.FoVx), W)

    if hasattr(viewpoint_cam, "Fy"):
        fy = float(viewpoint_cam.fy)
    else:
        fy = fov2focal(float(viewpoint_cam.FoVy), H)

    if hasattr(viewpoint_cam, "Cx"):
        cx = float(viewpoint_cam.cx)
    else:
        cx = W / 2.0

    if hasattr(viewpoint_cam, "Cy"):
        cy = float(viewpoint_cam.cy)
    else:
        cy = H / 2.0

    fx = torch.tensor(fx, dtype=torch.float32, device=device)
    fy = torch.tensor(fy, dtype=torch.float32, device=device)
    cx = torch.tensor(cx, dtype=torch.float32, device=device)
    cy = torch.tensor(cy, dtype=torch.float32, device=device)

    return fx, fy, cx, cy


# =========================================================
# 4. 构造每个像素的观察方向 v(x, y)
# 默认使用相机坐标系:
#   v = [(x - cx) / fx, (y - cy) / fy, 1]
# =========================================================
def build_view_dirs_camera(H, W, fx, fy, cx, cy, device="cuda"):
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij"
    )

    vx = (xs - cx) / fx
    vy = (ys - cy) / fy
    vz = torch.ones_like(vx)

    view_dirs = torch.stack([vx, vy, vz], dim=-1)
    view_dirs = F.normalize(view_dirs, dim=-1)

    return view_dirs


# =========================================================
# 5. 根据公式计算面积权重 w(x, y)
# =========================================================
def compute_surface_area_weight(
    depth_tensor,
    normal_tensor,
    valid_depth_mask,
    valid_normal_mask,
    view_dirs,
    eps=1e-6,
    clamp_quantile=0.99
):
    """
    depth_tensor: H x W
    normal_tensor: H x W x 3
    view_dirs: H x W x 3

    返回:
        area_weight: H x W，未归一化面积权重
        area_prob: H x W，归一化后的采样概率
        valid_mask: H x W，有效像素 mask
    """

    valid_mask = valid_depth_mask & valid_normal_mask

    # n(x, y) · v(x, y)
    cos_term = torch.sum(normal_tensor * view_dirs, dim=-1).abs()

    # w(x, y) = d^2 / (|n · v| + eps)
    area_weight = depth_tensor ** 2 / (cos_term + eps)

    # 无效像素权重置 0
    area_weight = torch.where(
        valid_mask,
        area_weight,
        torch.zeros_like(area_weight)
    )

    # 防止法线和观察方向近似垂直时，权重爆炸
    if clamp_quantile is not None:
        valid_weight = area_weight[area_weight > 0]
        if valid_weight.numel() > 0:
            max_weight = torch.quantile(valid_weight, clamp_quantile)
            area_weight = torch.clamp(area_weight, max=max_weight)

    # 归一化成采样概率
    weight_sum = area_weight.sum()
    if weight_sum > 0:
        area_prob = area_weight / weight_sum
    else:
        area_prob = torch.zeros_like(area_weight)

    return area_weight, area_prob, valid_mask
