import torch
from typing import Optional


def depth_order_weight_schedule(iteration: int, schedule: str = "default"):
    if schedule == "default":
        lambda_depth_order = 0.
        if iteration > 3_000:
            lambda_depth_order = 1.
        if iteration > 7_000:
            lambda_depth_order = 0.1
        if iteration > 15_000:
            lambda_depth_order = 0.01
        if iteration > 20_000:
            lambda_depth_order = 0.001
        if iteration > 25_000:
            lambda_depth_order = 0.0001

    elif schedule == "strong":
        lambda_depth_order = 1.

    elif schedule == "weak":
        lambda_depth_order = 0.
        if iteration > 3_000:
            lambda_depth_order = 0.1

    elif schedule == "none":
        lambda_depth_order = 0.

    else:
        raise ValueError(f"Invalid schedule: {schedule}")

    return lambda_depth_order


import torch
from typing import Optional
import torch.nn.functional as F

def _to_hw(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Convert tensor to [H, W].
    Supported shapes:
        [H, W]
        [1, H, W]
        [1, 1, H, W]
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")

    if x.dim() == 2:
        return x
    elif x.dim() == 3:
        if x.shape[0] != 1:
            raise ValueError(f"{name} with shape {tuple(x.shape)} cannot be safely squeezed to [H, W]")
        return x[0]
    elif x.dim() == 4:
        if x.shape[0] != 1 or x.shape[1] != 1:
            raise ValueError(f"{name} with shape {tuple(x.shape)} cannot be safely squeezed to [H, W]")
        return x[0, 0]
    else:
        raise ValueError(f"{name} must have shape [H,W], [1,H,W], or [1,1,H,W], got {tuple(x.shape)}")


def compute_depth_order_loss(
    depth: torch.Tensor,
    prior_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scene_extent: float = 1.0,
    max_pixel_shift_ratio: float = 0.05,
    normalize_loss: bool = True,
    log_space: bool = False,
    log_scale: float = 20.0,
    reduction: str = "mean",
    debug: bool = False,
    margin: float = 0.01,
    min_prior_diff: float = 0.002,
):
    depth = _to_hw(depth, "depth").float()
    prior_depth = _to_hw(prior_depth, "prior_depth").float()

    if depth.shape != prior_depth.shape:
        raise ValueError(f"depth and prior_depth must have the same shape, got {depth.shape} vs {prior_depth.shape}")

    if mask is not None:
        mask = _to_hw(mask, "mask")
        if mask.shape != depth.shape:
            raise ValueError(f"mask must have the same shape as depth, got {mask.shape} vs {depth.shape}")
        user_mask = mask > 0
    else:
        user_mask = torch.ones_like(depth, dtype=torch.bool)

    H, W = depth.shape
    device = depth.device
    N = H * W

    finite_mask = torch.isfinite(depth) & torch.isfinite(prior_depth)
    base_valid_mask = finite_mask & user_mask
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    prior_depth = torch.nan_to_num(prior_depth, nan=0.0, posinf=0.0, neginf=0.0)

    scene_extent = float(scene_extent)
    if scene_extent < 1e-6:
        scene_extent = 1e-6
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    pixel_coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)  # [N, 2]

    max_pixel_shift = max(int(round(max_pixel_shift_ratio * max(H, W))), 1)
    pixel_shifts = torch.randint(
        low=-max_pixel_shift,
        high=max_pixel_shift + 1,
        size=(N, 2),
        device=device
    )
    zero_shift = (pixel_shifts[:, 0] == 0) & (pixel_shifts[:, 1] == 0)
    retry = 0
    while zero_shift.any() and retry < 10:
        pixel_shifts[zero_shift] = torch.randint(
            low=-max_pixel_shift,
            high=max_pixel_shift + 1,
            size=(zero_shift.sum().item(), 2),
            device=device
        )
        zero_shift = (pixel_shifts[:, 0] == 0) & (pixel_shifts[:, 1] == 0)
        retry += 1
    if zero_shift.any():
        pixel_shifts[zero_shift, 1] = 1

    shifted_coords = pixel_coords + pixel_shifts  # [N, 2]
    in_bounds = (
        (shifted_coords[:, 0] >= 0) & (shifted_coords[:, 0] < H) &
        (shifted_coords[:, 1] >= 0) & (shifted_coords[:, 1] < W)
    )
    non_self = ~(
        (shifted_coords[:, 0] == pixel_coords[:, 0]) &
        (shifted_coords[:, 1] == pixel_coords[:, 1])
    )
    shifted_coords_safe = shifted_coords.clone()
    shifted_coords_safe[:, 0] = shifted_coords_safe[:, 0].clamp(0, H - 1)
    shifted_coords_safe[:, 1] = shifted_coords_safe[:, 1].clamp(0, W - 1)
    flat_depth = depth.reshape(-1)
    flat_prior = prior_depth.reshape(-1)
    flat_valid = base_valid_mask.reshape(-1)

    base_idx = torch.arange(N, device=device)
    shifted_idx = shifted_coords_safe[:, 0] * W + shifted_coords_safe[:, 1]

    d_i = flat_depth[base_idx]
    d_j = flat_depth[shifted_idx]
    p_i = flat_prior[base_idx]
    p_j = flat_prior[shifted_idx]
    valid_i = flat_valid[base_idx]
    valid_j = flat_valid[shifted_idx]
    diff = (d_i - d_j) / scene_extent
    prior_diff = (p_i - p_j) / scene_extent
    order_valid = prior_diff.abs() > float(min_prior_diff)
    pair_valid = valid_i & valid_j & in_bounds & non_self & order_valid

    if normalize_loss:
        target = torch.sign(prior_diff)  # -1 / 0 / +1
    else:
        target = torch.tanh(prior_diff)

    pair_loss = F.relu(float(margin) - target * diff)
    pair_loss = pair_loss * pair_valid.float()
    pair_loss = torch.nan_to_num(pair_loss, nan=0.0, posinf=0.0, neginf=0.0)
    if log_space:
        pair_loss = torch.log1p(float(log_scale) * pair_loss.clamp(min=0.0))
    valid_count = pair_valid.float().sum()

    if reduction == "mean":
        if valid_count.item() < 1:
            loss = depth.new_tensor(0.0)
        else:
            loss = pair_loss.sum() / valid_count.clamp(min=1.0)
    elif reduction == "sum":
        loss = pair_loss.sum()
    elif reduction == "none":
        loss = pair_loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    if isinstance(loss, torch.Tensor) and reduction != "none":
        if not torch.isfinite(loss):
            loss = depth.new_tensor(0.0)

    if debug:
        return {
            "loss": loss,
            "num_valid_pairs": valid_count,
            "valid_ratio": valid_count / max(float(N), 1.0),
            "pair_valid": pair_valid,
            "diff": diff,
            "prior_diff": prior_diff,
            "target": target,
            "shifted_coords": shifted_coords,
        }

    return loss



def sample_pixel_pairs(mask, num_pairs):
    """
    从掩码中随机采样像素对
    mask: H×W bool tensor
    return: (N, 2) index pairs in flattened index
    """
    # 展平，找到所有为true的像素的索引
    idx = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
    if idx.numel() < 2:
        return None
    # 随机采样
    perm = torch.randint(0, idx.numel(), (num_pairs * 2,), device=mask.device)
    u = idx[perm[:num_pairs]]
    v = idx[perm[num_pairs:]]
    return u, v

def depth_order_loss_(
    pred_depth,     # rendered_expected_depth (H×W)
    gt_depth,       # MoGe depth (H×W)
    mask,           # valid mask (H×W)
    num_pairs=8192,
    tau=0.02
):
    """
    tau: 相对深度阈值（后面我会解释怎么定）
    """
    if pred_depth.dim() == 3:
        pred_depth = pred_depth.squeeze(0)
    if gt_depth.dim() == 3:
        gt_depth = gt_depth.squeeze(0)
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    device = pred_depth.device
    H, W = pred_depth.shape
    pred = pred_depth.flatten()
    gt = gt_depth.flatten()

    pairs = sample_pixel_pairs(mask, num_pairs)
    if pairs is None:
        return torch.tensor(0.0, device=device)

    u, v = pairs

    # MoGe depth difference
    d_gt = gt[u] - gt[v]

    # ordinal label
    label = torch.zeros_like(d_gt)
    label[d_gt > tau] = 1.0
    label[d_gt < -tau] = -1.0

    valid = label != 0
    if valid.sum() < 16:
        return torch.tensor(0.0, device=device)

    d_pred = pred[u] - pred[v]

    # logistic ranking loss
    loss = torch.log1p(torch.exp(-label[valid] * d_pred[valid]))

    return loss.mean()