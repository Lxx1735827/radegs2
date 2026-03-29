import torch


def load_mask_list_torch(masks):
    """
    支持两种格式:
    1) [H, W] 标签图, 0为背景, 1/2/3...为不同块
    2) [N, H, W] 二值mask堆叠
    返回: list[H, W] bool mask
    """
    if masks.ndim == 2:
        ids = torch.unique(masks)
        ids = ids[ids != 0]
        mask_list = [(masks == idx) for idx in ids]
    elif masks.ndim == 3:
        mask_list = [m.bool() for m in masks]
    else:
        raise ValueError(f"Unsupported mask shape: {masks.shape}")
    return mask_list


def fit_scale_shift_torch(source, target):
    """
    最小二乘拟合:
        aligned = a * source + b
    使 source 对齐到 target
    source, target: [K]
    """
    x = source.reshape(-1).float()
    y = target.reshape(-1).float()

    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.numel() < 2:
        return None, None

    A = torch.stack([x, torch.ones_like(x)], dim=1)  # [K, 2]
    sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
    a, b = sol[0], sol[1]
    return a, b


def pearson_corr_torch(x, y, eps=1e-8):
    """
    x, y: [K]
    """
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()

    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.numel() < 2:
        return None

    x = x - x.mean()
    y = y - y.mean()

    denom = torch.sqrt(torch.sum(x * x) * torch.sum(y * y)) + eps
    corr = torch.sum(x * y) / denom
    return corr


def weighted_masked_pcc_loss(
    prior_depth,
    render_depth,
    region_masks,
    prior_valid_mask=None,
    render_valid_mask=None,
    min_pixels=10,
    eps=1e-8,
    detach_align=True,
    return_aligned_prior=False,
):
    """
    参数
    ----
    prior_depth: [H, W]
        先验深度，会被对齐到 render_depth
    render_depth: [H, W]
        渲染深度，最终被约束对象
    region_masks: [H, W] 或 [N, H, W]
        分块mask，标签图或mask堆叠
    prior_valid_mask: [H, W] bool，可选
        先验深度自己的有效mask
    render_valid_mask: [H, W] bool，可选
        渲染深度自己的有效mask
    min_pixels: int
        每个块最少有效像素数
    eps: float
        数值稳定项
    detach_align: bool
        True: 拟合对齐参数(a,b)时不参与梯度，通常更稳
    return_aligned_prior: bool
        是否返回对齐后的先验深度图

    返回
    ----
    total_loss
    aligned_prior (optional)
    details (optional, 可按需加)
    """
    if prior_depth.shape != render_depth.shape:
        raise ValueError(f"Shape mismatch: {prior_depth.shape} vs {render_depth.shape}")

    H, W = prior_depth.shape
    device = prior_depth.device
    dtype = prior_depth.dtype

    if prior_valid_mask is None:
        prior_valid_mask = torch.ones((H, W), device=device, dtype=torch.bool)
    else:
        prior_valid_mask = prior_valid_mask.bool()

    if render_valid_mask is None:
        render_valid_mask = torch.ones((H, W), device=device, dtype=torch.bool)
    else:
        render_valid_mask = render_valid_mask.bool()

    region_mask_list = load_mask_list_torch(region_masks)

    # 再额外叠加数值有效性约束
    prior_valid_mask = prior_valid_mask & torch.isfinite(prior_depth) & (prior_depth > 0)
    render_valid_mask = render_valid_mask & torch.isfinite(render_depth) & (render_depth > 0)

    valid_infos = []
    total_pixels = 0

    for i, block_mask in enumerate(region_mask_list):
        if block_mask.shape != (H, W):
            raise ValueError(f"Mask shape mismatch at block {i}: {block_mask.shape} vs {(H, W)}")

        valid = block_mask.bool() & prior_valid_mask & render_valid_mask
        n = int(valid.sum().item())

        if n >= min_pixels:
            valid_infos.append((i, valid, n))
            total_pixels += n

    if total_pixels == 0:
        zero_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if return_aligned_prior:
            aligned_prior = torch.full_like(prior_depth, float("nan"))
            return zero_loss, aligned_prior
        return zero_loss

    total_loss = torch.tensor(0.0, device=device, dtype=dtype)
    aligned_prior = torch.full_like(prior_depth, float("nan")) if return_aligned_prior else None

    for i, valid, n in valid_infos:
        prior_vals = prior_depth[valid]    # source
        render_vals = render_depth[valid]  # target

        # # 先验深度 -> 对齐到渲染深度
        # if detach_align:
        #     a, b = fit_scale_shift_torch(prior_vals.detach(), render_vals.detach())
        # else:
        #     a, b = fit_scale_shift_torch(prior_vals, render_vals)
        #
        # if a is None:
        #     continue
        #
        # aligned_vals = a * prior_vals + b

        if return_aligned_prior:
            aligned_prior[valid] = prior_vals

        corr = pearson_corr_torch(prior_vals, render_vals, eps=eps)
        if corr is None:
            continue

        block_loss = 1.0 - corr
        weight = n / total_pixels
        total_loss = total_loss + weight * block_loss

    return total_loss