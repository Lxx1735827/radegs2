import torch


def load_mask_list_torch(masks):
    """
    支持两种格式:
    1) [H, W] 标签图, 0为背景, 1/2/3...为不同块
    2) [N, H, W] 二值mask堆叠
    返回: list[[H, W] bool]
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

    A = torch.stack([x, torch.ones_like(x)], dim=1)   # [K, 2]
    sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
    a, b = sol[0], sol[1]
    return a, b


def pearson_corr_torch(x, y, eps=1e-8):
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


def weighted_global_aligned_pcc_loss(
    prior_depth,
    render_depth,
    region_masks,
    prior_valid_mask=None,
    render_valid_mask=None,
    min_pixels=10,
    eps=1e-8,
    detach_align=False,
    return_aligned_prior=False,
    verbose=False,
):
    """
    prior_depth: [H, W] 或 [1, H, W]
        先验深度，会先全局对齐到 render_depth
    render_depth: [H, W] 或 [1, H, W]
        渲染深度，最终被约束对象
    region_masks: [H, W] 或 [N, H, W]
        分块mask
    prior_valid_mask: [H, W] 或 [1, H, W]
        先验深度有效mask
    render_valid_mask: [H, W] 或 [1, H, W]
        渲染深度有效mask
    min_pixels:
        每个块最少有效像素数
    detach_align:
        是否在全局拟合 a,b 时切断梯度
    return_aligned_prior:
        是否返回全局对齐后的 prior_depth
    verbose:
        是否打印调试信息
    """
    # squeeze [1, H, W] -> [H, W]
    if prior_depth.ndim == 3 and prior_depth.shape[0] == 1:
        prior_depth = prior_depth.squeeze(0)
    if render_depth.ndim == 3 and render_depth.shape[0] == 1:
        render_depth = render_depth.squeeze(0)

    if prior_depth.shape != render_depth.shape:
        raise ValueError(f"Shape mismatch: {prior_depth.shape} vs {render_depth.shape}")

    H, W = prior_depth.shape
    device = prior_depth.device
    dtype = prior_depth.dtype

    if prior_valid_mask is None:
        prior_valid_mask = torch.ones((H, W), device=device, dtype=torch.bool)
    else:
        if prior_valid_mask.ndim == 3 and prior_valid_mask.shape[0] == 1:
            prior_valid_mask = prior_valid_mask.squeeze(0)
        prior_valid_mask = prior_valid_mask.bool()

    if render_valid_mask is None:
        render_valid_mask = torch.ones((H, W), device=device, dtype=torch.bool)
    else:
        if render_valid_mask.ndim == 3 and render_valid_mask.shape[0] == 1:
            render_valid_mask = render_valid_mask.squeeze(0)
        render_valid_mask = render_valid_mask.bool()

    region_mask_list = load_mask_list_torch(region_masks)

    # 全局有效区域
    prior_valid_mask = prior_valid_mask & torch.isfinite(prior_depth) & (prior_depth > 0)
    render_valid_mask = render_valid_mask & torch.isfinite(render_depth) & (render_depth > 0)
    global_valid = prior_valid_mask & render_valid_mask

    global_n = int(global_valid.sum().item())
    if verbose:
        print("global_valid_pixels:", global_n)

    if global_n < 2:
        zero_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if return_aligned_prior:
            aligned_prior = torch.full_like(prior_depth, float("nan"))
            return zero_loss, aligned_prior
        return zero_loss

    # ===== 全局只拟合一次 a,b =====
    prior_global = prior_depth[global_valid]
    render_global = render_depth[global_valid]

    if detach_align:
        a, b = fit_scale_shift_torch(prior_global.detach(), render_global.detach())
    else:
        a, b = fit_scale_shift_torch(prior_global, render_global)

    if a is None:
        zero_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if return_aligned_prior:
            aligned_prior = torch.full_like(prior_depth, float("nan"))
            return zero_loss, aligned_prior
        return zero_loss

    aligned_prior = a * prior_depth + b

    if verbose:
        print(f"global scale={a.item():.6f}, shift={b.item():.6f}")

    # ===== 分块统计有效像素 =====
    valid_infos = []
    total_pixels = 0

    for i, block_mask in enumerate(region_mask_list):
        if block_mask.shape != (H, W):
            raise ValueError(f"Mask shape mismatch at block {i}: {block_mask.shape} vs {(H, W)}")

        valid = block_mask.bool() & global_valid
        n = int(valid.sum().item())

        if verbose:
            print(f"block {i}: raw={int(block_mask.sum().item())}, valid={n}")

        if n >= min_pixels:
            valid_infos.append((i, valid, n))
            total_pixels += n

    if verbose:
        print("valid_block_count:", len(valid_infos), "total_pixels:", total_pixels)

    if total_pixels == 0:
        zero_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if return_aligned_prior:
            return zero_loss, aligned_prior
        return zero_loss

    # ===== 分块计算 PCC，再按块大小加权 =====
    total_loss = torch.tensor(0.0, device=device, dtype=dtype)

    for i, valid, n in valid_infos:
        aligned_vals = aligned_prior[valid]
        render_vals = render_depth[valid]

        corr = pearson_corr_torch(aligned_vals, render_vals, eps=eps)
        if corr is None:
            continue

        block_loss = 1.0 - corr
        weight = n / total_pixels
        total_loss = total_loss + weight * block_loss

        if verbose:
            print(
                f"block {i}: weight={weight:.6f}, "
                f"corr={corr.item():.6f}, "
                f"loss={block_loss.item():.6f}"
            )

    if return_aligned_prior:
        return total_loss, aligned_prior
    return total_loss