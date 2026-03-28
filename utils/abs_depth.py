import torch


def load_mask_list_torch(masks):
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
    x = source.reshape(-1).float()
    y = target.reshape(-1).float()

    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.numel() < 2:
        return None, None

    A = torch.stack([x, torch.ones_like(x)], dim=1)
    sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
    a, b = sol[0], sol[1]
    return a, b


def weighted_masked_l1_loss(
    prior_depth,
    render_depth,
    region_masks,
    prior_valid_mask=None,
    render_valid_mask=None,
    min_pixels=10,
    detach_align=False,
    return_aligned_prior=False,
):
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

    prior_valid_mask = prior_valid_mask & torch.isfinite(prior_depth) & (prior_depth > 0)
    render_valid_mask = render_valid_mask & torch.isfinite(render_depth) & (render_depth > 0)

    valid_infos = []
    total_pixels = 0

    for i, block_mask in enumerate(region_mask_list):
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
        prior_vals = prior_depth[valid]
        render_vals = render_depth[valid]

        if detach_align:
            a, b = fit_scale_shift_torch(prior_vals.detach(), render_vals.detach())
        else:
            a, b = fit_scale_shift_torch(prior_vals, render_vals)

        if a is None:
            continue

        aligned_vals = a * prior_vals + b

        if return_aligned_prior:
            aligned_prior[valid] = aligned_vals

        block_loss = torch.mean(torch.abs(aligned_vals - render_vals))
        weight = n / total_pixels
        total_loss = total_loss + weight * block_loss

    if return_aligned_prior:
        return total_loss, aligned_prior
    return total_loss