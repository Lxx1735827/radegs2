import torch

def pcc_loss(depth_1, depth_2, mask_1, mask_2, block_size=64, min_pixels=1e-3):
    """
    计算深度图之间的PCC损失，分块计算每个16x16区域的PCC损失，最后加权返回。

    Args:
        depth_1 (torch.Tensor): 第一个深度图 [H, W]。
        depth_2 (torch.Tensor): 第二个深度图 [H, W]。
        mask_1 (torch.Tensor): 第一个深度图的有效区域mask [H, W]。
        mask_2 (torch.Tensor): 第二个深度图的有效区域mask [H, W]。
        block_size (int): 每个块的大小，默认16x16。
        min_pixels (float): 最小有效像素数量，如果块内有效像素少于该值，则忽略该块。

    Returns:
        torch.Tensor: PCC损失，分块加权后的总损失。
    """
    H, W = depth_1.shape  # 获取深度图的高度和宽度

    # 计算每个block的数量
    n_blocks_h = H // block_size
    n_blocks_w = W // block_size

    # 初始化PCC损失
    total_pcc_loss = 0.0
    block_count = 0

    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            # 获取每个块的坐标范围
            start_h, end_h = i * block_size, (i + 1) * block_size
            start_w, end_w = j * block_size, (j + 1) * block_size

            # 获取当前块的深度值和mask
            block_depth_1 = depth_1[start_h:end_h, start_w:end_w]
            block_depth_2 = depth_2[start_h:end_h, start_w:end_w]
            block_mask_1 = mask_1[start_h:end_h, start_w:end_w]
            block_mask_2 = mask_2[start_h:end_h, start_w:end_w]

            # 忽略没有足够有效像素的块
            valid_pixels = torch.sum(block_mask_1 & block_mask_2)
            if valid_pixels < min_pixels:
                continue

            # 计算块内的PCC损失
            block_depth_1 = block_depth_1[block_mask_1 & block_mask_2]
            block_depth_2 = block_depth_2[block_mask_1 & block_mask_2]

            if block_depth_1.numel() == 0 or block_depth_2.numel() == 0:
                continue

            # 计算皮尔逊相关系数（PCC）
            mean_1 = torch.mean(block_depth_1)
            mean_2 = torch.mean(block_depth_2)

            numerator = torch.sum((block_depth_1 - mean_1) * (block_depth_2 - mean_2))
            denominator = torch.sqrt(torch.sum((block_depth_1 - mean_1)**2)) * torch.sqrt(torch.sum((block_depth_2 - mean_2)**2))

            pcc = numerator / (denominator + 1e-6)  # 避免除零错误

            # 计算当前块的PCC损失
            block_pcc_loss = 1 - pcc  # 1 - PCC 因为我们要最小化损失
            total_pcc_loss += block_pcc_loss
            block_count += 1

    # 返回加权后的总损失
    if block_count > 0:
        return total_pcc_loss / block_count
    else:
        return torch.tensor(0.0, device=depth_1.device)

