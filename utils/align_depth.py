from dataclasses import dataclass
from pathlib import Path
import os
import struct
import numpy as np


# =========================
# 1. COLMAP binary 读取
# =========================

CAMERA_MODEL_NUM_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE
    1: 4,   # PINHOLE
    2: 4,   # SIMPLE_RADIAL
    3: 5,   # RADIAL
    4: 8,   # OPENCV
    5: 8,   # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12,
    11: 14,
}


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


@dataclass
class Camera:
    id: int
    model_id: int
    width: int
    height: int
    params: np.ndarray


@dataclass
class ImageData:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray

    def qvec2rotmat(self):
        q = self.qvec.astype(np.float64)
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
        ], dtype=np.float64)


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
            if model_id not in CAMERA_MODEL_NUM_PARAMS:
                raise ValueError(f"不支持的相机 model_id={model_id}，请补充 CAMERA_MODEL_NUM_PARAMS")
            num_params = CAMERA_MODEL_NUM_PARAMS[model_id]
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = Camera(
                id=camera_id,
                model_id=model_id,
                width=width,
                height=height,
                params=np.array(params, dtype=np.float64),
            )
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5], dtype=np.float64)
            tvec = np.array(binary_image_properties[5:8], dtype=np.float64)
            camera_id = binary_image_properties[8]

            name_chars = []
            while True:
                char = read_next_bytes(fid, 1, "c")[0]
                if char == b"\x00":
                    break
                name_chars.append(char.decode("utf-8"))
            image_name = "".join(name_chars)

            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            if num_points2D > 0:
                x_y_id = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
                xys = np.column_stack([
                    np.array(x_y_id[0::3], dtype=np.float64),
                    np.array(x_y_id[1::3], dtype=np.float64),
                ])
                point3D_ids = np.array(x_y_id[2::3], dtype=np.int64)
            else:
                xys = np.zeros((0, 2), dtype=np.float64)
                point3D_ids = np.zeros((0,), dtype=np.int64)

            images[image_id] = ImageData(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_points3D_binary(path):
    points3D = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            point3D_id = read_next_bytes(fid, 8, "Q")[0]
            xyz = np.array(read_next_bytes(fid, 24, "ddd"), dtype=np.float64)
            rgb = np.array(read_next_bytes(fid, 3, "BBB"), dtype=np.uint8)
            error = read_next_bytes(fid, 8, "d")[0]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(8 * track_length, os.SEEK_CUR)

            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
            )
    return points3D


# =========================
# 2. 双线性采样
# =========================

def bilinear_sample(depth, xs, ys):
    h, w = depth.shape
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    valid = (
        np.isfinite(xs) & np.isfinite(ys) &
        (xs >= 0) & (xs <= w - 1) &
        (ys >= 0) & (ys <= h - 1)
    )

    sampled = np.full(xs.shape, np.nan, dtype=np.float64)
    if not np.any(valid):
        return sampled

    xv = xs[valid]
    yv = ys[valid]

    x0 = np.floor(xv).astype(np.int64)
    y0 = np.floor(yv).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    dx = xv - x0
    dy = yv - y0

    wa = (1.0 - dx) * (1.0 - dy)
    wb = dx * (1.0 - dy)
    wc = (1.0 - dx) * dy
    wd = dx * dy

    Ia = depth[y0, x0]
    Ib = depth[y0, x1]
    Ic = depth[y1, x0]
    Id = depth[y1, x1]

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id

    neighbor_valid = np.isfinite(Ia) & np.isfinite(Ib) & np.isfinite(Ic) & np.isfinite(Id)
    sampled_valid = np.full(xv.shape, np.nan, dtype=np.float64)
    sampled_valid[neighbor_valid] = out[neighbor_valid]
    sampled[valid] = sampled_valid
    return sampled


# =========================
# 3. RANSAC 拟合 scale + shift
# =========================

def fit_scale_shift_least_squares(x, y):
    A = np.column_stack([x, np.ones_like(x)])
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    scale, shift = sol
    return float(scale), float(shift)


def fit_scale_shift_ransac(x, y, num_iters=1000, residual_threshold=None, min_inlier_ratio=0.3):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        raise ValueError("有效匹配点少于 2 个，无法拟合 scale/shift")

    if residual_threshold is None:
        residual_threshold = max(0.02 * np.median(np.abs(y)), 0.02)

    best_inliers = None
    best_count = 0
    n = len(x)
    rng = np.random.default_rng(0)

    for _ in range(num_iters):
        i, j = rng.choice(n, size=2, replace=False)
        if abs(x[j] - x[i]) < 1e-12:
            continue

        scale = (y[j] - y[i]) / (x[j] - x[i])
        shift = y[i] - scale * x[i]

        if not np.isfinite(scale) or not np.isfinite(shift):
            continue
        if scale <= 0:
            continue

        residuals = np.abs((scale * x + shift) - y)
        inliers = residuals < residual_threshold
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < max(2, int(min_inlier_ratio * n)):
        scale, shift = fit_scale_shift_least_squares(x, y)
        inliers = np.ones_like(x, dtype=bool)
    else:
        scale, shift = fit_scale_shift_least_squares(x[best_inliers], y[best_inliers])
        inliers = best_inliers

    if scale <= 0:
        positive = x > 0
        if positive.sum() >= 1:
            scale = float(np.median(y[positive]) / np.median(x[positive]))
            shift = 0.0
        else:
            scale, shift = 1.0, 0.0

    return float(scale), float(shift), inliers


# =========================
# 4. 文件匹配
# =========================

def find_npy_file_by_image_name(folder_dir, image_name):
    """
    优先:
    folder_dir / 相对路径同名 .npy
    其次:
    folder_dir / stem.npy
    """
    image_rel = Path(image_name)

    candidate = Path(folder_dir) / image_rel.with_suffix(".npy")
    if candidate.exists():
        return candidate

    candidate2 = Path(folder_dir) / (image_rel.stem + ".npy")
    if candidate2.exists():
        return candidate2

    return None


# =========================
# 5. mask 处理
# =========================

def resize_mask_nearest(mask, out_h, out_w):
    """
    最近邻缩放 2D mask 到指定大小
    """
    in_h, in_w = mask.shape
    if in_h == out_h and in_w == out_w:
        return mask.astype(bool)

    ys = np.clip(np.round((np.arange(out_h) + 0.5) * in_h / out_h - 0.5).astype(np.int64), 0, in_h - 1)
    xs = np.clip(np.round((np.arange(out_w) + 0.5) * in_w / out_w - 0.5).astype(np.int64), 0, in_w - 1)
    resized = mask[ys[:, None], xs[None, :]]
    return resized.astype(bool)


def infer_and_prepare_masks(mask_array, target_h, target_w):
    """
    支持:
    - (H, W)
    - (N, H, W)
    - (H, W, N)
    最终统一输出为 (N, H, W) bool
    """
    arr = np.asarray(mask_array)

    if arr.ndim == 2:
        masks = arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[1] == target_h and arr.shape[2] == target_w:
            # (N, H, W)
            masks = arr
        elif arr.shape[0] == target_h and arr.shape[1] == target_w:
            # (H, W, N)
            masks = np.transpose(arr, (2, 0, 1))
        else:
            # 自动猜测 N 轴：通常 N 比 H/W 小很多
            axis_sizes = list(arr.shape)
            n_axis = int(np.argmin(axis_sizes))
            if n_axis == 0:
                masks = arr
            elif n_axis == 1:
                masks = np.transpose(arr, (1, 0, 2))
            else:
                masks = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"mask 维度不支持，当前 shape={arr.shape}")

    out_masks = []
    for i in range(masks.shape[0]):
        m = masks[i]
        m = m.astype(bool)
        m = resize_mask_nearest(m, target_h, target_w)
        out_masks.append(m)

    if len(out_masks) == 0:
        return np.zeros((0, target_h, target_w), dtype=bool)

    return np.stack(out_masks, axis=0).astype(bool)


# =========================
# 6. 构建某一张图的 depth 对齐对应关系
# =========================

def build_correspondences_for_image(image, camera, points3D, dense_depth, region_mask=None):
    """
    对当前 image:
    - 从 image.xys 和 point3D_ids 得到该图像上有 3D 对应关系的稀疏点
    - 计算这些 3D 点在当前相机坐标系下的真实深度 z_sparse
    - 在 dense_depth 上双线性采样得到 z_dense
    - 如果给了 region_mask，则只保留落在该 mask 内部的对应点

    返回:
        z_dense_valid, z_sparse_valid
    """
    R = image.qvec2rotmat()
    t = image.tvec.reshape(3, 1)

    cam_h, cam_w = int(camera.height), int(camera.width)
    dep_h, dep_w = dense_depth.shape

    if cam_w > 1 and dep_w > 1:
        sx = (dep_w - 1) / (cam_w - 1)
    else:
        sx = dep_w / max(cam_w, 1)

    if cam_h > 1 and dep_h > 1:
        sy = (dep_h - 1) / (cam_h - 1)
    else:
        sy = dep_h / max(cam_h, 1)

    xs_dense = []
    ys_dense = []
    z_sparse = []

    for xy, pid in zip(image.xys, image.point3D_ids):
        if pid == -1 or pid not in points3D:
            continue

        xyz_world = points3D[pid].xyz.reshape(3, 1)
        xyz_cam = (R @ xyz_world + t).reshape(3)
        z = float(xyz_cam[2])

        if z <= 1e-8:
            continue

        x, y = float(xy[0]), float(xy[1])
        x_d = x * sx
        y_d = y * sy

        if not (0 <= x_d <= dep_w - 1 and 0 <= y_d <= dep_h - 1):
            continue

        if region_mask is not None:
            xi = int(np.clip(np.round(x_d), 0, dep_w - 1))
            yi = int(np.clip(np.round(y_d), 0, dep_h - 1))
            if not region_mask[yi, xi]:
                continue

        xs_dense.append(x_d)
        ys_dense.append(y_d)
        z_sparse.append(z)

    if len(z_sparse) == 0:
        return np.array([]), np.array([])

    xs_dense = np.array(xs_dense, dtype=np.float64)
    ys_dense = np.array(ys_dense, dtype=np.float64)
    z_sparse = np.array(z_sparse, dtype=np.float64)

    z_dense = bilinear_sample(dense_depth.astype(np.float64), xs_dense, ys_dense)

    valid = np.isfinite(z_dense) & np.isfinite(z_sparse) & (z_dense > 0) & (z_sparse > 0)
    return z_dense[valid], z_sparse[valid]


# =========================
# 7. 应用 scale/shift
# =========================

def apply_scale_shift(depth, scale, shift):
    out = depth.astype(np.float32).copy()
    valid = np.isfinite(out) & (out > 0)
    out[valid] = (scale * out[valid] + shift).astype(np.float32)
    out[valid & (out <= 0)] = 0.0
    return out


# =========================
# 8. 单张 depth：全局 + 分块 对齐
# =========================

def align_one_depth_map_with_blocks(
    depth_path,
    mask_path,
    image,
    camera,
    points3D,
    out_path,
    ransac_iters=1000,
    min_block_matches=6,
):
    depth = np.load(depth_path)
    if depth.ndim != 2:
        raise ValueError(f"深度图必须是单通道 2D 数组，但 {depth_path} 的 shape={depth.shape}")

    dep_h, dep_w = depth.shape

    # ---------- 1) 全局拟合 ----------
    z_dense_global, z_sparse_global = build_correspondences_for_image(
        image=image,
        camera=camera,
        points3D=points3D,
        dense_depth=depth,
        region_mask=None,
    )

    if len(z_dense_global) < 2:
        print(f"[跳过] 全局匹配点太少: {depth_path}，有效点数={len(z_dense_global)}")
        return None

    global_scale, global_shift, global_inliers = fit_scale_shift_ransac(
        z_dense_global,
        z_sparse_global,
        num_iters=ransac_iters,
        residual_threshold=None,
        min_inlier_ratio=0.2,
    )

    global_aligned = apply_scale_shift(depth, global_scale, global_shift)

    result_info = {
        "global_scale": float(global_scale),
        "global_shift": float(global_shift),
        "global_num_matches": int(len(z_dense_global)),
        "global_num_inliers": int(global_inliers.sum()) if global_inliers is not None else int(len(z_dense_global)),
        "global_residual_before": float(np.median(np.abs(z_dense_global - z_sparse_global))),
        "global_residual_after": float(np.median(np.abs((global_scale * z_dense_global + global_shift) - z_sparse_global))),
        "num_blocks": 0,
        "num_local_success": 0,
        "num_global_fallback": 0,
        "block_infos": [],
    }

    # ---------- 2) 没有 mask：直接保存全局结果 ----------
    if mask_path is None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, global_aligned)
        return result_info

    masks_raw = np.load(mask_path)
    masks = infer_and_prepare_masks(masks_raw, dep_h, dep_w)
    result_info["num_blocks"] = int(masks.shape[0])

    if masks.shape[0] == 0:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, global_aligned)
        return result_info

    # ---------- 3) 分块拟合 ----------
    accum = np.zeros_like(depth, dtype=np.float64)
    weight = np.zeros_like(depth, dtype=np.float64)

    for block_idx in range(masks.shape[0]):
        block_mask = masks[block_idx].astype(bool)
        block_area = int(block_mask.sum())

        if block_area == 0:
            result_info["block_infos"].append({
                "block_idx": block_idx,
                "area": 0,
                "mode": "empty_skip",
                "scale": float(global_scale),
                "shift": float(global_shift),
                "num_matches": 0,
                "num_inliers": 0,
            })
            continue

        z_dense_blk, z_sparse_blk = build_correspondences_for_image(
            image=image,
            camera=camera,
            points3D=points3D,
            dense_depth=depth,
            region_mask=block_mask,
        )

        use_local = len(z_dense_blk) >= min_block_matches

        if use_local:
            try:
                local_scale, local_shift, local_inliers = fit_scale_shift_ransac(
                    z_dense_blk,
                    z_sparse_blk,
                    num_iters=ransac_iters,
                    residual_threshold=None,
                    min_inlier_ratio=0.2,
                )
                mode = "local"
                num_inliers = int(local_inliers.sum()) if local_inliers is not None else int(len(z_dense_blk))
                result_info["num_local_success"] += 1
            except Exception:
                local_scale, local_shift = global_scale, global_shift
                mode = "global_fallback"
                num_inliers = 0
                result_info["num_global_fallback"] += 1
        else:
            local_scale, local_shift = global_scale, global_shift
            mode = "global_fallback"
            num_inliers = 0
            result_info["num_global_fallback"] += 1

        block_aligned = apply_scale_shift(depth, local_scale, local_shift)
        valid_block = block_mask & np.isfinite(block_aligned) & (depth > 0)

        accum[valid_block] += block_aligned[valid_block]
        weight[valid_block] += 1.0

        block_info = {
            "block_idx": block_idx,
            "area": block_area,
            "mode": mode,
            "scale": float(local_scale),
            "shift": float(local_shift),
            "num_matches": int(len(z_dense_blk)),
            "num_inliers": int(num_inliers),
        }

        if len(z_dense_blk) > 0:
            block_info["residual_before"] = float(np.median(np.abs(z_dense_blk - z_sparse_blk)))
            block_info["residual_after"] = float(np.median(np.abs((local_scale * z_dense_blk + local_shift) - z_sparse_blk)))
        else:
            block_info["residual_before"] = None
            block_info["residual_after"] = None

        result_info["block_infos"].append(block_info)

    # ---------- 4) 融合分块结果 ----------
    final_aligned = global_aligned.copy()
    covered = weight > 0
    final_aligned[covered] = (accum[covered] / weight[covered]).astype(np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, final_aligned)

    return result_info


# =========================
# 9. 批量处理
# =========================

def align_all_depths_with_blocks(
    images_dir,
    depth_dir,
    mask_dir,
    images_bin,
    cameras_bin,
    points3D_bin,
    out_dir="depth_align",
    ransac_iters=1000,
    min_block_matches=6,
):
    images_dir = Path(images_dir)
    depth_dir = Path(depth_dir)
    mask_dir = Path(mask_dir) if mask_dir is not None else None
    out_dir = Path(out_dir)

    cameras = read_cameras_binary(cameras_bin)
    images = read_images_binary(images_bin)
    points3D = read_points3D_binary(points3D_bin)

    print(f"读取完成: cameras={len(cameras)}, images={len(images)}, points3D={len(points3D)}")

    results = {}
    ok = 0
    skip = 0

    for image_id, image in images.items():
        camera = cameras[image.camera_id]

        img_path = images_dir / image.name
        if not img_path.exists():
            print(f"[警告] images 里找不到图片: {img_path}，继续按重建名字匹配 depth / mask")

        depth_path = find_npy_file_by_image_name(depth_dir, image.name)
        if depth_path is None:
            print(f"[跳过] 找不到对应深度图: {image.name}")
            skip += 1
            continue

        mask_path = None
        if mask_dir is not None:
            mask_path = find_npy_file_by_image_name(mask_dir, image.name)
            if mask_path is None:
                print(f"[提示] 找不到对应 mask，将仅使用全局对齐: {image.name}")

        rel_out = Path(image.name).with_suffix(".npy")
        out_path = out_dir / rel_out

        try:
            info = align_one_depth_map_with_blocks(
                depth_path=depth_path,
                mask_path=mask_path,
                image=image,
                camera=camera,
                points3D=points3D,
                out_path=out_path,
                ransac_iters=ransac_iters,
                min_block_matches=min_block_matches,
            )

            if info is None:
                skip += 1
                continue

            results[image.name] = info
            ok += 1

            print(
                f"[完成] {image.name} | "
                f"global_matches={info['global_num_matches']} | "
                f"global_inliers={info['global_num_inliers']} | "
                f"global_scale={info['global_scale']:.6f} | "
                f"global_shift={info['global_shift']:.6f} | "
                f"blocks={info['num_blocks']} | "
                f"local_success={info['num_local_success']} | "
                f"fallback={info['num_global_fallback']}"
            )
        except Exception as e:
            print(f"[失败] {image.name}: {e}")
            skip += 1

    print(f"\n全部处理结束: 成功 {ok} 张，跳过/失败 {skip} 张")
    return results


# =========================
# 10. 最后只调用这一个函数
# =========================

def run_depth_alignment_with_blocks():
    return align_all_depths_with_blocks(
        images_dir=r"D:\your_project\images",
        depth_dir=r"D:\your_project\depth",
        mask_dir=r"D:\your_project\masks",   # 新增：mask 文件夹
        images_bin=r"D:\your_project\sparse\0\images.bin",
        cameras_bin=r"D:\your_project\sparse\0\cameras.bin",
        points3D_bin=r"D:\your_project\sparse\0\points3D.bin",
        out_dir=r"D:\your_project\depth_align",
        ransac_iters=1000,
        min_block_matches=6,
    )


if __name__ == "__main__":
    results = run_depth_alignment_with_blocks()
    print("\n前3个结果：")
    for k, v in list(results.items())[:3]:
        print(k, v)