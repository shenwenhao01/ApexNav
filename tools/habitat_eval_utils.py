"""Utility helpers for Habitat ObjectNav evaluation."""

from __future__ import annotations

import glob
import os
from typing import Optional

import cv2
import numpy as np
import rospy
from basic_utils.risk_utils import (
    compute_risk_from_voxels as _compute_risk_from_voxels,
    compute_dynamic_threshold as _compute_dynamic_threshold,
    select_publish_indices as _select_publish_indices,
)
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32MultiArray, Float64, Header, Int32, Int32MultiArray


def _pc2_centroid(pc_msg: PointCloud2) -> Optional[np.ndarray]:
    """Compute centroid (x,y,z) from a PointCloud2 message. Returns None if empty."""
    try:
        pts = []
        for x, y, z in pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append((float(x), float(y), float(z)))
        if len(pts) == 0:
            return None
        arr = np.asarray(pts, dtype=np.float32)
        return arr.mean(axis=0)
    except Exception:
        return None


def publish_int32(publisher, data):
    """发布Int32类型的消息"""
    msg = Int32()
    msg.data = data
    publisher.publish(msg)


def publish_float64(publisher, data):
    """发布Float64类型的消息"""
    msg = Float64()
    msg.data = data
    publisher.publish(msg)


def compute_risk_from_voxels(
    centroids: np.ndarray,
    voxel_conf: np.ndarray,
    disputed_list: Optional[list[tuple[np.ndarray, int, float]]],
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrapper of basic_utils.risk_utils.compute_risk_from_voxels for backward-compat."""
    return _compute_risk_from_voxels(centroids, voxel_conf, disputed_list, alpha=alpha, beta=beta)


def compute_dynamic_threshold(risk_vals: np.ndarray, base_threshold: float, quantile_target: float) -> float:
    """Wrapper for tests and internal reuse."""
    return _compute_dynamic_threshold(risk_vals, base_threshold, quantile_target)


def select_publish_indices(risk_vals: np.ndarray, dyn_threshold: float, max_points: int) -> np.ndarray:
    """Wrapper for tests and internal reuse."""
    return _select_publish_indices(risk_vals, dyn_threshold, max_points)


def _make_pointcloud2_xyzi(points_xyz: np.ndarray, intensity: np.ndarray, frame_id: str = "map") -> PointCloud2:
    """Create a PointCloud2 (PointXYZI) from Nx3 xyz and N intensities."""
    n = 0 if points_xyz is None else int(points_xyz.shape[0])
    if n == 0:
        hdr = Header(frame_id=frame_id)
        hdr.stamp = rospy.Time.now()
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("intensity", 12, PointField.FLOAT32, 1),
        ]
        return pc2.create_cloud(hdr, fields, [])
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(n, 3)
    inten = np.asarray(intensity, dtype=np.float32).reshape(n)
    rows = [(float(x), float(y), float(z), float(i)) for (x, y, z), i in zip(pts, inten)]
    hdr = Header(frame_id=frame_id)
    hdr.stamp = rospy.Time.now()
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("intensity", 12, PointField.FLOAT32, 1),
    ]
    return pc2.create_cloud(hdr, fields, rows)


def publish_int32_array(publisher, data_list):
    """发布Int32数组类型的消息"""
    msg = Int32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def publish_float32_array(publisher, data_list):
    """发布Float32数组类型的消息"""
    msg = Float32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def transform_rgb_bgr(image):
    """将RGB图像转换为BGR格式"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def transform_bgr_rgb(image):
    """将BGR图像转换为RGB格式"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _norm_scene_suffix(p: str) -> str:
    """Normalize a scene path string to the canonical suffix used by Habitat episodes."""
    try:
        s = str(p).replace("\\", "/").lstrip("./")
        key = "/scene_datasets/"
        i = s.find(key)
        if i >= 0:
            s = s[i + len(key) :]
        return s
    except Exception:
        return str(p)


def _candidate_scene_ids_from_local(scene_or_file: str) -> list[str]:
    """Return a list of candidate scene_id strings that should match env episodes."""
    s = _norm_scene_suffix(scene_or_file)
    cands: list[str] = []
    # Add both original and normalized versions
    if scene_or_file:
        cands.append(scene_or_file)  # 原始格式
    if s and s != scene_or_file:
        cands.append(s)  # 规范化后的格式
    base = os.path.basename(s)
    stem, ext = os.path.splitext(base)
    if base.lower().endswith(".basis.glb"):
        stem = base[: -len(".basis.glb")]
    elif ext.lower() == ".glb":
        stem = stem
    else:
        stem = os.path.splitext(base)[0]
    
    # Handle MP3D dataset format: mp3d/X7HyMhZNoso/X7HyMhZNoso.glb
    if s.startswith("mp3d/") or "mp3d" in s.lower():
        # Extract scene name from path like mp3d/X7HyMhZNoso/X7HyMhZNoso.glb
        parts = s.split("/")
        scene_name = None
        for part in parts:
            if part and part != "mp3d" and part != "val" and not part.endswith(".glb") and not part.startswith("data"):
                scene_name = part
                break
        
        if scene_name:
            # Add various possible MP3D scene ID formats
            # Note: _norm_scene_suffix removes /scene_datasets/ prefix, so we generate both formats
            mp3d_candidates = [
                s,  # Original format: mp3d/X7HyMhZNoso/X7HyMhZNoso.glb
                f"mp3d/{scene_name}/{scene_name}.glb",  # Simplified format
                f"mp3d/val/{scene_name}/{scene_name}.glb",  # With val
                f"data/scene_datasets/mp3d/{scene_name}/{scene_name}.glb",  # Full path (before normalization)
                f"scene_datasets/mp3d/{scene_name}/{scene_name}.glb",  # Without data/ prefix
                # Also add the normalized versions (what _norm_scene_suffix would produce)
                f"mp3d/{scene_name}/{scene_name}.glb",  # Already added above
                # Try with different extensions
                f"mp3d/{scene_name}/{scene_name}.basis.glb",
            ]
            for cand in mp3d_candidates:
                if cand not in cands:
                    cands.append(cand)
    
    roots = [
        os.path.join("data", "scene_datasets", "hm3d_v0.2", "val"),
        os.path.join("data", "scene_datasets", "hm3d", "val"),
        os.path.join("data", "scene_datasets", "mp3d"),
    ]
    for r in roots:
        try:
            if not os.path.isdir(r):
                continue
            for d in glob.glob(os.path.join(r, f"*-{stem}")):
                root_name = os.path.basename(os.path.dirname(r))
                folder = os.path.basename(d)
                if root_name == "scene_datasets":
                    # For MP3D, the structure might be different
                    if "mp3d" in r:
                        rel = f"mp3d/{folder}/{stem}.glb"
                    else:
                        rel = f"{root_name}/val/{folder}/{stem}.basis.glb"
                else:
                    rel = f"{root_name}/val/{folder}/{stem}.basis.glb"
                if rel not in cands:
                    cands.append(rel)
        except Exception:
            continue
    seen = set()
    out = []
    for x in cands:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _compute_preprocess_params(orig_h: int, orig_w: int, mode: str = "crop"):
    """Replicate Stream3R preprocess geometry to map preproc<->raw pixels."""
    target = 518
    if mode not in ("crop", "pad"):
        mode = "crop"
    if mode == "pad":
        if orig_w >= orig_h:
            new_w = target
            new_h = round(orig_h * (new_w / orig_w) / 14) * 14
        else:
            new_h = target
            new_w = round(orig_w * (new_h / orig_h) / 14) * 14
        sx = new_w / float(orig_w)
        sy = new_h / float(orig_h)
        h_pad = target - new_h
        w_pad = target - new_w
        off_x = int(w_pad // 2)
        off_y = int(h_pad // 2)
    else:
        new_w = target
        new_h = round(orig_h * (new_w / orig_w) / 14) * 14
        sx = new_w / float(orig_w)
        sy = new_h / float(orig_h)
        off_x = 0
        off_y = int(max(new_h - target, 0) // 2)
    return {"sx": sx, "sy": sy, "off_x": off_x, "off_y": off_y, "target": target, "mode": mode}


def _project_world_to_raw_uv(
    points_xyz: np.ndarray,
    extri: np.ndarray,
    intri: np.ndarray,
    pp: dict,
    raw_w: int,
    raw_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to raw image pixel coords (u,v) using Stream3R K,T."""
    if points_xyz.size == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.zeros((0,), dtype=bool),
        )
    R = extri[:, :3]
    t = extri[:, 3]
    X = points_xyz.astype(np.float32)
    X_cam = (X @ R.T) + t[None, :]
    z = X_cam[:, 2]
    valid = z > 1e-6
    fx, fy = intri[0, 0], intri[1, 1]
    cx, cy = intri[0, 2], intri[1, 2]
    u_pre = (fx * X_cam[:, 0] / z) + cx
    v_pre = (fy * X_cam[:, 1] / z) + cy
    if pp.get("mode", "crop") == "pad":
        u_raw = (u_pre - float(pp["off_x"])) / float(pp["sx"])
        v_raw = (v_pre - float(pp["off_y"])) / float(pp["sy"])
    else:
        u_raw = (u_pre) / float(pp["sx"])
        v_raw = (v_pre + float(pp["off_y"])) / float(pp["sy"])
    u_pix = np.floor(u_raw + 0.5).astype(np.int32)
    v_pix = np.floor(v_raw + 0.5).astype(np.int32)
    in_w = (u_pix >= 0) & (u_pix < int(raw_w))
    in_h = (v_pix >= 0) & (v_pix < int(raw_h))
    valid = valid & in_w & in_h
    u_pix = np.clip(u_pix, 0, int(raw_w) - 1)
    v_pix = np.clip(v_pix, 0, int(raw_h) - 1)
    return u_pix, v_pix, valid


def _preK_to_rawK(K_pre: np.ndarray, orig_h: int, orig_w: int, mode: str = "crop") -> np.ndarray:
    """将 518 预处理下的内参近似反变换为原始分辨率下的等效 K。"""
    pp = _compute_preprocess_params(orig_h, orig_w, mode=mode)
    fx_pre, fy_pre = float(K_pre[0, 0]), float(K_pre[1, 1])
    cx_pre, cy_pre = float(K_pre[0, 2]), float(K_pre[1, 2])
    sx, sy = float(pp["sx"]), float(pp["sy"])
    off_x, off_y = float(pp["off_x"]), float(pp["off_y"])
    if pp.get("mode", "crop") == "pad":
        fx_raw = fx_pre / sx
        fy_raw = fy_pre / sy
        cx_raw = (cx_pre - off_x) / sx
        cy_raw = (cy_pre - off_y) / sy
    else:
        fx_raw = fx_pre / sx
        fy_raw = fy_pre / sy
        cx_raw = cx_pre / sx
        cy_raw = (cy_pre + off_y) / sy
    K_raw = np.array([[fx_raw, 0.0, cx_raw], [0.0, fy_raw, cy_raw], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K_raw


def _depth_to_world_points(
    depth_img: np.ndarray,
    *,
    K: np.ndarray,
    extri_cw: np.ndarray,
    stride: int = 4,
    min_depth: float = 0.0,
    max_depth: float = 5.0,
) -> np.ndarray:
    """Backproject a depth image (Habitat normalized) to world-space points."""
    if depth_img is None or depth_img.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    H, W = depth_img.shape[:2]
    z = depth_img.reshape(H, W).astype(np.float32)
    z = z * float(max_depth - min_depth) + float(min_depth)
    ss = max(int(stride), 1)
    vs = np.arange(0, H, ss, dtype=np.int32)
    us = np.arange(0, W, ss, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)
    z_s = z[vv, uu]
    valid = np.isfinite(z_s) & (z_s > 1e-6)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    uu = uu[valid].astype(np.float32)
    vv = vv[valid].astype(np.float32)
    z_s = z_s[valid].astype(np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (uu - cx) * (z_s / fx)
    y = (vv - cy) * (z_s / fy)
    X_cam = np.stack([x, y, z_s], axis=1)
    R_cw = extri_cw[:, :3].astype(np.float32)
    t_cw = extri_cw[:, 3].astype(np.float32)
    R_wc = R_cw.T
    C_w = -R_wc @ t_cw
    X_w = (X_cam @ R_wc.T) + C_w[None, :]
    return X_w.astype(np.float32)
