"""
Habitat ObjectNav Evaluation Script for HM3D/MP3D Datasets

This script evaluates object navigation performance using the Habitat simulator
with support for HM3D-v1, HM3D-v2, and MP3D datasets. It communicates with ROS for
real-time planning and decision making, incorporates vision-language models
for object detection and image-text matching, and generates comprehensive
evaluation metrics.

Usage:
    # Run with HM3D-v1 dataset
    python habitat_evaluation.py --dataset hm3dv1

    # Run with HM3D-v2 dataset (default)
    python habitat_evaluation.py --dataset hm3dv2

    # Run with MP3D dataset
    python habitat_evaluation.py --dataset mp3d

    # Test specific episode
    python habitat_evaluation.py --dataset hm3dv2 test_epi_num=10

Author: Zager-Zhang
"""

# Standard library imports
import argparse
import gzip
import json
import os
import signal
import threading
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

# Third-party library imports
from hydra import initialize, compose
import numpy as np
import math
from tf.transformations import quaternion_matrix
import cv2
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from omegaconf import DictConfig, OmegaConf
from prettytable import PrettyTable
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32, Int32MultiArray, Float32MultiArray, Float64
from visualization_msgs.msg import Marker
import tqdm
import requests

# Habitat-related imports
import habitat
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)

# ROS message imports
from plan_env.msg import (
    MultipleMasksWithConfidence,
    SemanticObjectArray,
    SemanticObject,
    VoxelHotspotArray,
)
from plan_env.srv import (
    VerifySearchObject,
    VerifySearchObjectRequest,
    VerifySearchObjectResponse,
)

# Local project imports
from basic_utils.failure_check.count_files import count_files_in_directory
from basic_utils.failure_check.failure_check import check_failure, is_on_same_floor
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from basic_utils.record_episode.read_record import read_record
from basic_utils.point_cloud_memory import PointCloudMemory
from basic_utils.record_episode.write_record import write_record
from basic_utils.record_episode.extra_logger import EpisodeExtraLogger
from habitat2ros import habitat_publisher
from llm.answer_reader.answer_reader import read_answer
from llm.client.deepseek_answer import deepseek_respond
from llm.client.ollama_answer import ollama_respond
from params import HABITAT_STATE, ROS_STATE, ACTION, RESULT_TYPES
from basic_utils.label_mapping import LabelMapper
from basic_utils.voxel_fusion import VoxelSemanticFusion, FusionConfig
from basic_utils.scene_graph import ClusterData, VoxelContextAnalyzer
from basic_utils.llm_interface import generate_environment_prompt
from basic_utils.graph_reasoning import GraphReasoner, GraphConfig
from vlm.Labels import HM3D_ID_TO_NAME, MP3D_ID_TO_NAME
from vlm.utils.get_itm_message import get_itm_message_cosine
from vlm.utils.get_object_utils import get_object
from tools.habitat_eval_utils import (
    _candidate_scene_ids_from_local,
    _compute_preprocess_params,
    _depth_to_world_points,
    _make_pointcloud2_xyzi,
    _norm_scene_suffix,
    _pc2_centroid,
    _preK_to_rawK,
    _project_world_to_raw_uv,
    compute_dynamic_threshold,
    compute_risk_from_voxels,
    publish_float32_array,
    publish_float64,
    publish_int32,
    publish_int32_array,
    select_publish_indices,
    transform_bgr_rgb,
    transform_rgb_bgr,
)
from tools.habitat_eval_viz import (
    _apply_colored_masks,
    _draw_points_overlay,
    _hstack_resize,
    _render_uncertainty_bev,
)
from tools.search_object_verifier import SearchObjectVerifier


def _cfg_to_dict(cfg_obj) -> dict:
    if cfg_obj is None:
        return {}
    if isinstance(cfg_obj, dict):
        return dict(cfg_obj)
    try:
        return dict(OmegaConf.to_container(cfg_obj, resolve=True))  # type: ignore[arg-type]
    except Exception:
        data = {}
        for key in dir(cfg_obj):
            if key.startswith("_"):
                continue
            try:
                data[key] = getattr(cfg_obj, key)
            except Exception:
                continue
        return data


class TextualSummaryManager:
    """Periodically cluster voxel memory and request an LLM summary asynchronously."""

    def __init__(
        self,
        *,
        enabled: bool,
        every_steps: int,
        min_voxels: int,
        min_clusters: int,
        max_clusters: int,
        risk_alpha: float,
        risk_beta: float,
        risk_threshold: float,
        eps_scale: float,
        min_cluster_points: int,
        ignored_labels,
        voxel_size: float,
        log_root: str,
        label_mapper,
        llm_cfg: Optional[dict],
        fallback_llm_cfg: Optional[dict],
    ) -> None:
        self.enabled = bool(enabled)
        self.every_steps = max(int(every_steps), 1)
        self.min_voxels = max(int(min_voxels), 0)
        self.min_clusters = max(int(min_clusters), 1)
        self.max_clusters = max(int(max_clusters), 0)
        self.risk_alpha = float(risk_alpha)
        self.risk_beta = float(risk_beta)
        self.log_root = log_root
        default_log_root = self.log_root or os.path.join("logs", "textual_memory")
        ignored = ignored_labels if ignored_labels else ("empty", "noise", "unknown")
        self.analyzer = VoxelContextAnalyzer(
            voxel_size=voxel_size,
            eps_scale=eps_scale,
            min_samples=min_cluster_points,
            risk_threshold=risk_threshold,
            label_mapper=label_mapper,
            ignored_labels=ignored,
        )
        self.llm_cfg = self._merge_llm_cfg(fallback_llm_cfg, llm_cfg)
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._episode_log_path: Optional[str] = None
        self._episode_meta: dict = {}
        self._last_trigger_step = -1
        self._missing_cred_warned = False
        self.latest_result: Optional[dict] = None
        self._request_dump_dir = os.path.join(default_log_root, "request_dumps")
        os.makedirs(self._request_dump_dir, exist_ok=True)
        self._last_debug_request_path: Optional[str] = None

    @staticmethod
    def _merge_llm_cfg(
        base_cfg: Optional[dict], override_cfg: Optional[dict]
    ) -> Dict[str, object]:
        merged: Dict[str, object] = {}
        for cfg in (base_cfg, override_cfg):
            if not cfg:
                continue
            for key, val in cfg.items():
                if val not in (None, ""):
                    merged[key] = val
        return merged

    def start_episode(self, dataset: str, scene_id: str, episode_id: int) -> None:
        if not self.enabled:
            return
        safe_scene = self._sanitize(scene_id)
        log_dir = self.log_root or ""
        if not log_dir:
            log_dir = os.path.join("logs", "textual_memory")
        os.makedirs(log_dir, exist_ok=True)
        self._episode_log_path = os.path.join(
            log_dir, f"{dataset}_{safe_scene}_{int(episode_id)}.jsonl"
        )
        self._episode_meta = {
            "dataset": dataset,
            "scene_id": scene_id,
            "episode_id": int(episode_id),
        }

    def finish_episode(self) -> None:
        if not self.enabled:
            return
        worker = self._worker
        if worker and worker.is_alive():
            worker.join(timeout=2.0)
        self._worker = None

    def maybe_schedule(
        self,
        *,
        step_idx: int,
        timestamp: float,
        voxel_fuser: Optional[VoxelSemanticFusion],
    ) -> None:
        if (
            not self.enabled
            or voxel_fuser is None
            or (self._worker and self._worker.is_alive())
        ):
            return
        if (self._last_trigger_step >= 0) and (
            (step_idx - self._last_trigger_step) < self.every_steps
        ):
            return
        if not self._have_credentials():
            return
        snapshot = voxel_fuser.get_voxel_snapshot(
            timestamp=timestamp, risk_alpha=self.risk_alpha, risk_beta=self.risk_beta
        )
        count = int(snapshot.get("count", 0))
        if count < self.min_voxels:
            return
        clusters = self.analyzer.cluster_voxels(snapshot)
        if len(clusters) < self.min_clusters:
            return
        metadata = dict(self._episode_meta)
        metadata.update(
            {
                "step_idx": int(step_idx),
                "voxel_count": count,
                "timestamp": float(snapshot.get("timestamp", timestamp)),
            }
        )
        prompts = generate_environment_prompt(
            clusters, max_clusters=self.max_clusters, metadata=metadata
        )
        log_path = self._episode_log_path
        if not log_path:
            default_dir = self.log_root or os.path.join("logs", "textual_memory")
            os.makedirs(default_dir, exist_ok=True)
            log_path = os.path.join(default_dir, "textual_memory.jsonl")
            self._episode_log_path = log_path
        worker = threading.Thread(
            target=self._worker_loop,
            args=(step_idx, clusters, prompts, log_path, metadata),
            daemon=True,
        )
        self._worker = worker
        self._last_trigger_step = int(step_idx)
        worker.start()

    def _have_credentials(self) -> bool:
        base_url = str(self.llm_cfg.get("base_url", "") or "").strip()
        if not base_url:
            if not self._missing_cred_warned:
                try:
                    rospy.logwarn("[TextualSummary] base_url missing; skipping summaries")
                except Exception:
                    print("[TextualSummary] base_url missing; skipping summaries")
                self._missing_cred_warned = True
            return False
        key = self._resolve_api_key()
        if not key:
            if not self._missing_cred_warned:
                try:
                    rospy.logwarn("[TextualSummary] API key missing; skipping summaries")
                except Exception:
                    print("[TextualSummary] API key missing; skipping summaries")
                self._missing_cred_warned = True
            return False
        return True

    def _resolve_api_key(self) -> str:
        key = str(self.llm_cfg.get("api_key", "") or "").strip()
        if key:
            return key
        env_name = str(self.llm_cfg.get("api_key_env", "") or "").strip()
        if env_name:
            return str(os.getenv(env_name, "")).strip()
        return ""

    def _worker_loop(
        self,
        step_idx: int,
        clusters: List[ClusterData],
        prompts: Dict[str, str],
        log_path: str,
        metadata: Dict[str, object],
    ) -> None:
        start = time.time()
        response_text: Optional[str] = None
        response_json: Optional[dict] = None
        error_msg: Optional[str] = None
        debug_request_path: Optional[str] = None
        try:
            response_text, debug_request_path = self._call_llm(prompts)
            if response_text:
                response_json = json.loads(response_text)
        except Exception as exc:  # pragma: no cover - network failure path
            error_msg = str(exc)
            if debug_request_path is None:
                debug_request_path = self._last_debug_request_path
            try:
                rospy.logwarn(f"[TextualSummary] LLM request failed: {exc}")
            except Exception:
                print(f"[TextualSummary] LLM request failed: {exc}")
        entry = {
            "metadata": metadata,
            "step_idx": int(step_idx),
            "duration_sec": round(time.time() - start, 3),
            "prompt": prompts,
            "clusters": [c.to_dict() for c in clusters],
            "response_text": response_text,
            "response_json": response_json,
            "error": error_msg,
            "debug_request_path": debug_request_path,
        }
        try:
            self._append_log(log_path, entry)
        except Exception as exc:
            try:
                rospy.logwarn(f"[TextualSummary] Failed to write log: {exc}")
            except Exception:
                print(f"[TextualSummary] Failed to write log: {exc}")
        with self._lock:
            if response_json:
                self.latest_result = entry
            self._worker = None

    def _call_llm(self, prompts: Dict[str, str]) -> Tuple[str, Optional[str]]:
        base_url = str(self.llm_cfg.get("base_url", "") or "")
        api_key = self._resolve_api_key()
        model = self.llm_cfg.get("model") or "gpt-4o"
        temperature = float(self.llm_cfg.get("temperature", 0.2))
        timeout = float(self.llm_cfg.get("timeout", 20.0))
        force_json = bool(self.llm_cfg.get("force_json", True))
        endpoint = base_url.rstrip("/") + "/chat/completions"
        payload: Dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompts["system"]},
                {"role": "user", "content": prompts["user"]},
            ],
            "temperature": temperature,
        }
        if force_json:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        request_dump = {
            "endpoint": endpoint,
            "headers": headers,
            "payload": payload,
            "timeout": timeout,
        }
        debug_request_path = self._save_request_debug(request_dump)
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM response missing choices: {body}")
        message = choices[0].get("message") or {}
        return str(message.get("content") or "").strip(), debug_request_path

    def _save_request_debug(self, data: Dict[str, object]) -> Optional[str]:
        try:
            os.makedirs(self._request_dump_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            suffix = f"{int(time.time() * 1000)}_{threading.get_ident()}"
            path = os.path.join(self._request_dump_dir, f"request_{timestamp}_{suffix}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            self._last_debug_request_path = path
            return path
        except Exception:
            return None

    def _append_log(self, log_path: str, entry: dict) -> None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _sanitize(name: str) -> str:
        safe = name.replace("/", "_").replace(":", "_")
        return safe[:80]


def signal_handler(sig, frame):
    """处理Ctrl+C信号以优雅关闭程序"""
    print("Ctrl+C detected! Shutting down...")
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)


def publish_observations(event):
    """定时器回调函数，发布habitat观测数据和触发消息"""
    global msg_observations, fusion_threshold
    global ros_pub, trigger_pub, confidence_threshold_pub
    tmp = deepcopy(msg_observations)
    ros_pub.habitat_publish_ros_topic(tmp)
    publish_float64(confidence_threshold_pub, fusion_threshold)
    trigger = PoseStamped()
    trigger_pub.publish(trigger)


def ros_action_callback(msg):
    """ROS动作回调函数，接收来自ROS的规划动作"""
    global global_action
    global_action = msg.data
    try:
        print(f"[DBG] ros_action_callback: got /habitat/plan_action={int(global_action)}")
    except Exception:
        print("[DBG] ros_action_callback: got /habitat/plan_action (non-int)")





dbg_prev_ros_state = None  # debug: track and log state changes


def ros_state_callback(msg):
    """ROS状态回调函数，接收ROS系统状态"""
    global ros_state, dbg_prev_ros_state
    ros_state = msg.data
    if ros_state != dbg_prev_ros_state:
        dbg_prev_ros_state = ros_state
        print(f"[DBG] ros_state_callback: /ros/state -> {ros_state}")


def ros_final_state_callback(msg):
    """ROS最终状态回调函数"""
    global final_state
    final_state = msg.data


def ros_expl_result_callback(msg):
    """ROS探索结果回调函数"""
    global expl_result
    expl_result = msg.data

# ------------------------- K/T 差异统计（每步） -------------------------
_cpp_pose_latest = None  # type: ignore
_cpp_K = None  # type: ignore


def _sensor_pose_callback(msg):
    """订阅 C++ 地图侧使用的相机位姿（/habitat/sensor_pose）。"""
    global _cpp_pose_latest
    _cpp_pose_latest = msg  # nav_msgs/Odometry


def _ensure_cpp_intrinsics():
    """读取 C++ 侧使用的相机内参（若失败则回退到默认）"""
    global _cpp_K
    if _cpp_K is not None:
        return _cpp_K
    # 优先从参数服务器读取；若不存在则回退到 launch 中的默认值
    try:
        fx = float(rospy.get_param("/exploration_node/map_ros/fx"))
        fy = float(rospy.get_param("/exploration_node/map_ros/fy"))
        cx = float(rospy.get_param("/exploration_node/map_ros/cx"))
        cy = float(rospy.get_param("/exploration_node/map_ros/cy"))
    except Exception:
        # Fallback to defaults used in exploration.launch
        cx, cy = 320.0, 240.0
        fx, fy = 388.1910413097385, 422.0475153598262
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    _cpp_K = K
    return _cpp_K


def _odometry_to_extrinsic_cam_from_world(odom) -> np.ndarray:
    """将 nav_msgs/Odometry 转为相机-from-世界的 3x4 extrinsic [R|t]（OpenCV 约定）。"""
    # 位置（世界坐标）
    tx = float(odom.pose.pose.position.x)
    ty = float(odom.pose.pose.position.y)
    tz = float(odom.pose.pose.position.z)
    # 四元数（w,x,y,z）
    q = odom.pose.pose.orientation
    quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    # 世界->相机旋转：R_cw = R_wc^T
    T_wc = quaternion_matrix([quat[0], quat[1], quat[2], quat[3]])  # 4x4
    R_wc = T_wc[:3, :3]
    R_cw = R_wc.T
    C_w = np.array([tx, ty, tz], dtype=np.float64)
    t_cw = -R_cw @ C_w
    extri = np.concatenate([R_cw.astype(np.float32), t_cw.reshape(3, 1).astype(np.float32)], axis=1)
    return extri




# ----------------------- Habitat Depth -> World Points -----------------------


def _report_kt_diff(stream_intri: np.ndarray, stream_extri: np.ndarray, raw_h: int, raw_w: int, mode: str):
    """打印每步 C++ 与重建 K/T 的差异（fx,fy,cx,cy 与位姿误差）。"""
    # K 差异
    K_cpp = _ensure_cpp_intrinsics()
    K_stream_raw = _preK_to_rawK(stream_intri, raw_h, raw_w, mode)
    dK = K_stream_raw - K_cpp
    fx_err, fy_err = float(dK[0, 0]), float(dK[1, 1])
    cx_err, cy_err = float(dK[0, 2]), float(dK[1, 2])

    # T 差异（旋转角误差 + 相机中心位置误差）
    rot_err_deg = None
    trans_err = None
    if _cpp_pose_latest is not None:
        extri_cpp = _odometry_to_extrinsic_cam_from_world(_cpp_pose_latest)
        R_s = stream_extri[:, :3].astype(np.float64)
        t_s = stream_extri[:, 3].astype(np.float64)
        R_c = extri_cpp[:, :3].astype(np.float64)
        t_c = extri_cpp[:, 3].astype(np.float64)
        # 旋转误差
        dR = R_s @ R_c.T
        tr = np.trace(dR)
        tr = max(min((tr - 1.0) / 2.0, 1.0), -1.0)
        rot_err_rad = math.acos(tr)
        rot_err_deg = rot_err_rad * 180.0 / math.pi
        # 相机中心位置误差（世界系）
        Cw_s = -(R_s.T @ t_s)
        Cw_c = -(R_c.T @ t_c)
        trans_err = float(np.linalg.norm(Cw_s - Cw_c))

    print(
        f"[K/T Diff] fx={fx_err:+.3f} fy={fy_err:+.3f} cx={cx_err:+.3f} cy={cy_err:+.3f}"
        + (f" | dR(deg)={rot_err_deg:.3f} dCw(m)={trans_err:.3f}" if rot_err_deg is not None else "")
    )


def _parse_dataset_arg():
    """解析命令行参数以选择数据集，并捕获其余Hydra覆盖参数。

    支持一个轻量的 `--dry_run` 开关，用于仅装载配置并快速退出，
    便于在CI或预检阶段验证更改不会破坏主流程。
    注意：`--dry_run` 会被转换为 Hydra 覆盖项 `dry_run=true`。
    """
    parser = argparse.ArgumentParser(
        description="Habitat ObjectNav Evaluation", add_help=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hm3dv1", "hm3dv2", "mp3d"],
        default="hm3dv2",
        help="Choose dataset: hm3dv1, hm3dv2 or mp3d (default: hm3dv2)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load config and exit early without running the full evaluation",
    )
    # 保留未知参数，以便用户仍可以传递Hydra风格的覆盖参数（例如，key=value）
    args, unknown = parser.parse_known_args()
    # 将 --dry_run 转换为 Hydra 覆盖（便于在 main(cfg) 中访问 cfg.dry_run）
    if getattr(args, "dry_run", False):
        unknown.append("dry_run=true")
    return args.dataset, unknown


def main(cfg: DictConfig) -> None:
    # 全局变量声明，用于在函数间共享数据
    global msg_observations, global_action, ros_state, fusion_threshold
    global ros_pub, trigger_pub, obj_point_cloud_pub, confidence_threshold_pub
    global obj_summary_pub, hotspots_pub, risk_pub
    global final_state, expl_result

    # 统一的类别映射：与 ApexNav2 保持一致，统一使用 MP3D 规范名称空间
    # 说明：HM3D 的 episode 类别字符串（如 "sofa"、"tv_monitor"）将首先通过
    # MP3D 的官方映射 JSON 归一到 MP3D 的规范名（如 "couch"、"tv"）。
    label_space = "mp3d"
    label_mapper = LabelMapper(label_space=label_space)
    # Debug mapping: controlled by cfg.stream3r.memory.semantic.debug or env APEXNAV_DEBUG_MAPPING
    try:
        _sem_cfg_dbg = getattr(getattr(getattr(cfg, "stream3r", None), "memory", None), "semantic", None)
        sem_debug = bool(getattr(_sem_cfg_dbg, "debug", False)) if _sem_cfg_dbg is not None else False
    except Exception:
        sem_debug = False
    if not sem_debug:
        try:
            sem_debug = str(os.getenv("APEXNAV_DEBUG_MAPPING", "0")).strip() in ("1", "true", "True")
        except Exception:
            sem_debug = False
    if sem_debug:
        try:
            from vlm.Labels import HM3D_ID_TO_NAME, MP3D_ID_TO_NAME  # type: ignore
            _names = HM3D_ID_TO_NAME if label_space == "hm3d" else MP3D_ID_TO_NAME
            print("[MAP] Using label_space=", label_space, f"({len(_names)} classes)")
            print("[MAP] Canonical names:", ", ".join(_names))
        except Exception:
            print(f"[MAP] Using label_space={label_space}")

    # 记录程序开始时间
    start_time = time.time()

    # 初始化最终状态和探索结果变量
    final_state = 0
    expl_result = 0
    # 初始化结果列表，用于统计各类结果的数量
    result_list = [0] * len(RESULT_TYPES)

    # 修补配置以确保兼容性
    cfg = patch_config(cfg)
    # 预检/CI快速路径：仅加载配置并立即退出
    try:
        if bool(getattr(cfg, "dry_run", False)):
            print("[DRY-RUN] Config loaded and patched successfully; exiting early.")
            return
    except Exception:
        pass

    # 提取配置参数
    video_output_path = cfg.video_output_path.format(split=cfg.habitat.dataset.split)
    need_video = cfg.need_video  # 是否需要生成视频
    record_file_path = os.path.join(video_output_path, cfg.record_file_name)  # 记录文件路径
    continue_path = os.path.join(video_output_path, cfg.continue_file_name)  # 继续文件路径
    max_episode_steps = cfg.habitat.environment.max_episode_steps  # 每个episode的最大步数
    success_distance = cfg.habitat.task.measurements.success.success_distance  # 成功距离阈值

    # 检测器配置
    detector_cfg = cfg.detector

    # 大语言模型配置
    llm_cfg = cfg.llm
    llm_client = llm_cfg.llm_client
    llm_answer_path = llm_cfg.llm_answer_path
    llm_response_path = llm_cfg.llm_response_path

    # 单次测试参数
    env_num_once = cfg.test_epi_num  # 要测试的episode编号
    flag_once = env_num_once != -1  # 是否运行单次测试的标志

    # 创建必要的目录
    os.makedirs(os.path.dirname(llm_answer_path), exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)

    # 添加top_down_map和collisions可视化配置
    with habitat.config.read_write(cfg):
        cfg.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=256,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=79,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    # 创建habitat环境
    env = habitat.Env(cfg)
    print("Environment creation successful")

    # 可选：从外部 minibatch JSON 指定要跑的 (scene, episode_id) 列表
    minibatch_path = getattr(cfg, "minibatch", None)
    minibatch_items = []
    minibatch_mode = False
    if minibatch_path:
        try:
            import json as _json

            with open(str(minibatch_path), "r", encoding="utf-8") as f:
                batch = _json.load(f)
            eps = list(batch.get("episodes", []))
            # 规范化字段名
            for it in eps:
                sc = str(it.get("scene"))
                eid = int(it.get("episode_id"))
                minibatch_items.append({"scene": sc, "episode_id": eid})
            if len(minibatch_items) > 0:
                minibatch_mode = True
        except Exception as e:
            print(f"[Minibatch] WARN: failed to load minibatch from '{minibatch_path}': {e}")
            minibatch_items = []
            minibatch_mode = False

    # minibatch 模式下：为输出目录使用更明确的命名
    # 期望格式：/home/hdd2/chaiqi/Apexnav/videos/test_{dataset}_{split}_minibatch_{timestamp}
    # 若提供 cfg.minibatch_output_dir，则优先使用自定义目录（便于断点续跑）。
    if minibatch_mode:
        # 可选：外部指定固定输出目录，用于断点续跑
        try:
            _mb_outdir = str(getattr(cfg, "minibatch_output_dir", "") or "").strip()
        except Exception:
            _mb_outdir = ""

        if _mb_outdir:
            video_output_path = _mb_outdir
        else:
            # 自动根据数据集和 split 生成：test_{dataset}_{split}_minibatch_{timestamp}
            # dataset 取值：hm3dv1/hm3dv2/mp3d（从 data_path 猜测）
            try:
                dp = str(cfg.habitat.dataset.data_path).lower()
            except Exception:
                dp = ""
            if "mp3d" in dp:
                _ds = "mp3d"
            elif "/v1/" in dp or "hm3dv1" in dp:
                _ds = "hm3dv1"
            else:
                _ds = "hm3dv2"
            _split = str(cfg.habitat.dataset.split)
            ts = time.strftime("%Y%m%d_%H%M%S")
            video_output_path = f"/home/hdd2/chaiqi/Apexnav/videos/test_{_ds}_{_split}_minibatch_{ts}"

        os.makedirs(video_output_path, exist_ok=True)
        # 重新绑定记录/进度文件路径到新输出根目录
        record_file_path = os.path.join(video_output_path, cfg.record_file_name)
        continue_path = os.path.join(video_output_path, cfg.continue_file_name)

    # 设定 episode 总数（若启用 minibatch，则覆盖为列表大小）
    number_of_episodes = len(minibatch_items) if minibatch_mode else env.number_of_episodes

    # 额外日志实例（按 episode 追加到文件）
    extra_logger = EpisodeExtraLogger(out_dir=video_output_path)
    verifier_cfg = getattr(cfg, "search_verifier", None)
    verifier_enabled = bool(getattr(verifier_cfg, "enabled", False)) if verifier_cfg is not None else False
    verifier_service = (
        str(getattr(verifier_cfg, "service_name", "/search_object/verify_candidate"))
        if verifier_cfg is not None else "/search_object/verify_candidate"
    )
    search_verifier = SearchObjectVerifier(
        enabled=verifier_enabled,
        cfg=verifier_cfg,
        service_name=verifier_service,
        base_output_dir=video_output_path,
        extra_logger=extra_logger,
        llm_client=llm_client,
        label_mapper=label_mapper,
    )

    # 读取之前的记录并设置初始值（minibatch 模式下不沿用历史进度，强制从 0 开始）
    (
        num_total,  # 已完成的episode数量
        num_success,  # 成功的episode数量
        spl_all,  # 累计SPL值
        soft_spl_all,  # 累计Soft SPL值
        distance_to_goal_all,  # 累计目标距离
        distance_to_goal_reward_all,  # 累计目标距离奖励
        last_time,  # 上次记录的时间
    ) = read_record(continue_path, flag_once)
    # minibatch 默认不继承历史；若设置 cfg.minibatch_resume=true 则读取 continue.txt 继续
    try:
        _mb_resume = bool(getattr(cfg, "minibatch_resume", False)) if minibatch_mode else False
    except Exception:
        _mb_resume = False
    if minibatch_mode and not _mb_resume:
        num_total = 0
        num_success = 0
        spl_all = 0.0
        soft_spl_all = 0.0
        distance_to_goal_all = 0.0
        distance_to_goal_reward_all = 0.0
        last_time = 0.0

    # 检查是否已完成所有episode（仅非 minibatch 模式）
    if not minibatch_mode and num_total >= number_of_episodes:
        raise ValueError("Already finished all episodes.")

    # 创建进度条
    pbar = tqdm.tqdm(total=number_of_episodes)

    # 设置环境计数器（minibatch 模式下跳过前跳逻辑）
    if not minibatch_mode:
        env_count = num_total if not flag_once else env_num_once
        while env_count:
            pbar.update()
            env.current_episode = next(env.episode_iterator)
            env_count -= 1

    # 初始化ROS发布者、订阅者和定时器
    obj_point_cloud_pub = rospy.Publisher(
        "habitat/object_point_cloud", PointCloud2, queue_size=10
    )
    ros_pub = habitat_publisher.ROSPublisher()
    # 订阅 C++ 地图侧使用的位姿（用于和 Stream3R 进行 K/T 对齐比较）
    rospy.Subscriber("/habitat/sensor_pose", Odometry, _sensor_pose_callback, queue_size=10)
    # 订阅各种ROS主题以接收动作和状态信息
    rospy.Subscriber("/habitat/plan_action", Int32, ros_action_callback, queue_size=10)
    rospy.Subscriber("/ros/state", Int32, ros_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_state", Int32, ros_final_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_result", Int32, ros_expl_result_callback, queue_size=10)
    # 创建各种发布者以向ROS系统发布信息
    state_pub = rospy.Publisher("/habitat/state", Int32, queue_size=10)
    trigger_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    itm_score_pub = rospy.Publisher("/blip2/cosine_score", Float64, queue_size=10)
    confidence_threshold_pub = rospy.Publisher(
        "/detector/confidence_threshold", Float64, queue_size=10
    )
    cld_with_score_pub = rospy.Publisher(
        "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
    )
    # M7: boundary integration publishers (Python -> ROS summary)
    obj_summary_pub = rospy.Publisher(
        "/memory/objects_summary", SemanticObjectArray, queue_size=10
    )
    hotspots_pub = rospy.Publisher(
        "/memory/disputed_hotspots", VoxelHotspotArray, queue_size=10
    )
    # M1: Risk cloud publisher (PointXYZI, intensity=risk)
    _risk_cfg = getattr(cfg, "risk", None)
    risk_topic = str(getattr(_risk_cfg, "publish_topic", "/memory/risk_cloud")) if _risk_cfg is not None else "/memory/risk_cloud"
    risk_pub = rospy.Publisher(risk_topic, PointCloud2, queue_size=10)
    progress_pub = rospy.Publisher("/habitat/progress", Int32MultiArray, queue_size=10)
    record_pub = rospy.Publisher("/habitat/record", Float32MultiArray, queue_size=10)
    # Live RViz voxel visualization publisher (optional)
    # NOTE: this publisher depends on config flags parsed below, so we defer
    # its creation until after we read `stream3r.visualize.rviz_voxels`.
    voxels_vis_pub = None

    # ------------------------- Point Cloud Memory (Stream3R/Habitat) -------------------------
    # Optional config block `stream3r` in config/*.yaml controls the memory behavior
    mem_cfg = getattr(cfg, "stream3r", None)
    mapping_source = str(getattr(mem_cfg, "source", "stream3r")).lower() if mem_cfg is not None else "stream3r"
    # Safe defaults if config is absent
    mem_args = dict(
        port=getattr(mem_cfg, "port", 12185) if mem_cfg is not None else 12185,
        preprocess_mode=getattr(mem_cfg, "preprocess_mode", "crop") if mem_cfg is not None else "crop",
        points_stride=int(getattr(mem_cfg, "points_stride", 4)) if mem_cfg is not None else 4,
        return_depth=bool(getattr(mem_cfg, "return_depth", False)) if mem_cfg is not None else False,
        voxel_size=(float(getattr(mem_cfg, "voxel_size", 0.05)) if mem_cfg is not None else 0.05),
        max_points=int(getattr(mem_cfg, "max_points", 2_000_000)) if mem_cfg is not None else 2_000_000,
        evict_policy=str(getattr(mem_cfg, "evict_policy", "random")) if mem_cfg is not None else "random",
        quality=int(getattr(mem_cfg, "quality", 90)) if mem_cfg is not None else 90,
        enabled=bool(getattr(mem_cfg, "enabled", True)) if mem_cfg is not None else True,
    )
    # M1 memory nested config (robust to absence)
    _mem_mem = getattr(mem_cfg, "memory", None)
    _agg = getattr(_mem_mem, "aggregation", None) if _mem_mem is not None else None
    _feat = getattr(_mem_mem, "feature", None) if _mem_mem is not None else None
    _sem = getattr(_mem_mem, "semantic", None) if _mem_mem is not None else None
    mem_args.update(
        dict(
            aggregation_mode=str(getattr(_agg, "mode", "voxel_only")),
            feature_source=str(getattr(_feat, "source", "object_embedding")),
            feature_dim=int(getattr(_feat, "dim", 256)),
            feature_quantize=str(getattr(_feat, "quantize", "none")),
            semantic_enabled=bool(getattr(_sem, "enabled", False)),
            semantic_label_space=str(getattr(_sem, "label_space", label_space)),
        )
    )
    try:
        search_verifier.configure_memory_gate_geometry(
            voxel_size=float(mem_args.get("voxel_size", 0.05))
        )
    except Exception:
        pass
    mem_semantic_label_space = mem_args.get("semantic_label_space", label_space)
    if sem_debug:
        try:
            print(
                f"[MAP] Memory semantic config: enabled={mem_args.get('semantic_enabled', False)}, label_space={mem_semantic_label_space}"
            )
        except Exception:
            pass
    # Visualization config (M3)
    viz_cfg = getattr(mem_cfg, "visualize", None)
    viz_overlay = bool(getattr(viz_cfg, "overlay", False)) if viz_cfg is not None else False
    viz_overlay_mode = str(getattr(viz_cfg, "overlay_mode", "label")) if viz_cfg is not None else "label"
    viz_point_size = int(getattr(viz_cfg, "overlay_point_size", 2)) if viz_cfg is not None else 2
    viz_max_points = int(getattr(viz_cfg, "overlay_max_points", 3000)) if viz_cfg is not None else 3000
    viz_save_ply = bool(getattr(viz_cfg, "save_ply", False)) if viz_cfg is not None else False
    viz_ply_color_by = str(getattr(viz_cfg, "ply_color_by", "label")) if viz_cfg is not None else "label"

    # Extra composite visualization (side-by-side panels per step -> a per-episode video)
    viz_comp_cfg = getattr(viz_cfg, "composite", None) if viz_cfg is not None else None
    viz_comp_enable = bool(getattr(viz_comp_cfg, "enable", False)) if viz_comp_cfg is not None else False
    viz_comp_outdir = str(getattr(viz_comp_cfg, "output_dir", "")) if viz_comp_cfg is not None else ""
    viz_comp_fps = int(getattr(viz_comp_cfg, "fps", 6)) if viz_comp_cfg is not None else 6
    viz_comp_target_width = int(getattr(viz_comp_cfg, "target_width", 1920)) if viz_comp_cfg is not None else 1920

    # RViz voxel probability visualization (live)
    vox_vis_cfg = getattr(viz_cfg, "rviz_voxels", None) if viz_cfg is not None else None
    vox_vis_enable = bool(getattr(vox_vis_cfg, "enable", False)) if vox_vis_cfg is not None else False
    vox_vis_topic = str(getattr(vox_vis_cfg, "topic", "/memory/voxel_probs")) if vox_vis_cfg is not None else "/memory/voxel_probs"
    vox_vis_frame = str(getattr(vox_vis_cfg, "frame", "world")) if vox_vis_cfg is not None else "world"
    vox_vis_every = int(getattr(vox_vis_cfg, "publish_every", 5)) if vox_vis_cfg is not None else 5
    vox_vis_minp = float(getattr(vox_vis_cfg, "min_prob", 0.0)) if vox_vis_cfg is not None else 0.0
    vox_vis_cap = int(getattr(vox_vis_cfg, "max_voxels", 60000)) if vox_vis_cfg is not None else 60000
    vox_vis_alpha_by_prob = bool(getattr(vox_vis_cfg, "alpha_by_prob", True)) if vox_vis_cfg is not None else True
    vox_vis_size = float(getattr(vox_vis_cfg, "size", 0.0)) if vox_vis_cfg is not None else 0.0
    vox_vis_color_mode = str(getattr(vox_vis_cfg, "color_mode", "prob")) if vox_vis_cfg is not None else "prob"
    vox_vis_debug = bool(getattr(vox_vis_cfg, "debug_log", False)) if vox_vis_cfg is not None else False

    # Now that the config has been parsed, create the publisher if enabled.
    try:
        if vox_vis_enable:
            # Latch so RViz opened later still sees the latest marker
            voxels_vis_pub = rospy.Publisher(vox_vis_topic, Marker, queue_size=3, latch=True)
    except Exception:
        voxels_vis_pub = None

    # Uncertainty map saving (per-step 2D images, per-episode directory)
    uncert_cfg = getattr(viz_cfg, "uncertainty", None) if viz_cfg is not None else None
    uncert_enable = bool(getattr(uncert_cfg, "enable", False)) if uncert_cfg is not None else False
    uncert_every = int(getattr(uncert_cfg, "every_steps", 1)) if uncert_cfg is not None else 1
    uncert_outdir_override = str(getattr(uncert_cfg, "output_dir", "")) if uncert_cfg is not None else ""
    # BEV-specific
    uncert_bev_window_m = float(getattr(uncert_cfg, "window_size_m", 14.0)) if uncert_cfg is not None else 14.0
    uncert_bev_res_m = float(getattr(uncert_cfg, "resolution_m", 0.05)) if uncert_cfg is not None else 0.05
    uncert_bev_zmin = float(getattr(uncert_cfg, "z_min", -0.2)) if uncert_cfg is not None else -0.2
    uncert_bev_zmax = float(getattr(uncert_cfg, "z_max", 2.0)) if uncert_cfg is not None else 2.0
    uncert_bev_blur = float(getattr(uncert_cfg, "blur_sigma_px", 0.0)) if uncert_cfg is not None else 0.0
    uncert_bev_draw_agent = bool(getattr(uncert_cfg, "draw_agent", True)) if uncert_cfg is not None else True
    uncert_bev_draw_occupancy = bool(getattr(uncert_cfg, "draw_occupancy", True)) if uncert_cfg is not None else True
    uncert_bev_occ_max_points = int(getattr(uncert_cfg, "occ_max_points", 150000)) if uncert_cfg is not None else 150000
    uncert_bev_occ_thickness = int(getattr(uncert_cfg, "occ_line_thickness", 2)) if uncert_cfg is not None else 2
    uncert_bev_orientation = str(getattr(uncert_cfg, "orientation", "world_north_up")) if uncert_cfg is not None else "world_north_up"
    # Grid overlay config
    uncert_bev_draw_grid = bool(getattr(uncert_cfg, "draw_grid", True)) if uncert_cfg is not None else True
    uncert_bev_grid_spacing = int(getattr(uncert_cfg, "grid_spacing_cells", 1)) if uncert_cfg is not None else 1
    uncert_bev_grid_thickness = int(getattr(uncert_cfg, "grid_thickness", 1)) if uncert_cfg is not None else 1
    try:
        _gc = getattr(uncert_cfg, "grid_color", [220, 220, 220]) if uncert_cfg is not None else [220, 220, 220]
        uncert_bev_grid_color = (int(_gc[0]), int(_gc[1]), int(_gc[2]))
    except Exception:
        uncert_bev_grid_color = (220, 220, 220)

    # Auto-disable remote Stream3R client when mapping source is Habitat depth+pose.
    # This prevents any HTTP calls to port 12185 while keeping local memory usable.
    if mapping_source != "stream3r":
        try:
            mem_args["enabled"] = False
        except Exception:
            pass
    point_mem = PointCloudMemory(**mem_args)

    # M4: voxel-level fusion configuration and state
    _fusion = getattr(_mem_mem, "fusion", None) if _mem_mem is not None else None
    _corr = getattr(_mem_mem, "correction", None) if _mem_mem is not None else None
    fusion_conf = FusionConfig(
        use_view_angle=bool(getattr(_fusion, "weight_view_angle", True)) if _fusion is not None else True,
        use_distance=bool(getattr(_fusion, "weight_distance", True)) if _fusion is not None else True,
        use_confidence=bool(getattr(_fusion, "weight_confidence", True)) if _fusion is not None else True,
        time_decay_tau=float(getattr(_fusion, "time_decay_tau", 50.0)) if _fusion is not None else 50.0,
        angle_power=float(getattr(_fusion, "angle_power", 2.0)) if _fusion is not None else 2.0,
        dist_ref=float(getattr(_fusion, "dist_ref", 2.0)) if _fusion is not None else 2.0,
        dist_power=float(getattr(_fusion, "dist_power", 2.0)) if _fusion is not None else 2.0,
        # 支持新旧两套命名：优先读取 min_top1_prob/min_margin，
        # 回落到历史键 dispute_top1/dispute_margin 以保持兼容。
        min_top1_prob=(
            float(getattr(_fusion, "min_top1_prob", None))
            if (_fusion is not None and getattr(_fusion, "min_top1_prob", None) is not None)
            else float(getattr(_fusion, "dispute_top1", 0.60)) if _fusion is not None else 0.60
        ),
        min_margin=(
            float(getattr(_fusion, "min_margin", None))
            if (_fusion is not None and getattr(_fusion, "min_margin", None) is not None)
            else float(getattr(_fusion, "dispute_margin", 0.15)) if _fusion is not None else 0.15
        ),
        min_total_weight=float(getattr(_fusion, "min_total_weight", 1.0)) if _fusion is not None else 1.0,
        flip_window=int(getattr(_fusion, "flip_window", 30)) if _fusion is not None else 30,
        flip_count_thresh=int(getattr(_fusion, "flip_count", 2)) if _fusion is not None else 2,
        auto_correct=bool(getattr(_corr, "enabled", True)) if _corr is not None else True,
        auto_correct_min_top1=float(getattr(_corr, "auto_min_top1", 0.70)) if _corr is not None else 0.70,
        # New voxel confidence parameters (evidence x agreement)
        voxel_conf_weight_ref=float(getattr(_fusion, "voxel_conf_weight_ref", 2.0)) if _fusion is not None else 2.0,
        voxel_conf_alpha=float(getattr(_fusion, "voxel_conf_alpha", 1.0)) if _fusion is not None else 1.0,
        voxel_conf_beta=float(getattr(_fusion, "voxel_conf_beta", 1.0)) if _fusion is not None else 1.0,
    )
    fusion_observe_all = bool(getattr(_fusion, "observe_all_points", True)) if _fusion is not None else True
    # Determine class count from label space
    _num_classes = len(HM3D_ID_TO_NAME) if mem_semantic_label_space == "hm3d" else len(MP3D_ID_TO_NAME)
    voxel_fuser = VoxelSemanticFusion(
        voxel_size=float(mem_args.get("voxel_size", 0.05)),
        num_classes=_num_classes,
        label_mapper=label_mapper,
        config=fusion_conf,
    )
    textual_summary_mgr = None
    textual_cfg = getattr(cfg, "memory_textual", None)
    if textual_cfg is not None:
        try:
            ignored_labels = list(getattr(textual_cfg, "ignored_labels", []))
        except Exception:
            ignored_labels = ["empty", "noise", "unknown"]
        log_root = str(getattr(textual_cfg, "log_dir", "") or os.path.join(video_output_path, "textual_memory"))
        text_llm_cfg = _cfg_to_dict(getattr(textual_cfg, "llm_client", None))
        fallback_llm_cfg = _cfg_to_dict(getattr(llm_cfg, "llm_client", None))
        textual_summary_mgr = TextualSummaryManager(
            enabled=bool(getattr(textual_cfg, "enable", False)),
            every_steps=int(getattr(textual_cfg, "every_steps", 80)),
            min_voxels=int(getattr(textual_cfg, "min_voxels", 400)),
            min_clusters=int(getattr(textual_cfg, "min_clusters", 1)),
            max_clusters=int(getattr(textual_cfg, "max_clusters", 8)),
            risk_alpha=float(getattr(textual_cfg, "risk_alpha", 1.0)),
            risk_beta=float(getattr(textual_cfg, "risk_beta", 1.0)),
            risk_threshold=float(getattr(textual_cfg, "risk_threshold", 0.6)),
            eps_scale=float(getattr(textual_cfg, "eps_scale", 3.0)),
            min_cluster_points=int(getattr(textual_cfg, "min_cluster_points", 10)),
            ignored_labels=ignored_labels,
            voxel_size=float(mem_args.get("voxel_size", 0.05)),
            log_root=log_root,
            label_mapper=label_mapper,
            llm_cfg=text_llm_cfg,
            fallback_llm_cfg=fallback_llm_cfg,
        )
    try:
        search_verifier.configure_memory_gate_geometry(
            weight_cfg={
                "use_view_angle": fusion_conf.use_view_angle,
                "use_distance": fusion_conf.use_distance,
                "angle_power": fusion_conf.angle_power,
                "dist_ref": fusion_conf.dist_ref,
                "dist_power": fusion_conf.dist_power,
            }
        )
    except Exception:
        pass

    # M5: explainable graph reasoning configuration
    explain_cfg = getattr(mem_cfg, "explain", None) if mem_cfg is not None else None
    explain_enabled = bool(getattr(explain_cfg, "enabled", False)) if explain_cfg is not None else False
    explain_log_every = int(getattr(explain_cfg, "log_every", 10)) if explain_cfg is not None else 10
    explain_targets = list(getattr(explain_cfg, "targets", [])) if explain_cfg is not None else []
    # Instantiate reasoner (stateless across steps; safe to reuse)
    graph_reasoner = GraphReasoner(label_mapper=label_mapper, cfg=GraphConfig())

    # M1 risk parameters (read once)
    risk_alpha = 1.0
    risk_beta = 1.0
    risk_threshold = 0.5
    # Post-process normalization: none | unit | sigmoid
    risk_norm_mode = "none"
    risk_norm_k = 2.0
    risk_norm_center = 0.5  # default; will fallback to threshold if not provided
    # Publish strategy: quantile target, rate, max points, cold start steps
    risk_pub_quantile_target = 0.0  # 0 disables
    risk_pub_every_steps = 1        # 1 = publish every step
    risk_pub_max_points = 0         # <=0 disables capping
    risk_cold_start_steps = 0       # steps to delay risk publishing after episode start
    try:
        if _risk_cfg is not None:
            risk_alpha = float(getattr(_risk_cfg, "alpha", 1.0))
            risk_beta = float(getattr(_risk_cfg, "beta", 1.0))
            risk_threshold = float(getattr(_risk_cfg, "threshold", 0.5))
            # normalization sub-config
            _risk_norm = getattr(_risk_cfg, "normalize", None)
            if _risk_norm is not None:
                risk_norm_mode = str(getattr(_risk_norm, "mode", "none")).lower()
                risk_norm_k = float(getattr(_risk_norm, "sigmoid_k", 2.0))
                # If not explicitly set, use current threshold as center by default
                risk_norm_center = float(getattr(_risk_norm, "sigmoid_center", risk_threshold))
            # publish sub-config
            _risk_pub = getattr(_risk_cfg, "publish", None)
            if _risk_pub is not None:
                risk_pub_quantile_target = float(getattr(_risk_pub, "quantile_target", 0.0))
                risk_pub_every_steps = int(getattr(_risk_pub, "every_steps", 1))
                risk_pub_max_points = int(getattr(_risk_pub, "max_points", 0))
            # cold start delay (top-level risk config)
            risk_cold_start_steps = int(getattr(_risk_cfg, "cold_start_steps", 0))
    except Exception:
        pass

    # 主循环：遍历所有episode
    for epi in range(number_of_episodes - num_total):
        # 发布进度信息
        publish_int32_array(progress_pub, [num_total, number_of_episodes])

        # 选择 episode：
        # - minibatch 模式：按列表定位 (scene, episode_id)
        # - 单次测试模式：按 test_epi_num 偏移
        if minibatch_mode:
            try:
                # 运行 index = num_total（当前已完成后下一个）
                pick_idx = int(num_total)
                target = minibatch_items[pick_idx]
                want_scene_raw = str(target.get("scene"))
                want_eid = int(target.get("episode_id"))
                # 生成可能的 scene_id 候选（兼容 hm3d/hm3d_v0.2、绝对/相对路径、缺失前缀等）
                want_scene_cands = _candidate_scene_ids_from_local(want_scene_raw)
                # 遍历到指定 episode
                found = False
                for ep in env.episode_iterator:
                    env.current_episode = ep
                    ep_sid = _norm_scene_suffix(str(ep.scene_id))
                    if (ep_sid in want_scene_cands) and int(ep.episode_id) == want_eid:
                        found = True
                        break
                if not found:
                    # 打印更友好的诊断，包括候选匹配
                    try:
                        shown = want_scene_cands[0] if len(want_scene_cands) > 0 else want_scene_raw
                    except Exception:
                        shown = want_scene_raw
                    print(
                        f"[Minibatch] WARN: episode not found: scene={shown}, episode_id={want_eid}; skipping"
                    )
                    # 直接跳过，进入下一次循环
                    num_total += 1
                    pbar.update()
                    continue
            except Exception as e:
                print(f"[Minibatch] WARN: failed to select episode: {e}; skipping")
                num_total += 1
                pbar.update()
                continue
        elif flag_once:
            while env_count:
                env.current_episode = next(env.episode_iterator)
                env_count -= 1

        # 初始化episode变量
        pass_object = 0.0  # 是否经过目标对象
        near_object = 0.0  # 是否接近目标对象
        global_action = None  # 全局动作
        cld_with_score_msg = MultipleMasksWithConfidence()  # 点云消息
        count_steps = 0  # 步数计数器

        camera_pitch = 0.0  # 相机俯仰角
        # 重置环境并获取初始观测
        observations = env.reset()
        observations["camera_pitch"] = camera_pitch
        msg_observations = deepcopy(observations)
        del observations["camera_pitch"]
        # 解析 Habitat episode 的类别并映射到统一名称（与 C++ 一致）
        raw_category = env.current_episode.object_category
        label = label_mapper.episode_category_to_name(raw_category)
        if sem_debug:
            # Show episode category -> canonical name, and the id for the first alias
            try:
                right_aliases = list(map(str.strip, str(label).split("|"))) if isinstance(label, str) else []
                first_alias = right_aliases[0] if right_aliases else str(label)
                cid0 = label_mapper.name_to_id(first_alias)
                cname0 = label_mapper.id_to_name(cid0)
                print(
                    f"[MAP] Episode category='{raw_category}' -> canonical='{label}' | first_alias='{first_alias}' -> id={cid0}(:{cname0})"
                )
            except Exception as e:
                print(f"[MAP] WARN: failed to debug-print episode mapping: {e}")

        # 获取大语言模型的答案和融合阈值
        llm_answer, room, fusion_threshold = read_answer(
            llm_answer_path, llm_response_path, label, llm_client
        )

        # 启动本 episode 的额外日志
        try:
            scene_id = env.current_episode.scene_id
            episode_id = env.current_episode.episode_id
            ds_name = "mp3d" if "mp3d" in str(cfg.habitat.dataset.data_path).lower() else "hm3dv2"
            extra_logger.start_episode(dataset=ds_name, scene_id=scene_id, episode_id=episode_id, target_label=str(label))
            search_verifier.start_episode(
                dataset=ds_name,
                scene_id=scene_id,
                episode_id=episode_id,
                target_label=str(label),
                llm_candidates=llm_answer,
            )
            if textual_summary_mgr is not None:
                textual_summary_mgr.start_episode(ds_name, scene_id, episode_id)
        except Exception:
            pass

        # 缓存 Stop&Look 开关（用于识别环绕动作）
        try:
            _look_enabled_cached = bool(rospy.get_param("/exploration_node/look_around/enabled"))
        except Exception:
            try:
                _look_enabled_cached = bool(rospy.get_param("look_around/enabled"))
            except Exception:
                _look_enabled_cached = False

        # 初始化视频帧集合
        vis_frames = []
        # 初始化不确定性 BEV 图帧集合（按步保存为图片，并在收尾生成视频）
        uncert_frames = [] if uncert_enable else None
        # 若未在配置中指定 explain.targets，则默认使用本 episode 的目标类别作为解释对象
        ep_explain_targets = explain_targets if len(explain_targets) > 0 else [label]

        # 额外复合可视化（每步保存一个横向拼接图，episode 结束导出视频）
        comp_frames = [] if viz_comp_enable else None
        info = env.get_metrics()
        # 如果需要生成视频，则添加初始帧
        if need_video:
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames = [frame]

        # 为当前 episode 初始化记忆；若使用 Stream3R，同步重置其 KV。
        try:
            scene_id = env.current_episode.scene_id
            episode_id = env.current_episode.episode_id
            session_id = f"{os.path.basename(scene_id)}_{episode_id}"
        except Exception:
            session_id = f"episode_{num_total+1}"
        point_mem.reset(session_id=session_id, clear_memory=True)
        try:
            voxel_fuser.clear()
        except Exception:
            pass
        # 首帧：根据映射来源选择更新路径
        try:
            if mapping_source == "stream3r":
                init_bgr = transform_rgb_bgr(msg_observations["rgb"])  # RGB->BGR
                added, total_pts = point_mem.update(init_bgr)
                print(f"[Stream3R] Init frame fused: +{added} pts (total {total_pts}).")
            else:
                # Habitat 传感器：需要 C++ 侧的 K 与当前相机位姿
                if _cpp_pose_latest is None:
                    print("[HabitatMap] NOTE: sensor pose not yet available; skip initial fuse.")
                else:
                    K_cpp = _ensure_cpp_intrinsics()
                    extri_cpp = _odometry_to_extrinsic_cam_from_world(_cpp_pose_latest)
                    # 深度去归一化（使用 Habitat 配置）
                    dcfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
                    min_d = float(getattr(dcfg, "min_depth", 0.0))
                    max_d = float(getattr(dcfg, "max_depth", 5.0))
                    stride = int(getattr(mem_cfg, "points_stride", 4)) if mem_cfg is not None else 4
                    pts_w = _depth_to_world_points(
                        msg_observations["depth"][:, :, 0],
                        K=K_cpp,
                        extri_cw=extri_cpp,
                        stride=stride,
                        min_depth=min_d,
                        max_depth=max_d,
                    )
                    added, total_pts = point_mem.ingest(pts_w, intrinsics=K_cpp, extrinsics=extri_cpp)
                    print(f"[HabitatMap] Init frame fused: +{added} pts (total {total_pts}).")
        except Exception as e:
            print(f"[Mapping] WARN: initial memory update failed: {e}")

        # 开始发布基本信息和触发消息
        pub_timer = rospy.Timer(rospy.Duration(0.25), publish_observations)

        print("Agent is waiting in the environment!!!")

        # 等待ROS系统准备就绪
        rate = rospy.Rate(10)
        ros_state = ROS_STATE.INIT
        while ros_state == ROS_STATE.INIT or ros_state == ROS_STATE.WAIT_TRIGGER:
            if ros_state == ROS_STATE.INIT:
                print("Waiting for ROS to get odometry...")
            elif ros_state == ROS_STATE.WAIT_TRIGGER:
                print("Waiting for ROS trigger...")
            rate.sleep()

        # 停止定时器发布
        pub_timer.shutdown()

        print("Agent is ready to go!!!!")
        # Debug aids: throttle waiting logs inside main loop
        dbg_last_wait_log = time.time()
        dbg_wait_log_interval = 2.0  # seconds
        print("[DBG] Entering main loop. Waiting for actions from ROS...")

        # 主执行循环
        rate = rospy.Rate(10)
        action_label = "UNKNOWN"
        # Track last ROS state we observed to detect ENTER events for LOOK_AROUND
        ros_state_prev_for_log = None
        while not rospy.is_shutdown() and not env.episode_over:
            # 检查目标是否在同一楼层
            is_feasible = 0
            for goal in env.current_episode.goals:
                height = goal.position[1]
                is_feasible += is_on_same_floor(
                    height=height, episode=env.current_episode
                )
            if not is_feasible:
                break

            # 解析来自决策系统的动作
            action = None
            if global_action is not None:
                print(f"[DBG] Received ROS action code: {global_action}")
                # 如果达到最大步数，强制执行STOP动作
                if count_steps == max_episode_steps - 1:
                    global_action = ACTION.STOP

                # 将ROS动作映射到Habitat动作
                action_name = None
                if global_action == ACTION.MOVE_FORWARD:
                    action = HabitatSimActions.move_forward
                    action_name = "MOVE_FORWARD"
                elif global_action == ACTION.TURN_LEFT:
                    action = HabitatSimActions.turn_left
                    action_name = "TURN_LEFT"
                elif global_action == ACTION.TURN_RIGHT:
                    action = HabitatSimActions.turn_right
                    action_name = "TURN_RIGHT"
                elif global_action == ACTION.TURN_DOWN:
                    action = HabitatSimActions.look_down
                    camera_pitch = camera_pitch - np.pi / 6.0
                    action_name = "TURN_DOWN"
                elif global_action == ACTION.TURN_UP:
                    action = HabitatSimActions.look_up
                    camera_pitch = camera_pitch + np.pi / 6.0
                    action_name = "TURN_UP"
                elif global_action == ACTION.STOP:
                    action = HabitatSimActions.stop
                    action_name = "STOP"

                if action is not None:
                    print(f"[DBG] Mapped to Habitat action: {action_name}")
                    action_label = str(action_name)

                global_action = None

            # 如果没有动作，则继续等待（让出GIL给ROS回调处理）
            if action is None:
                # Important: sleep so rospy subscriber callbacks can run and set `global_action`
                now = time.time()
                if now - dbg_last_wait_log >= dbg_wait_log_interval:
                    print(f"[DBG] Waiting for action... ros_state={ros_state}")
                    dbg_last_wait_log = now
                rate.sleep()
                continue

            # 增加步数计数器
            count_steps += 1
            print(f"\n--------------Step: {count_steps}--------------")
            print(f"Finding [{label}]; Action: {action};")

            # 通知ROS系统开始执行动作
            print("[DBG] Publishing HABITAT_STATE.ACTION_EXEC")
            publish_int32(state_pub, HABITAT_STATE.ACTION_EXEC)

            # 执行动作并获取新观测
            _t_env0 = time.time()
            print("[DBG] Stepping environment...")
            observations = env.step(action)
            _t_env1 = time.time()
            print("[DBG] Environment stepped; got new observations")

            # 计算ITM余弦相似度分数
            _t_itm0 = time.time()
            print("[DBG] Computing ITM cosine score...")
            cosine = get_itm_message_cosine(observations["rgb"], label, room)
            _t_itm1 = time.time()
            print(f"Target related room: {room}")
            print(f"ITM cosine similarity: {cosine:.3f}")

            # 发布余弦相似度分数
            publish_float64(itm_score_pub, cosine)

            # 在当前观测中检测对象
            _t_det0 = time.time()
            print("[DBG] Running detector to get object masks...")
            # 保留原始观测图像（用于后续可视化拼接）
            raw_rgb_for_comp = observations["rgb"].copy()  # RGB
            # 检测/分割使用 BGR，并将结果转回 RGB 写回 observations
            det_input_bgr = transform_rgb_bgr(observations["rgb"])  # RGB->BGR
            det_bgr, score_list, object_masks_list, label_list, bbox_list, detector_indices = get_object(
                label, det_input_bgr, detector_cfg, llm_answer
            )
            _t_det1 = time.time()
            observations["rgb"] = transform_bgr_rgb(det_bgr)  # BGR->RGB，供下游使用
            try:
                n_masks = len(object_masks_list)
            except Exception:
                n_masks = 0
            print(f"[DBG] Detector done. masks={n_masks}")
            # 记录仅包含检测可视化的图像（在任何其它叠加之前，BGR 色彩空间）
            det_bgr_for_comp = det_bgr.copy()

            # 发布habitat观测到ROS
            print("[DBG] Publishing observations to ROS topics...")
            observations["camera_pitch"] = camera_pitch
            msg_observations = deepcopy(observations)
            del observations["camera_pitch"]
            ros_pub.habitat_publish_ros_topic(msg_observations)

            # 生成并发布对象点云
            print("[DBG] Building object point clouds...")
            obj_point_cloud_list = get_object_point_cloud(
                cfg, observations, object_masks_list
            )
            print(f"[DBG] Built {len(obj_point_cloud_list)} point cloud(s) from masks")

            # 发布检测相关信息
            print("[DBG] Publishing /detector/clouds_with_scores ...")
            cld_with_score_msg.point_clouds = obj_point_cloud_list
            cld_with_score_msg.confidence_scores = score_list
            # C++ ObjectMap2D expects labels in [0..4] (arrays sized 5). Clamp to avoid OOB.
            try:
                ros_label_indices = []
                for li in label_list:
                    ival = int(li)
                    if ival < 0:
                        ival = 0
                    if ival > 4:
                        ival = 4
                    ros_label_indices.append(ival)
            except Exception:
                ros_label_indices = [0 for _ in label_list]
            cld_with_score_msg.label_indices = ros_label_indices
            cld_with_score_msg.detection_indices = list(map(int, detector_indices))
            try:
                search_verifier.update_latest_frame(
                    step_idx=count_steps,
                    rgb=raw_rgb_for_comp,
                    masks=object_masks_list,
                    scores=score_list,
                    ros_labels=ros_label_indices,
                    bboxes_px=bbox_list,
                    detector_indices=detector_indices,
                )
            except Exception:
                pass
            cld_with_score_pub.publish(cld_with_score_msg)

            # M7: 发布对象级摘要（位置/类别/置信）
            try:
                obj_summary = SemanticObjectArray()
                obj_summary.header.stamp = rospy.Time.now()
                obj_summary.header.frame_id = "map"  # world frame
                objs = []
                # Reconstruct human-readable names from detector label_list
                # label_list: 0 -> target label; 1.. -> llm_answer[idx-1]
                right_label_list = list(map(str.strip, label.split("|")))
                for i, pc_msg in enumerate(obj_point_cloud_list):
                    # Derive name from detector labels
                    try:
                        li = int(label_list[i])
                    except Exception:
                        li = 0
                    if li == 0:
                        name = right_label_list[0] if len(right_label_list) > 0 else str(label)
                    else:
                        idx = li - 1
                        name = str(llm_answer[idx]) if 0 <= idx < len(llm_answer) else str(label)
                    cid = int(label_mapper.name_to_id(name))
                    # Centroid
                    cen = _pc2_centroid(pc_msg)
                    if cen is None:
                        continue
                    so = SemanticObject()
                    so.id = int(i)
                    so.label_id = cid
                    so.label_name = str(name)
                    try:
                        confv = float(score_list[i])
                    except Exception:
                        confv = 0.0
                    so.confidence = float(max(0.0, min(1.0, confv)))
                    so.position.x = float(cen[0])
                    so.position.y = float(cen[1])
                    so.position.z = float(cen[2])
                    objs.append(so)
                obj_summary.objects = objs
                if len(objs) > 0:
                    obj_summary_pub.publish(obj_summary)
            except Exception as e:
                print(f"[M7] WARN: object summary publish failed: {e}")

            # 更新记忆体（Stream3R 或 Habitat 传感器）
            try:
                _t_mem0 = time.time()
                if mapping_source == "stream3r":
                    print("[DBG] Stream3R: updating memory with current frame...")
                    cur_bgr = transform_rgb_bgr(observations["rgb"])  # RGB->BGR
                    added, total_pts = point_mem.update(cur_bgr)
                    print(f"[Stream3R] Step fused: +{added} pts (total {total_pts}).")
                else:
                    # Habitat：利用深度 + C++ K/T 回投
                    if _cpp_pose_latest is None:
                        print("[HabitatMap] WARN: sensor pose not ready; skip this step.")
                        added = 0
                    else:
                        K_cpp = _ensure_cpp_intrinsics()
                        extri_cpp = _odometry_to_extrinsic_cam_from_world(_cpp_pose_latest)
                        dcfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
                        min_d = float(getattr(dcfg, "min_depth", 0.0))
                        max_d = float(getattr(dcfg, "max_depth", 5.0))
                        stride = int(getattr(mem_cfg, "points_stride", 4)) if mem_cfg is not None else 4
                        pts_w = _depth_to_world_points(
                            observations["depth"][:, :, 0],
                            K=K_cpp,
                            extri_cw=extri_cpp,
                            stride=stride,
                            min_depth=min_d,
                            max_depth=max_d,
                        )
                        added, total_pts = point_mem.ingest(pts_w, intrinsics=K_cpp, extrinsics=extri_cpp)
                        print(f"[HabitatMap] Step fused: +{added} pts (total {total_pts}).")
                _t_mem1 = time.time()
            except Exception as e:
                print(f"[Mapping] WARN: step memory update failed: {e}")
                _t_mem0 = _t_mem1 = None

            # 计算本步需要的投影：
            # - new_pts（绑定内存属性、视频叠加）
            # - frame_pts（可选，用于 M4 observe_all 融合）
            u_pix = v_pix = valid = None  # for new_pts
            u_all = v_all = valid_all = None  # for frame_pts
            labels_np = None  # for new_pts
            new_pts = None
            frame_pts = None
            try:
                raw_h, raw_w = observations["rgb"].shape[:2]
                extri = point_mem.get_poses()[-1]
                intri = point_mem.get_intrinsics()[-1]
                if added > 0:
                    pts_all = point_mem.get_points()
                    new_pts = pts_all[-added:]
                frame_pts = point_mem.get_last_frame_points()
                # Choose preprocess mapping
                if mapping_source == "stream3r":
                    proj_mode = str(getattr(mem_cfg, "preprocess_mode", "crop")) if mem_cfg is not None else "crop"
                    pp = _compute_preprocess_params(raw_h, raw_w, mode=proj_mode)
                    if added > 0 and new_pts is not None:
                        u_pix, v_pix, valid = _project_world_to_raw_uv(new_pts, extri, intri, pp, raw_w, raw_h)
                        _report_kt_diff(intri, extri, raw_h, raw_w, proj_mode)
                    if fusion_observe_all and frame_pts is not None and frame_pts.size > 0:
                        u_all, v_all, valid_all = _project_world_to_raw_uv(frame_pts, extri, intri, pp, raw_w, raw_h)
                else:
                    # Habitat 无裁剪/填充
                    pp = {"sx": 1.0, "sy": 1.0, "off_x": 0.0, "off_y": 0.0, "mode": "crop"}
                    if added > 0 and new_pts is not None:
                        u_pix, v_pix, valid = _project_world_to_raw_uv(new_pts, extri, intri, pp, raw_w, raw_h)
                    if fusion_observe_all and frame_pts is not None and frame_pts.size > 0:
                        u_all, v_all, valid_all = _project_world_to_raw_uv(frame_pts, extri, intri, pp, raw_w, raw_h)
                try:
                    search_verifier.update_memory_projection_cache(
                        step_idx=count_steps,
                        extrinsic_cam_from_world=extri,
                        intrinsic_preproc=intri,
                        image_shape=(raw_h, raw_w),
                        preprocess=pp,
                        frame_points_world=frame_pts,
                    )
                except Exception:
                    pass
            except Exception:
                u_pix = v_pix = valid = None
                u_all = v_all = valid_all = None

            # M3: 为本步新增点绑定语义标签（可选，按配置开关）
            try:
                sem_cfg = getattr(cfg, "stream3r", None)
                sem_enabled = False
                if sem_cfg is not None:
                    _mem = getattr(sem_cfg, "memory", None)
                    _sem = getattr(_mem, "semantic", None) if _mem is not None else None
                    sem_enabled = bool(getattr(_sem, "enabled", False))

                has_masks = len(object_masks_list) > 0
                if sem_enabled and has_masks:
                    if sem_debug:
                        try:
                            print(
                                f"[MAP] Semantic step: added={added}, masks={len(object_masks_list)}, label_space=mapper={label_mapper.label_space}, mem={mem_semantic_label_space}, observe_all={fusion_observe_all}"
                            )
                            if label_mapper.label_space != mem_semantic_label_space:
                                print("[MAP][WARN] label_space mismatch between mapper and memory; mapped ids may not align expected space.")
                        except Exception:
                            pass
                    # Reconstruct phrases from label_list
                    right_label_list = list(map(str.strip, label.split("|")))
                    all_answer = right_label_list + llm_answer
                    phrases = []
                    for li in label_list:
                        if int(li) == 0:
                            phrases.append(right_label_list[0] if len(right_label_list) > 0 else label)
                        else:
                            idx = int(li) - 1
                            if 0 <= idx < len(llm_answer):
                                phrases.append(str(llm_answer[idx]))
                            else:
                                phrases.append(label)  # fallback to target

                    if sem_debug:
                        print(f"[MAP] right_label_list={right_label_list}")
                        print(f"[MAP] llm_answer={llm_answer}")

                    # 1) 内存属性绑定：仅对新增点
                    if added > 0 and u_pix is not None:
                        labels_np = np.full((added,), -1, dtype=np.int32)
                        conf_np = np.zeros((added,), dtype=np.float32)
                        assigned_total = 0
                        for msk, score, phrase in zip(object_masks_list, score_list, phrases):
                            if msk is None:
                                continue
                            cid = label_mapper.name_to_id(str(phrase))
                            if cid == label_mapper.unknown_id:
                                if sem_debug:
                                    print(f"[MAP] phrase='{phrase}' -> id={cid} (UNKNOWN); skip")
                                continue
                            msk_bool = msk.astype(bool)
                            hits = msk_bool[v_pix, u_pix] & valid
                            better = hits & (score > conf_np)
                            if np.any(better):
                                labels_np[better] = int(cid)
                                conf_np[better] = float(score)
                                assigned_total += int(np.count_nonzero(better))
                            if sem_debug:
                                cname = label_mapper.id_to_name(int(cid))
                                print(f"[MAP] phrase='{phrase}' -> id={cid}(:{cname}); hits={int(np.count_nonzero(hits))}, newly_assigned={int(np.count_nonzero(better))}, score={float(score):.3f}")
                        # 纠错并追加
                        if np.any(conf_np > 0):
                            try:
                                if voxel_fuser is not None and fusion_conf.auto_correct and new_pts is not None:
                                    changed = voxel_fuser.auto_correct_points(new_pts, labels_np)
                                    if sem_debug and changed > 0:
                                        print(f"[M4] auto-corrected {changed}/{added} labels by voxel majority before fusion")
                            except Exception:
                                pass
                            point_mem.append_attributes(labels=labels_np, label_conf=conf_np)
                            if sem_debug:
                                known_now = int(np.count_nonzero(labels_np >= 0))
                                print(f"[MAP] appended labels for {known_now}/{added} new points (assigned_total={assigned_total})")
                        elif sem_debug:
                            print("[MAP] No points assigned a label for new points in this step")

                    # 2) 体素融合：可选使用整帧点（observe_all）
                    try:
                        step_ts = float(point_mem.stats().last_step_idx)
                        if fusion_observe_all and frame_pts is not None and u_all is not None:
                            n_all = int(frame_pts.shape[0])
                            labels_all = np.full((n_all,), -1, dtype=np.int32)
                            conf_all = np.zeros((n_all,), dtype=np.float32)
                            for msk, score, phrase in zip(object_masks_list, score_list, phrases):
                                if msk is None:
                                    continue
                                cid = label_mapper.name_to_id(str(phrase))
                                if cid == label_mapper.unknown_id:
                                    continue
                                msk_bool = msk.astype(bool)
                                hits = msk_bool[v_all, u_all] & valid_all
                                better = hits & (score > conf_all)
                                if np.any(better):
                                    labels_all[better] = int(cid)
                                    conf_all[better] = float(score)
                            voxel_fuser.update(
                                frame_pts,
                                labels=labels_all,
                                conf=conf_all,
                                extrinsic_cam_from_world=extri,
                                timestamp=step_ts,
                            )
                        elif added > 0 and labels_np is not None and new_pts is not None:
                            voxel_fuser.update(
                                new_pts,
                                labels=labels_np,
                                conf=conf_np,
                                extrinsic_cam_from_world=extri,
                                timestamp=step_ts,
                            )
                        if sem_debug and (point_mem.stats().steps % 5 == 0):
                            dv = voxel_fuser.get_disputed_voxels(timestamp=step_ts)
                            print(f"[M4] disputed_voxels={len(dv)} (sample printed every 5 steps)")
                    except Exception as e:
                        if sem_debug:
                            print(f"[M4] WARN: voxel fusion update skipped due to error: {e}")
                    # Live voxel probability visualization to RViz
                    try:
                        if voxels_vis_pub is not None and vox_vis_enable:
                            steps_now = int(point_mem.stats().steps)
                            if vox_vis_every <= 1 or (steps_now % max(vox_vis_every, 1) == 0):
                                # Apply a global time decay tick so even non-touched
                                # voxels age visually with `time_decay_tau`.
                                try:
                                    voxel_fuser.tick(timestamp=step_ts)
                                except Exception:
                                    pass
                                cents, labs, probs = voxel_fuser.export_voxels()
                                if cents is not None and cents.size > 0:
                                    # Filter and cap to keep RViz responsive
                                    mask = probs >= float(vox_vis_minp)
                                    cents_v = cents[mask]
                                    probs_v = probs[mask]
                                    labs_v = labs[mask] if labs is not None and labs.shape[0] == cents.shape[0] else None
                                    if cents_v.shape[0] > vox_vis_cap > 0:
                                        idx = np.random.choice(cents_v.shape[0], size=vox_vis_cap, replace=False)
                                        cents_v = cents_v[idx]
                                        probs_v = probs_v[idx]
                                        if labs_v is not None:
                                            labs_v = labs_v[idx]
                                    # Delete previous marker to avoid leftover cubes
                                    mk_del = Marker()
                                    mk_del.header.frame_id = vox_vis_frame
                                    mk_del.header.stamp = rospy.Time.now()
                                    mk_del.ns = "voxel_probs"
                                    mk_del.id = 0
                                    mk_del.type = Marker.CUBE_LIST
                                    mk_del.action = Marker.DELETE
                                    voxels_vis_pub.publish(mk_del)
                                    # Publish new cube list
                                    mk = Marker()
                                    mk.header.frame_id = vox_vis_frame
                                    mk.header.stamp = rospy.Time.now()
                                    mk.ns = "voxel_probs"
                                    mk.id = 0
                                    mk.type = Marker.CUBE_LIST
                                    mk.action = Marker.ADD
                                    sz = float(vox_vis_size if vox_vis_size > 0 else float(mem_args.get("voxel_size", 0.05)))
                                    mk.scale.x = sz
                                    mk.scale.y = sz
                                    mk.scale.z = sz
                                    mk.pose.orientation.w = 1.0
                                    from geometry_msgs.msg import Point as _Pt
                                    from std_msgs.msg import ColorRGBA as _Clr
                                    # Choose color scheme: prob | label | label_prob
                                    use_label = vox_vis_color_mode in ("label", "label_prob") and (labs_v is not None)
                                    for i in range(cents_v.shape[0]):
                                        x, y, z = map(float, cents_v[i])
                                        p = float(probs_v[i])
                                        mk.points.append(_Pt(x=x, y=y, z=z))
                                        if use_label:
                                            lid = int(labs_v[i])
                                            # hash label id -> RGB in [0,1]
                                            r = ((37 * lid + 17) % 255) / 255.0
                                            g = ((57 * lid + 31) % 255) / 255.0
                                            b = ((97 * lid + 73) % 255) / 255.0
                                            if vox_vis_color_mode == "label_prob":
                                                # dim by probability
                                                r *= p
                                                g *= p
                                                b *= p
                                        else:
                                            # probability colormap: red->green
                                            r = max(0.0, 1.0 - p)
                                            g = max(0.0, p)
                                            b = 0.0
                                        a = (0.2 + 0.8 * p) if vox_vis_alpha_by_prob else 0.8
                                        mk.colors.append(_Clr(r=float(r), g=float(g), b=float(b), a=float(a)))
                                    voxels_vis_pub.publish(mk)
                                    if vox_vis_debug:
                                        try:
                                            print(
                                                f"[RViz] voxel_probs: sent {len(mk.points)} cubes | frame={vox_vis_frame} size={mk.scale.x:.3f} color_mode={vox_vis_color_mode} min_prob={vox_vis_minp}"
                                            )
                                        except Exception:
                                            pass
                    except Exception as e:
                        if sem_debug:
                            print(f"[M4] WARN: voxel RViz publish skipped: {e}")
            except Exception as e:
                # 保持鲁棒性：语义绑定失败不影响主流程
                print(f"[M3] WARN: semantic binding skipped due to error: {e}")

            # 每步统一应用一次时间衰减（受 time_decay_tau 控制）。
            # 放在主循环中，保证可视化/热区/解释关闭时也持续衰减。
            try:
                step_ts = float(point_mem.stats().last_step_idx)
                voxel_fuser.tick(timestamp=step_ts)
            except Exception:
                pass

            # M5: 可解释图结构推理（按需）
            try:
                if explain_enabled and (point_mem.stats().steps % max(explain_log_every, 1) == 0):
                    try:
                        voxel_fuser.tick(timestamp=point_mem.stats().last_step_idx)
                    except Exception:
                        pass
                    cents, labs, probs = voxel_fuser.export_voxels()
                    agent_pos = None
                    try:
                        # 从当前 extrinsic 计算相机中心 C_w = -R^T t
                        R_cw = extri[:, :3].astype(np.float32)
                        t_cw = extri[:, 3].astype(np.float32)
                        agent_pos = (-R_cw.T @ t_cw).astype(np.float32)
                    except Exception:
                        agent_pos = None
                    graph_reasoner.build_from_voxels(cents, labs, probs, agent_position=agent_pos)
                    # 目标类说明
                    for tgt in (ep_explain_targets or []):
                        exp = graph_reasoner.explain_why_go_to(str(tgt))
                        print(f"[M5] why-go-to[{tgt}]: {exp.get('text', '')}")
                    # 前沿探索建议
                    exp_explore = graph_reasoner.explain_should_explore("frontier")
                    print(f"[M5] explore(frontier): {exp_explore.get('text', '')}")
            except Exception as e:
                if sem_debug:
                    print(f"[M5] WARN: graph reasoning skipped due to error: {e}")
            
            # M7: Publish disputed voxel hotspots (optional, lightweight handoff)
            try:
                # Age voxels before computing hotspots to reflect time decay
                try:
                    voxel_fuser.tick(timestamp=point_mem.stats().last_step_idx)
                except Exception:
                    pass
                _t_hot0 = time.time()
                disputed = voxel_fuser.get_disputed_voxels(timestamp=point_mem.stats().last_step_idx)
                disputed_count_this_step = len(disputed) if disputed is not None else 0
                if disputed and len(disputed) > 0:
                    hs = VoxelHotspotArray()
                    hs.header.stamp = rospy.Time.now()
                    hs.header.frame_id = "map"
                    from geometry_msgs.msg import Point  # local import to avoid init cost at module import
                    pts = []
                    lbls = []
                    confs = []
                    for cen, lid, p1 in disputed:
                        pts.append(Point(x=float(cen[0]), y=float(cen[1]), z=float(cen[2])))
                        lbls.append(int(lid))
                        confs.append(float(p1))
                    hs.positions = pts
                    hs.label_ids = lbls
                    hs.confidence = confs
                    hotspots_pub.publish(hs)
                _t_hot1 = time.time()
            except Exception as e:
                if sem_debug:
                    print(f"[M7] WARN: hotspot publish skipped: {e}")
                _t_hot0 = _t_hot1 = None
                disputed_count_this_step = 0

            # M1: RiskCloud 发布（PointXYZI，intensity=risk）
            try:
                # 导出体素与置信度，并根据争议体素计算 risk
                try:
                    voxel_fuser.tick(timestamp=point_mem.stats().last_step_idx)
                except Exception:
                    pass
                _t_risk0 = time.time()
                cents, labs, probs = voxel_fuser.export_voxels()
                disputed_list = voxel_fuser.get_disputed_voxels(timestamp=point_mem.stats().last_step_idx)
                risk_vals, is_disp = compute_risk_from_voxels(
                    cents, probs, disputed_list, alpha=risk_alpha, beta=risk_beta
                )
                try:
                    search_verifier.update_memory_risk_snapshot(
                        centroids=cents,
                        voxel_conf=probs,
                        disputed_list=disputed_list,
                        risk_alpha=risk_alpha,
                        risk_beta=risk_beta,
                        risk_values=risk_vals,
                    )
                except Exception:
                    pass
                steps_now = int(point_mem.stats().steps)
                # 发布节流与冷启动控制
                if risk_cold_start_steps > 0 and steps_now < risk_cold_start_steps:
                    raise RuntimeError("skip_publish_cold_start")
                if risk_pub_every_steps > 1 and (steps_now % max(risk_pub_every_steps, 1) != 0):
                    raise RuntimeError("skip_publish_rate_limit")
                if cents is not None and cents.size > 0 and risk_vals.size == cents.shape[0]:
                    # 归一化 / 压缩（可选）
                    if risk_norm_mode == "unit":
                        denom = max(risk_alpha + risk_beta, 1e-6)
                        risk_vals = np.clip(risk_vals / denom, 0.0, 1.0)
                    elif risk_norm_mode == "sigmoid":
                        # Smooth step around center; map to (0,1)
                        x = (risk_vals - float(risk_norm_center)) * float(risk_norm_k)
                        risk_vals = 1.0 / (1.0 + np.exp(-x))
                    # 动态阈值（可选）：按分位数调节，永不低于静态阈值
                    dyn_thr = float(risk_threshold)
                    if 0.0 < risk_pub_quantile_target <= 1.0:
                        try:
                            qv = float(np.quantile(risk_vals, risk_pub_quantile_target))
                            if qv > dyn_thr:
                                dyn_thr = qv
                        except Exception:
                            pass
                    # 选点并限量
                    idx_all = np.where(risk_vals > dyn_thr)[0]
                    if idx_all.size > 0:
                        if risk_pub_max_points > 0 and idx_all.size > risk_pub_max_points:
                            # 取 risk 值最大的前K个
                            rsub = risk_vals[idx_all]
                            # argpartition to get top-K indices
                            K = int(risk_pub_max_points)
                            part = np.argpartition(rsub, -K)[-K:]
                            idx_all = idx_all[part]
                            # 可选：按风险降序排序（便于读数）
                            order = np.argsort(risk_vals[idx_all])[::-1]
                            idx_all = idx_all[order]
                        t0 = time.time()
                        msg = _make_pointcloud2_xyzi(cents[idx_all], risk_vals[idx_all], frame_id="map")
                        risk_pub.publish(msg)
                        t1 = time.time()
                        if sem_debug and (steps_now % 10 == 0):
                            try:
                                r_mean = float(risk_vals[idx_all].mean()) if idx_all.size > 0 else 0.0
                                r_max = float(risk_vals[idx_all].max()) if idx_all.size > 0 else 0.0
                                r_p95 = float(np.quantile(risk_vals, 0.95)) if risk_vals.size > 0 else 0.0
                                print(
                                    f"[M1] risk_cloud: N={int(idx_all.size)} thr={dyn_thr:.3f} mean={r_mean:.3f} max={r_max:.3f} p95={r_p95:.3f} mode={risk_norm_mode} dt={int((t1-t0)*1000)}ms"
                                )
                            except Exception:
                                pass
            except Exception as e:
                if sem_debug:
                    # filter out internal control flow exceptions used to skip publish
                    if str(e) not in ("skip_publish_cold_start", "skip_publish_rate_limit"):
                        print(f"[M1] WARN: risk cloud publish skipped: {e}")
                _t_risk0 = None

            # Per-step uncertainty BEV image saving (independent of risk publish rate limits)
            try:
                if uncert_enable:
                    # Optionally decimate by steps
                    steps_now = int(point_mem.stats().steps)
                    if uncert_every <= 1 or (steps_now % max(uncert_every, 1) == 0):
                        # Export latest voxels and compute risk without gating
                        try:
                            voxel_fuser.tick(timestamp=point_mem.stats().last_step_idx)
                        except Exception:
                            pass
                        cents_u, labs_u, probs_u = voxel_fuser.export_voxels()
                        if cents_u is not None and cents_u.size > 0 and probs_u is not None and probs_u.size == cents_u.shape[0]:
                            disputed_u = None
                            try:
                                disputed_u = voxel_fuser.get_disputed_voxels(timestamp=point_mem.stats().last_step_idx)
                            except Exception:
                                disputed_u = None
                            risk_u, _ = compute_risk_from_voxels(cents_u, probs_u, disputed_u, alpha=risk_alpha, beta=risk_beta)
                            if risk_norm_mode == "unit":
                                denom = max(risk_alpha + risk_beta, 1e-6)
                                risk_u = np.clip(risk_u / denom, 0.0, 1.0)
                            elif risk_norm_mode == "sigmoid":
                                x = (risk_u - float(risk_norm_center)) * float(risk_norm_k)
                                risk_u = 1.0 / (1.0 + np.exp(-x))
                            # Prepare current extrinsic (pose) for BEV frame
                            if mapping_source == "stream3r":
                                try:
                                    extri = point_mem.get_poses()[-1]
                                except Exception:
                                    extri = None
                            else:
                                extri = _odometry_to_extrinsic_cam_from_world(_cpp_pose_latest) if _cpp_pose_latest is not None else None
                            if extri is not None:
                                # Global map points for occupancy boundary (optional)
                                try:
                                    map_pts = point_mem.get_points()
                                except Exception:
                                    map_pts = None
                                bev_bgr = _render_uncertainty_bev(
                                    cents_w=cents_u,
                                    risk_vals=risk_u,
                                    extri_cw=extri,
                                    window_size_m=uncert_bev_window_m,
                                    resolution_m=uncert_bev_res_m,
                                    z_min=uncert_bev_zmin,
                                    z_max=uncert_bev_zmax,
                                    blur_sigma_px=uncert_bev_blur,
                                    draw_agent=uncert_bev_draw_agent,
                                    map_points=map_pts,
                                    draw_occupancy=uncert_bev_draw_occupancy,
                                    occ_max_points=uncert_bev_occ_max_points,
                                    occ_line_thickness=uncert_bev_occ_thickness,
                                    orientation=uncert_bev_orientation,
                                    draw_grid=uncert_bev_draw_grid,
                                    grid_spacing_cells=uncert_bev_grid_spacing,
                                    grid_color=uncert_bev_grid_color,
                                    grid_thickness=uncert_bev_grid_thickness,
                                )
                                if uncert_frames is not None:
                                    # store as RGB for images_to_video
                                    uncert_frames.append(transform_bgr_rgb(bev_bgr))
            except Exception as e:
                if sem_debug:
                    print(f"[M1] WARN: uncertainty image render failed: {e}")

            # Textual environment summary (non-blocking LLM call)
            try:
                if textual_summary_mgr is not None:
                    step_ts = float(point_mem.stats().last_step_idx)
                    textual_summary_mgr.maybe_schedule(
                        step_idx=count_steps,
                        timestamp=step_ts,
                        voxel_fuser=voxel_fuser,
                    )
            except Exception as e:
                if sem_debug:
                    print(f"[LLM] WARN: skipped textual summary: {e}")

            # 额外复合可视化（按需）：每步保存一个拼接图
            try:
                if viz_comp_enable and comp_frames is not None:
                    # 保证使用 BGR 颜色空间进行绘制与拼接
                    raw_bgr = transform_rgb_bgr(raw_rgb_for_comp)
                    det_bgr = det_bgr_for_comp
                    # Panel A: 原始观测 + 点云可视化（叠加到同一张图）
                    if u_pix is not None and v_pix is not None:
                        # Compute coverage by any semantic mask for coloring logic
                        covered_mask = None
                        try:
                            if object_masks_list is not None and len(object_masks_list) > 0:
                                covered_mask = np.zeros((u_pix.shape[0],), dtype=bool)
                                for m in object_masks_list:
                                    if m is None:
                                        continue
                                    m_bool = np.asarray(m).astype(bool)
                                    # Guard shape mismatches via try/except
                                    try:
                                        hits = m_bool[v_pix, u_pix]
                                        if valid is not None:
                                            hits = hits & valid
                                        covered_mask |= hits
                                    except Exception:
                                        continue
                        except Exception:
                            covered_mask = None
                        pcl_overlay_img = _draw_points_overlay(
                            raw_bgr,
                            u_pix,
                            v_pix,
                            valid,
                            point_size=viz_point_size,
                            labels_np=labels_np,
                            mode=viz_overlay_mode,
                            uniform_color=(0, 255, 255),
                            covered_mask=covered_mask,
                        )
                    else:
                        pcl_overlay_img = raw_bgr.copy()
                    panel_a = pcl_overlay_img

                    # Panel B: 原始观测 + 目标检测可视化（叠加图本身）
                    panel_b = det_bgr

                    # Panel C: 原始观测 + 分割可视化（以掩码半透明上色）
                    seg_overlay_img = _apply_colored_masks(raw_bgr, object_masks_list)
                    panel_c = seg_overlay_img

                    composite = _hstack_resize([panel_a, panel_b, panel_c])
                    # 控制宽度（避免 6x 宽度过大）
                    if viz_comp_target_width and viz_comp_target_width > 0:
                        h, w = composite.shape[:2]
                        if w > viz_comp_target_width:
                            scale = float(viz_comp_target_width) / float(w)
                            new_h = max(1, int(round(h * scale)))
                            composite = cv2.resize(composite, (viz_comp_target_width, new_h), interpolation=cv2.INTER_AREA)
                    composite = transform_bgr_rgb(composite)  # BGR->RGB
                    comp_frames.append(composite)
            except Exception:
                pass

            # 可视化叠加：解耦语义门控；即使无掩码/语义失败，也以统一色叠加几何
            try:
                if viz_overlay and added > 0 and u_pix is not None:
                    mask = valid if valid is not None else None
                    if mask is None or np.count_nonzero(mask) > 0:
                        if mask is None:
                            draw_indices = np.arange(added, dtype=np.int64)
                        else:
                            draw_indices = np.nonzero(mask)[0]
                        if draw_indices.size > viz_max_points:
                            sel = np.random.choice(draw_indices.size, viz_max_points, replace=False)
                            draw_indices = draw_indices[sel]
                        # 配色：有语义且 overlay_mode=label -> 已知=哈希色；未知(被掩码覆盖)=灰；未被语义覆盖=黄
                        if viz_overlay_mode == "label" and labels_np is not None:
                            labs = labels_np[draw_indices]
                            # Base: yellow (BGR) for all points
                            colors = np.tile(np.array([[0, 255, 255]], dtype=np.uint8), (draw_indices.size, 1))
                            # Known labels -> hashed BGR
                            ok_known = labs >= 0
                            if np.any(ok_known):
                                lid = labs[ok_known].astype(np.int64)
                                r = (37 * lid + 17) % 255
                                g = (57 * lid + 31) % 255
                                b = (97 * lid + 73) % 255
                                colors[ok_known, 0] = b.astype(np.uint8)  # BGR for cv2
                                colors[ok_known, 1] = g.astype(np.uint8)
                                colors[ok_known, 2] = r.astype(np.uint8)
                            # Unknown labels -> gray only where covered by any mask
                            try:
                                covered_mask = None
                                if object_masks_list is not None and len(object_masks_list) > 0:
                                    covered_mask = np.zeros((u_pix.shape[0],), dtype=bool)
                                    for m in object_masks_list:
                                        if m is None:
                                            continue
                                        m_bool = np.asarray(m).astype(bool)
                                        try:
                                            hits = m_bool[v_pix, u_pix]
                                            if valid is not None:
                                                hits = hits & valid
                                            covered_mask |= hits
                                        except Exception:
                                            continue
                                if covered_mask is not None:
                                    covered_sub = covered_mask[draw_indices]
                                    unk = (labs < 0) & covered_sub
                                    if np.any(unk):
                                        colors[unk] = np.array([128, 128, 128], dtype=np.uint8)
                                else:
                                    # No coverage info -> all unknown gray
                                    unk = labs < 0
                                    if np.any(unk):
                                        colors[unk] = np.array([128, 128, 128], dtype=np.uint8)
                            except Exception:
                                # Robust fallback: unknown -> gray
                                unk = labs < 0
                                if np.any(unk):
                                    colors[unk] = np.array([128, 128, 128], dtype=np.uint8)
                        else:
                            colors = np.tile(np.array([[0, 255, 255]], dtype=np.uint8), (draw_indices.size, 1))
                        # 在 BGR 上绘制，再转回 RGB，避免通道污染
                        frame_bgr = transform_rgb_bgr(observations["rgb"])  # RGB->BGR
                        for uu, vv, col in zip(u_pix[draw_indices], v_pix[draw_indices], colors):
                            cv2.circle(frame_bgr, (int(uu), int(vv)), viz_point_size,
                                       color=(int(col[0]), int(col[1]), int(col[2])), thickness=-1)
                        observations["rgb"] = transform_bgr_rgb(frame_bgr)  # BGR->RGB
            except Exception:
                pass

            # 生成视频帧
            info = env.get_metrics()
            if need_video:
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            # 记录本步额外日志（仅特殊事件 + 汇总耗时）
            try:
                timings = {}
                if '_t_env0' in locals() and '_t_env1' in locals() and _t_env0 is not None and _t_env1 is not None:
                    timings['env_step'] = float(_t_env1 - _t_env0)
                if '_t_itm0' in locals() and '_t_itm1' in locals():
                    timings['itm'] = float(_t_itm1 - _t_itm0)
                if '_t_det0' in locals() and '_t_det1' in locals():
                    timings['detector'] = float(_t_det1 - _t_det0)
                if '_t_mem0' in locals() and '_t_mem1' in locals() and _t_mem0 is not None and _t_mem1 is not None:
                    timings['memory'] = float(_t_mem1 - _t_mem0)
                if '_t_hot0' in locals() and '_t_hot1' in locals() and _t_hot0 is not None and _t_hot1 is not None:
                    timings['hotspots'] = float(_t_hot1 - _t_hot0)
                if '_t_risk0' in locals() and _t_risk0 is not None:
                    timings['risk_export'] = float(time.time() - _t_risk0)
                # Look-around ENTER detection: capture only the moment FSM enters LOOK_AROUND.
                # Guard against init orientation: only consider TURN_LEFT/RIGHT steps.
                cur_rs = int(ros_state) if isinstance(ros_state, int) else None
                look_suspect = bool(
                    _look_enabled_cached
                    and cur_rs is not None
                    and cur_rs == ROS_STATE.LOOK_AROUND
                    and (ros_state_prev_for_log is None or int(ros_state_prev_for_log) != ROS_STATE.LOOK_AROUND)
                    and (action_label == 'TURN_LEFT' or action_label == 'TURN_RIGHT')
                )
                extra_logger.log_step(
                    step_idx=int(count_steps),
                    action_name=str(action_label),
                    ros_state=int(ros_state) if isinstance(ros_state, int) else None,
                    disputed_count=int(disputed_count_this_step) if 'disputed_count_this_step' in locals() else 0,
                    is_look_around_suspect=look_suspect,
                    timings=timings,
                )
                # Update previous-state snapshot after logging
                ros_state_prev_for_log = cur_rs
            except Exception:
                pass

            # 跟踪代理是否经过目标附近
            distance_to_goal = info["distance_to_goal"]
            if distance_to_goal <= success_distance and pass_object == 0:
                pass_object = 1

            # 通知ROS系统动作执行完成
            print("[DBG] Publishing HABITAT_STATE.ACTION_FINISH")
            publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)
            rate.sleep()

        # 通知ROS系统当前episode评估完成
        publish_int32(state_pub, HABITAT_STATE.EPISODE_FINISH)

        # 收集评估指标
        info = env.get_metrics()
        spl = info["spl"]  # SPL指标
        soft_spl = info["soft_spl"]  # Soft SPL指标
        distance_to_goal = info["distance_to_goal"]  # 到目标的距离
        distance_to_goal_reward = info["distance_to_goal_reward"]  # 距离奖励
        success = info["success"]  # 是否成功

        # 检查代理是否接近目标对象
        if distance_to_goal <= success_distance:
            near_object = 1

        # 确定episode结果
        if success == 1:
            num_success += 1
            result_text = "success"
        else:
            # 检查失败原因
            result_text = check_failure(
                env.current_episode,
                final_state,
                expl_result,
                count_steps,
                max_episode_steps,
                pass_object,
                near_object,
            )

        # 更新累积统计数据
        num_total += 1
        spl_all += spl
        soft_spl_all += soft_spl
        distance_to_goal_all += distance_to_goal
        distance_to_goal_reward_all += distance_to_goal_reward

        # 生成视频文件
        scene_id = env.current_episode.scene_id
        episode_id = env.current_episode.episode_id
        video_name = f"{os.path.basename(scene_id)}_{episode_id}"
        time_spend = time.time() - start_time + last_time

        img2video_output_path = os.path.join(video_output_path, result_text)

        if flag_once:
            img2video_output_path = "/home/hdd2/chaiqi/Apexnav/videos"
            video_name = "video_once"

        # 如果需要视频，则生成视频文件（在文件名中附加点云来源）
        if need_video:
            _src_tag = "stream3r" if mapping_source == "stream3r" else "habitat"
            video_name_src = f"{video_name}_src-{_src_tag}"
            images_to_video(
                vis_frames, img2video_output_path, video_name_src, fps=6, quality=9
            )
        vis_frames.clear()

        # 写出不确定性 BEV 图（逐步 PNG + 组装视频）
        try:
            if uncert_enable and uncert_frames is not None and len(uncert_frames) > 0:
                # out dir: override or <video_output_path>/<result_text>/uncertainty_bev/<video_name>
                base_dir = uncert_outdir_override if len(uncert_outdir_override) > 0 else os.path.join(video_output_path, result_text, "uncertainty_bev")
                out_dir = os.path.join(base_dir, f"{video_name}")
                os.makedirs(out_dir, exist_ok=True)
                for i, img_rgb in enumerate(uncert_frames):
                    fn = os.path.join(out_dir, f"step_{i:04d}.png")
                    try:
                        cv2.imwrite(fn, transform_rgb_bgr(img_rgb))
                    except Exception:
                        pass
                # Also export a video assembled from the BEV frames
                _src_tag = "stream3r" if mapping_source == "stream3r" else "habitat"
                bev_name = f"{video_name}_src-{_src_tag}_uncertainty_bev"
                images_to_video(uncert_frames, out_dir, bev_name, fps=6, quality=9)
        except Exception as e:
            print(f"[M1] WARN: failed to save per-step uncertainty images: {e}")
        if uncert_frames is not None:
            uncert_frames.clear()

        # 额外复合可视化视频（可选）
        try:
            if viz_comp_enable and comp_frames is not None and len(comp_frames) > 0:
                comp_dir = viz_comp_outdir if viz_comp_outdir else os.path.join(video_output_path, "composite")
                os.makedirs(comp_dir, exist_ok=True)
                _src_tag = "stream3r" if mapping_source == "stream3r" else "habitat"
                comp_name = f"{video_name}_src-{_src_tag}_composite"
                images_to_video(comp_frames, comp_dir, comp_name, fps=viz_comp_fps, quality=9)
        except Exception as e:
            print(f"[M3] WARN: failed to save composite video: {e}")
        if comp_frames is not None:
            comp_frames.clear()

        # 显示平均性能指标
        table1 = PrettyTable(["Metric", "Average"])
        table1.add_row(["Average Success", f"{num_success/num_total * 100:.2f}%"])
        table1.add_row(["Average SPL", f"{spl_all/num_total * 100:.2f}%"])
        table1.add_row(["Average Soft SPL", f"{soft_spl_all/num_total * 100:.2f}%"])
        table1.add_row(
            ["Average Distance to Goal", f"{distance_to_goal_all/num_total:.4f}"]
        )
        print(table1)
        print(f"Episode {num_total} data written to {record_file_path}")
        print(f"Result: {result_text}")

        # 显示总性能指标
        table2 = PrettyTable(["Metric", "Total"])
        table2.add_row(["Total Success", f"{num_success}"])
        table2.add_row(["Total SPL", f"{spl_all:.2f}"])
        table2.add_row(["Total Soft SPL", f"{soft_spl_all:.2f}"])
        table2.add_row(["Total Distance to Goal", f"{distance_to_goal_all:.4f}"])

        # 保存本 episode 的点云（可选）
        try:
            save_every_episode = bool(getattr(mem_cfg, "save_every_episode", False)) if mem_cfg is not None else False
            mem_output_dir = getattr(mem_cfg, "output_dir", None) if mem_cfg is not None else None
            storage_backend = "npz"
            storage_mode = "full"
            storage_cfg = getattr(mem_cfg, "storage", None) if mem_cfg is not None else None
            if storage_cfg is not None:
                storage_backend = str(getattr(storage_cfg, "backend", "npz")).lower()
                storage_mode = str(getattr(storage_cfg, "mode", "full")).lower()
            if save_every_episode:
                # Derive save path
                if mem_output_dir is None or len(str(mem_output_dir)) == 0:
                    mem_output_dir = os.path.join(video_output_path, "pointclouds")
                os.makedirs(mem_output_dir, exist_ok=True)
                # Choose extension based on backend
                if storage_backend in ("hdf5", "h5", "h5py"):
                    pc_path = os.path.join(mem_output_dir, f"{os.path.basename(scene_id)}_{episode_id}.h5")
                else:
                    pc_path = os.path.join(mem_output_dir, f"{os.path.basename(scene_id)}_{episode_id}.npz")
                try:
                    from basic_utils.storage_backend import get_backend  # lazy import

                    be = get_backend(storage_backend)
                    be.save(point_mem, pc_path, mode=storage_mode)  # type: ignore[arg-type]
                except Exception:
                    # Fallback to legacy NPZ save
                    pc_path = os.path.join(mem_output_dir, f"{os.path.basename(scene_id)}_{episode_id}.npz")
                    point_mem.save_npz(pc_path)
                print(f"[Stream3R] Saved memory to {pc_path} ({point_mem.stats().total_points} pts)")
            # Optional: export colored PLY for visualization
            if viz_save_ply:
                if mem_output_dir is None or len(str(mem_output_dir)) == 0:
                    ply_dir = os.path.join(video_output_path, "pointclouds")
                else:
                    ply_dir = mem_output_dir
                os.makedirs(ply_dir, exist_ok=True)
                pc_ply = os.path.join(ply_dir, f"{os.path.basename(scene_id)}_{episode_id}.ply")
                point_mem.save_ply(pc_ply, with_colors=True, color_by=viz_ply_color_by)
                print(f"[Stream3R] Saved colored PLY to {pc_ply}")
        except Exception as e:
            print(f"[Stream3R] WARN: failed to save memory: {e}")



        # 将结果写入记录文件
        write_record(
            scene_id,
            episode_id,
            table1,
            result_text,
            label,
            num_total,
            time_spend,
            record_file_path,
        )

        # 将结果写入继续文件
        write_record(
            scene_id,
            episode_id,
            table2,
            result_text,
            label,
            num_total,
            time_spend,
            continue_path,
        )
        # 结束额外日志，并在控制台输出：争议体素出现的 step 以及各部分总时长与占比
        try:
            _ep_summary = extra_logger.end_episode()
            try:
                disp_steps = _ep_summary.get('disputed_steps', []) if isinstance(_ep_summary, dict) else []
                ttot = _ep_summary.get('timing_totals', {}) if isinstance(_ep_summary, dict) else {}
                tperc = _ep_summary.get('timing_percent', {}) if isinstance(_ep_summary, dict) else {}
                if isinstance(disp_steps, list):
                    print(f"[EP] Disputed voxel steps: {disp_steps}")
                # Pretty breakdown
                if isinstance(ttot, dict) and len(ttot) > 0:
                    print("[EP] Time breakdown (sum | share):")
                    # stable order by key
                    for k in sorted(ttot.keys()):
                        v = float(ttot.get(k, 0.0))
                        p = float(tperc.get(k, 0.0)) * 100.0
                        print(f"  - {k}: {v:.3f}s | {p:.1f}%")
            except Exception:
                pass
        except Exception:
            pass
        if textual_summary_mgr is not None:
            try:
                textual_summary_mgr.finish_episode()
            except Exception:
                pass

        # 如果是单次测试模式，则退出循环
        if flag_once:
            break
        # 统计每个结果类别文件夹中的文件数量
        for i in range(len(RESULT_TYPES)):
            folder = RESULT_TYPES[i]  # 获取当前类别（文件夹名称）
            folder_path = os.path.join(video_output_path, folder)  # 构建文件夹路径
            file_count = count_files_in_directory(folder_path)  # 统计文件夹中的文件数
            result_list[i] = file_count

        # 发布综合记录数据
        record_data = [
            num_success / num_total * 100,  # 成功率百分比
            spl_all / num_total * 100,      # SPL百分比
            soft_spl_all / num_total * 100, # Soft SPL百分比
            distance_to_goal_all / num_total, # 平均距离到目标
        ]
        record_data.extend(result_list)
        publish_float32_array(record_pub, record_data)

        # 更新进度条并切换到下一个episode
        pbar.update()
        if not minibatch_mode:
            env.current_episode = next(env.episode_iterator)
        rospy.sleep(0.1)  # 等待片刻

    # 关闭环境和进度条
    env.close()
    pbar.close()


# 程序入口点
if __name__ == "__main__":
    # 设置信号处理器以处理Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    # 初始化ROS节点
    rospy.init_node("habitat_eval_node", anonymous=True)

    try:
        # 解析数据集参数
        dataset, overrides = _parse_dataset_arg()
        cfg_name = f"habitat_eval_{dataset}"
        # 组合选择的配置并传递额外的Hydra覆盖参数
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        main(cfg)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
