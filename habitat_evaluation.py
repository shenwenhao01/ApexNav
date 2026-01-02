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
import time
from copy import deepcopy

# Third-party library imports
from hydra import initialize, compose
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from omegaconf import DictConfig
from prettytable import PrettyTable
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32, Int32MultiArray, Float32MultiArray, Float64
import tqdm

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
from plan_env.msg import MultipleMasksWithConfidence

# Local project imports
from basic_utils.failure_check.count_files import count_files_in_directory
from basic_utils.failure_check.failure_check import check_failure, is_on_same_floor
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from basic_utils.record_episode.read_record import read_record
from basic_utils.record_episode.write_record import write_record
from habitat2ros import habitat_publisher
from llm.answer_reader.answer_reader import read_answer
from params import HABITAT_STATE, ROS_STATE, ACTION, RESULT_TYPES
from vlm.Labels import MP3D_ID_TO_NAME
from vlm.utils.get_itm_message import get_itm_message_cosine
from vlm.utils.get_object_utils import get_object
from tools.habitat_eval_utils import (
    _candidate_scene_ids_from_local,
    _norm_scene_suffix,
)


def publish_int32(publisher, data):
    msg = Int32()
    msg.data = data
    publisher.publish(msg)


def publish_float64(publisher, data):
    msg = Float64()
    msg.data = data
    publisher.publish(msg)


def publish_int32_array(publisher, data_list):
    msg = Int32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def publish_float32_array(publisher, data_list):
    msg = Float32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def signal_handler(sig, frame):
    """Handle Ctrl+C signal for graceful shutdown"""
    print("Ctrl+C detected! Shutting down...")
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)


def transform_rgb_bgr(image):
    """Convert RGB image to BGR format"""
    return image[:, :, [2, 1, 0]]


def publish_observations(event):
    """Timer callback to publish habitat observations and trigger messages"""
    global msg_observations, fusion_threshold
    global ros_pub, trigger_pub, confidence_threshold_pub
    tmp = deepcopy(msg_observations)
    ros_pub.habitat_publish_ros_topic(tmp)
    publish_float64(confidence_threshold_pub, fusion_threshold)
    trigger = PoseStamped()
    trigger_pub.publish(trigger)


def ros_action_callback(msg):
    global global_action
    global_action = msg.data


def ros_state_callback(msg):
    global ros_state
    ros_state = msg.data


def ros_final_state_callback(msg):
    global final_state
    final_state = msg.data


def ros_expl_result_callback(msg):
    global expl_result
    expl_result = msg.data


def _parse_dataset_arg():
    """Parse CLI to choose dataset and capture remaining Hydra overrides."""
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
    # Keep unknown so users can still pass Hydra-style overrides (e.g., key=value)
    args, unknown = parser.parse_known_args()
    return args.dataset, unknown


def main(cfg: DictConfig) -> None:
    global msg_observations, global_action, ros_state, fusion_threshold
    global ros_pub, trigger_pub, obj_point_cloud_pub, confidence_threshold_pub
    global final_state, expl_result

    # Load MP3D validation data for object category mapping
    with gzip.open(
        "data/datasets/objectnav/mp3d/v1/val/val.json.gz", "rt", encoding="utf-8"
    ) as f:
        val_data = json.load(f)
    category_to_coco = val_data.get("category_to_mp3d_category_id", {})
    id_to_name = {
        category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
        for idx, cat in enumerate(category_to_coco)
    }

    start_time = time.time()

    final_state = 0
    expl_result = 0
    result_list = [0] * len(RESULT_TYPES)

    cfg = patch_config(cfg)

    # Extract configuration parameters
    video_output_path = cfg.video_output_path.format(split=cfg.habitat.dataset.split)
    need_video = cfg.need_video
    record_file_path = os.path.join(video_output_path, cfg.record_file_name)
    continue_path = os.path.join(video_output_path, cfg.continue_file_name)
    max_episode_steps = cfg.habitat.environment.max_episode_steps
    success_distance = cfg.habitat.task.measurements.success.success_distance

    detector_cfg = cfg.detector

    llm_cfg = cfg.llm
    llm_client = llm_cfg.llm_client
    llm_answer_path = llm_cfg.llm_answer_path
    llm_response_path = llm_cfg.llm_response_path

    # Single test parameters
    env_num_once = cfg.test_epi_num  # Which episode to test for single run
    flag_once = env_num_once != -1  # Whether to run single test

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(llm_answer_path), exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)

    # Add top_down_map and collisions visualization
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
            video_output_path = f"./videos/test_{_ds}_{_split}_minibatch_{ts}"

        os.makedirs(video_output_path, exist_ok=True)
        # 重新绑定记录/进度文件路径到新输出根目录
        record_file_path = os.path.join(video_output_path, cfg.record_file_name)
        continue_path = os.path.join(video_output_path, cfg.continue_file_name)

    # 设定 episode 总数（若启用 minibatch，则覆盖为列表大小）
    number_of_episodes = len(minibatch_items) if minibatch_mode else env.number_of_episodes
    
    # 在 minibatch 模式下，预先加载所有 episodes 到列表（避免 iterator 被消耗）
    all_episodes_list = None
    if minibatch_mode:
        try:
            print("[Minibatch] Loading all episodes into memory...")
            all_episodes_list = list(env.episode_iterator)
            print(f"[Minibatch] Loaded {len(all_episodes_list)} episodes")
        except Exception as e:
            print(f"[Minibatch] WARN: Failed to preload episodes: {e}, will use iterator directly")
            all_episodes_list = None

    # Read previous records and set initial values
    (
        num_total,
        num_success,
        spl_all,
        soft_spl_all,
        distance_to_goal_all,
        distance_to_goal_reward_all,
        last_time,
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

    pbar = tqdm.tqdm(total=number_of_episodes)

    # 设置环境计数器（minibatch 模式下跳过前跳逻辑）
    if not minibatch_mode:
        env_count = num_total if not flag_once else env_num_once
        while env_count:
            pbar.update()
            env.current_episode = next(env.episode_iterator)
            env_count -= 1

    # Initialize ROS publishers, subscribers, and timers
    obj_point_cloud_pub = rospy.Publisher(
        "habitat/object_point_cloud", PointCloud2, queue_size=10
    )
    ros_pub = habitat_publisher.ROSPublisher()
    rospy.Subscriber("/habitat/plan_action", Int32, ros_action_callback, queue_size=10)
    rospy.Subscriber("/ros/state", Int32, ros_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_state", Int32, ros_final_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_result", Int32, ros_expl_result_callback, queue_size=10)
    state_pub = rospy.Publisher("/habitat/state", Int32, queue_size=10)
    trigger_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    itm_score_pub = rospy.Publisher("/blip2/cosine_score", Float64, queue_size=10)
    confidence_threshold_pub = rospy.Publisher(
        "/detector/confidence_threshold", Float64, queue_size=10
    )
    cld_with_score_pub = rospy.Publisher(
        "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
    )
    progress_pub = rospy.Publisher("/habitat/progress", Int32MultiArray, queue_size=10)
    record_pub = rospy.Publisher("/habitat/record", Float32MultiArray, queue_size=10)

    for epi in range(number_of_episodes - num_total):
        # Publish progress information
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
                scene_found = False
                # 提取 scene name 用于匹配（例如从 mp3d/X7HyMhZNoso/X7HyMhZNoso.glb 提取 X7HyMhZNoso）
                scene_name = None
                if "/" in want_scene_raw:
                    parts = want_scene_raw.split("/")
                    for part in parts:
                        if part and part != "mp3d" and not part.endswith(".glb"):
                            scene_name = part
                            break
                
                sample_scene_ids = []  # 收集包含相同 scene name 的实际 scene_id 示例
                any_scene_ids = []  # 收集任意 scene_id 作为示例
                episode_count = 0
                # 在 minibatch 模式下，使用预先加载的 episodes 列表
                episodes_to_iterate = all_episodes_list if all_episodes_list is not None else env.episode_iterator
                
                for ep in episodes_to_iterate:
                    env.current_episode = ep
                    ep_sid_raw = str(ep.scene_id)  # 原始 scene_id
                    ep_sid = _norm_scene_suffix(ep_sid_raw)  # 规范化后的 scene_id
                    episode_count += 1
                    
                    # 收集任意 scene_id 作为示例（最多5个，确保能收集到）
                    if len(any_scene_ids) < 5:
                        any_scene_ids.append(f"{ep_sid} (raw: {ep_sid_raw})")
                    # 如果包含相同的 scene name，收集作为示例
                    if scene_name and scene_name in ep_sid and len(sample_scene_ids) < 5:
                        sample_scene_ids.append(f"{ep_sid} (raw: {ep_sid_raw})")
                    
                    # 同时检查原始和规范化后的 scene_id，以及通过 scene_name 匹配
                    if ep_sid in want_scene_cands or ep_sid_raw in want_scene_cands:
                        scene_found = True
                        if int(ep.episode_id) == want_eid:
                            found = True
                            break
                    # 也尝试通过 scene_name 匹配（更灵活的匹配）
                    elif scene_name and scene_name in ep_sid:
                        scene_found = True
                        if int(ep.episode_id) == want_eid:
                            found = True
                            break
                    
                    # 如果已经检查了很多 episode 还没找到，提前停止（避免无限循环）
                    if episode_count > 10000 and not scene_found:
                        break
                if not found:
                    # 打印更友好的诊断，包括候选匹配
                    try:
                        shown = want_scene_cands[0] if len(want_scene_cands) > 0 else want_scene_raw
                    except Exception:
                        shown = want_scene_raw
                    if scene_found:
                        print(
                            f"[Minibatch] WARN: episode not found: scene={shown}, episode_id={want_eid}; "
                            f"scene exists but episode_id not found; skipping"
                        )
                    else:
                        msg = f"[Minibatch] WARN: episode not found: scene={shown}, episode_id={want_eid}; "
                        msg += f"scene not found (tried: {want_scene_cands[:3]})"
                        if sample_scene_ids:
                            msg += f"; similar scene_ids: {sample_scene_ids[:3]}"
                        if any_scene_ids:
                            msg += f"; sample scene_ids in dataset: {any_scene_ids[:3]}"
                        if scene_name:
                            msg += f" (looking for scene_name: {scene_name})"
                        msg += f" (checked {episode_count} episodes); skipping"
                        print(msg)
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
            env_count = env_num_once
            while env_count:
                env.current_episode = next(env.episode_iterator)
                env_count -= 1

        # Initialize episode variables
        pass_object = 0.0
        near_object = 0.0
        global_action = None
        cld_with_score_msg = MultipleMasksWithConfidence()
        count_steps = 0

        camera_pitch = 0.0
        observations = env.reset()
        observations["camera_pitch"] = camera_pitch
        msg_observations = deepcopy(observations)
        del observations["camera_pitch"]
        label = env.current_episode.object_category

        # Convert object category to coco name format
        if label in category_to_coco:
            coco_id = category_to_coco[label]
            label = id_to_name.get(coco_id, label)

        # Get LLM answer and fusion threshold for the target object
        llm_answer, room, fusion_threshold = read_answer(
            llm_answer_path, llm_response_path, label, llm_client
        )

        # Initialize video frame collection
        vis_frames = []
        info = env.get_metrics()
        if need_video:
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames = [frame]

        # Start publishing basic information and trigger messages
        pub_timer = rospy.Timer(rospy.Duration(0.25), publish_observations)

        print("Agent is waiting in the environment!!!")

        # Wait for ROS system to be ready
        rate = rospy.Rate(10)
        ros_state = ROS_STATE.INIT
        while ros_state == ROS_STATE.INIT or ros_state == ROS_STATE.WAIT_TRIGGER:
            if ros_state == ROS_STATE.INIT:
                print("Waiting for ROS to get odometry...")
            elif ros_state == ROS_STATE.WAIT_TRIGGER:
                print("Waiting for ROS trigger...")
            rate.sleep()

        # Stop timer publishing when starting action execution
        pub_timer.shutdown()

        print("Agent is ready to go!!!!")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not env.episode_over:
            # Skip episode if target is not on the same floor
            is_feasible = 0
            for goal in env.current_episode.goals:
                height = goal.position[1]
                is_feasible += is_on_same_floor(
                    height=height, episode=env.current_episode
                )
            if not is_feasible:
                break

            # Parse action from decision system
            action = None
            if global_action is not None:
                if count_steps == max_episode_steps - 1:
                    global_action = ACTION.STOP

                if global_action == ACTION.MOVE_FORWARD:
                    action = HabitatSimActions.move_forward
                elif global_action == ACTION.TURN_LEFT:
                    action = HabitatSimActions.turn_left
                elif global_action == ACTION.TURN_RIGHT:
                    action = HabitatSimActions.turn_right
                elif global_action == ACTION.TURN_DOWN:
                    action = HabitatSimActions.look_down
                    camera_pitch = camera_pitch - np.pi / 6.0
                elif global_action == ACTION.TURN_UP:
                    action = HabitatSimActions.look_up
                    camera_pitch = camera_pitch + np.pi / 6.0
                elif global_action == ACTION.STOP:
                    action = HabitatSimActions.stop

                global_action = None

            if action is None:
                continue

            count_steps += 1
            print(f"\n--------------Step: {count_steps}--------------")
            print(f"Finding [{label}]; Action: {action};")

            # Notify ROS system that action execution is starting
            publish_int32(state_pub, HABITAT_STATE.ACTION_EXEC)

            observations = env.step(action)

            # Calculate ITM cosine similarity score
            cosine = get_itm_message_cosine(observations["rgb"], label, room)
            print(f"Target related room: {room}")
            print(f"ITM cosine similarity: {cosine:.3f}")

            publish_float64(itm_score_pub, cosine)

            # Detect objects in the current observation
            observations["rgb"], score_list, object_masks_list, label_list = get_object(
                label, observations["rgb"], detector_cfg, llm_answer
            )

            # Publish habitat observations to ROS
            observations["camera_pitch"] = camera_pitch
            msg_observations = deepcopy(observations)
            del observations["camera_pitch"]
            ros_pub.habitat_publish_ros_topic(msg_observations)

            # Generate and publish object point clouds
            obj_point_cloud_list = get_object_point_cloud(
                cfg, observations, object_masks_list
            )

            # Publish detection-related information
            cld_with_score_msg.point_clouds = obj_point_cloud_list
            cld_with_score_msg.confidence_scores = score_list
            cld_with_score_msg.label_indices = label_list
            cld_with_score_pub.publish(cld_with_score_msg)

            # Generate video frame
            info = env.get_metrics()
            if need_video:
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            # Track if agent has passed close to the target
            distance_to_goal = info["distance_to_goal"]
            if distance_to_goal <= success_distance and pass_object == 0:
                pass_object = 1

            # Notify ROS system that action execution is complete
            publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)
            rate.sleep()

        # Notify ROS system that current episode evaluation is complete
        publish_int32(state_pub, HABITAT_STATE.EPISODE_FINISH)

        # Collect evaluation metrics
        info = env.get_metrics()
        spl = info["spl"]
        soft_spl = info["soft_spl"]
        distance_to_goal = info["distance_to_goal"]
        distance_to_goal_reward = info["distance_to_goal_reward"]
        success = info["success"]

        # Check if agent got close to the target object
        if distance_to_goal <= success_distance:
            near_object = 1

        # Determine episode result
        if success == 1:
            num_success += 1
            result_text = "success"
        else:
            result_text = check_failure(
                env.current_episode,
                final_state,
                expl_result,
                count_steps,
                max_episode_steps,
                pass_object,
                near_object,
            )

        # Update cumulative statistics
        num_total += 1
        spl_all += spl
        soft_spl_all += soft_spl
        distance_to_goal_all += distance_to_goal
        distance_to_goal_reward_all += distance_to_goal_reward

        # Generate video file
        scene_id = env.current_episode.scene_id
        episode_id = env.current_episode.episode_id
        video_name = f"{os.path.basename(scene_id)}_{episode_id}"
        time_spend = time.time() - start_time + last_time

        img2video_output_path = os.path.join(video_output_path, result_text)

        if flag_once:
            img2video_output_path = "videos"
            video_name = "video_once"

        if need_video:
            images_to_video(
                vis_frames, img2video_output_path, video_name, fps=6, quality=9
            )
        vis_frames.clear()

        # Display average performance metrics
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

        # Display total performance metrics
        table2 = PrettyTable(["Metric", "Total"])
        table2.add_row(["Total Success", f"{num_success}"])
        table2.add_row(["Total SPL", f"{spl_all:.2f}"])
        table2.add_row(["Total Soft SPL", f"{soft_spl_all:.2f}"])
        table2.add_row(["Total Distance to Goal", f"{distance_to_goal_all:.4f}"])

        if flag_once:
            break

        # Write results to record file
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

        # Write results to continue file
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

        # Count files in each result category folder
        for i in range(len(RESULT_TYPES)):
            folder = RESULT_TYPES[i]  # Get current category (folder name)
            folder_path = os.path.join(video_output_path, folder)  # Build folder path
            file_count = count_files_in_directory(folder_path)  # Count files in folder
            result_list[i] = file_count

        # Publish comprehensive record data
        record_data = [
            num_success / num_total * 100,
            spl_all / num_total * 100,
            soft_spl_all / num_total * 100,
            distance_to_goal_all / num_total,
        ]
        record_data.extend(result_list)
        publish_float32_array(record_pub, record_data)

        # 更新进度条并切换到下一个episode
        pbar.update()
        if not minibatch_mode:
            env.current_episode = next(env.episode_iterator)
        rospy.sleep(0.1)  # wait a moment

    env.close()
    pbar.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_eval_node", anonymous=True)

    try:
        dataset, overrides = _parse_dataset_arg()
        cfg_name = f"habitat_eval_{dataset}"
        # Compose the chosen config and pass through extra Hydra overrides
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        main(cfg)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
