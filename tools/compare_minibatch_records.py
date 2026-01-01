#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Optional, Tuple

from minibatch_metrics_lib import load_run_entries, summarize


def _print_table(label: str, summary):
    print(f"+--------------------------+---------+  ({label})")
    print("|          Metric          | Average |")
    print("+--------------------------+---------+")
    print(f"|     Average Success      |  {summary['success_pct']:.2f}% |")
    print(f"|       Average SPL        |  {summary['spl_pct']:.2f}% |")
    print(f"|     Average Soft SPL     |  {summary['soft_spl_pct']:.2f}% |")
    print(f"| Average Distance to Goal |  {summary['avg_dist']:.4f} |")
    print("+--------------------------+---------+")
    print(f"Episodes covered: {summary['n']}")


def _build_map(entries: List[Dict]) -> Dict[Tuple[str, int], Dict]:
    mapping: Dict[Tuple[str, int], Dict] = {}
    for item in entries:
        scene = str(item.get("canonical_scene") or item.get("scene") or "").strip()
        ep_raw = item.get("episode_id")
        try:
            ep_id = int(ep_raw)
        except (TypeError, ValueError):
            continue
        if not scene:
            continue
        mapping[(scene, ep_id)] = item
    return mapping


def _success_flag(entry: Optional[Dict]) -> Optional[int]:
    if entry is None:
        return None
    return int(round(float(entry.get("success", 0.0))))


def _format_name(entry_a: Optional[Dict], entry_b: Optional[Dict]) -> str:
    entry = entry_a or entry_b
    if entry is None:
        return "unknown"
    scene = entry.get("canonical_scene") or entry.get("scene") or "scene"
    base = entry.get("scene_basename") or os.path.basename(scene)
    ep = entry.get("episode_id", "?")
    return f"{base}:{ep}"


def _subset_keys_by_success(mapping: Dict[Tuple[str, int], Dict], flag: int) -> List[Tuple[str, int]]:
    return [key for key, entry in mapping.items() if _success_flag(entry) == flag]


def _summarize_subset(mapping: Dict[Tuple[str, int], Dict], keys: List[Tuple[str, int]]):
    subset = [mapping[k] for k in keys if k in mapping]
    return summarize(subset)


def _print_subset(title: str, keys: List[Tuple[str, int]], map_a, map_b, label_a, label_b):
    print(f"\n=== {title} ===")
    total = len(keys)
    print(f"选中 episodes: {total}")
    if total == 0:
        print("该子集中无 episode")
        return
    summary_a = _summarize_subset(map_a, keys)
    present_a = summary_a["n"]
    summary_b = _summarize_subset(map_b, keys)
    present_b = summary_b["n"]
    missing_b = total - present_b
    missing_a = total - present_a
    if missing_a:
        print(f"{label_a} 缺失 {missing_a} 条记录")
    _print_table(label_a, summary_a)
    if present_b == 0:
        print(f"{label_b} 在该子集中缺少全部记录")
    else:
        if missing_b:
            print(f"{label_b} 缺失 {missing_b} 条记录")
        _print_table(label_b, summary_b)


def _confusion(ref_map, other_map, label_ref: str, label_other: str) -> Dict[str, int]:
    counts = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "missing": 0}
    for key, ref in ref_map.items():
        ref_flag = _success_flag(ref)
        other = other_map.get(key)
        if other is None:
            counts["missing"] += 1
            continue
        other_flag = _success_flag(other)
        if ref_flag == 1 and other_flag == 1:
            counts["tp"] += 1
        elif ref_flag == 1 and other_flag == 0:
            counts["fn"] += 1
        elif ref_flag == 0 and other_flag == 1:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    print(f"\n以 {label_ref} 为真值的成功/失败对比：")
    print("+-----------------------+-----------------+-----------------+")
    col_success = f"{label_other} 成功"
    col_fail = f"{label_other} 失败"
    print(f"|                       | {col_success:^15} | {col_fail:^15} |")
    print("+-----------------------+-----------------+-----------------+")
    print(f"| {label_ref:^7} 成功        | {counts['tp']:>13} | {counts['fn']:>13} |")
    print("+-----------------------+-----------------+-----------------+")
    print(f"| {label_ref:^7} 失败        | {counts['fp']:>13} | {counts['tn']:>13} |")
    print("+-----------------------+-----------------+-----------------+")
    print(f"缺少 {label_other} 数据：{counts['missing']}")
    return counts


def _collect_differences(map_a, map_b):
    diffs = []
    keys = set(map_a.keys()) | set(map_b.keys())
    for key in sorted(keys):
        entry_a = map_a.get(key)
        entry_b = map_b.get(key)
        flag_a = _success_flag(entry_a)
        flag_b = _success_flag(entry_b)
        if flag_a == flag_b:
            continue
        diffs.append(
            {
                "key": key,
                "entry_a": entry_a,
                "entry_b": entry_b,
                "flag_a": flag_a,
                "flag_b": flag_b,
            }
        )
    return diffs


def _format_status(entry: Optional[Dict], flag: Optional[int], correct: bool) -> str:
    if entry is None:
        return "缺失"
    mark = "对" if correct else "错"
    label = "success" if flag == 1 else "failure"
    text = entry.get("result_text") or label
    return f"{mark}({text})"


def _print_differences(diffs, assume: str, label_a: str, label_b: str):
    assert assume in ("A", "B")
    if not diffs:
        print("  无差异")
        return
    for diff in diffs:
        name = _format_name(diff["entry_a"], diff["entry_b"])
        if assume == "A":
            status_a = _format_status(diff["entry_a"], diff["flag_a"], True)
            status_b = _format_status(diff["entry_b"], diff["flag_b"], False)
            print(f"  - {name}: {label_a}={status_a} | {label_b}={status_b}")
        else:
            status_a = _format_status(diff["entry_a"], diff["flag_a"], False)
            status_b = _format_status(diff["entry_b"], diff["flag_b"], True)
            print(f"  - {name}: {label_a}={status_a} | {label_b}={status_b}")


def main():
    parser = argparse.ArgumentParser(description="比较同一 minibatch 的两份 record 结果")
    parser.add_argument("--dir-a", required=True, help="记录文件夹 A")
    parser.add_argument("--dir-b", required=True, help="记录文件夹 B")
    parser.add_argument("--minibatch", required=True, help="minibatch JSON")
    parser.add_argument("--record-name", default="record.txt", help="record 文件名")
    parser.add_argument(
        "--continue-name",
        default="continue.txt",
        help="continue 文件名（若不存在则忽略）",
    )
    parser.add_argument("--label-a", default="A", help="A 的标签名")
    parser.add_argument("--label-b", default="B", help="B 的标签名")
    args = parser.parse_args()

    def _paths(root: str):
        record_path = os.path.join(root, args.record_name)
        cont_path = os.path.join(root, args.continue_name)
        if not os.path.exists(cont_path):
            cont_path = None
        return record_path, cont_path

    rec_a, cont_a = _paths(args.dir_a)
    rec_b, cont_b = _paths(args.dir_b)

    entries_a = load_run_entries(rec_a, cont_a, args.minibatch)
    entries_b = load_run_entries(rec_b, cont_b, args.minibatch)
    if not entries_a:
        raise SystemExit("A 中未找到任何匹配的 episode")
    if not entries_b:
        raise SystemExit("B 中未找到任何匹配的 episode")

    map_a = _build_map(entries_a)
    map_b = _build_map(entries_b)

    print("=== 全量统计 ===")
    _print_table(args.label_a, summarize(list(map_a.values())))
    _print_table(args.label_b, summarize(list(map_b.values())))

    keys_a_success = _subset_keys_by_success(map_a, 1)
    keys_a_fail = _subset_keys_by_success(map_a, 0)

    _print_subset("A 准确 (success=1)", keys_a_success, map_a, map_b, args.label_a, args.label_b)
    _print_subset("A 不准确 (success=0)", keys_a_fail, map_a, map_b, args.label_a, args.label_b)

    diffs = _collect_differences(map_a, map_b)
    _confusion(map_a, map_b, args.label_a, args.label_b)
    _confusion(map_b, map_a, args.label_b, args.label_a)

    print("\n差异 episodes（假设 A 正确）：")
    _print_differences(diffs, "A", args.label_a, args.label_b)

    print("\n差异 episodes（假设 B 正确）：")
    _print_differences(diffs, "B", args.label_a, args.label_b)


if __name__ == "__main__":
    main()
