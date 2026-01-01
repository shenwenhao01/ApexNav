#!/usr/bin/env python3
"""Shared utilities for parsing ApexNav record/continue files."""

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


def _split_blocks(text: str) -> List[str]:
    blocks = re.split(r"\n(?=Scene ID:)", text.strip())
    return [b for b in blocks if b.strip()]


def _parse_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def parse_record_file(path: str) -> List[Dict]:
    """Parse record.txt (Average table). Newest entry appears first."""

    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    records = _split_blocks(content)
    out: List[Dict] = []
    for block in records:
        scene = re.search(r"Scene ID:\s*(.*)", block)
        ep = re.search(r"Episode ID:\s*(\d+)", block)
        idx = re.search(r"No\.(\d+) task is finished", block)
        success_avg = re.search(r"Average Success\s*\|\s*([0-9.]+)%", block)
        spl_avg = re.search(r"Average SPL\s*\|\s*([0-9.]+)%", block)
        soft_spl_avg = re.search(r"Average Soft SPL\s*\|\s*([0-9.]+)%", block)
        dist_avg = re.search(r"Average Distance to Goal\s*\|\s*([0-9.]+)", block)
        result = re.search(r"success or not:\s*(.*)", block)

        if not (scene and ep and idx and success_avg and spl_avg and soft_spl_avg and dist_avg):
            continue

        out.append(
            {
                "scene": scene.group(1).strip(),
                "episode_id": _parse_int(ep.group(1)),
                "idx": _parse_int(idx.group(1)),
                "avg_success_pct": _parse_float(success_avg.group(1)),
                "avg_spl_pct": _parse_float(spl_avg.group(1)),
                "avg_soft_spl_pct": _parse_float(soft_spl_avg.group(1)),
                "avg_dist": _parse_float(dist_avg.group(1)),
                "result_text": result.group(1).strip() if result else None,
            }
        )

    return out


def parse_continue_file(path: Optional[str]) -> List[Dict]:
    """Parse continue.txt (Total table)."""

    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    records = _split_blocks(content)
    out: List[Dict] = []
    for block in records:
        scene = re.search(r"Scene ID:\s*(.*)", block)
        ep = re.search(r"Episode ID:\s*(\d+)", block)
        idx = re.search(r"No\.(\d+) task is finished", block)
        tot_success = re.search(r"Total Success\s*\|\s*(\d+)", block)
        tot_spl = re.search(r"Total SPL\s*\|\s*([0-9.]+)", block)
        tot_soft = re.search(r"Total Soft SPL\s*\|\s*([0-9.]+)", block)
        tot_dist = re.search(r"Total Distance to Goal\s*\|\s*([0-9.]+)", block)
        if not (scene and ep and idx and tot_success and tot_spl and tot_soft and tot_dist):
            continue
        out.append(
            {
                "scene": scene.group(1).strip(),
                "episode_id": _parse_int(ep.group(1)),
                "idx": _parse_int(idx.group(1)),
                "tot_success": _parse_int(tot_success.group(1)),
                "tot_spl": _parse_float(tot_spl.group(1)),
                "tot_soft_spl": _parse_float(tot_soft.group(1)),
                "tot_dist": _parse_float(tot_dist.group(1)),
            }
        )

    return out


def reconstruct_per_episode(rec_blocks: List[Dict], cont_blocks: List[Dict]) -> List[Dict]:
    """Reconstruct single-episode metrics from cumulative records."""

    out: List[Dict] = []
    cont_map: Dict[Tuple[str, int, int], Dict] = {}
    for cb in cont_blocks:
        key = (cb["scene"], cb["episode_id"], cb["idx"])
        cont_map[key] = cb

    runs: List[List[Dict]] = []
    cur: List[Dict] = []
    last_idx: Optional[int] = None
    for rb in rec_blocks:
        idx = int(rb.get("idx") or 0)
        if last_idx is not None and idx >= last_idx:
            if cur:
                runs.append(cur)
            cur = []
        cur.append(rb)
        last_idx = idx
    if cur:
        runs.append(cur)

    for seg in runs:
        seg_sorted = sorted(seg, key=lambda x: int(x.get("idx") or 0))
        prev_tot_success = 0.0
        prev_tot_spl = 0.0
        prev_tot_soft = 0.0
        prev_tot_dist = 0.0

        for rb in seg_sorted:
            key = (rb["scene"], rb["episode_id"], rb["idx"])
            if key in cont_map:
                cb = cont_map[key]
                tot_success = float(cb.get("tot_success") or 0.0)
                tot_spl = float(cb.get("tot_spl") or 0.0)
                tot_soft = float(cb.get("tot_soft_spl") or 0.0)
                tot_dist = float(cb.get("tot_dist") or 0.0)
            else:
                n = float(rb.get("idx") or 0)
                tot_success = n * float((rb.get("avg_success_pct") or 0.0) / 100.0)
                tot_spl = n * float((rb.get("avg_spl_pct") or 0.0) / 100.0)
                tot_soft = n * float((rb.get("avg_soft_spl_pct") or 0.0) / 100.0)
                tot_dist = n * float(rb.get("avg_dist") or 0.0)

            delta_success = tot_success - prev_tot_success
            delta_spl = tot_spl - prev_tot_spl
            delta_soft = tot_soft - prev_tot_soft
            delta_dist = tot_dist - prev_tot_dist

            if rb.get("result_text"):
                succ = 1.0 if str(rb["result_text"]).strip().lower().startswith("success") else 0.0
            else:
                succ = float(int(round(delta_success)))

            out.append(
                {
                    "scene": rb["scene"],
                    "episode_id": rb["episode_id"],
                    "idx": rb["idx"],
                    "success": float(succ),
                    "spl": float(delta_spl),
                    "soft_spl": float(delta_soft),
                    "dist": float(delta_dist),
                    "result_text": rb.get("result_text"),
                }
            )

            prev_tot_success = tot_success
            prev_tot_spl = tot_spl
            prev_tot_soft = tot_soft
            prev_tot_dist = tot_dist

    return out


def _load_minibatch(minibatch_path: str) -> Tuple[Dict[Tuple[str, int], str], Dict[Tuple[str, int], str]]:
    with open(minibatch_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    want_full: Dict[Tuple[str, int], str] = {}
    want_base: Dict[Tuple[str, int], str] = {}
    episodes = payload.get("episodes", [])
    for ep in episodes:
        scene = str(ep.get("scene", "")).strip()
        try:
            episode_id = int(ep.get("episode_id"))
        except Exception:
            continue
        if not scene:
            continue
        want_full[(scene, episode_id)] = scene
        want_base[(os.path.basename(scene), episode_id)] = scene
    return want_full, want_base


def _load_minibatch_union(
    minibatch_paths: Sequence[str],
) -> Tuple[Dict[Tuple[str, int], str], Dict[Tuple[str, int], str]]:
    want_full: Dict[Tuple[str, int], str] = {}
    want_base: Dict[Tuple[str, int], str] = {}
    for path in minibatch_paths:
        if not path:
            continue
        part_full, part_base = _load_minibatch(path)
        want_full.update(part_full)
        want_base.update(part_base)
    return want_full, want_base


def filter_by_minibatch(
    per_episode: List[Dict], minibatch_path: Union[str, Sequence[str]]
) -> List[Dict]:
    """Filter reconstructed metrics to the provided minibatch episodes."""

    if isinstance(minibatch_path, (list, tuple, set)):
        paths: Iterable[str] = minibatch_path
    else:
        paths = [minibatch_path]

    want_full, want_base = _load_minibatch_union(list(paths))
    selected: List[Dict] = []
    seen: set = set()

    for item in per_episode:
        scene = str(item.get("scene", ""))
        ep_id = int(item.get("episode_id") or 0)
        key_full = (scene, ep_id)
        key_base = (os.path.basename(scene), ep_id)

        canonical = None
        if key_full in want_full:
            canonical = want_full[key_full]
        elif key_base in want_base:
            canonical = want_base[key_base]
        else:
            continue

        canon_key = (canonical, ep_id)
        if canon_key in seen:
            continue

        entry = dict(item)
        entry["canonical_scene"] = canonical
        entry["scene_basename"] = os.path.basename(canonical)
        selected.append(entry)
        seen.add(canon_key)

    return selected


def summarize(entries: List[Dict]) -> Dict[str, float]:
    total = float(len(entries)) if entries else 0.0
    if total == 0.0:
        return {"n": 0, "success_pct": 0.0, "spl_pct": 0.0, "soft_spl_pct": 0.0, "avg_dist": 0.0}

    success_sum = sum(float(ep.get("success", 0.0)) for ep in entries)
    spl_sum = sum(float(ep.get("spl", 0.0)) for ep in entries)
    soft_sum = sum(float(ep.get("soft_spl", 0.0)) for ep in entries)
    dist_sum = sum(float(ep.get("dist", 0.0)) for ep in entries)

    return {
        "n": int(total),
        "success_pct": 100.0 * success_sum / total,
        "spl_pct": 100.0 * spl_sum / total,
        "soft_spl_pct": 100.0 * soft_sum / total,
        "avg_dist": dist_sum / total,
    }


MinibatchSelector = Optional[Union[str, Sequence[str]]]


def load_run_entries(
    record_path: str, continue_path: Optional[str], minibatch_path: MinibatchSelector
) -> List[Dict]:
    rec = parse_record_file(record_path)
    cont = parse_continue_file(continue_path) if continue_path else []
    per_ep = reconstruct_per_episode(rec, cont)
    if minibatch_path:
        per_ep = filter_by_minibatch(per_ep, minibatch_path)
    return per_ep
