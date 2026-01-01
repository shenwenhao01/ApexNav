#!/usr/bin/env python3
"""Split a Habitat dataset JSON into sequential minibatches.

Usage example:
  python tools/split_minibatch_sequential.py \
      --chunks 4 \
      --dataset-file data/datasets/objectnav/hm3d/v1/val/val.json.gz \
      --output-prefix minibatch_hm3dv1_part \
      --dataset-name hm3dv1

The script accepts either a single Habitat dataset JSON(.gz) or a directory
like `.../val/content` that stores per-scene JSON files. It automatically
falls back to the sibling `content/` folder if the top-level JSON has an empty
episode list (e.g. hm3dv1 release).

Each minibatch JSON follows the schema expected by habitat_evaluation.py:
{
  "dataset": "hm3dv1",
  "episodes": [
    {"scene": "...", "episode_id": 0},
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import List, Sequence


def _load_json_file(path: Path) -> Sequence[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        return []
    return episodes


def _load_content_dir(content_dir: Path) -> Sequence[dict]:
    if not content_dir.is_dir():
        return []
    episodes: List[dict] = []
    # Habitat loads per-scene JSONs alphabetically, so we mimic that order.
    files = sorted(p for p in content_dir.glob("*.json*") if p.is_file())
    for file_path in files:
        episodes.extend(_load_json_file(file_path))
    return episodes


def _read_dataset(path: Path) -> Sequence[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    if path.is_dir():
        episodes = _load_content_dir(path)
        if not episodes:
            raise ValueError(f"No episode JSON files found under directory: {path}")
        return episodes

    episodes = _load_json_file(path)
    if episodes:
        return episodes

    # Some Habitat datasets (hm3dv1) ship episodes in a sibling 'content' folder.
    content_dir = path.parent / "content"
    episodes = _load_content_dir(content_dir)
    if episodes:
        return episodes

    raise ValueError(
        "Dataset JSON contains no episodes and no content directory was found."
    )


def _write_batch(out_path: Path, dataset_name: str, episodes: Sequence[dict]) -> None:
    payload = {
        "dataset": dataset_name,
        "episodes": [
            {
                "scene": ep["scene_id"],
                "episode_id": ep["episode_id"],
            }
            for ep in episodes
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def split_minibatches(
    dataset_path: Path,
    dataset_name: str,
    chunks: int,
    output_dir: Path,
    output_prefix: str,
) -> List[Path]:
    episodes = _read_dataset(dataset_path)
    total = len(episodes)
    if total == 0:
        raise ValueError("Dataset file contains zero episodes")
    if chunks <= 0:
        raise ValueError("--chunks must be positive")
    per_chunk = math.ceil(total / chunks)
    outputs: List[Path] = []
    for idx in range(chunks):
        start = idx * per_chunk
        if start >= total:
            break
        end = min((idx + 1) * per_chunk, total)
        part_eps = episodes[start:end]
        out_path = output_dir / f"{output_prefix}{idx + 1}.json"
        _write_batch(out_path, dataset_name, part_eps)
        print(
            f"Saved {out_path} | episodes {start}-{end - 1} (count={len(part_eps)})"
        )
        outputs.append(out_path)
    return outputs


def build_argparser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    default_dataset = repo_root / "data/datasets/objectnav/hm3d/v1/val/val.json.gz"
    parser = argparse.ArgumentParser(
        description="Split a Habitat dataset JSON into sequential minibatch files",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=default_dataset,
        help="Habitat dataset JSON(.gz) or a content directory",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="hm3dv1",
        help="Dataset field to embed inside minibatch JSON",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=4,
        help="Number of sequential minibatches to produce",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to write minibatch JSON files",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="minibatch_hm3dv1_part",
        help="Filename prefix for minibatch JSONs",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    split_minibatches(
        dataset_path=args.dataset_file,
        dataset_name=args.dataset_name,
        chunks=args.chunks,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
