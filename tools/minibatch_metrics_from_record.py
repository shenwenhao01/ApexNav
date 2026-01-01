#!/usr/bin/env python3
import argparse
import os

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


def main():
    parser = argparse.ArgumentParser(
        description="Compute minibatch metrics from record/continue files"
    )
    parser.add_argument("--record", help="record.txt 路径（单 run 模式）")
    parser.add_argument(
        "--minibatch",
        required=True,
        nargs="+",
        help="一个或多个 minibatch JSON 路径；传多个时将自动求并集",
    )
    parser.add_argument(
        "--continue_file",
        default=None,
        help="continue.txt 路径，可选（若省略则只用 record.txt）",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=None,
        help="多个输出目录；脚本会自动读取其中的 record/continue 并合并",
    )
    parser.add_argument(
        "--record-name", default="record.txt", help="run 目录下 record 文件名"
    )
    parser.add_argument(
        "--continue-name", default="continue.txt", help="run 目录下 continue 文件名"
    )
    args = parser.parse_args()

    minibatch_arg = [os.path.abspath(p) for p in args.minibatch]
    if len(minibatch_arg) == 1:
        minibatch_input = minibatch_arg[0]
    else:
        minibatch_input = minibatch_arg

    entries = []
    if args.run_dirs:
        for d in args.run_dirs:
            root = os.path.abspath(d)
            record_path = os.path.join(root, args.record_name)
            if not os.path.exists(record_path):
                raise SystemExit(f"未找到 record 文件: {record_path}")
            cont_path = os.path.join(root, args.continue_name)
            if not os.path.exists(cont_path):
                cont_path = None
            entries.extend(load_run_entries(record_path, cont_path, minibatch_input))
    else:
        if not args.record:
            raise SystemExit("必须指定 --record 或 --run-dirs 中的一种输入方式")
        record_path = os.path.abspath(args.record)
        cont_path = os.path.abspath(args.continue_file) if args.continue_file else None
        entries = load_run_entries(record_path, cont_path, minibatch_input)
    if not entries:
        raise SystemExit("未能在记录文件中找到任何与 minibatch 匹配的 episode")

    summary = summarize(entries)
    _print_table("Minibatch", summary)


if __name__ == "__main__":
    main()
