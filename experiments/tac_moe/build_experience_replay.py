#!/usr/bin/env python3
"""Build a simple continual-learning Experience Replay train file.

This is an offline replay-memory constructor. It does not change the training
interface: the generated file has the same JSONL schema as the existing train
files and can be passed directly to --train_file.

Default policy:
  train_phase2 = all new-task samples + a stratified memory buffer from old tasks

The memory buffer is sampled by task/label strata to avoid accidental collapse
to a single large old task or class.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


Record = dict[str, Any]


def read_records(path: Path) -> list[Record]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data

    rows = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"Line {lineno} in {path} is not a JSON object")
        rows.append(obj)
    return rows


def write_jsonl(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def field_value(record: Record, field: str) -> str:
    value = record.get(field)
    if value is None:
        return "<missing>"
    return str(value)


def task_name(record: Record) -> str:
    for field in ("task_dataset", "task_name", "dataset"):
        if field in record:
            return str(record[field])
    if "task_id" in record:
        return f"task{record['task_id']}"
    return "<missing_task>"


def parse_task_ratios(raw: str) -> dict[str, float]:
    ratios: dict[str, float] = {}
    if not raw:
        return ratios
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bad task ratio item {item!r}; expected TASK:RATIO")
        task, ratio = item.split(":", 1)
        ratios[task.strip()] = float(ratio)
    return ratios


def allocate_quota(sizes: dict[tuple[str, ...], int], total: int) -> dict[tuple[str, ...], int]:
    if total <= 0 or not sizes:
        return {key: 0 for key in sizes}
    available = sum(sizes.values())
    total = min(total, available)

    raw = {key: total * size / available for key, size in sizes.items()}
    quota = {key: min(sizes[key], int(math.floor(value))) for key, value in raw.items()}
    remaining = total - sum(quota.values())

    order = sorted(
        sizes,
        key=lambda key: (raw[key] - math.floor(raw[key]), sizes[key]),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for key in order:
            if quota[key] >= sizes[key]:
                continue
            quota[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return quota


def stratified_sample(
    records: list[Record],
    size: int,
    fields: list[str],
    rng: random.Random,
) -> list[Record]:
    if size >= len(records):
        result = list(records)
        rng.shuffle(result)
        return result

    buckets: dict[tuple[str, ...], list[Record]] = defaultdict(list)
    for record in records:
        key = tuple(field_value(record, field) for field in fields)
        buckets[key].append(record)

    quota = allocate_quota({key: len(value) for key, value in buckets.items()}, size)
    sampled: list[Record] = []
    for key, bucket in buckets.items():
        k = quota[key]
        if k <= 0:
            continue
        sampled.extend(rng.sample(bucket, k))
    rng.shuffle(sampled)
    return sampled


def count_by(records: list[Record], fields: list[str]) -> Counter[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for record in records:
        counts[tuple(field_value(record, field) for field in fields)] += 1
    return counts


def print_counts(title: str, records: list[Record], fields: list[str]) -> None:
    print(title)
    total = len(records)
    print(f"  total: {total}")
    for key, count in sorted(count_by(records, fields).items()):
        label = " | ".join(key)
        pct = 100 * count / total if total else 0.0
        print(f"  {label:<40} {count:>7}  {pct:>6.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline Experience Replay file.")
    parser.add_argument("--combined_file", default="", help="Combined Phase-2 train file. Split by --new_task.")
    parser.add_argument("--old_file", default="", help="JSONL/JSON list containing old-task train samples.")
    parser.add_argument("--new_file", default="", help="JSONL/JSON list containing new-task train samples.")
    parser.add_argument(
        "--new_task",
        default="SemEval2018",
        help="New task name used when --combined_file is provided.",
    )
    parser.add_argument("--output", required=True, help="Output mixed replay train file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--memory_ratio",
        type=float,
        default=None,
        help="Sample this fraction of the old pool as memory. Ignored if --memory_size is set.",
    )
    parser.add_argument(
        "--memory_size",
        type=int,
        default=None,
        help="Total old-task memory size. Overrides --memory_ratio.",
    )
    parser.add_argument(
        "--task_memory_ratios",
        default="",
        help="Optional per-task ratios, e.g. IHD:1.0,IACV2:0.5. Overrides global memory size/ratio.",
    )
    parser.add_argument(
        "--stratify_fields",
        default="task_dataset,target",
        help="Comma-separated fields used for stratified sampling.",
    )
    parser.add_argument("--no_shuffle_output", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if args.combined_file:
        combined_records = read_records(Path(args.combined_file))
        old_records = [record for record in combined_records if task_name(record) != args.new_task]
        new_records = [record for record in combined_records if task_name(record) == args.new_task]
        if not new_records:
            raise ValueError(f"No new-task records found for {args.new_task!r} in {args.combined_file}")
        if not old_records:
            raise ValueError(f"No old-task records found in {args.combined_file}")
    else:
        if not args.old_file or not args.new_file:
            raise ValueError("Provide either --combined_file or both --old_file and --new_file.")
        old_records = read_records(Path(args.old_file))
        new_records = read_records(Path(args.new_file))
    fields = [item.strip() for item in args.stratify_fields.split(",") if item.strip()]
    if not fields:
        raise ValueError("--stratify_fields must contain at least one field")

    task_ratios = parse_task_ratios(args.task_memory_ratios)
    if task_ratios:
        by_task: dict[str, list[Record]] = defaultdict(list)
        for record in old_records:
            by_task[task_name(record)].append(record)
        memory: list[Record] = []
        for task, records in sorted(by_task.items()):
            ratio = task_ratios.get(task, 0.0)
            size = min(len(records), int(round(len(records) * ratio)))
            memory.extend(stratified_sample(records, size, fields, rng))
    else:
        if args.memory_size is not None:
            memory_size = args.memory_size
        elif args.memory_ratio is not None:
            memory_size = int(round(len(old_records) * args.memory_ratio))
        else:
            raise ValueError("Provide one of --memory_size, --memory_ratio, or --task_memory_ratios.")
        memory_size = max(0, min(memory_size, len(old_records)))
        memory = stratified_sample(old_records, memory_size, fields, rng)

    mixed = list(new_records) + memory
    if not args.no_shuffle_output:
        rng.shuffle(mixed)
    write_jsonl(Path(args.output), mixed)

    print_counts("Old pool", old_records, fields)
    print_counts("New task", new_records, fields)
    print_counts("Replay memory", memory, fields)
    print_counts("Mixed output", mixed, fields)
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
