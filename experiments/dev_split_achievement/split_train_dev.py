#!/usr/bin/env python3
"""Create a stratified train/dev split from a MOELoRA JSONL train file."""
import argparse
import json
import os
import random
from collections import Counter, defaultdict


def parse_ratio(value):
    value = str(value).strip()
    if value.endswith("%"):
        ratio = float(value[:-1]) / 100.0
    else:
        ratio = float(value)
    if ratio <= 0 or ratio >= 1:
        raise argparse.ArgumentTypeError("--ratio must be in (0, 1), e.g. 0.05")
    return ratio


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_split_source_lineno"] = lineno
            rows.append(obj)
    return rows


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            out = dict(obj)
            out.pop("_split_source_lineno", None)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def make_key(row, fields):
    return tuple(str(row.get(field, "")) for field in fields)


def split_stratified(rows, ratio, seed, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[make_key(row, fields)].append(row)

    dev_rows = []
    train_rows = []
    rng = random.Random(seed)

    for key in sorted(groups):
        items = list(groups[key])
        rng.shuffle(items)
        n = len(items)
        if n <= 1:
            n_dev = 0
        else:
            n_dev = int(round(n * ratio))
            if n_dev == 0:
                n_dev = 1
            if n_dev >= n:
                n_dev = n - 1
        dev_rows.extend(items[:n_dev])
        train_rows.extend(items[n_dev:])

    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    return train_rows, dev_rows


def print_distribution(name, rows, fields):
    dist = Counter(make_key(row, fields) for row in rows)
    total = len(rows)
    print(f"\n{name}: {total} samples")
    for key in sorted(dist):
        count = dist[key]
        pct = 100.0 * count / total if total else 0.0
        key_text = " | ".join(key)
        print(f"  {key_text:<32} {count:>6}  {pct:>6.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL train file")
    parser.add_argument("--train_out", required=True, help="Output JSONL train split")
    parser.add_argument("--dev_out", required=True, help="Output JSONL dev split")
    parser.add_argument(
        "--ratio",
        type=parse_ratio,
        required=True,
        help="Dev ratio, e.g. 0.05, 0.10, 15%%",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify_fields",
        default="task_dataset,target",
        help="Comma-separated JSON fields used as strata",
    )
    args = parser.parse_args()

    fields = [field.strip() for field in args.stratify_fields.split(",") if field.strip()]
    if not fields:
        raise ValueError("--stratify_fields cannot be empty")

    rows = read_jsonl(args.input)
    train_rows, dev_rows = split_stratified(rows, args.ratio, args.seed, fields)

    write_jsonl(train_rows, args.train_out)
    write_jsonl(dev_rows, args.dev_out)

    print(f"Input:     {args.input}")
    print(f"Train out: {args.train_out}")
    print(f"Dev out:   {args.dev_out}")
    print(f"Ratio:     {args.ratio:.4f}")
    print(f"Seed:      {args.seed}")
    print(f"Strata:    {', '.join(fields)}")
    print_distribution("Original train", rows, fields)
    print_distribution("New train", train_rows, fields)
    print_distribution("Dev", dev_rows, fields)


if __name__ == "__main__":
    main()
