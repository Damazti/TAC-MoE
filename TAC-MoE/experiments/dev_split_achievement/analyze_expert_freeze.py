#!/usr/bin/env python3
"""Analyze whether Phase 2 frozen experts match Phase 1 old-task routing.

The Phase 2 partial-freeze recipe should freeze experts that Phase 1 routes
old tasks into most strongly. This script compares the current freeze set with
the top old-task experts from the Phase 1 checkpoint for each dev ratio.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch


TASK_IDS = {"IHD": 1, "IACV2": 2, "SemEval2018": 3}


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_by(rows, *fields):
    counts = Counter()
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in fields)
        counts[key] += 1
    return counts


def softmax(logits):
    logits = logits.float()
    logits = logits - logits.max()
    probs = torch.exp(logits)
    return probs / probs.sum()


def entropy(probs):
    probs = probs.float().clamp_min(1e-12)
    return float(-(probs * probs.log()).sum().item())


def parse_freeze_ids(value):
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def checkpoint_path(work_root, tag, phase, step):
    saved_root = work_root / "saved"
    candidates = [
        saved_root / f"devsplit_{tag}_{phase}_step{step}" / f"checkpoint-{step}" / "adapter_model.bin",
        saved_root / f"devsplit_{tag}_{phase}_step{step}" / "adapter_model.bin",
    ]
    candidates.extend(sorted(saved_root.glob(f"devsplit_{tag}_{phase}_step{step}*/checkpoint-{step}/adapter_model.bin")))
    candidates.extend(sorted(saved_root.glob(f"devsplit_{tag}_{phase}_step{step}*/adapter_model.bin")))
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt
    return None


def load_routing(ckpt_path, task_ids):
    weights = torch.load(str(ckpt_path), map_location="cpu")
    gate_w = weights["lora_gate.GateL.weight"].float()
    task_emb = weights["lora_task_embedding.weight"].float()
    routings = {}
    for task_name, task_id in task_ids.items():
        if task_id >= task_emb.shape[0]:
            continue
        probs = softmax(gate_w @ task_emb[task_id])
        routings[task_name] = probs
    return gate_w, task_emb, routings


def top_ids(scores, k=4):
    return [int(i) for i in torch.argsort(scores, descending=True)[:k].tolist()]


def fmt_probs(probs):
    return " ".join(f"E{i}:{float(v):.4f}" for i, v in enumerate(probs.tolist()))


def fmt_top(probs, k=4):
    order = torch.argsort(probs, descending=True)[:k].tolist()
    return ", ".join(f"E{int(i)}={float(probs[i]):.4f}" for i in order)


def print_counts(title, rows):
    task_counts = count_by(rows, "task_dataset")
    label_counts = count_by(rows, "task_dataset", "target")
    total = len(rows)
    print(f"{title}: {total} samples")
    if not total:
        return
    for (task,), count in sorted(task_counts.items()):
        print(f"  {task:<12} {count:>6}  {100.0 * count / total:>6.2f}%")
    for (task, target), count in sorted(label_counts.items()):
        task_total = task_counts[(task,)]
        print(f"    {task:<12} target={target:<4} {count:>6}  {100.0 * count / task_total:>6.2f}% within task")


def analyze_ratio(work_root, tag, freeze_ids, p1_step, p2_step):
    data_root = work_root / "data" / tag
    p1_train = read_jsonl(data_root / "phase1" / "train.json")
    p1_dev = read_jsonl(data_root / "phase1" / "dev.json")
    p2_train = read_jsonl(data_root / "phase2" / "train.json")
    p2_test = read_jsonl(data_root / "phase2" / "test.json")

    print("=" * 100)
    print(f"{tag}")
    print("=" * 100)
    print_counts("Phase 1 train", p1_train)
    print_counts("Phase 1 dev", p1_dev)
    print_counts("Phase 2 train", p2_train)
    print_counts("Phase 2 test", p2_test)

    p1_ckpt = checkpoint_path(work_root, tag, "p1", p1_step)
    if p1_ckpt is None:
        print(f"Phase 1 checkpoint not found for {tag}")
        return

    _, task_emb, routings = load_routing(p1_ckpt, {"IHD": 1, "IACV2": 2})
    print(f"\nPhase 1 routing checkpoint: {p1_ckpt}")
    for task_name in ("IHD", "IACV2"):
        probs = routings[task_name]
        print(f"  {task_name:<8} entropy={entropy(probs):.4f} top4: {fmt_top(probs)}")
        print(f"           all: {fmt_probs(probs)}")

    p1_task_counts = count_by(p1_train, "task_dataset")
    ihd_n = p1_task_counts.get(("IHD",), 0)
    iac_n = p1_task_counts.get(("IACV2",), 0)
    total_old = max(ihd_n + iac_n, 1)

    balanced_old = 0.5 * (routings["IHD"] + routings["IACV2"])
    weighted_old = (ihd_n * routings["IHD"] + iac_n * routings["IACV2"]) / total_old

    for name, scores in (("balanced old-task score", balanced_old), ("train-weighted old-task score", weighted_old)):
        strongest = top_ids(scores, k=4)
        overlap = sorted(set(strongest) & set(freeze_ids))
        missing = [x for x in strongest if x not in freeze_ids]
        extra = [x for x in freeze_ids if x not in strongest]
        print(f"\n  {name}:")
        print(f"    top4      : {strongest} ({fmt_top(scores)})")
        print(f"    freeze ids: {freeze_ids}")
        print(f"    overlap   : {overlap}")
        print(f"    missing   : {missing}")
        print(f"    extra     : {extra}")

    if task_emb.shape[0] >= 3:
        cos = torch.nn.functional.cosine_similarity(task_emb[1], task_emb[2], dim=0).item()
        print(f"\n  Cosine(task embedding IHD, IACV2): {cos:+.4f}")

    p2_ckpt = checkpoint_path(work_root, tag, "p2", p2_step)
    if p2_ckpt is not None:
        _, p2_emb, p2_routings = load_routing(p2_ckpt, TASK_IDS)
        print(f"\nPhase 2 routing checkpoint: {p2_ckpt}")
        for task_name in ("IHD", "IACV2", "SemEval2018"):
            if task_name not in p2_routings:
                continue
            probs = p2_routings[task_name]
            frozen_mass = float(probs[freeze_ids].sum().item())
            trainable_ids = [i for i in range(probs.shape[0]) if i not in freeze_ids]
            trainable_mass = float(probs[trainable_ids].sum().item())
            print(
                f"  {task_name:<12} top4: {fmt_top(probs)} | "
                f"frozen_mass={frozen_mass:.4f} trainable_mass={trainable_mass:.4f}"
            )
        if p2_emb.shape[0] >= 4:
            cos_31 = torch.nn.functional.cosine_similarity(p2_emb[3], p2_emb[1], dim=0).item()
            cos_32 = torch.nn.functional.cosine_similarity(p2_emb[3], p2_emb[2], dim=0).item()
            print(f"  Cosine(SemEval2018 emb, IHD):   {cos_31:+.4f}")
            print(f"  Cosine(SemEval2018 emb, IACV2): {cos_32:+.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work_root", default="experiments/dev_split_achievement/work")
    parser.add_argument("--ratios", default="r05,r10,r15")
    parser.add_argument("--freeze_expert_ids", default="0,2,4,5")
    parser.add_argument("--p1_step", type=int, default=285)
    parser.add_argument("--p2_step", type=int, default=585)
    args = parser.parse_args()

    work_root = Path(args.work_root)
    tags = [x.strip() for x in args.ratios.split(",") if x.strip()]
    freeze_ids = parse_freeze_ids(args.freeze_expert_ids)

    print(f"Work root: {work_root}")
    print(f"Ratios: {tags}")
    print(f"Current freeze ids: {freeze_ids}")
    print()

    for tag in tags:
        analyze_ratio(work_root, tag, freeze_ids, args.p1_step, args.p2_step)


if __name__ == "__main__":
    main()
