# -*- encoding: utf-8 -*-
"""
Convert datasets to MOELoRA training/test JSONL format.

Step 1: Convert each dataset → its own train.jsonl / test.jsonl (under dataset/<name>/)
Step 2: Merge selected datasets → data/train.json + data/test.json (shuffled)

Usage:
    # Convert all and merge all
    python convert_datasets.py

    # Only merge specific datasets
    python convert_datasets.py --datasets INLI,SemEval2018,IHC

    # Merge Metap/Hypo datasets with oversampling
    python convert_datasets.py --datasets LCC,Trofi,HypoData,HypoL --oversample

    # Change split seed
    python convert_datasets.py --seed 123

    # List available datasets
    python convert_datasets.py --list
"""
import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
METAP_HYPO_DIR = os.path.join(os.path.dirname(__file__),
                               "multitask_hyperbole_metaphor_detection-main", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


# ============================================================
# Instruction Prompts
# ============================================================

NLI_INSTRUCTION = (
    "You are an expert in natural language inference. "
    "Given a premise and a hypothesis, determine their relationship. "
    "Choose from the following:\n"
    "A) implied entailment - the hypothesis is indirectly supported by the premise\n"
    "B) explicit entailment - the hypothesis is directly stated or closely paraphrased from the premise\n"
    "C) neutral - the hypothesis is neither supported nor contradicted\n"
    "D) contradiction - the hypothesis contradicts the premise\n\n"
    "Example:\n"
    "Premise: She slammed the door so hard the walls shook.\n"
    "Hypothesis: She was angry.\n"
    "Output: A\n\n"
    "Strictly output only 'A', 'B', 'C', or 'D' with no other characters."
)

IHD_INSTRUCTION = (
    "You are an expert in hate speech detection. "
    "Given a social media post, classify it as hateful or not hateful.\n\n"
    "Example:\n"
    "Post: Some people just don't belong in this country.\n"
    "Output: 1\n\n"
    "Output '1' if the post contains hate speech and '0' if it does not. "
    "Strictly output only '0' or '1' with no other characters."
)

SARCA_INSTRUCTION = (
    "You are an expert in linguistics and sarcasm detection. "
    "Please analyze whether the given text contains sarcasm. "
    "Sarcasm is a form of irony that is intended to mock or convey contempt.\n\n"
    "Example:\n"
    "Text: Oh great, another Monday morning. Just what I needed.\n"
    "Output: 1\n\n"
    "Output '1' if the text is sarcastic and '0' if it is literal. "
    "Strictly output only '0' or '1' with no other characters."
)

IACV2_INSTRUCTION = (
    "You are an expert in linguistics and sarcasm detection. "
    "Please analyze whether the given text from an online debate contains sarcasm. "
    "Sarcasm is a form of irony that is intended to mock or convey contempt.\n\n"
    "Example:\n"
    "Text: Oh sure, because your internet argument is definitely going to change the world.\n"
    "Output: 1\n\n"
    "Output '1' if the text is sarcastic and '0' if it is literal. "
    "Strictly output only '0' or '1' with no other characters.\n"
)


IRONY_INSTRUCTION = (
    "You are an expert in irony detection. "
    "Please analyze whether the given tweet contains irony. "
    "Irony is the use of words to convey the opposite of their literal meaning, "
    "often for humorous or emphatic effect.\n\n"
    "Example:\n"
    "Text: I just love being stuck in traffic for hours. Best day ever!\n"
    "Output: 1\n\n"
    "Output '1' if the tweet is ironic and '0' if it is literal. "
    "Strictly output only '0' or '1' with no other characters."
)

# --- New: Metap & Hypo (output constraints first, example last) ---

METAP_INSTRUCTION = (
    "You are an expert in figurative language analysis. "
    "Given a sentence, determine whether it contains a metaphor. "
    "A metaphor is a figure of speech that describes something by comparing it "
    "to something else without using 'like' or 'as'.\n\n"
    "Output '1' if the sentence contains a metaphor and '0' if it does not. "
    "Strictly output only '0' or '1' with no other characters.\n\n"
    "Example:\n"
    "Sentence: Time is money.\n"
    "Output: 1"
)

HYPO_INSTRUCTION = (
    "You are an expert in figurative language analysis. "
    "Given a sentence, determine whether it contains hyperbole. "
    "Hyperbole is a figure of speech that involves exaggeration for emphasis or effect.\n\n"
    "Output '1' if the sentence contains hyperbole and '0' if it does not. "
    "Strictly output only '0' or '1' with no other characters.\n\n"
    "Example:\n"
    "Sentence: I've told you a million times.\n"
    "Output: 1"
)

NLI_LABEL_MAP = {
    "implied_entailment": "A",
    "explicit_entailment": "B",
    "neutral": "C",
    "contradiction": "D",
}

IHD_LABEL_MAP = {
    "normal": "0",
    "hate": "1",
}


# ============================================================
# Helper
# ============================================================

def write_jsonl(samples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Written {len(samples)} samples → {path}")


def make_sample(input_text, target, task_dataset, source_dataset=""):
    return {
        "input": input_text,
        "target": target,
        "task_dataset": task_dataset,
        "source_dataset": source_dataset,
        "task_type": "",
        "sample_id": "",
        "answer_choices": [],
    }


def train_test_split(rows, seed=42, test_ratio=0.2):
    """Split a list of rows into train and test sets."""
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    split_idx = int(len(rows) * (1 - test_ratio))
    return rows[:split_idx], rows[split_idx:]


def oversample_minority(samples, seed=42):
    """Oversample minority class within each task to balance label distribution."""
    rng = random.Random(seed)
    task_samples = defaultdict(list)
    for s in samples:
        task_samples[s["task_dataset"]].append(s)

    balanced = []
    for task in sorted(task_samples):
        label_groups = defaultdict(list)
        for s in task_samples[task]:
            label_groups[s["target"]].append(s)

        max_count = max(len(v) for v in label_groups.values())
        for label in sorted(label_groups):
            items = label_groups[label]
            if len(items) < max_count:
                extra = rng.choices(items, k=max_count - len(items))
                balanced.extend(items + extra)
                print(f"  Oversampled {task} label [{label}]: {len(items)} → {max_count} (+{len(extra)})")
            else:
                balanced.extend(items)

    return balanced


def oversample_tasks(samples, seed=42):
    """Oversample minority tasks to match the largest task's sample count."""
    rng = random.Random(seed)
    task_samples = defaultdict(list)
    for s in samples:
        task_samples[s["task_dataset"]].append(s)

    max_count = max(len(v) for v in task_samples.values())
    balanced = []
    for task in sorted(task_samples):
        items = task_samples[task]
        if len(items) < max_count:
            extra = rng.choices(items, k=max_count - len(items))
            balanced.extend(items + extra)
            print(f"  Oversampled task [{task}]: {len(items)} → {max_count} (+{len(extra)})")
        else:
            balanced.extend(items)

    return balanced


# ============================================================
# 1. INLI → NLI task (4-class)
# ============================================================

def convert_INLI(**kwargs):
    """INLI has official train/val/test splits."""
    base = os.path.join(DATASET_DIR, "INLI")

    def read_split(split_file):
        path = os.path.join(base, split_file)
        label_cols = ["implied_entailment", "explicit_entailment", "neutral", "contradiction"]
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                premise = row["premise"].strip()
                for label in label_cols:
                    hypothesis = row[label].strip()
                    input_text = (
                        f"{NLI_INSTRUCTION}\n"
                        f"Premise: {premise}\n"
                        f"Hypothesis: {hypothesis}"
                    )
                    samples.append(make_sample(input_text, NLI_LABEL_MAP[label], "NLI", source_dataset="INLI"))
        return samples

    train = read_split("train.csv")
    test = read_split("test.csv")

    write_jsonl(train, os.path.join(base, "train.jsonl"))
    write_jsonl(test, os.path.join(base, "test.jsonl"))
    return train, test


# ============================================================
# 2. IHC → IHD task (2-class, has train/valid splits)
# ============================================================

def convert_IHC(**kwargs):
    """IHC has train/valid CSV splits with raw_text and label columns."""
    base = os.path.join(DATASET_DIR, "IHC")

    def read_csv(path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                post = row["raw_text"].strip()
                label = IHD_LABEL_MAP[row["label"].strip()]
                input_text = f"{IHD_INSTRUCTION}\nPost: {post}\nOutput:"
                samples.append(make_sample(input_text, label, "IHD", source_dataset="IHC"))
        return samples

    train = read_csv(os.path.join(base, "IHC_train.csv"))
    test = read_csv(os.path.join(base, "IHC_valid.csv"))

    write_jsonl(train, os.path.join(base, "train.jsonl"))
    write_jsonl(test, os.path.join(base, "test.jsonl"))
    return train, test


# ============================================================
# 3. SemEval2018 → Sarca task (binary)
# ============================================================

def convert_SemEval2018(**kwargs):
    """SemEval2018 Task 3A — emoji + hashtag version.
    Train: remove ALL hashtags (prevent shortcut learning).
    Test:  keep all hashtags unchanged.
    """
    base = os.path.join(DATASET_DIR, "SemEval2018")

    def read_tsv(path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            text_col = next((c for c in reader.fieldnames if c.lower() == "tweet text"), None)
            label_col = next((c for c in reader.fieldnames if c.lower() == "label"), None)
            for row in reader:
                tweet = row[text_col].strip()
                label = str(int(row[label_col]))
                input_text = f"{IRONY_INSTRUCTION}\n{tweet}\nOutput:"
                samples.append(make_sample(input_text, label, "SemEval2018", source_dataset="SemEval2018"))
        return samples

    train = read_tsv(os.path.join(base, "train", "SemEval2018-T3-train-taskA_emoji.txt"))
    test = read_tsv(os.path.join(base, "test", "SemEval2018-T3_gold_test_taskA_emoji.txt"))

    write_jsonl(train, os.path.join(base, "train.jsonl"))
    write_jsonl(test, os.path.join(base, "test.jsonl"))
    return train, test


# ============================================================
# 4. SemEval2022 → Sarca task (binary)
# ============================================================

def convert_SemEval2022(**kwargs):
    base = os.path.join(DATASET_DIR, "SemEval2022")

    train_samples = []
    with open(os.path.join(base, "train", "train.En.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweet = row["tweet"].strip()
            label = str(int(row["sarcastic"]))
            input_text = f"{SARCA_INSTRUCTION}\n{tweet}\nOutput:"
            train_samples.append(make_sample(input_text, label, "SemEval2022", source_dataset="SemEval2022"))

    test_samples = []
    with open(os.path.join(base, "test", "task_A_En_test.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweet = row["text"].strip()
            label = str(int(row["sarcastic"]))
            input_text = f"{SARCA_INSTRUCTION}\n{tweet}\nOutput:"
            test_samples.append(make_sample(input_text, label, "SemEval2022", source_dataset="SemEval2022"))

    write_jsonl(train_samples, os.path.join(base, "train.jsonl"))
    write_jsonl(test_samples, os.path.join(base, "test.jsonl"))
    return train_samples, test_samples


# ============================================================
# 5. iSarcasm → Sarca task (binary, no tweet text)
# ============================================================

def convert_iSarcasm(**kwargs):
    """iSarcasm only has tweet IDs, no text. Skip if no text column found."""
    base = os.path.join(DATASET_DIR, "iSarcasm")

    label_map = {"not_sarcastic": "0", "sarcastic": "1"}

    def read_csv(path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            text_col = None
            for col in ["tweet", "text", "tweet_text", "content"]:
                if col in cols:
                    text_col = col
                    break
            if text_col is None:
                print(f"  WARNING: {path} has no text column (columns: {cols}), skipping")
                return samples

            for row in reader:
                tweet = row[text_col].strip()
                if not tweet:
                    continue
                label = label_map.get(row["sarcasm_label"].strip(), row["sarcasm_label"].strip())
                input_text = f"{SARCA_INSTRUCTION}\n{tweet}"
                samples.append(make_sample(input_text, label, "Sarca", source_dataset="SemEval2018"))
        return samples

    train = read_csv(os.path.join(base, "isarcasm_train.csv"))
    test = read_csv(os.path.join(base, "isarcasm_test.csv"))

    if train:
        write_jsonl(train, os.path.join(base, "train.jsonl"))
    if test:
        write_jsonl(test, os.path.join(base, "test.jsonl"))
    return train, test


# ============================================================
# 6. IACV2 → IACV2 task (binary sarcasm, separate from Sarca)
# ============================================================

def convert_IACV2(**kwargs):
    """IACV2 (Internet Argument Corpus v2) — sarcasm in online debates.
    Has its own train/test CSV splits. Uses distinct prompt to allow
    per-dataset evaluation separate from SemEval Sarca."""
    base = os.path.join(DATASET_DIR, "IACV2")

    def read_csv(path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["Text"].strip()
                label = str(int(row["Label"]))
                input_text = f"{IACV2_INSTRUCTION}{text}\nOutput:"
                samples.append(make_sample(input_text, label, "IACV2", source_dataset="IACV2"))
        return samples

    train = read_csv(os.path.join(base, "train_iacv2.csv"))
    test = read_csv(os.path.join(base, "test_iacv2.csv"))

    write_jsonl(train, os.path.join(base, "train.jsonl"))
    write_jsonl(test, os.path.join(base, "test.jsonl"))
    return train, test


# ============================================================
# 7-10. Metap/Hypo dual-task datasets
#   Each CSV has: Sentence, Hyperbole, Metaphor
#   Each row → 2 samples (one Metap, one Hypo)
#   No official train/test split → 80/20 split
# ============================================================

def _read_metap_hypo_csv(csv_path, source_name=""):
    """Read a Metap/Hypo CSV, return list of (sentence, hypo_label, metap_label, source_name)."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = row["Sentence"].strip()
            hypo_label = row["Hyperbole"].strip()
            metap_label = row["Metaphor"].strip()
            if not sentence or hypo_label not in ("0", "1") or metap_label not in ("0", "1"):
                continue
            rows.append((sentence, hypo_label, metap_label, source_name))
    return rows


def _rows_to_samples(row_list):
    """Convert (sentence, hypo_label, metap_label, source_name) rows to Metap + Hypo samples."""
    samples = []
    for sentence, hypo_label, metap_label, source_name in row_list:
        input_metap = f"{METAP_INSTRUCTION}\nSentence: {sentence}"
        samples.append(make_sample(input_metap, metap_label, "Metap", source_name))
        input_hypo = f"{HYPO_INSTRUCTION}\nSentence: {sentence}"
        samples.append(make_sample(input_hypo, hypo_label, "Hypo", source_name))
    return samples


def _convert_metap_hypo_csv(csv_path, dataset_label, seed=42):
    """Generic converter for Metap/Hypo CSV files (single split).
    Returns (train_samples, test_samples)."""
    rows = _read_metap_hypo_csv(csv_path, source_name=dataset_label)
    train_rows, test_rows = train_test_split(rows, seed=seed)

    train_samples = _rows_to_samples(train_rows)
    test_samples = _rows_to_samples(test_rows)

    base_dir = os.path.join(DATASET_DIR, dataset_label)
    write_jsonl(train_samples, os.path.join(base_dir, "train.jsonl"))
    write_jsonl(test_samples, os.path.join(base_dir, "test.jsonl"))

    return train_samples, test_samples


def convert_LCC(seed=42, **kwargs):
    """LCC_Label_Balanced.csv → Metap + Hypo"""
    return _convert_metap_hypo_csv(
        os.path.join(METAP_HYPO_DIR, "LCC_Label_Balanced.csv"), "LCC", seed=seed)


def convert_Trofi(seed=42, **kwargs):
    """Trofi_Label_Balanced.csv → Metap + Hypo"""
    return _convert_metap_hypo_csv(
        os.path.join(METAP_HYPO_DIR, "Trofi_Label_Balanced.csv"), "Trofi", seed=seed)


def convert_HypoData(seed=42, **kwargs):
    """hypo.csv → Metap + Hypo"""
    return _convert_metap_hypo_csv(
        os.path.join(METAP_HYPO_DIR, "hypo.csv"), "HypoData", seed=seed)


def convert_HypoL(seed=42, **kwargs):
    """hypo-l.csv → Metap + Hypo"""
    return _convert_metap_hypo_csv(
        os.path.join(METAP_HYPO_DIR, "hypo-l.csv"), "HypoL", seed=seed)


# ============================================================
# Registry
# ============================================================

CONVERTERS = {
    "INLI": convert_INLI,
    "IHC": convert_IHC,
    "SemEval2018": convert_SemEval2018,
    "SemEval2022": convert_SemEval2022,
    "iSarcasm": convert_iSarcasm,
    "IACV2": convert_IACV2,
    "LCC": convert_LCC,
    "Trofi": convert_Trofi,
    "HypoData": convert_HypoData,
    "HypoL": convert_HypoL,
}

# Dataset → Task mapping (list for multi-task datasets)
DATASET_TO_TASK = {
    "INLI": "NLI",
    "IHC": "IHD",
    "SemEval2018": "Irony",
    "SemEval2022": "Sarca",
    "iSarcasm": "Sarca",
    "IACV2": "Sarca",
    "LCC": "Metap, Hypo",
    "Trofi": "Metap, Hypo",
    "HypoData": "Metap, Hypo",
    "HypoL": "Metap, Hypo",
}


# ============================================================
# Merge & Statistics
# ============================================================

def print_stats(samples, label):
    task_counts = Counter(s["task_dataset"] for s in samples)
    print(f"\n=== {label} ({len(samples)} samples) ===")
    for task in sorted(task_counts):
        count = task_counts[task]
        pct = count / len(samples) * 100
        dist = Counter(s["target"] for s in samples if s["task_dataset"] == task)
        dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(dist.items()))
        print(f"  {task}: {count} ({pct:.1f}%)  [{dist_str}]")


TASK_INFO = {
    "NLI": ("4-class classification (natural language inference)", ["A", "B", "C", "D"]),
    "IHD": ("Binary classification (hate speech detection)", ["0", "1"]),
    "SemEval2018": ("Binary classification (sarcasm detection — Twitter)", ["0", "1"]),
    "IACV2": ("Binary classification (sarcasm detection — online debates)", ["0", "1"]),
    "Sarca": ("Binary classification (sarcasm detection)", ["0", "1"]),
    "Metap": ("Binary classification (metaphor detection)", ["0", "1"]),
    "Hypo": ("Binary classification (hyperbole detection)", ["0", "1"]),
}


def write_statistics(all_train, all_test, task_map, selected):
    lines = []
    lines.append("Dataset Statistics - MOELoRA Multi-Task")
    lines.append("=" * 40)
    lines.append("")

    # Record which datasets were used
    lines.append("=== Datasets Used ===")
    for name in selected:
        task = DATASET_TO_TASK.get(name, "?")
        lines.append(f"  {name} → {task}")
    lines.append("")

    for split_name, samples in [("TRAIN", all_train), ("TEST", all_test)]:
        task_counts = Counter(s["task_dataset"] for s in samples)
        lines.append(f"=== {split_name} ({len(samples):,} samples) ===")
        lines.append("")

        for task in sorted(task_counts):
            count = task_counts[task]
            pct = count / len(samples) * 100
            desc, _ = TASK_INFO.get(task, ("", []))
            lines.append(f"  {task} ({count:,} samples, {pct:.1f}%)    {desc}")

            dist = Counter(s["target"] for s in samples if s["task_dataset"] == task)
            for label in sorted(dist):
                lcount = dist[label]
                lpct = lcount / count * 100
                lines.append(f"    label [{label}]: {lcount:,} ({lpct:.1f}%)")
            lines.append("")

    # Task ratio
    train_counts = Counter(s["task_dataset"] for s in all_train)
    tasks_sorted = sorted(train_counts)
    min_count = min(train_counts.values())
    ratios = " : ".join(f"{train_counts[t]/min_count:.1f}" for t in tasks_sorted)
    lines.append("=== Task Ratio (Train) ===")
    lines.append(f"  {' : '.join(tasks_sorted)} = {ratios}")
    lines.append("")

    # Task ID mapping
    lines.append("=== Task ID Mapping ===")
    for tid, tname in sorted(task_map["id2str"].items(), key=lambda x: int(x[0])):
        lines.append(f"  {tid}: {tname}")

    stat_path = os.path.join(OUTPUT_DIR, "dataset_statistics.txt")
    with open(stat_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  dataset_statistics.txt → {stat_path}")


def _write_fold_outputs(train_samples, test_samples, selected, fold_output_dir):
    """Write train/test/task map/statistics for one fold."""
    os.makedirs(fold_output_dir, exist_ok=True)

    train_path = os.path.join(fold_output_dir, "train.json")
    test_path = os.path.join(fold_output_dir, "test.json")
    write_jsonl(train_samples, train_path)
    write_jsonl(test_samples, test_path)

    # Build task_dataset.json with task-level IDs (not dataset-level)
    ds_names = sorted(set(s["task_dataset"] for s in train_samples + test_samples))
    ds_to_task = {ds: DATASET_TO_TASK.get(ds, ds) for ds in ds_names}
    unique_tasks = sorted(set(ds_to_task.values()))
    task_name_to_id = {t: i + 1 for i, t in enumerate(unique_tasks)}
    task_map = {
        "str2id": {ds: task_name_to_id[task] for ds, task in ds_to_task.items()},
        "id2str": {str(i + 1): t for i, t in enumerate(unique_tasks)},
    }
    task_path = os.path.join(fold_output_dir, "task_dataset.json")
    with open(task_path, "w", encoding="utf-8") as f:
        json.dump(task_map, f, ensure_ascii=False, indent=4)

    print(f"  task_dataset.json → {task_path}")
    print(f"  Tasks: {task_map['str2id']}")

    # Also write fold-specific stats file
    stat_lines = []
    stat_lines.append(f"Dataset Statistics - Fold")
    stat_lines.append("=" * 40)
    stat_lines.append("")
    stat_lines.append("=== Datasets Used ===")
    for name in selected:
        task = DATASET_TO_TASK.get(name, "?")
        stat_lines.append(f"  {name} → {task}")
    stat_lines.append("")

    for split_name, samples in [("TRAIN", train_samples), ("TEST", test_samples)]:
        task_counts = Counter(s["task_dataset"] for s in samples)
        stat_lines.append(f"=== {split_name} ({len(samples):,} samples) ===")
        stat_lines.append("")
        for task in sorted(task_counts):
            count = task_counts[task]
            pct = count / len(samples) * 100
            dist = Counter(s["target"] for s in samples if s["task_dataset"] == task)
            stat_lines.append(f"  {task} ({count:,} samples, {pct:.1f}%)")
            for label in sorted(dist):
                lcount = dist[label]
                lpct = lcount / count * 100
                stat_lines.append(f"    label [{label}]: {lcount:,} ({lpct:.1f}%)")
            stat_lines.append("")

    stat_path = os.path.join(fold_output_dir, "dataset_statistics.txt")
    with open(stat_path, "w", encoding="utf-8") as f:
        f.write("\n".join(stat_lines) + "\n")
    print(f"  dataset_statistics.txt → {stat_path}")


def build_kfold_metap_hypo(selected, kfold=1, seed=42, do_oversample=False, test_ratio=0.1):
    """Build one random-seed split (90/10 by default) for Metap/Hypo datasets.
    Each dataset is split independently to guarantee balanced test ratio per source."""
    if kfold != 1:
        raise ValueError("Current mode supports only one split: please set --kfold 1")

    metap_hypo_datasets = {"LCC", "Trofi", "HypoData", "HypoL"}
    bad = [d for d in selected if d not in metap_hypo_datasets]
    if bad:
        raise ValueError(f"--kfold only supports Metap/Hypo datasets now. Invalid: {bad}")

    dataset_to_path = {
        "LCC": os.path.join(METAP_HYPO_DIR, "LCC_Label_Balanced.csv"),
        "Trofi": os.path.join(METAP_HYPO_DIR, "Trofi_Label_Balanced.csv"),
        "HypoData": os.path.join(METAP_HYPO_DIR, "hypo.csv"),
        "HypoL": os.path.join(METAP_HYPO_DIR, "hypo-l.csv"),
    }

    all_train_samples = []
    all_test_samples = []
    total_train_rows = 0
    total_test_rows = 0

    print(f"\n[Per-dataset Split] seed={seed}, test_ratio={test_ratio}")
    for name in selected:
        rows = _read_metap_hypo_csv(dataset_to_path[name], source_name=name)
        train_rows, test_rows = train_test_split(rows, seed=seed, test_ratio=test_ratio)
        train_samples = _rows_to_samples(train_rows)
        test_samples = _rows_to_samples(test_rows)
        all_train_samples.extend(train_samples)
        all_test_samples.extend(test_samples)
        total_train_rows += len(train_rows)
        total_test_rows += len(test_rows)
        pct = len(test_rows) / len(rows) * 100 if rows else 0
        print(f"  {name}: {len(rows)} rows → train={len(train_rows)}, test={len(test_rows)} ({pct:.1f}%)")

    if do_oversample:
        print("  Oversampling minority classes in train split")
        all_train_samples = oversample_minority(all_train_samples, seed=seed)

    random.Random(seed).shuffle(all_train_samples)
    random.Random(seed + 1).shuffle(all_test_samples)

    merge_and_write(all_train_samples, all_test_samples, seed=seed, selected=selected, do_oversample=False)

    print(f"\nSingle split generated under: {OUTPUT_DIR}/train.json and {OUTPUT_DIR}/test.json")
    print(f"Rows: train={total_train_rows}, test={total_test_rows}")


def merge_and_write(all_train, all_test, seed, selected, do_oversample=False):
    """Standard merge: write single train.json + test.json."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if do_oversample:
        print("\n--- Oversampling minority classes in training data ---")
        all_train = oversample_minority(all_train, seed=seed)

    random.seed(seed)
    random.shuffle(all_train)
    random.shuffle(all_test)

    train_path = os.path.join(OUTPUT_DIR, "train.json")
    write_jsonl(all_train, train_path)

    test_path = os.path.join(OUTPUT_DIR, "test.json")
    write_jsonl(all_test, test_path)

    # Build task_dataset.json: map dataset names → task-level IDs
    # Datasets sharing the same task (e.g., IACV2 & SemEval2018 → Sarca)
    # get the SAME task_id so MOELoRA routes them identically.
    # Evaluation still uses dataset names for per-dataset metrics.
    DATASET_TO_TASK_ID = {}  # dataset_name → task_name
    for ds_name in sorted(set(s["task_dataset"] for s in all_train + all_test)):
        task_name = DATASET_TO_TASK.get(ds_name, ds_name)  # fallback to ds_name
        DATASET_TO_TASK_ID[ds_name] = task_name

    # Assign numeric IDs per unique task (not per dataset)
    unique_tasks = sorted(set(DATASET_TO_TASK_ID.values()))
    task_name_to_id = {t: i + 1 for i, t in enumerate(unique_tasks)}

    task_map = {
        "str2id": {ds: task_name_to_id[task] for ds, task in DATASET_TO_TASK_ID.items()},
        "id2str": {str(i + 1): t for i, t in enumerate(unique_tasks)},
    }
    task_path = os.path.join(OUTPUT_DIR, "task_dataset.json")
    with open(task_path, "w", encoding="utf-8") as f:
        json.dump(task_map, f, ensure_ascii=False, indent=4)
    print(f"\n  task_dataset.json → {task_path}")
    print(f"  Tasks (dataset→task_id): {task_map['str2id']}")
    print(f"  Unique tasks: {unique_tasks}")

    print_stats(all_train, "Train")
    print_stats(all_test, "Test")

    write_statistics(all_train, all_test, task_map, selected)



# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and merge datasets for MOELoRA")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names to merge (default: all). "
                             f"Available: {', '.join(CONVERTERS.keys())}")
    parser.add_argument("--seed", type=int, default=3407,
                        help="Random seed for split and merge shuffling (default: 3407)")
    parser.add_argument("--ihc_split", type=str, default="8:1:1",
                        help="IHC train:val:test ratio (default: 8:1:1)")
    parser.add_argument("--oversample", action="store_true",
                        help="Oversample minority class in training data to balance labels")
    parser.add_argument("--kfold", type=int, default=0,
                        help="Build one 90/10 split for Metap/Hypo datasets (set to 1).")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets and exit")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name in CONVERTERS:
            print(f"  {name}")
        exit(0)

    # Determine which datasets to process
    if args.datasets:
        selected = [d.strip() for d in args.datasets.split(",")]
        for d in selected:
            if d not in CONVERTERS:
                print(f"ERROR: Unknown dataset '{d}'. Available: {', '.join(CONVERTERS.keys())}")
                exit(1)
    else:
        selected = list(CONVERTERS.keys())

    print(f"Datasets: {', '.join(selected)}")
    print(f"Seed: {args.seed}")
    if args.kfold:
        print(f"Single split mode")
    if args.oversample:
        print("Oversampling: ENABLED")
    print()

    if args.kfold:
        if args.kfold < 1:
            raise ValueError("--kfold must be >= 1")
        print("=" * 50)
        print("Building single random split...")
        build_kfold_metap_hypo(selected, kfold=args.kfold, seed=args.seed,
                               do_oversample=args.oversample)
    else:
        # Step 1: Convert each dataset
        all_train = []
        all_test = []

        for name in selected:
            print(f"[{name}]")
            train, test = CONVERTERS[name](seed=args.seed)
            all_train.extend(train)
            all_test.extend(test)
            print()

        # Step 2: Merge
        print("=" * 50)
        print("Merging...")
        merge_and_write(all_train, all_test, seed=args.seed, selected=selected,
                        do_oversample=args.oversample)
