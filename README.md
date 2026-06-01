# TAC-MoE Dataset Conversion

This repository uses `convert_datasets.py` to convert raw datasets into the
JSONL format used by TAC-MoE training.

- `IACV2`
- `SemEval2018`
- `IHC`

## Directory Layout

Place the raw files under `dataset/` with the following structure:

```text
dataset/
  IACV2/
    train_iacv2.csv
    test_iacv2.csv
  SemEval2018/
    train/
      SemEval2018-T3-train-taskA_emoji.txt
    test/
      SemEval2018-T3_gold_test_taskA_emoji.txt
  IHC/
    IHC_train.csv
    IHC_valid.csv
```

## Input Formats

### IACV2

`IACV2` uses CSV files:

- `dataset/IACV2/train_iacv2.csv`
- `dataset/IACV2/test_iacv2.csv`

Required columns:

| Column | Description |
| --- | --- |
| `Text` | Online debate text |
| `Label` | Binary label: `0` for non-sarcastic, `1` for sarcastic |

Converted samples use `task_dataset: "IACV2"` and `source_dataset: "IACV2"`.
When the merged `task_dataset.json` is written, `IACV2` is mapped to the task
name `Sarca`.

### SemEval2018

`SemEval2018` uses the official Task 3A emoji-version TSV files:

- `dataset/SemEval2018/train/SemEval2018-T3-train-taskA_emoji.txt`
- `dataset/SemEval2018/test/SemEval2018-T3_gold_test_taskA_emoji.txt`

Required columns:

| Column | Description |
| --- | --- |
| `Tweet text` | Tweet text |
| `Label` | Binary label: `0` for non-ironic, `1` for ironic |

Column matching is case-insensitive. Converted samples use
`task_dataset: "SemEval2018"` and `source_dataset: "SemEval2018"`. When the
merged `task_dataset.json` is written, `SemEval2018` is mapped to the task name
`Irony`.

### IHC

`IHC` uses CSV files:

- `dataset/IHC/IHC_train.csv`
- `dataset/IHC/IHC_valid.csv`

Required columns:

| Column | Description |
| --- | --- |
| `raw_text` | Social media text |
| `label` | Raw label: `normal` or `hate` |

Label mapping:

| Raw label | Output label |
| --- | --- |
| `normal` | `0` |
| `hate` | `1` |

Converted samples use `task_dataset: "IHD"` and `source_dataset: "IHC"`. When
the merged `task_dataset.json` is written, `IHD` is mapped to the task name
`IHD`.

## Conversion Commands

Run the conversion from the project root:

```bash
python convert_datasets.py --datasets IACV2,SemEval2018,IHC
```