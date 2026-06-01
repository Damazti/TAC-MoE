# MOELoRA-peft 项目代码指南

## 一、项目概述

本项目基于 ChatGLM3-6b 大模型，使用 **MoELoRA (Mixture of Experts LoRA)** 方法进行多任务指令微调。
当前已迁移至 3 个新任务：NLI（自然语言推理）、Metap（隐喻检测）、Sarca（讽刺检测）。
训练损失使用 **Achievement-based Multi-task Loss** 动态调权。

---

## 二、整体数据流

```
train.json (JSONL)
    │
    ▼
chatglm.py (tokenize + 读取 task_dataset.json 得到 task_id)
    │
    ▼
collator.py (padding + 组装 batch: input_ids, labels, task_id)
    │
    ▼
modeling_chatglm.py 模型 forward:
    input_ids → Embedding → GLMTransformer(28层) → output_layer → logits
    task_id → 每层 MMOELoraLinearS.forward() 中作为 expert 权重
    │
    ▼
trainer.py compute_loss:
    logits + labels → per-sample CE loss → achievement_loss 加权 → 反向传播
```

---

## 三、目录结构与文件说明

### 3.1 入口与启动

| 文件 | 作用 |
|------|------|
| `run_mlora.py` | **程序入口**。设置 CUDA_VISIBLE_DEVICES，解析命令行参数（ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments），调用 `main()` |
| `experiments/train_new_tasks.bash` | **新任务训练脚本**（当前使用）。配置 3 任务、achievement loss、deepspeed 单卡训练 |
| `experiments/moelora.bash` | 原始训练脚本（16 任务 PromptCBLUE，4 卡，8000 步），仅供参考 |

### 3.2 核心训练逻辑 (`src/MLoRA/`)

| 文件 | 作用 | 关键内容 |
|------|------|----------|
| `main.py` | **主训练流程**。加载数据 → 加载模型 → 构建 PEFT → 创建 Trainer → 训练/评估/预测 | - L115: `AutoConfig.from_pretrained` 加载模型配置<br>- L127: `AutoModel.from_pretrained` 加载模型权重<br>- L152-158: `lora_name == "moelora"` 时使用 `MMOELoraConfigS`，task_type 设为 `CAUSAL_LMS`<br>- L173: `get_peft_model()` 注入 LoRA 层<br>- L212: 模型名检查，只支持 `chatglm-6b` 和 `chatglm3-6b`<br>- L335: 创建 `Seq2SeqTrainer`<br>- L347-354: 启用 achievement loss 时挂载到 trainer |
| `arguments.py` | **参数定义**。两个 dataclass：`ModelArguments` 和 `DataTrainingArguments` | - `task_num` (默认16，CLI 传 3)<br>- `expert_num` (默认4)<br>- `task_embedding_dim` (默认64)<br>- `lora_name` (默认"lora"，CLI 传 "moelora")<br>- `use_achievement_loss` / `achievement_gamma` / `achievement_margin`<br>- `trainable` (LoRA 作用的模块名) |
| `trainer.py` | **HuggingFace Trainer 子类**（约 2800 行）。核心在 `compute_loss` 方法 | - L673: `__init__` 中 `self.achievement_loss = None`<br>- L2702-2747: **compute_loss**：<br>&nbsp;&nbsp;① 有 `task_ids` 且 `achievement_loss` 不为 None → 走 achievement 路径<br>&nbsp;&nbsp;② 从 `inputs.get("labels")` 获取 labels（不 pop，让模型正常运行）<br>&nbsp;&nbsp;③ logits 转 fp32 → 计算 per-sample CE loss → achievement 加权<br>&nbsp;&nbsp;④ 否则走原始 loss 路径 |
| `trainer_seq2seq.py` | 继承 `Trainer`，添加 `evaluate` / `predict` / `prediction_step` 的生成逻辑 | - L204-209: prediction_step 中传递 `task_id`、`depart` 到 `model.generate()` |
| `achievement_loss.py` | **Achievement 多任务损失加权模块** | - 公式：`w = (1 - score / (margin * P))^gamma`，softmax 归一化<br>- `compute_weighted_loss(per_sample_loss, task_ids)`：按 task 分组加权<br>- `update_score(task_id, score)`：更新任务 F1，触发权重重算<br>- 初始状态：所有任务 F1=0 → 权重均匀 |
| `test.py` | 测试脚本（可忽略） | |
| `main_offline.py` | 离线版 main（可忽略） | |

### 3.3 数据处理 (`src/data_processor/`)

| 文件 | 作用 | 关键内容 |
|------|------|----------|
| `chatglm.py` | **数据预处理**，将 JSONL 样本 tokenize 为模型输入 | **chatglm1_train**:<br>- L33: 读取 `data/task_dataset.json` 获取 task_id 映射<br>- L51-52: 对 input 和 target 分别 encode<br>- L62-67: chatglm3 分支设置 context_length<br>- L69: labels = `[-100]*context_length + answer_tokens`（-100 不计损失）<br>- L82-83: `task_id = task_dict[examples['task_dataset'][i]]`<br><br>**chatglm1_eval**: 类似但用于评估 |
| `collator.py` | **DataCollator**，将多个样本 padding 组装为 batch | **LongestSequenceCollator**:<br>- pad input_ids (用 pad_token_id)<br>- pad labels (用 -100)<br>- 当 `task_flag=True` 时组装 `task_id` tensor<br>- 当 `depart_flag=True` 时额外组装 `depart`、`entity` |

### 3.4 PEFT / MoELoRA (`src/MLoRA/peft/`)

| 文件 | 作用 | 关键内容 |
|------|------|----------|
| `__init__.py` | 包导出 | 导出所有 Config/Model 类，包括 `MMOELoraConfigS`, `MMOELoraModelS` |
| `mapping.py` | **PEFT 类型映射** | - `MODEL_TYPE_TO_PEFT_MODEL_MAPPING`: task_type → PeftModel 子类<br>&nbsp;&nbsp;`CAUSAL_LMS` → `PeftModelForCausalLMShared`<br>- `get_peft_model(model, peft_config)`: 根据 config 包装模型 |
| `peft_model.py` | **PeftModel 基类**及各任务子类 | - `PeftModel`: 基础包装，from_pretrained/save_pretrained<br>- `PeftModelForCausalLMShared`: MoELoRA 使用的子类，forward 中传递 task_id/kwargs |
| `shared.py` | **Gate 网络** | - `Gate(PeftConfig)`: task_embedding → Linear(te_dim, expert_num) → Softmax → expert 权重<br>- `GateN`: 简化版 Gate |

### 3.5 MoELoRA 核心 (`src/MLoRA/peft/tuners/`)

| 文件 | 作用 | 关键内容 |
|------|------|----------|
| `__init__.py` | 导出 tuner 类 | 包括 `MMOELoraConfigS`, `MMOELoraModelS` |
| `lora.py` | **标准 LoRA** 实现 | `LoraConfig`, `LoraModel`, `LoraLayer`, `LoraLinear` 基类 |
| `mmoelora.py` | **MMOELoRA 基础版**（含 Gate） | - `MMOELoraConfig`: 继承 LoraConfig，加 task_num/expert_num/task_embedding_dim<br>- `MMOELoraModel`: 替换 Linear 为 MMOELoraLinear<br>- `MMOELoraLinear`: **包含 Gate + TaskEmbedding** 的 LoRA 层<br>&nbsp;&nbsp;forward: task_id → Embedding → Gate → expert_weight → 加权多专家输出<br>- `MMOELinearA` / `MMOELinearB`: 多专家 LoRA A/B 矩阵<br>- `Expert`: 单个专家（一个 Linear 层） |
| `mmoeloraS.py` | **MMOELoRA 共享版**（当前使用） | - `MMOELoraConfigS`: 继承 LoraConfig，peft_type = MMOELORAS<br>- `MMOELoraModelS`: 继承 MMOELoraModel，替换为 MMOELoraLinearS<br>- `MMOELoraLinearS`: **不包含 Gate**，forward 接收预计算的 expert_weight（来自 kwargs["task_id"]）<br>&nbsp;&nbsp;L157: `expert_weight = kwargs["task_id"]`<br>&nbsp;&nbsp;L171-178: 遍历 expert，加权合并结果 |
| `adalora.py` | AdaLoRA 实现（未使用） | |
| `adaption_prompt.py` | Adaption Prompt（未使用） | |
| `prefix_tuning.py` | Prefix Tuning（未使用） | |
| `prompt_tuning.py` | Prompt Tuning（未使用） | |
| `p_tuning.py` | P-Tuning（未使用） | |

### 3.6 模型文件

| 文件 | 作用 |
|------|------|
| `根目录/modeling_chatglm.py` | **修改版 ChatGLM3 模型**。在 Attention/MLP 层传递 kwargs(task_id)，forward 返回 `CausalLMOutputWithPastAndConLoss`。已移除 NCELoss 对比损失 |
| `resources/modeling_chatglm.py` | 原始参考版本（旧，不直接使用） |
| `resources/chatglm3-6b/` | **模型权重目录**（服务器上），包含 config.json、tokenizer、权重文件。需将根目录的 modeling_chatglm.py 复制到此 |

### 3.7 DeepSpeed 配置

| 文件 | 作用 |
|------|------|
| `src/ds.config` | DeepSpeed 配置：ZeRO-2，fp16，AdamW，CPU offload optimizer |

### 3.8 数据转换

| 文件 | 作用 |
|------|------|
| `convert_datasets.py` | **数据转换脚本**。将 NLI/Metap/Sarca 原始 CSV/TSV 转为 JSONL 格式 |
| `data/train.json` | 训练数据（JSONL，当前 30 条 demo 样本） |
| `data/task_dataset.json` | 任务ID映射：`{"str2id": {"NLI":1, "Metap":2, "Sarca":3}, "id2str": {...}}` |
| `data/test.json` | 测试数据（待生成） |

### 3.9 评估（原始 PromptCBLUE，需适配）

| 文件 | 作用 |
|------|------|
| `results/evaluation.py` | 评估脚本（针对原 16 个医学任务，需重写） |
| `results/post_generate_process.py` | 预测结果后处理（针对原 16 个医学任务） |
| `results/utils.py` | 评估工具函数 |

---

## 四、MoELoRA 核心机制

### 4.1 task_id 数据流

```
训练数据 {"task_dataset": "NLI"}
    │
    ▼ chatglm.py: task_dict["NLI"] = 1
task_id = 1 (integer)
    │
    ▼ collator.py: torch.LongTensor([1, 2, 3, 1, ...])  # batch 中的 task_ids
    │
    ▼ model forward (modeling_chatglm.py):
      kwargs["task_id"] = tensor([1, 2, 3, 1])
    │
    ▼ 每一层 SelfAttention & MLP 中的 MMOELoraLinearS:
      kwargs["task_id"] 实际上是 expert_weight (已被 Gate 预处理)
      → 加权合并多个 expert 的 LoRA 输出
```

**注意**：在 `mmoeloraS.py` (共享版) 中，`kwargs["task_id"]` 实际传递的是已经过 Gate 计算的 expert_weight（不是原始 task_id 整数）。Gate 计算在 `PeftModelForCausalLMShared` 的 forward 中完成。

### 4.2 Expert 权重计算

```
task_id (int) → TaskEmbedding → task_vector (64维)
    │
    ▼ Gate = Linear(64, expert_num) + Softmax
expert_weight = [0.3, 0.2, 0.4, 0.1]  # 每个 expert 的权重
    │
    ▼ LoRA output = Σ expert_weight[i] * (B_i @ A_i @ x) * scaling
```

### 4.3 Achievement Loss

```
每个样本的 CE loss → 按 task 分组求均值 → 乘以 achievement 权重 → 求和

权重公式:
  raw_w[task] = (1 - F1_score / (margin * target))^gamma
  weight[task] = softmax(raw_w)[task]

效果: F1 越低的任务 → 权重越大 → 模型更关注弱任务
```

---

## 五、训练数据格式

### 5.1 train.json (JSONL，每行一个 JSON)

```json
{
  "input": "指令+输入文本",
  "target": "期望输出",
  "task_dataset": "NLI",
  "task_type": "",
  "sample_id": "",
  "answer_choices": []
}
```

- **必需字段**: `input`, `target`, `task_dataset`
- **`task_dataset`**: 必须与 `task_dataset.json` 中的 key 一致
- 其他字段设为空即可

### 5.2 task_dataset.json

```json
{
  "str2id": {"NLI": 1, "Metap": 2, "Sarca": 3},
  "id2str": {"1": "NLI", "2": "Metap", "3": "Sarca"}
}
```

---

## 六、关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora_name moelora` | "lora" | 使用 MoELoRA（必须设为 moelora） |
| `--task_num 3` | 16 | 任务数量 |
| `--expert_num 4` | 4 | 每层 LoRA 的 expert 数量 |
| `--task_embedding_dim 64` | 64 | 任务 embedding 维度 |
| `--lora_rank 16` | 8 | LoRA rank（必须能被 expert_num 整除） |
| `--trainable` | "q_proj,v_proj" | LoRA 作用的模块名。ChatGLM3 用 `query_key_value,dense,dense_h_to_4h,dense_4h_to_h` |
| `--use_achievement_loss` | False | 启用 achievement 多任务损失 |
| `--achievement_gamma 2.0` | 2.0 | 聚焦参数，越大越关注弱任务 |
| `--achievement_margin 1.2` | 1.2 | 目标余量（>1 鼓励超过目标） |
| `--model_name_or_path` | - | 模型路径，如 `resources/chatglm3-6b` |
| `--fp16` | False | 半精度训练 |
| `--deepspeed src/ds.config` | - | DeepSpeed 配置文件 |

---

## 七、服务器部署步骤

```bash
# 1. 进入项目目录
cd /root/autodl-tmp/MOELoRA-peft-master

# 2. 确保 modeling_chatglm.py 在 resources/chatglm3-6b/ 中
cp modeling_chatglm.py resources/chatglm3-6b/modeling_chatglm.py

# 3. 生成训练数据（如需重新生成）
python convert_datasets.py

# 4. 开始训练
bash experiments/train_new_tasks.bash

# 5. 训练完成后，checkpoint 保存在 saved/moelora/
```

---

## 八、已做的修改（相对原始项目）

1. **chatglm.py**: 添加 chatglm3 分支（context_length 计算），task_dataset.json 路径改为 `data/`
2. **main.py**: 模型名检查支持 chatglm3-6b，导入并初始化 AchievementWeightedLoss
3. **arguments.py**: 新增 `use_achievement_loss` / `achievement_gamma` / `achievement_margin`
4. **trainer.py**: `__init__` 初始化 `self.achievement_loss = None`；`compute_loss` 增加 achievement loss 路径（不 pop labels，fp32 计算 per-sample loss）
5. **achievement_loss.py**: 新文件，实现 achievement 多任务损失加权
6. **convert_datasets.py**: 新文件，转换 NLI/Metap/Sarca 数据
7. **train_new_tasks.bash**: 新文件，3 任务训练脚本
8. **modeling_chatglm.py（根目录）**: 移除 NCELoss 对比损失，`con_loss` 初始化为 None

---

## 九、待完成事项

1. **完整数据集转换**: 修改 `convert_datasets.py` 中的 `n=10` 为实际数据量
2. **评估脚本适配**: `results/evaluation.py` 需要重写以支持 NLI/Metap/Sarca 的评估指标
3. **F1 动态更新**: 训练过程中定期评估并调用 `trainer.achievement_loss.update_scores()` 更新权重
4. **config.json 检查**: 服务器上 `resources/chatglm3-6b/config.json` 需包含 `max_length` 字段（官方 chatglm3 自带则无需修改）
