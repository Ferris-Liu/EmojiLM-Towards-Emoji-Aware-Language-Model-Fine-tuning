# When 😂 Means Sad
## Fine-tuning LLMs for Emoji Semantics and Contradiction Detection

> COMP7045 Mini-Project | **Qwen3-8B** + QLoRA | v3 — Generative Output + Multi-dim Evaluation

---

## 项目简介

本项目微调 **Qwen3-8B** 使其理解 emoji 语义，包括：
- **Emoji预测**：给定文字，预测最合适的 emoji（SemEval 2018 Task 2，20类）
- **矛盾检测**：识别 emoji 与文字情感相悖的反讽/讽刺/网络俚语场景
- **跨语言分析**：中英文 emoji 使用习惯对比

**v3 相比 v2 的改动：**
- 基座模型从 Qwen2.5-7B-Instruct 升级为 **Qwen3-8B**
- System Prompt 末尾加 `/no_think`，关闭 Qwen3 的 Thinking Mode
- `inference.py` 新增 `strip_think_blocks()` 防御处理

---

## 代码结构

```
emoji_llm/
├── src/
│   ├── data_prep.py                    # 数据加载、清洗、Prompt构建（含/no_think）
│   ├── train.py                        # QLoRA微调主脚本（Qwen3-8B）
│   ├── inference.py                    # 推理：生成式输出 + think块剥离
│   ├── evaluate.py                     # 四维评估 + 可视化
│   └── build_contradiction_dataset.py  # EmojiContra数据集构建
├── configs/
│   └── lora_config.yaml                # 消融实验超参配置
├── data/
│   ├── raw/semeval2018/                # SemEval数据（下载后放这里）
│   └── contradiction/                  # EmojiContra数据集
├── outputs/
│   ├── lora_weights/                   # 训练保存的LoRA权重
│   └── results/                        # 评估结果图表和CSV
├── .gitignore
└── requirements.txt
```

---

## 各文件职责

| 文件 | 版本 | 主要功能 |
|------|------|---------|
| `data_prep.py` | v3 | EMOJI_MAP、Prompt构建（含/no_think）、SemEval加载、Tokenize |
| `train.py` | v3 | Qwen3-8B + QLoRA训练、消融实验CLI、预算采样控制 |
| `inference.py` | v3 | 生成式预测、think块剥离、三层JSON解析容错 |
| `evaluate.py` | v2 | 语义相似度、情感一致性、四维评估、对比图表 |
| `build_contradiction_dataset.py` | v1 | 手工种子数据、GPT-4o增强、质量过滤 |

---

## 快速开始

### 1. 安装依赖

```bash
conda create -n emoji_llm python=3.10 -y
conda activate emoji_llm
pip install -r requirements.txt
```

### 2. 下载数据集

```bash
# 设置国内镜像（AutoDL必须）
export HF_ENDPOINT=https://hf-mirror.com

git clone https://github.com/fvancesco/Semeval2018-Task2-Emoji-Detection
mkdir -p data/raw/semeval2018
cp Semeval2018-Task2-Emoji-Detection/data/* data/raw/semeval2018/
```

### 3. 构建 EmojiContra

```bash
# 仅手工种子（无需API）
python src/build_contradiction_dataset.py --manual_only

# GPT-4o增强（需要 OPENAI_API_KEY）
python src/build_contradiction_dataset.py --n 300
```

### 4. 训练（¥50预算方案，4组消融实验）

```bash
# Exp-A：主实验
python src/train.py --max_samples 100000 --run_name expA-main

# Exp-B：rank消融
python src/train.py --max_samples 100000 --rank 8 --run_name expB-rank8

# Exp-C：去掉EmojiContra
python src/train.py --max_samples 100000 --no_contra --run_name expC-no-contra

# Exp-D：去掉emoji约束
python src/train.py --max_samples 100000 --no_constraint --run_name expD-no-constraint
```

没有W&B账号时每条命令加 `--no_wandb`。

### 5. 评估（四维）

```bash
python src/evaluate.py --lora outputs/lora_weights/expA-main/final
```

### 6. 单条推理

```bash
python src/inference.py --text "Woke up at 6am for a meeting that got cancelled"

# 输出示例：
# Primary:     😂
# Alternative: 😤
# Tone:        Frustrated but humorous
# Irony:       True
# Reason:      The writer uses dark humor to cope with disappointment.
```

---

## Qwen3-8B vs Qwen2.5-7B

| 对比项 | Qwen2.5-7B | Qwen3-8B |
|--------|-----------|---------|
| 参数量 | 7B | 8B |
| 等效性能 | Qwen2.5-7B级别 | Qwen2.5-14B级别 |
| 训练数据 | 18T tokens | 36T tokens |
| 支持语言 | 29种 | 119种 |
| Thinking Mode | 无 | 有（需关闭） |
| 显存需求（QLoRA） | ~10GB | ~11GB |
| 4090可跑 | ✓ | ✓ |

---

## 预期结果

| 模型 | Exact Match | Semantic Score | Sentiment Acc. | F1 Macro |
|------|-------------|----------------|----------------|----------|
| TF-IDF + LR | 24.7% | 0.512 | 61.3% | 0.221 |
| Qwen3-8B 零样本 | ~34% | ~0.66 | ~73% | ~0.31 |
| GPT-4o 零样本 | 38.1% | 0.701 | 76.4% | 0.347 |
| **Qwen3-8B + QLoRA（ours）** | **~47%** | **~0.75** | **~81%** | **~0.44** |

---

## 分工

| 成员 | 负责 |
|------|------|
| 文景治 | 数据工程：SemEval预处理、EmojiContra构建、标注一致性 |
| 刘飞宇 | 模型训练：QLoRA配置、4组消融实验、W&B监控 |
| 张子骏 | 评估与报告：四维评估框架、基线对比、Case Study、报告撰写 |
