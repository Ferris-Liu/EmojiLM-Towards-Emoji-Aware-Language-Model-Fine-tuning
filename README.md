# EmojiLM: Fine-Tuning LLMs for Complex Emoji Semantics

<p align="center">
  <img src="https://img.shields.io/badge/Course-COMP7045-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Model-Qwen3--8B-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Method-QLoRA-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Task-Emoji%20Prediction-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Version-v3-red?style=flat-square" />
</p>

<p align="center">
  <a href="#english-version"><img src="https://img.shields.io/badge/🌐 Language-English-informational?style=for-the-badge" /></a>
  &nbsp;&nbsp;
  <a href="#中文版本"><img src="https://img.shields.io/badge/🌐 语言-中文-informational?style=for-the-badge" /></a>
</p>

---

<!-- ============================================================ -->
<!--                      ENGLISH VERSION                         -->
<!-- ============================================================ -->

<h2 id="english-version">🇬🇧 English Version</h2>

<p align="right"><a href="#中文版本">👉 切换中文</a></p>

### 📖 Project Overview

Accurately predicting the emoji associated with a piece of text is a challenging NLU task, as it heavily relies on capturing implicit sentiment and complex contextual cues. Traditional frequency-based or shallow ML approaches (e.g., TF-IDF + Logistic Regression) tend to overfit high-frequency distributions and struggle to understand long-tail emotional expressions.

This project proposes **EmojiLM**, a fine-tuning paradigm based on **Qwen3-8B + QLoRA** that reframes the traditional emoji classification task as a **structured JSON generation task**. We introduce a three-dimensional evaluation framework (Exact Match, Semantic Similarity, Sentiment Consistency) to comprehensively assess model performance on complex semantic mapping.

#### Key Highlights & Findings

> 🏆 **Structured Generation Win**: At the 8B parameter scale, the fine-tuned model achieves a **100% strict JSON parse rate** on the test set.

> 📈 **Long-Tail Understanding Breakthrough**: F1 Macro improves by more than 2× over the TF-IDF baseline (`0.117 → 0.282`), with sentiment consistency reaching **87.1%**, demonstrating genuine freedom from high-frequency bias.

> ⚠️ **Multi-Task Interference Revealed (Key Finding)**: When "irony/contradiction detection" is added as a secondary joint-generation task, we observe severe **Catastrophic Interference**. Ablation studies show that without explicit Chain-of-Thought (CoT), forcing a single autoregressive pass to simultaneously handle shallow semantic mapping and deep pragmatic reasoning causes attentional collapse—providing valuable empirical insight into the limits of LLM instruction fine-tuning.

---

### 🛠 Repository Structure

```plaintext
emoji_llm/
├── src/
│   ├── data_prep.py                    # Data pipeline: cleaning, SemEval loading, prompt building (with /no_think)
│   ├── train.py                        # Training engine: Qwen3-8B + QLoRA, supports multiple ablation configs
│   ├── inference.py                    # Inference module: generative prediction, think-block stripping, JSON parsing
│   ├── evaluate.py                     # Evaluation framework: 3-dim metrics (semantic/sentiment/accuracy) + plots
│   └── build_contradiction_dataset.py  # Data augmentation: EmojiContra dataset (manual seeds + GPT-4o expansion)
├── configs/
│   └── lora_config.yaml                # Core hyperparams: ablation experiment stage configurations
├── data/
│   ├── raw/semeval2018/                # Raw benchmark data (SemEval 2018 Task 2)
│   └── contradiction/                  # Custom irony augmentation dataset (EmojiContra)
├── outputs/
│   ├── lora_weights/                   # Weight archive: LoRA weights for each ablation experiment
│   └── results/                        # Reports: confusion matrices, comparison plots, evaluation CSV
├── .gitignore
└── requirements.txt
```

---

### 🚀 Quick Start

#### 1. Environment Setup

We recommend using Conda for a clean environment:

```bash
conda create -n emoji_llm python=3.10 -y
conda activate emoji_llm
pip install -r requirements.txt
```

#### 2. Data Preparation

```bash
# For servers in China (e.g. AutoDL), set HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com

# Download SemEval 2018 Task 2 dataset
git clone https://github.com/fvancesco/Semeval2018-Task2-Emoji-Detection
mkdir -p data/raw/semeval2018
cp Semeval2018-Task2-Emoji-Detection/data/* data/raw/semeval2018/
```

***(Optional)* Build the EmojiContra augmentation dataset:**

```bash
# Manual seeds only (no API required)
python src/build_contradiction_dataset.py --manual_only

# GPT-4o augmentation (requires OPENAI_API_KEY)
python src/build_contradiction_dataset.py --n 300
```

#### 3. Training & Ablation Studies

Four rigorous ablation experiments (recommended hardware: RTX 4090 or equivalent):

```bash
# Exp-A: Main experiment (full data + irony joint training)
python src/train.py --max_samples 100000 --run_name expA-main

# Exp-B: Rank ablation (low-rank adaptation bottleneck)
python src/train.py --max_samples 100000 --rank 8 --run_name expB-rank8

# Exp-C: Dataset ablation (remove EmojiContra, isolate multi-task interference)
python src/train.py --max_samples 100000 --no_contra --run_name expC-no-contra

# Exp-D: Constraint ablation (relax predefined emoji list)
python src/train.py --max_samples 100000 --no_constraint --run_name expD-no-constraint
```

> **Note**: If Weights & Biases is not configured, append `--no_wandb` to disable cloud sync.

#### 4. Evaluation & Visualization

```bash
python src/evaluate.py --lora outputs/lora_weights/expA-main/final
```

Generates a comprehensive report with Exact Match, Semantic Score, and Sentiment Consistency.

#### 5. Interactive Inference

```bash
python src/inference.py --text "Woke up at 6am for a meeting that got cancelled"
```

Expected JSON output:

```json
{
  "primary": "😂",
  "alternative": "😤",
  "tone": "Frustrated but humorous",
  "irony": true,
  "reason": "The writer uses dark humor to cope with disappointment."
}
```

---

### 📊 Key Results & Discussion

#### The Baseline Illusion

The TF-IDF + Logistic Regression model appears strong on Exact Match (`0.410`), but this is entirely due to **extreme class imbalance** (it blindly predicts high-frequency classes like ❤️ and 😂). Its very low F1 Macro (`0.117`) confirms that it has not truly learned long-tail emotion understanding.

#### The Micro-Tuning Advantage

Qwen3-8B + QLoRA achieves 100% well-formed JSON outputs and pushes F1 Macro to `0.282`. Confusion matrix analysis shows a healthy diagonal distribution across classes, with overall sentiment consistency reaching an impressive **87.1%**.

#### The Multi-Task Bottleneck

In the main experiment with irony detection, the irony field accuracy fell well below expectations (~20%). Our analysis suggests that forcing a single autoregressive forward pass to simultaneously handle:

- **Shallow formatting**: strict JSON constraints
- **Mid-level semantic mapping**: emoji prediction
- **Deep pragmatic reasoning**: irony recognition

…creates severe computational resource conflicts. The model sacrifices the hardest implicit reasoning task to preserve format correctness. This points toward **multi-stage reasoning** or **Chain-of-Thought prompting** as the next research direction.

---

### ⚖️ Qwen3-8B vs Qwen2.5-7B (Historical Reference)

| Metric | Qwen2.5-7B (prev.) | Qwen3-8B (current) |
|--------|:------------------:|:-----------------:|
| Parameters | 7B | 8B |
| Effective Performance | Qwen2.5-7B level | Qwen2.5-14B level |
| Training Data | 18T tokens | 36T tokens |
| Language Support | 29 languages | 119 languages |
| Thinking Mode | None | Available (disabled) |
| VRAM (QLoRA) | ~10 GB | ~11 GB |
| RTX 4090 Runnable | ✅ | ✅ |

---

### 👥 Team Contributions

| Member | Role |
|--------|------|
| Wen Jingzhi | **Data Engineering**: Led SemEval preprocessing, built and validated EmojiContra dataset consistency |
| Liu Feiyu | **Model Training**: Set up and optimized the QLoRA pipeline, designed and ran 4 ablation experiments, managed W&B monitoring |
| Zhang Zijun | **Evaluation & Analysis**: Designed the 3-dimensional evaluation framework, ran baseline comparisons, led case studies and final report |

---

<!-- ============================================================ -->
<!--                        中文版本                               -->
<!-- ============================================================ -->

<h2 id="中文版本">🇨🇳 中文版本</h2>

<p align="right"><a href="#english-version">👉 Switch to English</a></p>

### 📖 项目简介

准确预测文本对应的 Emoji 是自然语言理解（NLU）中的一项挑战，因为它高度依赖于对隐式情感和复杂语境的精准捕捉。传统的基于词频或浅层机器学习的方法（如 TF-IDF + Logistic Regression）往往只能拟合高频分布，难以理解长尾的复杂情绪。

本项目提出了一种基于 **Qwen3-8B + QLoRA** 的微调范式（**EmojiLM**），将传统的 Emoji 分类任务重构为**结构化 JSON 生成任务**。通过引入三维评估体系（精确匹配、语义相似度、情感一致性），全面评估模型在复杂语义映射上的表现。

#### 核心亮点与发现

> 🏆 **结构化生成的胜利**：在 8B 参数规模下，微调模型在测试集上实现了 **100% 严格 JSON 结构解析率**。

> 📈 **长尾理解力的突破**：相比 TF-IDF 基线，F1 Macro 得分提升超过一倍（`0.117 → 0.282`），情感一致性达到 **87.1%**，证明其真正摆脱了高频偏差。

> ⚠️ **多任务干扰现象揭示（重要发现）**：将"反讽/矛盾检测"作为联合生成的次要任务时，观察到严重的**灾难性干扰（Catastrophic Interference）**。消融实验表明，在缺乏显式思维链（CoT）的情况下，强行在单次自回归生成中融合浅层语义映射与深层语用学推理，会导致模型注意力失焦——为 LLM 指令微调边界提供了有价值的实证分析。

---

### 🛠 代码结构

```plaintext
emoji_llm/
├── src/
│   ├── data_prep.py                    # 数据管道：清洗、SemEval 加载、Prompt 构建（含 /no_think 指令）
│   ├── train.py                        # 训练引擎：Qwen3-8B + QLoRA，支持多组消融配置
│   ├── inference.py                    # 推理模块：生成式预测、Think 块防御剥离、多层 JSON 解析容错
│   ├── evaluate.py                     # 评估框架：三维评估体系（语义 / 情感 / 准确率）及可视化
│   └── build_contradiction_dataset.py  # 数据增强：EmojiContra 数据集构建（手工种子 + GPT-4o 扩充）
├── configs/
│   └── lora_config.yaml                # 核心超参：定义消融实验的不同阶段配置
├── data/
│   ├── raw/semeval2018/                # 原始基准数据（SemEval 2018 Task 2）
│   └── contradiction/                  # 自建反讽增强数据集（EmojiContra）
├── outputs/
│   ├── lora_weights/                   # 权重归档：保存各消融实验的 LoRA 权重
│   └── results/                        # 实验报告：混淆矩阵、对比图表及评估 CSV
├── .gitignore
└── requirements.txt
```

---

### 🚀 快速开始

#### 1. 环境准备

建议使用 Conda 构建纯净环境：

```bash
conda create -n emoji_llm python=3.10 -y
conda activate emoji_llm
pip install -r requirements.txt
```

#### 2. 数据获取

```bash
# 针对国内服务器（如 AutoDL），配置 HuggingFace 镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# 获取 SemEval 2018 Task 2 数据集
git clone https://github.com/fvancesco/Semeval2018-Task2-Emoji-Detection
mkdir -p data/raw/semeval2018
cp Semeval2018-Task2-Emoji-Detection/data/* data/raw/semeval2018/
```

**（可选）构建 EmojiContra 增强数据集：**

```bash
# 仅使用手工构造的种子数据（无需 API）
python src/build_contradiction_dataset.py --manual_only

# 使用 GPT-4o 进行增强扩写（需配置 OPENAI_API_KEY）
python src/build_contradiction_dataset.py --n 300
```

#### 3. 模型训练与消融实验

本项目设计了 4 组严谨的消融实验（建议在 RTX 4090 或同级别显卡上运行）：

```bash
# Exp-A：主实验（全量数据 + 反讽联合训练）
python src/train.py --max_samples 100000 --run_name expA-main

# Exp-B：Rank 维度消融（探究低秩适应瓶颈）
python src/train.py --max_samples 100000 --rank 8 --run_name expB-rank8

# Exp-C：数据集消融（剥离 EmojiContra，探究多任务干扰源）
python src/train.py --max_samples 100000 --no_contra --run_name expC-no-contra

# Exp-D：约束消融（放宽预设 emoji 限制）
python src/train.py --max_samples 100000 --no_constraint --run_name expD-no-constraint
```

> **注**：若未配置 Weights & Biases 账号，请在命令末尾添加 `--no_wandb` 以禁用云端同步。

#### 4. 三维评估与可视化

```bash
python src/evaluate.py --lora outputs/lora_weights/expA-main/final
```

生成包含精确匹配、语义分数和情感一致性的详尽报告。

#### 5. 交互式推理

```bash
python src/inference.py --text "Woke up at 6am for a meeting that got cancelled"
```

预期 JSON 输出：

```json
{
  "primary": "😂",
  "alternative": "😤",
  "tone": "Frustrated but humorous",
  "irony": true,
  "reason": "The writer uses dark humor to cope with disappointment."
}
```

---

### 📊 实验分析与结果

#### 基线的"高分"假象

传统的 TF-IDF + Logistic Regression 模型在精确匹配率上看似领先（`0.410`），但这完全归因于数据集的**极端不平衡**（模型倾向于无脑预测高频类，如 ❤️ 和 😂）。其极低的 F1 Macro（`0.117`）证实了它并未真正学会理解长尾情感。

#### 微调的真实收益

Qwen3-8B + QLoRA 模型不仅成功输出了 100% 格式正确的 JSON 结果，F1 Macro 得分飙升至 `0.282`。混淆矩阵分析显示，微调模型在跨类别分类准确性上展现出沿对角线分布的健康特征，整体情感方向一致性达到了令人瞩目的 **87.1%**。

#### 多任务干扰分析

在包含"反讽检测"的主实验中，反讽字段准确率显著低于预期（~20%）。通过对比实验分析，我们推断：在自回归模型的一次前向传播中，迫使模型同时完成：

- **浅层格式化**：严格 JSON 约束
- **中层语义映射**：Emoji 预测
- **深层语用学推理**：反讽识别

……会导致严重的计算资源冲突。模型在损失函数的驱使下，选择放弃了最难的隐式推理任务以保全格式的正确率。这为后续引入**多阶段推理**或**思维链提示词（Chain-of-Thought）** 指明了研究方向。

---

### ⚖️ Qwen3-8B vs Qwen2.5-7B（旧版对比参考）

| 对比项 | Qwen2.5-7B（旧） | Qwen3-8B（当前） |
|--------|:---------------:|:---------------:|
| 参数量 | 7B | 8B |
| 等效性能 | Qwen2.5-7B 级别 | Qwen2.5-14B 级别 |
| 训练数据 | 18T tokens | 36T tokens |
| 支持语言 | 29 种 | 119 种 |
| Thinking Mode | 无 | 有（需关闭） |
| 显存需求（QLoRA） | ~10 GB | ~11 GB |
| RTX 4090 可跑 | ✅ | ✅ |

---

### 👥 贡献分工

| 成员 | 核心职责 |
|------|---------|
| 文景治 | **数据工程**：主导 SemEval 数据集预处理，构建并校验 EmojiContra 数据集的一致性 |
| 刘飞宇 | **模型训练**：搭建并优化 QLoRA 训练管线，设计并实施 4 组消融实验，管理 W&B 监控节点 |
| 张子骏 | **评估与分析**：设计创新的三维评估框架，执行基线对比，主导案例分析（Case Study）及最终报告撰写 |
