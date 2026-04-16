EmojiLM: Fine-Tuning LLMs for Complex Emoji Semantics and the Limits of Multi-Task Generation

    COMP7045 Mini-Project | Qwen3-8B + QLoRA | v3 — Generative Output & Task Interference Analysis

📖 项目简介 (Project Overview)

准确预测文本对应的 Emoji 是自然语言理解（NLU）中的一项挑战，因为它高度依赖于对隐式情感和复杂语境的精准捕捉。传统的基于词频或浅层机器学习的方法（如 TF-IDF + Logistic Regression）往往只能拟合高频分布，难以理解长尾的复杂情绪。

本项目提出了一种基于 Qwen3-8B 和 QLoRA 的微调范式（EmojiLM），将传统的 Emoji 分类任务重构为结构化的 JSON 生成任务。通过引入三维评估体系（精确匹配、语义相似度、情感一致性），我们全面评估了模型在复杂语义映射上的表现。

核心亮点与发现 (Key Highlights & Insights)：

    结构化生成的胜利：在 8B 参数规模下，微调模型在测试集上实现了 100% 的严格 JSON 结构解析率。

    长尾理解力的突破：相比于 TF-IDF 基线，我们的微调模型在 F1 Macro 得分上实现了超过一倍的提升 (0.117 -> 0.282)，情感一致性达到 87.1%，证明其摆脱了高频偏差，真正理解了表情的复杂语义。

    多任务干扰现象揭示 (重要发现)：在探索将“反讽/矛盾检测”作为联合生成的次要任务时，我们观察到了严重的灾难性干扰 (Catastrophic Interference)。消融实验表明，在缺乏显式思维链 (CoT) 的情况下，强行在单次自回归生成中融合浅层语义映射与深层语用学推理，会导致模型注意力失焦。这一发现为大语言模型的指令微调边界提供了有价值的实证分析。

🛠 代码结构 (Repository Structure)
Plaintext

emoji_llm/
├── src/
│   ├── data_prep.py                    # 数据管道：清洗、SemEval加载、Prompt构建（含/no_think 指令）
│   ├── train.py                        # 训练引擎：Qwen3-8B + QLoRA，支持多组消融配置
│   ├── inference.py                    # 推理模块：生成式预测、Think块防御剥离、多层JSON解析容错
│   ├── evaluate.py                     # 评估框架：三维评估体系（语义/情感/准确率）及可视化
│   └── build_contradiction_dataset.py  # 数据增强：EmojiContra数据集构建（手工种子 + GPT-4o 扩充）
├── configs/
│   └── lora_config.yaml                # 核心超参：定义消融实验的不同阶段配置
├── data/
│   ├── raw/semeval2018/                # 原始基准数据（基于 SemEval 2018 Task 2）
│   └── contradiction/                  # 自建反讽增强数据集 (EmojiContra)
├── outputs/
│   ├── lora_weights/                   # 权重归档：保存各消融实验的 LoRA 权重
│   └── results/                        # 实验报告：混淆矩阵、对比图表及评估 CSV
├── .gitignore
└── requirements.txt

🚀 快速开始 (Quick Start)
1. 环境准备 (Environment Setup)

建议使用 Conda 构建纯净环境：
Bash

conda create -n emoji_llm python=3.10 -y
conda activate emoji_llm
pip install -r requirements.txt

2. 数据获取 (Data Preparation)
Bash

# 针对国内服务器 (如 AutoDL)，配置 HuggingFace 镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# 获取 SemEval 2018 Task 2 数据集
git clone https://github.com/fvancesco/Semeval2018-Task2-Emoji-Detection
mkdir -p data/raw/semeval2018
cp Semeval2018-Task2-Emoji-Detection/data/* data/raw/semeval2018/

(可选) 构建 EmojiContra 增强数据集：
Bash

# 仅使用手工构造的种子数据（无需 API）
python src/build_contradiction_dataset.py --manual_only

# 使用 GPT-4o 进行增强扩写（需配置 OPENAI_API_KEY）
python src/build_contradiction_dataset.py --n 300

3. 模型训练与消融实验 (Training & Ablation Studies)

本项目设计了 4 组严谨的消融实验以探究模型的性能边界（建议在 RTX 4090 或同级别显卡上运行）：
Bash

# Exp-A: 主实验 (全量数据 + 反讽联合训练)
python src/train.py --max_samples 100000 --run_name expA-main

# Exp-B: Rank 维度消融 (探究低秩适应瓶颈)
python src/train.py --max_samples 100000 --rank 8 --run_name expB-rank8

# Exp-C: 数据集消融 (剥离 EmojiContra，探究多任务干扰源)
python src/train.py --max_samples 100000 --no_contra --run_name expC-no-contra

# Exp-D: 约束消融 (放宽预设 emoji 限制)
python src/train.py --max_samples 100000 --no_constraint --run_name expD-no-constraint

(注：若未配置 Weights & Biases 账号，请在上述命令末尾添加 --no_wandb 以禁用云端同步。)
4. 三维评估与可视化 (Evaluation)

执行定制化的评估脚本，生成包含精确匹配、语义分数和情感一致性的详尽报告：
Bash

python src/evaluate.py --lora outputs/lora_weights/expA-main/final

5. 交互式推理 (Inference Showcase)
Bash

python src/inference.py --text "Woke up at 6am for a meeting that got cancelled"

预期 JSON 输出：
JSON

{
  "primary": "😂",
  "alternative": "😤",
  "tone": "Frustrated but humorous",
  "irony": true,
  "reason": "The writer uses dark humor to cope with disappointment."
}

📊 实验分析与结果洞察 (Key Results & Discussion)
The Baseline Illusion (基线的“高分”假象)

传统的 TF-IDF + Logistic Regression 模型在精确匹配率（Accuracy）上看似领先（0.410），但这完全归因于数据集的极端不平衡（模型倾向于无脑预测高频类，如 ❤️ 和 😂）。其极低的 F1 Macro（0.117）证实了它并未真正学会理解长尾情感。
The Micro-Tuning Advantage (微调的真实收益)

Qwen3-8B + QLoRA 模型不仅成功输出了 100% 格式正确的 JSON 结果，其 F1 Macro 得分飙升至 0.282。混淆矩阵分析显示，微调模型在跨类别的分类准确性上展现出沿对角线分布的健康特征，其整体情感方向一致性达到了令人瞩目的 87.1%。
The Multi-Task Bottleneck (多任务干扰分析)

在包含“反讽检测”的主实验中，我们观察到反讽字段的准确率显著低于预期（~20%）。通过对比实验分析，我们推断：在自回归模型的一次前向传播中，迫使模型同时完成极其严格的浅层格式化（JSON 约束）、中层语义映射（Emoji 预测）以及深层语用学推理（反讽识别），会导致严重的计算资源冲突。 模型在损失函数的驱使下，选择放弃了最难的隐式推理任务以保全格式的正确率。这为后续引入多阶段推理或思维链提示词（Chain-of-Thought）指明了研究方向。
👥 贡献分工 (Team Contributions)
成员	核心职责
文景治	数据工程：主导 SemEval 数据集预处理，构建并校验 EmojiContra 数据集的一致性。
刘飞宇	模型训练：搭建并优化 QLoRA 训练管线，设计并实施 4 组消融实验，管理 W&B 监控节点。
张子骏	评估与分析：设计创新的三维评估框架，执行基线对比，主导案例分析（Case Study）及最终报告撰写。