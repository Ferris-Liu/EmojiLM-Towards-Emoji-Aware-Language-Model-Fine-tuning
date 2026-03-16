# When 😂 Means Sad: Fine-tuning LLMs for Emoji Semantics and Contradiction Detection

> COMP7045 Mini-Project | Fine-tuning Qwen2.5-7B with QLoRA on SemEval 2018 Task 2

---

## Overview

This project fine-tunes a Large Language Model (Qwen2.5-7B-Instruct) using QLoRA to better understand emoji semantics, including:
- **Emoji Prediction**: Given text, predict the most fitting emoji
- **Contradiction Detection**: Detect when emoji and text carry conflicting sentiment (irony, sarcasm, slang)
- **Cross-lingual Analysis**: Compare emoji usage patterns in English vs Chinese

---

## Project Structure

```
emoji_llm/
├── src/
│   ├── data_prep.py                  # Data loading & preprocessing
│   ├── train.py                      # QLoRA fine-tuning
│   ├── evaluate.py                   # Metrics & model comparison
│   ├── inference.py                  # Single/batch prediction
│   └── build_contradiction_dataset.py  # Custom dataset builder
├── configs/
│   └── lora_config.yaml              # Hyperparameter configs
├── notebooks/
│   ├── 01_exploration.ipynb          # EDA
│   └── 02_case_study.ipynb           # Case study analysis
├── data/
│   └── contradiction/                # Custom contradiction dataset
└── outputs/
    └── results/                      # Evaluation charts & tables
```

---

## Quickstart

### 1. Install dependencies

```bash
conda create -n emoji_llm python=3.10 -y
conda activate emoji_llm
pip install -r requirements.txt
```

### 2. Download SemEval 2018 Task 2 data

```bash
git clone https://github.com/fvancesco/Semeval2018-Task2-Emoji-Detection
mkdir -p data/raw/semeval2018
cp Semeval2018-Task2-Emoji-Detection/data/* data/raw/semeval2018/
```

### 3. Build contradiction dataset

```bash
# Manual seeds only (no API needed):
python src/build_contradiction_dataset.py --manual_only

# With GPT-4o (requires OPENAI_API_KEY):
python src/build_contradiction_dataset.py --n 300 --api_key YOUR_KEY
```

### 4. Train

```bash
# Default config (rank=16, lr=2e-4, 3 epochs)
python src/train.py

# Custom hyperparameters (for ablation study)
python src/train.py --rank 8  --lr 2e-4 --epochs 3  # Exp 1
python src/train.py --rank 16 --lr 2e-4 --epochs 3  # Exp 2 (recommended)
python src/train.py --rank 32 --lr 1e-4 --epochs 5  # Exp 3
```

### 5. Evaluate

```bash
python src/evaluate.py --lora outputs/lora_weights/final_r16
```

### 6. Inference

```bash
python src/inference.py --text "Just got my dream job!" --mode all
```

---

## Results (Expected)

| Model                        | Accuracy | F1 (Macro) |
|------------------------------|----------|------------|
| TF-IDF + Logistic Regression | ~24.7%   | ~0.221     |
| Qwen2.5-7B (zero-shot)       | ~31.2%   | ~0.284     |
| GPT-4o (zero-shot)           | ~38.1%   | ~0.347     |
| **Qwen2.5-7B + QLoRA (ours)**| **~45%** | **~0.42**  |

---

## Team

| Member | Responsibilities |
|--------|-----------------|
| A | Data collection, preprocessing, contradiction dataset |
| B | Model training, QLoRA config, ablation experiments |
| C | Evaluation, baselines, case study, report |
