"""
data_prep.py
------------
Data loading, cleaning, and tokenization for Emoji LLM project.
Handles SemEval 2018 Task 2 dataset + custom contradiction dataset.
"""

import re
import json
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# ── Emoji label mapping (SemEval 2018 Task 2, 20 classes) ─────────────────────
EMOJI_MAP = {
    0:  "❤️",  1:  "😍",  2:  "😂",  3:  "💕",  4:  "🔥",
    5:  "😊",  6:  "😎",  7:  "✨",  8:  "💙",  9:  "😘",
    10: "📷",  11: "🇺🇸", 12: "☀️",  13: "💜",  14: "😉",
    15: "💯",  16: "😁",  17: "🎄",  18: "📸",  19: "😜",
}
EMOJI_TO_LABEL = {v: k for k, v in EMOJI_MAP.items()}

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert in emoji semantics and social media language. "
    "Your task is to predict the most appropriate emoji for a given text. "
    "Consider the emotional tone, context, and cultural nuances of the text. "
    f"Choose from these 20 emoji only: {' '.join(EMOJI_MAP.values())}"
)


# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Clean raw tweet text."""
    text = text.strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)        # Remove URLs
    text = re.sub(r"@\w+", "@user", text)               # Anonymize mentions
    text = re.sub(r"#(\w+)", r"\1", text)               # Remove # but keep word
    text = re.sub(r"\s+", " ", text).strip()            # Normalize whitespace
    return text


# ── Prompt builder (Qwen2.5 chat format) ──────────────────────────────────────
def build_chat_prompt(text: str, label_emoji: str = None) -> list[dict]:
    """
    Build prompt in Qwen2.5-Instruct chat format.
    If label_emoji is provided (training), the assistant reply is included.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Text: {text}\nPredict the emoji:"},
    ]
    if label_emoji is not None:
        messages.append({"role": "assistant", "content": label_emoji})
    return messages


# ── SemEval loader ─────────────────────────────────────────────────────────────
def load_semeval(text_path: str, label_path: str) -> pd.DataFrame:
    """
    Load SemEval 2018 Task 2 data.
    Expected files: us_train.text / us_train.labels
    """
    texts  = open(text_path,  encoding="utf-8").read().strip().split("\n")
    labels = open(label_path, encoding="utf-8").read().strip().split("\n")

    df = pd.DataFrame({"text": texts, "label": [int(l) for l in labels]})
    df["text"]  = df["text"].apply(clean_text)
    df["emoji"] = df["label"].map(EMOJI_MAP)

    # Filter extreme lengths
    df = df[df["text"].str.len().between(10, 280)].reset_index(drop=True)

    print(f"[SemEval] Loaded {len(df):,} samples")
    print(df["label"].value_counts().to_string())
    return df


# ── Contradiction dataset loader ───────────────────────────────────────────────
def load_contradiction(json_path: str) -> pd.DataFrame:
    """
    Load custom contradiction dataset.
    JSON format: [{"text": "...", "emoji": "😂", "type": "irony", "true_sentiment": "negative"}, ...]
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["text"]  = df["text"].apply(clean_text)
    df["label"] = df["emoji"].map(EMOJI_TO_LABEL).fillna(-1).astype(int)
    print(f"[Contradiction] Loaded {len(df):,} samples | Types: {df['type'].value_counts().to_dict()}")
    return df


# ── Tokenization ───────────────────────────────────────────────────────────────
def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_length: int = 256,
) -> Dataset:
    """Convert DataFrame to tokenized HuggingFace Dataset."""

    def tokenize_batch(examples):
        prompts = [
            build_chat_prompt(t, e)
            for t, e in zip(examples["text"], examples["emoji"])
        ]
        formatted = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=False)
            for p in prompts
        ]
        result = tokenizer(
            formatted,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    ds = Dataset.from_pandas(df[["text", "emoji"]].reset_index(drop=True))
    ds = ds.map(tokenize_batch, batched=True, remove_columns=ds.column_names)
    return ds


# ── Full pipeline ──────────────────────────────────────────────────────────────
def prepare_all(
    train_text_path: str,
    train_label_path: str,
    tokenizer: AutoTokenizer,
    contradiction_path: str = None,
    test_size: float = 0.1,
    max_length: int = 256,
):
    """
    End-to-end data preparation.
    Returns (train_dataset, val_dataset, test_df).
    """
    df = load_semeval(train_text_path, train_label_path)

    # Optionally merge contradiction data
    if contradiction_path:
        contra_df = load_contradiction(contradiction_path)
        df = pd.concat([df, contra_df], ignore_index=True)
        print(f"[Merged] Total samples: {len(df):,}")

    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label"].clip(0, 19)
    )
    print(f"[Split] Train: {len(train_df):,} | Val: {len(val_df):,}")

    train_ds = tokenize_dataset(train_df, tokenizer, max_length)
    val_ds   = tokenize_dataset(val_df,   tokenizer, max_length)

    return train_ds, val_ds, val_df  # val_df used for evaluation


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Verify prompt format without loading model
    sample = build_chat_prompt("Just got promoted at work today!", "🔥")
    for msg in sample:
        print(f"[{msg['role']}] {msg['content']}")
