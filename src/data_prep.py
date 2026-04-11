"""
data_prep.py  (v3 — Qwen3-8B)
-------------------------------
Qwen3 关键改动：
  SYSTEM_PROMPT_V2 末尾加了 /no_think
  → 关闭 Qwen3 的 Thinking Mode，确保输出干净 JSON，不产生 <think> 块
  其余逻辑与 v2 完全相同。
"""

import re
import json
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# ── Emoji 标签映射（SemEval 2018 Task 2，20类）────────────────────────────────
EMOJI_MAP = {
    0:  "❤️",  1:  "😍",  2:  "😂",  3:  "💕",  4:  "🔥",
    5:  "😊",  6:  "😎",  7:  "✨",  8:  "💙",  9:  "😘",
    10: "📷",  11: "🇺🇸", 12: "☀️",  13: "💜",  14: "😉",
    15: "💯",  16: "😁",  17: "🎄",  18: "📸",  19: "😜",
}
EMOJI_TO_LABEL = {v: k for k, v in EMOJI_MAP.items()}
VALID_EMOJI    = set(EMOJI_MAP.values())

# ── 情感极性分组（方案三：情感一致性评估）────────────────────────────────────
POSITIVE_EMOJI = {
    "❤️", "😍", "💕", "🔥", "😊", "😎", "✨", "💙", "😘",
    "☀️", "💜", "😉", "💯", "😁", "🎄", "😜"
}
NEUTRAL_EMOJI  = {"😂", "📷", "🇺🇸", "📸"}

def emoji_sentiment(e: str) -> str:
    if e in POSITIVE_EMOJI:
        return "positive"
    return "neutral_or_negative"


# ══════════════════════════════════════════════════════════════════════════════
#  System Prompts
# ══════════════════════════════════════════════════════════════════════════════

# v1：旧版，单 emoji 输出（保留用于消融实验对比）
SYSTEM_PROMPT_V1 = (
    "You are an expert in emoji semantics and social media language. "
    "Your task is to predict the most appropriate emoji for a given text. "
    "Consider the emotional tone, context, and cultural nuances of the text. "
    f"Choose from these 20 emoji only: {' '.join(EMOJI_MAP.values())}"
)

# v2/v3：生成式 JSON 输出
# 关键改动：末尾加 /no_think，关闭 Qwen3 的 Thinking Mode
# 如果使用 Qwen2.5，/no_think 会被忽略，不影响功能
SYSTEM_PROMPT_V2 = f"""You are an expert in emoji semantics and social media language.

Given a piece of text, you must:
1. Briefly analyze the emotional tone (1 sentence)
2. Detect if there is any irony, sarcasm, or internet slang present
3. Predict the PRIMARY emoji (most fitting)
4. Suggest one ALTERNATIVE emoji
5. Give a one-sentence reason

You MUST choose emoji only from this list:
{' '.join(EMOJI_MAP.values())}

Respond in this exact JSON format:
{{
  "tone": "<brief emotional analysis>",
  "irony": <true or false>,
  "primary": "<emoji>",
  "alternative": "<emoji>",
  "reason": "<one sentence explanation>"
}}

/no_think"""
# ↑ /no_think 是 Qwen3 的指令，告诉模型跳过 <think> 推理块，直接输出答案
# 对 Qwen2.5 无效果（会被当作普通文本忽略）

# 默认使用 v2
SYSTEM_PROMPT = SYSTEM_PROMPT_V2


# ── 训练标签构建（v2 格式）────────────────────────────────────────────────────
def build_label_v2(emoji: str, text: str) -> str:
    """
    为训练数据构造结构化的 assistant 回复。
    SemEval 只有 emoji 标签，tone/irony/reason 用规则启发式生成。
    """
    sentiment = emoji_sentiment(emoji)

    irony_keywords   = ["great", "wonderful", "love", "perfect", "amazing",
                        "just what i needed", "totally", "absolutely"]
    negative_context = any(w in text.lower() for w in
                           ["cancel", "fail", "late", "stuck", "broke", "lost",
                            "miss", "bad", "terrible", "worst", "monday"])
    has_irony_word   = any(w in text.lower() for w in irony_keywords)
    irony = negative_context and has_irony_word

    ALTERNATIVES = {
        "❤️": "💕",  "💕": "❤️",  "💙": "💜",  "💜": "💙",
        "😍": "😊",  "😊": "😁",  "😁": "😍",  "😂": "😜",
        "😜": "😉",  "😉": "😂",  "🔥": "💯",  "💯": "🔥",
        "✨": "☀️",  "☀️": "✨",  "📷": "📸",  "📸": "📷",
        "😎": "😊",  "😘": "❤️",  "🎄": "✨",  "🇺🇸": "☀️",
    }
    alternative = ALTERNATIVES.get(emoji, "😊")

    label = {
        "tone":        f"{'Ironic or sarcastic' if irony else sentiment.capitalize()} emotional tone",
        "irony":       irony,
        "primary":     emoji,
        "alternative": alternative,
        "reason":      f"The text conveys a {sentiment} sentiment, best matched by {emoji}.",
    }
    return json.dumps(label, ensure_ascii=False)


# ── Prompt 构建 ────────────────────────────────────────────────────────────────
def build_chat_prompt(text: str, label_emoji: str = None,
                      use_v2: bool = True) -> list:
    """
    构建对话格式的 prompt。
    训练时：label_emoji 不为 None，assistant 回复为结构化 JSON。
    推理时：label_emoji 为 None，让模型自由生成。
    use_v2=False 退回旧版单 emoji 输出（消融实验用）。
    """
    system = SYSTEM_PROMPT_V2 if use_v2 else SYSTEM_PROMPT_V1

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Text: {text}\nAnalyze and predict:"},
    ]
    if label_emoji is not None:
        assistant_content = (
            build_label_v2(label_emoji, text) if use_v2 else label_emoji
        )
        messages.append({"role": "assistant", "content": assistant_content})
    return messages


# ── 文本清洗 ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── SemEval 加载 ───────────────────────────────────────────────────────────────
def load_semeval(text_path: str, label_path: str) -> pd.DataFrame:
    texts  = open(text_path,  encoding="utf-8").read().strip().split("\n")
    labels = open(label_path, encoding="utf-8").read().strip().split("\n")

    df = pd.DataFrame({"text": texts, "label": [int(l) for l in labels]})
    df["text"]  = df["text"].apply(clean_text)
    df["emoji"] = df["label"].map(EMOJI_MAP)
    df = df[df["text"].str.len().between(10, 280)].reset_index(drop=True)

    print(f"[SemEval] Loaded {len(df):,} samples")
    return df


# ── Contradiction 数据集加载 ───────────────────────────────────────────────────
def load_contradiction(json_path: str) -> pd.DataFrame:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["text"]  = df["text"].apply(clean_text)
    df["label"] = df["emoji"].map(EMOJI_TO_LABEL).fillna(-1).astype(int)
    print(f"[Contradiction] Loaded {len(df):,} samples")
    return df


# ── Tokenize ───────────────────────────────────────────────────────────────────
def tokenize_dataset(df: pd.DataFrame, tokenizer,
                     max_length: int = 256,
                     use_v2: bool = True) -> Dataset:
    def tokenize_batch(examples):
        prompts = [
            build_chat_prompt(t, e, use_v2=use_v2)
            for t, e in zip(examples["text"], examples["emoji"])
        ]
        formatted = [
            tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=False
            )
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
    return ds.map(tokenize_batch, batched=True, remove_columns=ds.column_names)


# ── 完整 pipeline ──────────────────────────────────────────────────────────────
def prepare_all(train_text_path: str, train_label_path: str, tokenizer,
                contradiction_path: str = None, test_size: float = 0.1,
                max_length: int = 256, use_v2: bool = True):
    df = load_semeval(train_text_path, train_label_path)
    if contradiction_path:
        contra_df = load_contradiction(contradiction_path)
        df = pd.concat([df, contra_df], ignore_index=True)

    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=42,
        stratify=df["label"].clip(0, 19)
    )
    print(f"[Split] Train: {len(train_df):,} | Val: {len(val_df):,}")

    train_ds = tokenize_dataset(train_df, tokenizer, max_length, use_v2)
    val_ds   = tokenize_dataset(val_df,   tokenizer, max_length, use_v2)
    return train_ds, val_ds, val_df


if __name__ == "__main__":
    # 验证 Prompt 格式（不需要加载模型）
    msgs = build_chat_prompt("Just got promoted!", "🔥", use_v2=True)
    for m in msgs:
        print(f"\n[{m['role'].upper()}]")
        print(m["content"])
