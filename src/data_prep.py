"""Data loading and prompt construction for the EmojiLM JSON task."""

import re
import json
import pandas as pd
from datasets import Dataset

EMOJI_MAP = {
    0:  "❤️",  1:  "😍",  2:  "😂",  3:  "💕",  4:  "🔥",
    5:  "😊",  6:  "😎",  7:  "✨",  8:  "💙",  9:  "😘",
    10: "📷",  11: "🇺🇸", 12: "☀️",  13: "💜",  14: "😉",
    15: "💯",  16: "😁",  17: "🎄",  18: "📸",  19: "😜",
}
EMOJI_TO_LABEL = {v: k for k, v in EMOJI_MAP.items()}
VALID_EMOJI    = set(EMOJI_MAP.values())

POSITIVE_EMOJI = {
    "❤️", "😍", "💕", "🔥", "😊", "😎", "✨", "💙", "😘",
    "☀️", "💜", "😉", "💯", "😁", "🎄", "😜"
}

def emoji_sentiment(e: str) -> str:
    if e in POSITIVE_EMOJI:
        return "positive"
    return "neutral_or_negative"


def build_system_prompt(constrain_emoji: bool = True) -> str:
    """Build the final Qwen3 prompt used by training, inference, and evaluation."""
    constraint = ""
    if constrain_emoji:
        constraint = (
            "\nYou MUST choose emoji only from this list:\n"
            f"{' '.join(EMOJI_MAP.values())}\n"
        )

    return f"""You are an expert in emoji semantics and social media language.

Given a piece of text, you must:
1. Briefly analyze the emotional tone (1 sentence)
2. Detect if there is any irony, sarcasm, or internet slang present
3. Predict the PRIMARY emoji (most fitting)
4. Suggest one ALTERNATIVE emoji
5. Give a one-sentence reason

{constraint}
Respond in this exact JSON format:
{{
  "tone": "<brief emotional analysis>",
  "irony": <true or false>,
  "primary": "<emoji>",
  "alternative": "<emoji>",
  "reason": "<one sentence explanation>"
}}

/no_think"""


def build_training_label(emoji: str, text: str) -> str:
    """Build the structured assistant answer used as supervised fine-tuning label."""
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


def build_chat_prompt(text: str, label_emoji: str = None,
                      constrain_emoji: bool = True) -> list:
    """
    Build chat messages.
    Training passes label_emoji and receives a JSON assistant target.
    Inference omits label_emoji and lets the model generate the JSON.
    """
    messages = [
        {"role": "system", "content": build_system_prompt(constrain_emoji)},
        {"role": "user",   "content": f"Text: {text}\nAnalyze and predict:"},
    ]
    if label_emoji is not None:
        messages.append({"role": "assistant", "content": build_training_label(label_emoji, text)})
    return messages


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_semeval(text_path: str, label_path: str) -> pd.DataFrame:
    with open(text_path, encoding="utf-8") as f:
        texts = f.read().strip().split("\n")
    with open(label_path, encoding="utf-8") as f:
        labels = f.read().strip().split("\n")

    df = pd.DataFrame({"text": texts, "label": [int(l) for l in labels]})
    df["text"]  = df["text"].apply(clean_text)
    df["emoji"] = df["label"].map(EMOJI_MAP)
    df = df[df["text"].str.len().between(10, 280)].reset_index(drop=True)

    print(f"[SemEval] Loaded {len(df):,} samples")
    return df


def load_contradiction(json_path: str) -> pd.DataFrame:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["text"]  = df["text"].apply(clean_text)
    df["label"] = df["emoji"].map(EMOJI_TO_LABEL).fillna(-1).astype(int)
    print(f"[Contradiction] Loaded {len(df):,} samples")
    return df


def tokenize_dataset(df: pd.DataFrame, tokenizer,
                     max_length: int = 256,
                     constrain_emoji: bool = True) -> Dataset:
    def tokenize_batch(examples):
        prompts = [
            build_chat_prompt(t, e, constrain_emoji=constrain_emoji)
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


if __name__ == "__main__":
    msgs = build_chat_prompt("Just got promoted!", "🔥")
    for m in msgs:
        print(f"\n[{m['role'].upper()}]")
        print(m["content"])
