"""
inference.py
------------
Single-sample and batch inference for fine-tuned EmojiLM.
Supports fine-tuned model, base zero-shot model, and GPT-4o baseline.

Usage:
    python src/inference.py --text "Just got my dream job!"
    python src/inference.py --text "I love Mondays 😭" --mode all
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from data_prep import EMOJI_MAP, SYSTEM_PROMPT, build_chat_prompt

MODEL_NAME  = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH   = "outputs/lora_weights/final_r16"
VALID_EMOJI = set(EMOJI_MAP.values())


# ── Model loaders ──────────────────────────────────────────────────────────────
def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_base_model(model_name: str = MODEL_NAME):
    """Load base model (for zero-shot baseline)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(base_model_name: str = MODEL_NAME, lora_path: str = LORA_PATH):
    """Load fine-tuned model with LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    print(f"✓ Fine-tuned model loaded from: {lora_path}")
    return model, tokenizer


# ── Core inference ─────────────────────────────────────────────────────────────
def predict_emoji(
    text: str,
    model,
    tokenizer,
    max_new_tokens: int = 10,
) -> str:
    """
    Predict emoji for a given text.
    Returns the first valid emoji found in the output, or '❓' if none.
    """
    messages  = build_chat_prompt(text)          # No label = inference mode
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,             # Greedy = deterministic, good for eval
            repetition_penalty=1.1,
        )

    # Decode only newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Extract first valid emoji from output
    for char in raw:
        if char in VALID_EMOJI:
            return char
    return raw[:2] if raw else "❓"   # Fallback: take first 2 chars


def predict_batch(
    texts: list[str],
    model,
    tokenizer,
    batch_size: int = 8,
) -> list[str]:
    """Batch prediction for evaluation."""
    from tqdm import tqdm
    predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch = texts[i : i + batch_size]
        preds = [predict_emoji(t, model, tokenizer) for t in batch]
        predictions.extend(preds)
    return predictions


# ── GPT-4o baseline ────────────────────────────────────────────────────────────
def predict_gpt4o(text: str, client=None) -> str:
    """
    GPT-4o zero-shot prediction (requires openai API key).
    Set OPENAI_API_KEY environment variable before use.
    """
    if client is None:
        import openai, os
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    valid_list = " ".join(EMOJI_MAP.values())
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": (
                f"Text: {text}\n"
                f"Reply with ONE emoji from this list only: {valid_list}"
            )},
        ],
        max_tokens=5,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# ── TF-IDF baseline ────────────────────────────────────────────────────────────
def train_tfidf_baseline(train_df):
    """Train TF-IDF + Logistic Regression baseline."""
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)),
    ])
    clf.fit(train_df["text"], train_df["label"])
    print("✓ TF-IDF baseline trained")
    return clf


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",  type=str, required=True)
    parser.add_argument("--mode",  type=str, default="finetuned",
                        choices=["finetuned", "zeroshot", "all"])
    parser.add_argument("--lora",  type=str, default=LORA_PATH)
    args = parser.parse_args()

    print(f"\nInput: \"{args.text}\"\n")

    if args.mode in ("zeroshot", "all"):
        model, tokenizer = load_base_model()
        pred = predict_emoji(args.text, model, tokenizer)
        print(f"  Zero-shot (Qwen2.5-7B):  {pred}")
        del model

    if args.mode in ("finetuned", "all"):
        model, tokenizer = load_finetuned_model(lora_path=args.lora)
        pred = predict_emoji(args.text, model, tokenizer)
        print(f"  Fine-tuned (QLoRA):       {pred}")
