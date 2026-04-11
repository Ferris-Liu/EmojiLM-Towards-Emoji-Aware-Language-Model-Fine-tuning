"""
build_contradiction_dataset.py
-------------------------------
Build custom contradiction dataset where emoji and text sentiment conflict.
Two strategies:
  1. Manual seed examples (high quality)
  2. GPT-4o assisted generation (scale up to 300-500 samples)

Usage:
    python src/build_contradiction_dataset.py --n 300 --out data/contradiction/contradiction_en.json
    python src/build_contradiction_dataset.py --manual_only  # No API needed
"""

import argparse
import json
import os
import random
import time

# ── Manual seed examples (always included, high quality) ──────────────────────
MANUAL_SEEDS = [
    # --- Irony: negative situation + positive emoji ---
    {"text": "Woke up at 6am for a meeting that got cancelled",     "emoji": "😂", "type": "irony",    "true_sentiment": "negative"},
    {"text": "My flight just got cancelled, perfect timing",         "emoji": "✨", "type": "irony",    "true_sentiment": "negative"},
    {"text": "Lost my umbrella on the rainiest day of the year",     "emoji": "😂", "type": "irony",    "true_sentiment": "negative"},
    {"text": "Stepped on Lego barefoot at 3am",                     "emoji": "🔥", "type": "irony",    "true_sentiment": "negative"},
    {"text": "The WiFi is out again right before my deadline",       "emoji": "😍", "type": "irony",    "true_sentiment": "negative"},

    # --- Sarcasm: positive word + negative emoji ---
    {"text": "Another Monday, great",                               "emoji": "🙄", "type": "sarcasm",  "true_sentiment": "negative"},
    {"text": "Great, the printer is jammed again",                  "emoji": "🙄", "type": "sarcasm",  "true_sentiment": "negative"},
    {"text": "Oh wonderful, another group project",                  "emoji": "🙄", "type": "sarcasm",  "true_sentiment": "negative"},
    {"text": "I absolutely love sitting in traffic for 2 hours",    "emoji": "❤️", "type": "sarcasm",  "true_sentiment": "negative"},
    {"text": "Love how my alarm goes off 5 minutes before I fall asleep", "emoji": "❤️", "type": "sarcasm", "true_sentiment": "negative"},

    # --- Dark humor / coping ---
    {"text": "Failed my exam for the third time this semester",     "emoji": "😂", "type": "dark_humor","true_sentiment": "mixed"},
    {"text": "Accidentally deleted 3 hours of work",                "emoji": "😂", "type": "dark_humor","true_sentiment": "negative"},
    {"text": "My dog ate my assignment again",                      "emoji": "😂", "type": "dark_humor","true_sentiment": "mixed"},

    # --- Gen-Z / internet slang emoji ─────────────────────────────────────────
    # 💀 = dying of laughter (positive in slang)
    {"text": "My professor said the exam was easy",                 "emoji": "💀", "type": "slang",    "true_sentiment": "positive"},
    {"text": "No cap this song hits different at 3am",              "emoji": "💀", "type": "slang",    "true_sentiment": "positive"},
    {"text": "The way I just tripped in front of my whole class",   "emoji": "💀", "type": "slang",    "true_sentiment": "positive"},
    # 🧢 (cap) = lying in slang
    {"text": "No cap this is the best pizza I have ever had",       "emoji": "🧢", "type": "slang",    "true_sentiment": "positive"},
    # 😭 = extremely funny / overwhelmed (not just sad)
    {"text": "This meme is sending me",                             "emoji": "😭", "type": "slang",    "true_sentiment": "positive"},
    {"text": "The plot twist in that movie got me",                 "emoji": "😭", "type": "slang",    "true_sentiment": "mixed"},

    # --- Bittersweet / complex emotion ---
    {"text": "Just dropped my kid off at college",                  "emoji": "😢", "type": "bittersweet","true_sentiment": "mixed"},
    {"text": "Last day at a job I loved",                           "emoji": "😊", "type": "bittersweet","true_sentiment": "mixed"},
    {"text": "Watched my childhood home get sold today",            "emoji": "☀️", "type": "bittersweet","true_sentiment": "mixed"},

    # --- Chinese examples (cross-lingual) ---
    {"text": "终于交完作业了，感觉整个人都不好了",                       "emoji": "😂", "type": "irony",    "true_sentiment": "mixed"},
    {"text": "今天又被老板夸了，真是太开心了",                           "emoji": "🙄", "type": "sarcasm",  "true_sentiment": "negative"},
    {"text": "考试寄了",                                              "emoji": "💀", "type": "slang",    "true_sentiment": "negative"},
    {"text": "这个梗笑死我了",                                         "emoji": "😭", "type": "slang",    "true_sentiment": "positive"},
]

# ── GPT-4o generation ──────────────────────────────────────────────────────────
GPT_PROMPT_TEMPLATE = """Generate {n} diverse examples of text+emoji pairs where the emoji contradicts or adds irony to the text meaning.

Rules:
- The emoji must CONFLICT with the surface-level sentiment of the text
- Include a mix of: irony, sarcasm, dark_humor, slang (internet slang where emoji meaning differs from literal)
- Keep texts realistic and social-media-style (1-2 sentences)
- true_sentiment should reflect the ACTUAL emotion, not the surface reading

Respond with a valid JSON array ONLY, no other text. Format:
[
  {{"text": "...", "emoji": "...", "type": "irony|sarcasm|dark_humor|slang|bittersweet", "true_sentiment": "positive|negative|mixed"}}
]
"""


def generate_with_gpt4o(n_batches: int = 10, batch_size: int = 10, api_key: str = None) -> list[dict]:
    import openai
    client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    results = []
    for i in range(n_batches):
        print(f"  GPT-4o batch {i+1}/{n_batches}...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": GPT_PROMPT_TEMPLATE.format(n=batch_size)
                }],
                temperature=0.9,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            batch = json.loads(raw)
            results.extend(batch)
            time.sleep(1)   # Rate limiting
        except Exception as e:
            print(f"  Error in batch {i+1}: {e}")

    return results


# ── Quality filtering ──────────────────────────────────────────────────────────
def filter_quality(samples: list[dict]) -> list[dict]:
    """Remove low-quality or duplicate samples."""
    from data_prep import EMOJI_MAP
    valid_emoji = set(EMOJI_MAP.values())
    seen_texts  = set()
    clean       = []

    for s in samples:
        text  = s.get("text", "").strip()
        emoji = s.get("emoji", "").strip()

        if not text or len(text) < 10:
            continue
        if emoji not in valid_emoji:
            continue
        if text.lower() in seen_texts:
            continue

        seen_texts.add(text.lower())
        clean.append({
            "text":           text,
            "emoji":          emoji,
            "type":           s.get("type", "unknown"),
            "true_sentiment": s.get("true_sentiment", "unknown"),
        })

    return clean


# ── Main ───────────────────────────────────────────────────────────────────────
def build_dataset(
    target_n: int = 300,
    out_path: str = "data/contradiction/contradiction_en.json",
    use_gpt4o: bool = True,
    api_key: str = None,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[1] Starting with {len(MANUAL_SEEDS)} manual seed examples")
    all_samples = list(MANUAL_SEEDS)

    if use_gpt4o and len(all_samples) < target_n:
        needed     = target_n - len(all_samples)
        n_batches  = (needed // 10) + 1
        print(f"[2] Generating ~{n_batches * 10} more via GPT-4o...")
        gpt_samples = generate_with_gpt4o(n_batches=n_batches, api_key=api_key)
        all_samples.extend(gpt_samples)
        print(f"    Generated: {len(gpt_samples)} samples")
    else:
        print("[2] Skipping GPT-4o generation (manual_only mode)")

    # Filter
    all_samples = filter_quality(all_samples)
    random.shuffle(all_samples)
    print(f"[3] After quality filtering: {len(all_samples)} samples")

    # Type distribution
    from collections import Counter
    types = Counter(s["type"] for s in all_samples)
    print(f"    Types: {dict(types)}")

    # Save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved {len(all_samples)} samples to: {out_path}")
    return all_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int,  default=300)
    parser.add_argument("--out",         type=str,  default="data/contradiction/contradiction_en.json")
    parser.add_argument("--manual_only", action="store_true", help="Skip GPT-4o, use manual seeds only")
    parser.add_argument("--api_key",     type=str,  default=None)
    args = parser.parse_args()

    build_dataset(
        target_n=args.n,
        out_path=args.out,
        use_gpt4o=not args.manual_only,
        api_key=args.api_key,
    )
