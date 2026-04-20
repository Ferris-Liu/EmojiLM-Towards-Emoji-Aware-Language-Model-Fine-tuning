"""Inference utilities for the final Qwen3 JSON emoji predictor."""

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import (
    VALID_EMOJI,
    build_chat_prompt,
)

MODEL_NAME = "Qwen/Qwen3-8B"
LORA_PATH  = "outputs/lora_weights/expA-main/final"


def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def load_base_model(model_name=MODEL_NAME):
    """Load the base model for zero-shot comparison."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"✓ Base model loaded: {model_name}")
    return model, tokenizer

def load_finetuned_model(base_model_name=MODEL_NAME, lora_path=LORA_PATH):
    """Load the LoRA fine-tuned model."""
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    print(f"✓ Fine-tuned model loaded: {lora_path}")
    return model, tokenizer


def strip_think_blocks(text: str) -> str:
    """Remove any Qwen3 thinking blocks before JSON parsing."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def parse_generated_output(raw: str) -> dict:
    """Parse model JSON with two fallbacks: scan emoji, then unknown marker."""
    result = {
        "primary":       None,
        "alternative":   None,
        "tone":          "",
        "irony":         False,
        "reason":        "",
        "parse_success": False,
    }

    raw = strip_think_blocks(raw)

    try:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group())
            result["tone"]          = data.get("tone", "")
            result["irony"]         = bool(data.get("irony", False))
            result["reason"]        = data.get("reason", "")
            result["parse_success"] = True

            primary = data.get("primary", "").strip()
            alt     = data.get("alternative", "").strip()
            result["primary"]     = primary if primary in VALID_EMOJI else None
            result["alternative"] = alt     if alt     in VALID_EMOJI else None
    except (json.JSONDecodeError, ValueError):
        pass

    if result["primary"] is None:
        found = find_valid_emoji(raw)
        if found:
            result["primary"] = found[0]
            if len(found) > 1:
                result["alternative"] = found[1]

    if result["primary"] is None:
        result["primary"] = "❓"

    return result


def find_valid_emoji(text: str) -> list:
    found = []
    candidates = sorted(VALID_EMOJI, key=len, reverse=True)
    for i in range(len(text)):
        for emoji in candidates:
            if text.startswith(emoji, i):
                found.append(emoji)
                break
    return found


def predict_emoji(text: str, model, tokenizer,
                  max_new_tokens: int = 150) -> dict:
    """Generate and parse the final structured prediction."""
    messages  = build_chat_prompt(text)
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # 贪心解码，保证评估可复现
            repetition_penalty=1.1,
            temperature=1.0,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    result = parse_generated_output(raw)
    result["raw"] = raw
    return result


def predict_batch(texts: list, model, tokenizer,
                  batch_size: int = 4) -> list:
    from tqdm import tqdm
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        for t in texts[i: i + batch_size]:
            results.append(predict_emoji(t, model, tokenizer))
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",  required=True,  help="输入文本")
    parser.add_argument("--mode",  default="finetuned",
                        choices=["finetuned", "zeroshot", "all"])
    parser.add_argument("--lora",  default=LORA_PATH)
    args = parser.parse_args()

    print(f'\n输入："{args.text}"\n')

    if args.mode in ("zeroshot", "all"):
        model, tok = load_base_model()
        r = predict_emoji(args.text, model, tok)
        print(f"  [零样本 Qwen3-8B]")
        print(f"  Primary:     {r['primary']}")
        print(f"  Alternative: {r['alternative']}")
        print(f"  Irony:       {r['irony']}")
        print(f"  Reason:      {r['reason']}\n")
        del model

    if args.mode in ("finetuned", "all"):
        model, tok = load_finetuned_model(lora_path=args.lora)
        r = predict_emoji(args.text, model, tok)
        print(f"  [微调后 Qwen3-8B + QLoRA]")
        print(f"  Primary:     {r['primary']}")
        print(f"  Alternative: {r['alternative']}")
        print(f"  Irony:       {r['irony']}")
        print(f"  Reason:      {r['reason']}\n")
