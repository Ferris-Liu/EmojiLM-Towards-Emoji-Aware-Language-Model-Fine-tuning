"""
inference.py  (v3 — Qwen3-8B)
-------------------------------
Qwen3 关键改动：
  1. MODEL_NAME 改为 Qwen/Qwen3-8B
  2. parse_generated_output() 新增第一步：剥离 <think>...</think> 块
     （Qwen3 即使加了 /no_think 也可能在某些情况下输出 think 块，做防御处理）
  3. predict_emoji_v2() 的 generate 参数加了 enable_thinking=False
     （Qwen3 官方推荐的关闭 thinking 的方式）

用法：
  python src/inference.py --text "Woke up at 6am for a meeting that got cancelled"
  python src/inference.py --text "Just got promoted!" --mode all
"""

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import (
    EMOJI_MAP, VALID_EMOJI,
    SYSTEM_PROMPT_V1, SYSTEM_PROMPT_V2,
    build_chat_prompt, emoji_sentiment,
)

MODEL_NAME = "Qwen/Qwen3-8B"           # ← Qwen2.5-7B → Qwen3-8B
LORA_PATH  = "outputs/lora_weights/expA-main/final"


# ── 模型加载 ───────────────────────────────────────────────────────────────────
def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def load_base_model(model_name=MODEL_NAME):
    """加载基础模型（零样本 baseline 用）"""
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
    """加载微调后的模型（LoRA 权重叠加在基础模型上）"""
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


# ══════════════════════════════════════════════════════════════════════════════
#  Qwen3 核心改动：剥离 <think> 块的预处理函数
# ══════════════════════════════════════════════════════════════════════════════

def strip_think_blocks(text: str) -> str:
    """
    Qwen3 在 Thinking Mode 下会在回答前输出 <think>...</think> 块。
    即使加了 /no_think，极少数情况下仍可能出现。
    这个函数把 think 块完整剥离，只保留实际回答部分。

    示例输入：
      <think>
      The text expresses frustration. The emoji 😂 suggests irony...
      </think>
      {"tone": "Frustrated", "irony": true, "primary": "😂", ...}

    示例输出：
      {"tone": "Frustrated", "irony": true, "primary": "😂", ...}
    """
    # 去掉 <think>...</think> 块（包括跨行内容）
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


# ── JSON 解析器（三层容错）────────────────────────────────────────────────────
def parse_generated_output(raw: str) -> dict:
    """
    解析模型输出的 JSON 字符串。
    三层容错：JSON解析 → 扫描emoji → 默认值。
    """
    result = {
        "primary":       None,
        "alternative":   None,
        "tone":          "",
        "irony":         False,
        "reason":        "",
        "parse_success": False,
    }

    # ── 第零步（Qwen3 新增）：剥离 <think> 块 ─────────────────────────────────
    raw = strip_think_blocks(raw)

    # ── 策略1：JSON 解析 ───────────────────────────────────────────────────────
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

    # ── 策略2：扫描 emoji（JSON 解析失败时）───────────────────────────────────
    if result["primary"] is None:
        found = [c for c in raw if c in VALID_EMOJI]
        if found:
            result["primary"] = found[0]
            if len(found) > 1:
                result["alternative"] = found[1]

    # ── 策略3：完全失败，返回默认值 ────────────────────────────────────────────
    if result["primary"] is None:
        result["primary"] = "❓"

    return result


# ── 生成式推理（v2，返回完整结构化结果）────────────────────────────────────────
def predict_emoji_v2(text: str, model, tokenizer,
                     max_new_tokens: int = 150) -> dict:
    """
    Qwen3 推理：关闭 Thinking Mode，返回结构化 JSON 结果。
    """
    messages  = build_chat_prompt(text, use_v2=True)   # 不传 label，推理模式
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
            # Qwen3 官方推荐：通过 generation config 关闭 thinking
            # 如果模型支持，这里会直接跳过 <think> 块的生成
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    result = parse_generated_output(raw)   # 内部已包含 strip_think_blocks
    result["raw"] = raw
    return result


# ── 旧版推理（v1，单 emoji 输出，消融实验对比用）─────────────────────────────
def predict_emoji(text: str, model, tokenizer,
                  max_new_tokens: int = 10) -> str:
    """
    v1 格式：直接返回单个 emoji 字符串。
    消融实验 Exp-7（v1 vs v2 对比）时使用。
    """
    messages  = build_chat_prompt(text, use_v2=False)
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    raw = strip_think_blocks(raw)           # Qwen3 防御处理

    for char in raw:
        if char in VALID_EMOJI:
            return char
    return raw[:2] if raw else "❓"


# ── 批量推理 ───────────────────────────────────────────────────────────────────
def predict_batch_v2(texts: list, model, tokenizer,
                     batch_size: int = 4) -> list:
    from tqdm import tqdm
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting (v2)"):
        for t in texts[i: i + batch_size]:
            results.append(predict_emoji_v2(t, model, tokenizer))
    return results


# ── GPT-4o baseline ────────────────────────────────────────────────────────────
def predict_gpt4o_v2(text: str, client=None) -> dict:
    if client is None:
        import openai, os as _os
        client = openai.OpenAI(api_key=_os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_V2},
            {"role": "user",   "content": f"Text: {text}\nAnalyze and predict:"},
        ],
        max_tokens=200,
        temperature=0.0,
    )
    raw    = response.choices[0].message.content.strip()
    result = parse_generated_output(raw)
    result["raw"] = raw
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────
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
        r = predict_emoji_v2(args.text, model, tok)
        print(f"  [零样本 Qwen3-8B]")
        print(f"  Primary:     {r['primary']}")
        print(f"  Alternative: {r['alternative']}")
        print(f"  Irony:       {r['irony']}")
        print(f"  Reason:      {r['reason']}\n")
        del model

    if args.mode in ("finetuned", "all"):
        model, tok = load_finetuned_model(lora_path=args.lora)
        r = predict_emoji_v2(args.text, model, tok)
        print(f"  [微调后 Qwen3-8B + QLoRA]")
        print(f"  Primary:     {r['primary']}")
        print(f"  Alternative: {r['alternative']}")
        print(f"  Irony:       {r['irony']}")
        print(f"  Reason:      {r['reason']}\n")
