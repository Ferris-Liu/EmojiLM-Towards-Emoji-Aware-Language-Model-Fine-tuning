"""
train.py  (v3 — Qwen3-8B)
--------------------------
基座模型：Qwen3-8B（QLoRA 微调）。

Qwen3 关键变化：
  1. 模型名称：Qwen/Qwen3-8B
  2. Thinking Mode：Qwen3 默认会输出 <think>...</think> 推理块
     → 训练时在 system prompt 末尾加 /no_think 关闭
     → 保证输出格式干净，不影响 JSON 解析

用法：
  # 主实验（Exp-A）
  python src/train.py --max_samples 100000 --run_name expA-main

  # 消融：rank=8（Exp-B）
  python src/train.py --max_samples 100000 --rank 8 --run_name expB-rank8

  # 消融：去掉EmojiContra（Exp-C）
  python src/train.py --max_samples 100000 --no_contra --run_name expC-no-contra

  # 消融：去掉emoji列表约束（Exp-D）
  python src/train.py --max_samples 100000 --no_constraint --run_name expD-no-constraint

  # 从checkpoint续训
  python src/train.py --max_samples 100000 --resume outputs/lora_weights/expA-main/checkpoint-5000
"""

import argparse
import os
import sys
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import (
    load_semeval, load_contradiction,
    tokenize_dataset, EMOJI_MAP,
    SYSTEM_PROMPT_V2,
)

# ── 路径配置 ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen3-8B"
TRAIN_TEXT  = "data/raw/semeval2018/us_train.text"
TRAIN_LABEL = "data/raw/semeval2018/us_train.labels"
CONTRA_PATH = "data/contradiction/contradiction_en.json"
OUTPUT_DIR  = "outputs/lora_weights"


# ── 命令行参数 ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="EmojiLM QLoRA Fine-tuning (Qwen3-8B)")

    parser.add_argument("--rank",        type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch",       type=int,   default=4)
    parser.add_argument("--grad_accum",  type=int,   default=4)
    parser.add_argument("--max_len",     type=int,   default=256)
    parser.add_argument("--max_samples", type=int,   default=100000)
    parser.add_argument("--no_contra",      action="store_true")
    parser.add_argument("--no_constraint",  action="store_true")
    parser.add_argument("--run_name",    type=str,   default="emoji-qwen3-8b")
    parser.add_argument("--resume",      type=str,   default=None)
    parser.add_argument("--no_wandb",    action="store_true")
    parser.add_argument("--test_size",   type=float, default=0.1)

    return parser.parse_args()


# ── 数据准备 ────────────────────────────────────────────────────────────────────
def prepare_data(args, tokenizer):
    from sklearn.model_selection import train_test_split

    print("  加载SemEval数据...")
    df = load_semeval(TRAIN_TEXT, TRAIN_LABEL)

    if args.max_samples and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        print(f"  已采样：{len(df):,} 条")
    else:
        print(f"  使用全量数据：{len(df):,} 条")

    if not args.no_contra and os.path.exists(CONTRA_PATH):
        contra_df = load_contradiction(CONTRA_PATH)
        df = pd.concat([df, contra_df], ignore_index=True)
        print(f"  加入EmojiContra后：{len(df):,} 条")
    elif args.no_contra:
        print("  [消融] 已跳过EmojiContra数据集")
    else:
        print(f"  [警告] 未找到EmojiContra文件，跳过")

    train_df, val_df = train_test_split(
        df, test_size=args.test_size, random_state=42,
        
    )
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    print(f"  Tokenizing（max_len={args.max_len}）...")
    train_ds = tokenize_dataset(train_df, tokenizer, args.max_len, use_v2=True)
    val_ds   = tokenize_dataset(val_df,   tokenizer, args.max_len, use_v2=True)

    return train_ds, val_ds


# ── 主函数 ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  EmojiLM Training — Qwen3-8B")
    print(f"{'='*60}")
    print(f"  rank={args.rank}  lr={args.lr}  epochs={args.epochs}")
    print(f"  batch={args.batch}  grad_accum={args.grad_accum}"
          f"  -> 等效batch={args.batch * args.grad_accum}")
    print(f"  max_samples={args.max_samples}  max_len={args.max_len}")
    print(f"  no_contra={args.no_contra}  no_constraint={args.no_constraint}")
    print(f"  run_name={args.run_name}")
    if args.resume:
        print(f"  续训自：{args.resume}")
    print(f"{'='*60}\n")

    # 消融Exp-D：去掉emoji列表约束
    if args.no_constraint:
        print("[消融] 已去掉System Prompt中的emoji列表约束")
        import data_prep as dp
        emoji_list_line = (
            f"\nYou MUST choose emoji only from this list:\n"
            f"{' '.join(EMOJI_MAP.values())}"
        )
        dp.SYSTEM_PROMPT = dp.SYSTEM_PROMPT_V2.replace(emoji_list_line, "")

    # ── Step 1: Tokenizer ──────────────────────────────────────────────────────
    print("[1/5] 加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Step 2: 数据 ───────────────────────────────────────────────────────────
    print("[2/5] 准备数据集...")
    train_ds, val_ds = prepare_data(args, tokenizer)

    # ── Step 3: 模型（4bit量化）────────────────────────────────────────────────
    print("[3/5] 加载Qwen3-8B（4bit量化）...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
        print("  flash_attention_2 可用")
    except ImportError:
        attn_impl = "sdpa"
        print("  flash_attn未安装，使用sdpa")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    if torch.cuda.is_available():
        mem_gb   = torch.cuda.memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  显存：{mem_gb:.1f} GB / {total_gb:.0f} GB")

    # ── Step 4: LoRA ───────────────────────────────────────────────────────────
    print("[4/5] 配置LoRA...")
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Step 5: 训练 ───────────────────────────────────────────────────────────
    print("[5/5] 开始训练...")

    save_dir = os.path.join(OUTPUT_DIR, args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=save_dir,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        bf16=True,
        tf32=True,

        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        max_length=args.max_len,

        report_to="none" if args.no_wandb else "wandb",
        run_name=f"{args.run_name}-r{args.rank}-lr{args.lr}",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    if args.resume:
        print(f"  从checkpoint续训：{args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    final_path = os.path.join(save_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    best_loss = trainer.state.best_metric
    print(f"\n{'='*60}")
    print(f"  训练完成")
    print(f"  模型保存至：{final_path}")
    if best_loss:
        print(f"  最佳验证集loss：{best_loss:.4f}")
    print(f"{'='*60}\n")
    print(f"下一步——运行评估：")
    print(f"  python src/evaluate.py --lora {final_path}\n")


if __name__ == "__main__":
    main()
