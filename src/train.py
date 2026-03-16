"""
train.py
--------
QLoRA fine-tuning of Qwen2.5-7B-Instruct on emoji prediction task.
Uses TRL SFTTrainer for supervised fine-tuning.

Usage:
    python src/train.py
    python src/train.py --rank 8 --lr 1e-4 --epochs 2   # custom hyperparams
"""

import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from data_prep import prepare_all

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

TRAIN_TEXT  = "data/raw/semeval2018/us_train.text"
TRAIN_LABEL = "data/raw/semeval2018/us_train.labels"
CONTRA_PATH = "data/contradiction/contradiction_en.json"   # optional
OUTPUT_DIR  = "outputs/lora_weights"


# ── CLI arguments (for ablation experiments) ──────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank",   type=int,   default=16,   help="LoRA rank r")
    parser.add_argument("--lr",     type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int,   default=3,    help="Training epochs")
    parser.add_argument("--batch",  type=int,   default=4,    help="Per-device batch size")
    parser.add_argument("--max_len",type=int,   default=256,  help="Max sequence length")
    parser.add_argument("--run_name", type=str, default="emoji-qwen25-lora")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\n{'='*50}")
    print(f"  EmojiLM Training — rank={args.rank}, lr={args.lr}, epochs={args.epochs}")
    print(f"{'='*50}\n")

    # ── 1. Tokenizer ───────────────────────────────────────────────────────────
    print("[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",   # Critical for causal LM training
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Data ────────────────────────────────────────────────────────────────
    print("[2/5] Preparing datasets...")
    train_ds, val_ds, _ = prepare_all(
        train_text_path=TRAIN_TEXT,
        train_label_path=TRAIN_LABEL,
        tokenizer=tokenizer,
        contradiction_path=CONTRA_PATH,
        max_length=args.max_len,
    )

    # ── 3. Model (4-bit quantized) ─────────────────────────────────────────────
    print("[3/5] Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # Nested quant: saves ~0.4 GB
        bnb_4bit_quant_type="nf4",       # NormalFloat4 > int4 for LLMs
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Remove if flash-attn not installed
    )
    print(f"  GPU memory after load: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── 4. LoRA ────────────────────────────────────────────────────────────────
    print("[4/5] Configuring LoRA...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,      # Convention: alpha = 2 * r
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
            "gate_proj", "up_proj", "down_proj",        # MLP
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 5. Train ───────────────────────────────────────────────────────────────
    print("[5/5] Starting training...")
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,

        # Epochs & batching
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=4,        # Effective batch = batch * 4

        # Learning rate
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        # Precision
        bf16=True,
        tf32=True,

        # Logging & checkpointing
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Sequence length
        max_seq_length=args.max_len,

        # W&B
        report_to="wandb",
        run_name=f"{args.run_name}-r{args.rank}-lr{args.lr}",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = f"{OUTPUT_DIR}/final_r{args.rank}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n✓ Model saved to: {save_path}")
    print(f"  Final eval loss: {trainer.state.best_metric:.4f}")


if __name__ == "__main__":
    main()
