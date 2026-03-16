"""
evaluate.py
-----------
Full evaluation pipeline: metrics, confusion matrix, per-class breakdown,
and model comparison table.

Usage:
    python src/evaluate.py --lora outputs/lora_weights/final_r16
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data_prep import EMOJI_MAP, load_semeval
from inference import (
    load_finetuned_model,
    load_base_model,
    train_tfidf_baseline,
    predict_batch,
)

VALID_LABELS = list(EMOJI_MAP.values())
RESULTS_DIR  = "outputs/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Core evaluation ────────────────────────────────────────────────────────────
def evaluate_model(
    predictions: list[str],
    ground_truths: list[str],
    model_name: str,
    save: bool = True,
) -> dict:
    """Compute and display all metrics for one model."""

    acc         = accuracy_score(ground_truths, predictions)
    f1_macro    = f1_score(ground_truths, predictions,
                           average="macro",    labels=VALID_LABELS, zero_division=0)
    f1_weighted = f1_score(ground_truths, predictions,
                           average="weighted", labels=VALID_LABELS, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy    : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 Macro    : {f1_macro:.4f}")
    print(f"  F1 Weighted : {f1_weighted:.4f}")
    print(f"\n{classification_report(ground_truths, predictions, labels=VALID_LABELS, zero_division=0)}")

    if save:
        # Confusion matrix heatmap
        cm = confusion_matrix(ground_truths, predictions, labels=VALID_LABELS)
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=VALID_LABELS, yticklabels=VALID_LABELS,
        )
        plt.title(f"Confusion Matrix — {model_name}", fontsize=14)
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_").replace("/", "-")
        plt.savefig(f"{RESULTS_DIR}/confusion_{safe_name}.png", dpi=150)
        plt.close()
        print(f"  Confusion matrix saved.")

    return {
        "model":       model_name,
        "accuracy":    round(acc, 4),
        "f1_macro":    round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
    }


# ── Per-class breakdown ────────────────────────────────────────────────────────
def per_class_accuracy(predictions: list[str], ground_truths: list[str]) -> dict:
    """Accuracy per emoji class."""
    results = {}
    for emoji in VALID_LABELS:
        idxs = [i for i, g in enumerate(ground_truths) if g == emoji]
        if not idxs:
            continue
        correct = sum(predictions[i] == emoji for i in idxs)
        results[emoji] = round(correct / len(idxs), 4)
    return dict(sorted(results.items(), key=lambda x: -x[1]))


def plot_per_class(results: dict, model_name: str):
    emojis = list(results.keys())
    accs   = list(results.values())
    colors = ["#7c6aff" if a >= 0.5 else "#ff6a9b" if a >= 0.3 else "#888" for a in accs]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(emojis, [a * 100 for a in accs], color=colors, edgecolor="none")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Per-Emoji Accuracy — {model_name}")
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{acc*100:.0f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    safe = model_name.lower().replace(" ", "_")
    plt.savefig(f"{RESULTS_DIR}/per_class_{safe}.png", dpi=150)
    plt.close()


# ── Error analysis ─────────────────────────────────────────────────────────────
def error_analysis(
    predictions: list[str],
    ground_truths: list[str],
    texts: list[str],
    top_n: int = 10,
):
    """Find and display the most common error patterns."""
    from collections import Counter

    errors = [
        (gt, pred, txt)
        for gt, pred, txt in zip(ground_truths, predictions, texts)
        if gt != pred
    ]
    pair_counts = Counter((gt, pred) for gt, pred, _ in errors)

    print(f"\n── Top {top_n} Confusion Pairs ──")
    for (gt, pred), count in pair_counts.most_common(top_n):
        pct = count / len(ground_truths) * 100
        print(f"  {gt} → {pred}:  {count:4d}×  ({pct:.1f}%)")

    # Sample wrong examples for case study
    print(f"\n── Sample Errors (first 5) ──")
    for gt, pred, txt in errors[:5]:
        print(f"  Text: \"{txt[:60]}...\"")
        print(f"    True: {gt}  |  Predicted: {pred}\n")

    return pair_counts


# ── Full comparison pipeline ───────────────────────────────────────────────────
def run_full_comparison(
    test_text_path: str,
    test_label_path: str,
    lora_path: str,
):
    """
    Run all 4 models and produce comparison table.
    Models: TF-IDF, Zero-shot, GPT-4o (optional), Fine-tuned.
    """
    # Load test data
    test_df = load_semeval(test_text_path, test_label_path)
    texts   = test_df["text"].tolist()
    gt      = test_df["emoji"].tolist()

    all_results = []

    # ── Baseline 1: TF-IDF ────────────────────────────────────────────────────
    print("\n[1/3] Evaluating TF-IDF baseline...")
    from data_prep import load_semeval as _ls
    train_df = _ls(
        test_text_path.replace("trial", "train"),
        test_label_path.replace("trial", "train"),
    )
    clf    = train_tfidf_baseline(train_df)
    tfidf_preds = [EMOJI_MAP[p] for p in clf.predict(texts)]
    all_results.append(evaluate_model(tfidf_preds, gt, "TF-IDF + Logistic Regression"))

    # ── Baseline 2: Zero-shot ─────────────────────────────────────────────────
    print("\n[2/3] Evaluating zero-shot Qwen2.5-7B...")
    base_model, base_tok = load_base_model()
    zero_preds = predict_batch(texts, base_model, base_tok)
    all_results.append(evaluate_model(zero_preds, gt, "Qwen2.5-7B (zero-shot)"))
    del base_model

    # ── Our model: QLoRA fine-tuned ───────────────────────────────────────────
    print("\n[3/3] Evaluating fine-tuned model (QLoRA)...")
    ft_model, ft_tok = load_finetuned_model(lora_path=lora_path)
    ft_preds = predict_batch(texts, ft_model, ft_tok)
    all_results.append(evaluate_model(ft_preds, gt, "Qwen2.5-7B + QLoRA (ours)"))

    # Per-class breakdown for our model
    pc = per_class_accuracy(ft_preds, gt)
    plot_per_class(pc, "Qwen2.5-7B + QLoRA (ours)")

    # Error analysis for our model
    error_analysis(ft_preds, gt, texts)

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = pd.DataFrame(all_results)
    print("\n" + "="*55)
    print("  FINAL COMPARISON TABLE")
    print("="*55)
    print(summary.to_string(index=False))

    summary.to_csv(f"{RESULTS_DIR}/comparison_table.csv", index=False)
    print(f"\n✓ Results saved to {RESULTS_DIR}/")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_text",  default="data/raw/semeval2018/us_trial.text")
    parser.add_argument("--test_label", default="data/raw/semeval2018/us_trial.labels")
    parser.add_argument("--lora",       default="outputs/lora_weights/final_r16")
    args = parser.parse_args()

    run_full_comparison(args.test_text, args.test_label, args.lora)
