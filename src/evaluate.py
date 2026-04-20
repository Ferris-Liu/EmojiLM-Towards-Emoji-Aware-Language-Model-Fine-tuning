"""Evaluate EmojiLM with exact, semantic, sentiment, and JSON-quality metrics."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
)

from data_prep import EMOJI_MAP, emoji_sentiment, load_semeval
from inference import (
    predict_emoji,
    load_finetuned_model, load_base_model,
    predict_batch,
)

VALID_LABELS = list(EMOJI_MAP.values())
RESULTS_DIR  = "outputs/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


EMOJI_SEMANTIC_SIMILARITY = {
    ("❤️", "💕"): 0.91, ("💕", "❤️"): 0.91,
    ("❤️", "💙"): 0.72, ("💙", "❤️"): 0.72,
    ("❤️", "💜"): 0.73, ("💜", "❤️"): 0.73,
    ("💕", "💙"): 0.70, ("💙", "💕"): 0.70,
    ("💙", "💜"): 0.85, ("💜", "💙"): 0.85,
    ("😊", "😁"): 0.88, ("😁", "😊"): 0.88,
    ("😊", "😍"): 0.75, ("😍", "😊"): 0.75,
    ("😁", "😍"): 0.72, ("😍", "😁"): 0.72,
    ("😂", "😜"): 0.68, ("😜", "😂"): 0.68,
    ("😜", "😉"): 0.74, ("😉", "😜"): 0.74,
    ("😂", "😉"): 0.60, ("😉", "😂"): 0.60,
    ("📷", "📸"): 0.93, ("📸", "📷"): 0.93,
    ("🔥", "💯"): 0.76, ("💯", "🔥"): 0.76,
    ("🔥", "✨"): 0.58, ("✨", "🔥"): 0.58,
    ("✨", "☀️"): 0.65, ("☀️", "✨"): 0.65,
    ("😘", "❤️"): 0.78, ("❤️", "😘"): 0.78,
    ("😘", "💕"): 0.80, ("💕", "😘"): 0.80,
    ("😎", "😊"): 0.55, ("😊", "😎"): 0.55,
}

def emoji_semantic_similarity(pred: str, truth: str) -> float:
    if pred == truth:
        return 1.0
    return EMOJI_SEMANTIC_SIMILARITY.get((pred, truth), 0.0)


def semantic_similarity_score(predictions: list, ground_truths: list) -> float:
    scores = [
        emoji_semantic_similarity(p, g)
        for p, g in zip(predictions, ground_truths)
    ]
    return float(np.mean(scores))


def sentiment_consistency_accuracy(predictions: list, ground_truths: list) -> float:
    consistent = sum(
        emoji_sentiment(p) == emoji_sentiment(g)
        for p, g in zip(predictions, ground_truths)
    )
    return consistent / len(predictions)


def evaluate_generative_quality(
    generative_results: list,
    ground_truths: list,
    contradiction_df=None,
) -> dict:
    """
    评估生成式输出的额外维度：
    1. JSON解析成功率
    2. 备选emoji的语义相似度
    3. 反讽检测准确率（需要EmojiContra数据集提供真实标签）
    """
    parse_success = sum(r["parse_success"] for r in generative_results) / len(generative_results)

    # 备选emoji评估
    alt_scores = []
    for r, truth in zip(generative_results, ground_truths):
        if r["alternative"] and r["alternative"] != "❓":
            alt_scores.append(emoji_semantic_similarity(r["alternative"], truth))
    avg_alt_score = float(np.mean(alt_scores)) if alt_scores else 0.0

    metrics = {
        "json_parse_success_rate": round(parse_success, 4),
        "alternative_semantic_score": round(avg_alt_score, 4),
    }

    # 反讽检测（如果有标注数据）
    if contradiction_df is not None and "type" in contradiction_df.columns:
        irony_preds = [r["irony"] for r in generative_results[:len(contradiction_df)]]
        irony_truth = [row["type"] in ("irony", "sarcasm", "dark_humor", "slang")
                       for _, row in contradiction_df.iterrows()]
        if len(irony_preds) == len(irony_truth):
            irony_acc = accuracy_score(irony_truth, irony_preds)
            metrics["irony_detection_accuracy"] = round(irony_acc, 4)

    return metrics


def evaluate_model_full(
    predictions: list,
    ground_truths: list,
    model_name: str,
    save: bool = True,
) -> dict:
    acc         = accuracy_score(ground_truths, predictions)
    f1_macro    = f1_score(ground_truths, predictions, average="macro",
                           labels=VALID_LABELS, zero_division=0)
    f1_weighted = f1_score(ground_truths, predictions, average="weighted",
                           labels=VALID_LABELS, zero_division=0)

    sem_score = semantic_similarity_score(predictions, ground_truths)
    sent_acc = sentiment_consistency_accuracy(predictions, ground_truths)

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  [Exact Match]")
    print(f"    Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    F1 Macro       : {f1_macro:.4f}")
    print(f"    F1 Weighted    : {f1_weighted:.4f}")
    print(f"  [Semantic Metrics]")
    print(f"    Semantic Score : {sem_score:.4f}  ← 语义相似度")
    print(f"    Sentiment Acc. : {sent_acc:.4f}  ← 情感方向一致性")

    metrics = {
        "model":        model_name,
        "accuracy":     round(acc, 4),
        "f1_macro":     round(f1_macro, 4),
        "f1_weighted":  round(f1_weighted, 4),
        "semantic_score": round(sem_score, 4),
        "sentiment_acc":  round(sent_acc, 4),
    }

    if save:
        _save_confusion_matrix(predictions, ground_truths, model_name)

    return metrics


def _save_confusion_matrix(predictions, ground_truths, model_name):
    cm = confusion_matrix(ground_truths, predictions, labels=VALID_LABELS)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=VALID_LABELS, yticklabels=VALID_LABELS)
    plt.title(f"Confusion Matrix — {model_name}", fontsize=13)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    safe = model_name.lower().replace(" ", "_").replace("/", "-")
    plt.savefig(f"{RESULTS_DIR}/confusion_{safe}.png", dpi=150)
    plt.close()
    print(f"  Confusion matrix saved.")


def plot_comparison(all_results: list):
    models  = [r["model"].replace(" (ours)", "★").replace("Qwen3.0-8B + QLoRA", "QLoRA★") for r in all_results]
    metrics = {
        "Exact Match Acc.":    [r["accuracy"]      for r in all_results],
        "Semantic Score":      [r["semantic_score"] for r in all_results],
        "Sentiment Acc.":      [r["sentiment_acc"]  for r in all_results],
        "F1 Macro":            [r["f1_macro"]       for r in all_results],
    }
    colors = ["#ff6a9b", "#ffaa6a", "#6affda", "#7c6aff"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Model Comparison — Four Evaluation Dimensions", fontsize=14, y=1.02)

    for ax, (metric_name, values), color in zip(axes, metrics.items(), colors):
        bars = ax.bar(range(len(models)), values, color=color, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax.set_title(metric_name, fontsize=11, pad=8)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/four_dimension_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Four-dimension comparison chart saved.")


def plot_semantic_distribution(predictions: list, ground_truths: list, model_name: str):
    scores = [emoji_semantic_similarity(p, g) for p, g in zip(predictions, ground_truths)]
    bins   = [0, 0.01, 0.5, 0.7, 0.85, 1.01]
    labels = ["Wrong\n(0)", "Unrelated\n(0~0.5)",
              "Somewhat\nSimilar\n(0.5~0.7)",
              "Very\nSimilar\n(0.7~0.85)", "Exact\nMatch\n(1.0)"]
    counts = [sum(bins[i] <= s < bins[i+1] for s in scores) for i in range(len(bins)-1)]
    pcts   = [c / len(scores) * 100 for c in counts]
    colors = ["#ff6a9b", "#ffaa6a", "#ffe066", "#6affda", "#7c6aff"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, pcts, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Semantic Similarity Distribution — {model_name}")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    safe = model_name.lower().replace(" ", "_")
    plt.savefig(f"{RESULTS_DIR}/semantic_dist_{safe}.png", dpi=150)
    plt.close()


def run_full_comparison(test_text_path, test_label_path, lora_path,
                        contradiction_json=None):
    test_df = load_semeval(test_text_path, test_label_path)
    texts   = test_df["text"].tolist()
    gt      = test_df["emoji"].tolist()

    all_results = []

    print("\n[1/4] TF-IDF baseline...")
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    train_df = load_semeval(
        test_text_path.replace("trial", "train"),
        test_label_path.replace("trial", "train"),
    )
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1,2))),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)),
    ])
    clf.fit(train_df["text"], train_df["label"])
    tfidf_preds = [EMOJI_MAP[p] for p in clf.predict(texts)]
    all_results.append(evaluate_model_full(tfidf_preds, gt, "TF-IDF + Logistic Regression"))

    print("\n[2/4] Zero-shot Qwen3-8B...")
    base_model, base_tok = load_base_model()
    zero_results = [predict_emoji(t, base_model, base_tok) for t in tqdm(texts, desc="Zero-shot")]
    zero_preds = [r["primary"] for r in zero_results]
    all_results.append(evaluate_model_full(zero_preds, gt, "Qwen3-8B (zero-shot)"))
    del base_model

    print("\n[3/4] Fine-tuned model (QLoRA)...")
    ft_model, ft_tok = load_finetuned_model(lora_path=lora_path)
    ft_results = predict_batch(texts, ft_model, ft_tok)
    ft_preds = [r["primary"] for r in ft_results]

    contra_df = None
    if contradiction_json and os.path.exists(contradiction_json):
        import json as _json
        with open(contradiction_json, encoding="utf-8") as f:
            contra_df = pd.DataFrame(_json.load(f))

    gen_quality = evaluate_generative_quality(ft_results, gt, contra_df)
    main_metrics = evaluate_model_full(ft_preds, gt, "Qwen3-8B + QLoRA (ours)")
    main_metrics.update(gen_quality)
    all_results.append(main_metrics)

    print(f"\n  [Generation Quality]")
    print(f"    JSON解析成功率      : {gen_quality['json_parse_success_rate']:.2%}")
    print(f"    备选emoji语义相似度  : {gen_quality['alternative_semantic_score']:.4f}")
    if "irony_detection_accuracy" in gen_quality:
        print(f"    反讽检测准确率       : {gen_quality['irony_detection_accuracy']:.2%}")

    plot_semantic_distribution(ft_preds, gt, "Qwen3-8B + QLoRA")
    plot_comparison(all_results)

    summary = pd.DataFrame(all_results)
    print(f"\n{'='*65}")
    print("  FINAL COMPARISON — FOUR DIMENSIONS")
    print(f"{'='*65}")
    cols = ["model", "accuracy", "semantic_score", "sentiment_acc", "f1_macro"]
    print(summary[cols].to_string(index=False))

    summary.to_csv(f"{RESULTS_DIR}/comparison_table.csv", index=False)
    print(f"\n✓ All results saved to {RESULTS_DIR}/")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_text",    default="data/raw/semeval2018/us_trial.text")
    parser.add_argument("--test_label",   default="data/raw/semeval2018/us_trial.labels")
    parser.add_argument("--lora",         default="outputs/lora_weights/expA-main/final")
    parser.add_argument("--contradiction",default="data/contradiction/contradiction_en.json")
    args = parser.parse_args()

    run_full_comparison(args.test_text, args.test_label,
                        args.lora, args.contradiction)
