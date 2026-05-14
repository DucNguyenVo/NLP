"""
Giai đoạn 3: vẽ biểu đồ cột so sánh F1 (LLM zero-shot / fuzzy vs PubMedBERT prompt-tuned).

Đọc hai file JSON do validate.py và evaluate_prompt_seqeval.py xuất (trường "f1").
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_f1(path: Path) -> float:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["f1"])


def main():
    ap = argparse.ArgumentParser(description="So sánh F1 LLM vs PubMedBERT (biểu đồ cột).")
    ap.add_argument(
        "--llm-json",
        type=Path,
        default=Path("metrics/llm_vs_gt.json"),
        help="JSON từ validate.py --json-out",
    )
    ap.add_argument(
        "--bert-json",
        type=Path,
        default=Path("metrics/prompt_tuned_seqeval.json"),
        help="JSON từ evaluate_prompt_seqeval.py --json-out",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("figures/f1_comparison.png"),
        help="Đường dẫn lưu hình PNG",
    )
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("Cần cài matplotlib: pip install matplotlib")
        raise e

    if not args.llm_json.is_file():
        print(f"[Lỗi] Thiếu file LLM metrics: {args.llm_json}")
        print("Gợi ý: python src/evaluation/validate.py --json-out metrics/llm_vs_gt.json")
        sys.exit(1)
    if not args.bert_json.is_file():
        print(f"[Lỗi] Thiếu file BERT metrics: {args.bert_json}")
        print(
            "Gợi ý: python src/evaluation/evaluate_prompt_seqeval.py --json-out metrics/prompt_tuned_seqeval.json"
        )
        sys.exit(1)

    f1_llm = load_f1(args.llm_json)
    f1_bert = load_f1(args.bert_json)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    labels = ["LLM (Gemini)\nvs GT (fuzzy entity)", "PubMedBERT\n(prompt-tune + seqeval)"]
    values = [f1_llm, f1_bert]
    colors = ["#4C72B0", "#55A868"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0, max(1.0, max(values) * 1.15))
    ax.set_title("BioNER few-shot: so sánh F1")
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"[OK] Đã lưu biểu đồ: {args.output}")


if __name__ == "__main__":
    main()
