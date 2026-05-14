"""
Giai đoạn 3: đánh giá mô hình Prompt-Tuned (PubMedBERT) bằng seqeval.

Chiến lược **oracle span**: biên span lấy từ nhãn vàng CoNLL; mô hình chỉ
phân loại loại thực thể cho từng span. Chuỗi nhãn token được dựng lại từ các
span để dùng `seqeval` (đo chunk BIO giống NER chuẩn trên biên đúng).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.model_training import openprompt_compat  # noqa: F401 — patch transformers trước OpenPrompt

import torch
from openprompt.data_utils import InputExample
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from src.evaluation.conll_utils import (
    extract_entity_spans,
    parse_conll_sentences,
    tags_for_sentence,
)
from src.model_training.prompt_ner_model import (
    load_prompt_model_for_eval,
    make_eval_dataloader,
    predict_labels_batch,
)


def run_eval(
    conll_path: Path,
    source: str,
    checkpoint_dir: Path,
    max_sentences: int | None = None,
):
    source_key = source.strip().replace(" ", "_")
    sentences = parse_conll_sentences(str(conll_path))
    if max_sentences is not None:
        sentences = sentences[: max_sentences]

    examples: list[InputExample] = []
    meta: list[tuple[int, int, int, str]] = []

    for si, sent in enumerate(sentences):
        tokens = sent["tokens"]
        labels = sent["labels"]
        text = " ".join(tokens)
        spans = extract_entity_spans(tokens, labels, source_key)
        for start, end, gold_cls in spans:
            phrase = " ".join(tokens[start : end + 1])
            examples.append(
                InputExample(
                    guid=f"{si}_{start}_{end}",
                    text_a=text,
                    meta={"entity": phrase},
                    label=0,
                )
            )
            meta.append((si, start, end, gold_cls))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_model, template, verbalizer, tokenizer, WrapperClass, classes = (
        load_prompt_model_for_eval(checkpoint_dir, device=device)
    )

    if not examples:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "note": "Không có span thực thể trong tập đã chọn.",
            "num_sentences": len(sentences),
            "num_spans": 0,
        }

    loader = make_eval_dataloader(
        examples, template, tokenizer, WrapperClass, shuffle=False
    )
    pred_indices = predict_labels_batch(prompt_model, loader, device)

    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []

    for sent in sentences:
        tokens = sent["tokens"]
        labels = sent["labels"]
        spans = extract_entity_spans(tokens, labels, source_key)
        y_true.append(tags_for_sentence(len(tokens), spans))

    pred_tags_per_sent: list[list[str]] = [
        ["O"] * len(s["tokens"]) for s in sentences
    ]
    for pred_i, (si, st, en, _gold_cls) in zip(pred_indices, meta):
        pcls = classes[pred_i]
        pred_tags_per_sent[si][st] = f"B-{pcls}"
        for k in range(st + 1, en + 1):
            pred_tags_per_sent[si][k] = f"I-{pcls}"

    for si, sent in enumerate(sentences):
        y_pred.append(pred_tags_per_sent[si])

    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "num_sentences": len(sentences),
        "num_spans": len(examples),
        "checkpoint": str(checkpoint_dir),
        "conll": str(conll_path),
        "source": source_key,
        "eval_mode": "oracle_span_seqeval",
        "classification_report": report,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Đánh giá PubMedBERT prompt-tuned với seqeval (oracle span)."
    )
    ap.add_argument(
        "--conll",
        type=Path,
        default=Path("data/raw/BC5CDR/train.tsv"),
        help="Đường dẫn file CoNLL .tsv (train hoặc test).",
    )
    ap.add_argument(
        "--source",
        type=str,
        default="BC5CDR",
        help="Tên nguồn: BC5CDR | JNLPBA | NCBI_Disease (dùng cho ánh xạ nhãn).",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/prompt_ner"),
        help="Thư mục chứa pytorch_model.bin và classes.json.",
    )
    ap.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Giới hạn số câu (tùy chọn, để chạy nhanh khi thử).",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Ghi metrics ra JSON (ví dụ metrics/prompt_tuned_seqeval.json).",
    )
    args = ap.parse_args()

    if not args.conll.is_file():
        print(f"[Lỗi] Không tìm thấy file CoNLL: {args.conll}")
        sys.exit(1)
    if not (args.checkpoint / "pytorch_model.bin").is_file():
        print(
            f"[Lỗi] Chưa có checkpoint. Hãy chạy trước: python src/model_training/prompt_tuning.py\n"
            f"Thiếu: {args.checkpoint / 'pytorch_model.bin'}"
        )
        sys.exit(1)

    metrics = run_eval(
        args.conll,
        args.source,
        args.checkpoint,
        max_sentences=args.max_sentences,
    )

    print("\n" + "=" * 50)
    print("seqeval (oracle span — biên vàng CoNLL)")
    print("=" * 50)
    print(f"Số câu: {metrics['num_sentences']} | Số span: {metrics['num_spans']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    if metrics.get("classification_report"):
        print("\n" + metrics["classification_report"])

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        out = {k: v for k, v in metrics.items() if k != "classification_report"}
        out["classification_report"] = metrics.get("classification_report", "")
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Đã ghi JSON: {args.json_out}")


if __name__ == "__main__":
    main()
