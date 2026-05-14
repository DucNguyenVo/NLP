"""
Giai đoạn 2: Prompt-Tuning PubMedBERT (OpenPrompt) trên silver labels từ Gemini.

Nạp trọng số BERT + đầu MLM (tương đương AutoModelForMaskedLM) qua OpenPrompt;
huấn luyện few-shot với batch nhỏ, LR thấp, đóng băng tầng dưới.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.model_training import openprompt_compat  # noqa: F401

import torch
from torch.optim import AdamW
from openprompt.data_utils import InputExample

from src.model_training.prompt_config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EPOCHS,
    LEARNING_RATE,
    NER_CLASSES,
    normalize_silver_entity_label,
)
from src.model_training.prompt_ner_model import (
    build_prompt_model,
    make_eval_dataloader,
    save_checkpoint,
)


def load_silver_training_data(file_path: Path):
    """Mỗi thực thể trong JSON silver → một InputExample (phân loại span)."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    classes = list(NER_CLASSES)
    labels_map = {label: i for i, label in enumerate(classes)}

    for item in data:
        guid = item["id"]
        text = item["text"]
        if not item.get("entities"):
            continue
        for idx, ent in enumerate(item["entities"]):
            word = ent["word"]
            canon = normalize_silver_entity_label(str(ent.get("label", "")))
            if canon is None:
                continue
            label_id = labels_map[canon]
            examples.append(
                InputExample(
                    guid=f"{guid}_{idx}_{hash(word) & 0xFFFF}",
                    text_a=text,
                    meta={"entity": word},
                    label=label_id,
                )
            )
    return examples, classes


def train_prompt_model(
    silver_json: Path | None = None,
    out_dir: Path | None = None,
):
    silver_json = silver_json or (_ROOT / "data/silver_labels/few_shot_train_multilabel.json")
    dataset, classes = load_silver_training_data(silver_json)
    print(f"[*] Đã tải {len(dataset)} mẫu thực thể (silver) để huấn luyện.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_model, template, verbalizer, tokenizer, WrapperClass, classes = (
        build_prompt_model(classes=classes, device=device)
    )
    print(f"[*] Thiết bị: {device}")

    train_dataloader = make_eval_dataloader(
        dataset, template, tokenizer, WrapperClass, shuffle=True
    )

    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in prompt_model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in prompt_model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    print(
        f"\n[*] BẮT ĐẦU PROMPT-TUNING | epochs={EPOCHS} | batch_size={BATCH_SIZE} | lr={LEARNING_RATE}"
    )
    prompt_model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step, inputs in enumerate(train_dataloader):
            inputs = inputs.to(device)
            logits = prompt_model(inputs)
            labels = inputs["label"]
            loss = loss_func(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | Step {step} | Loss: {loss.item():.4f}"
                )
        avg = total_loss / max(len(train_dataloader), 1)
        print(f"--> Epoch {epoch + 1} kết thúc | avg loss: {avg:.4f}\n")

    save_checkpoint(
        prompt_model,
        classes,
        out_dir=out_dir or CHECKPOINT_DIR,
        extra_config={"silver_data": str(silver_json)},
    )
    print(f"[Done] Đã lưu checkpoint tại: {out_dir or CHECKPOINT_DIR}")


if __name__ == "__main__":
    train_prompt_model()
