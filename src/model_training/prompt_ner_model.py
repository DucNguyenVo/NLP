"""
Xây dựng / lưu / tải mô hình PromptForClassification trên PubMedBERT.

OpenPrompt tải BERT (Masked LM) và dùng đầu MLM cho [MASK] — tương đương
dùng năng lực sinh từ AutoModelForMaskedLM trên kiến trúc BERT.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os

import torch

from src.model_training import openprompt_compat  # noqa: F401 — patch transformers trước OpenPrompt

from openprompt import PromptForClassification, PromptDataLoader
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer

from src.model_training.prompt_config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CLASSES_NAME,
    DECODER_MAX_LENGTH,
    EPOCHS,
    FREEZE_EMBEDDING_AND_FIRST_LAYERS,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    NER_CLASSES,
    PUBMEDBERT_MODEL,
    TRAIN_CONFIG_NAME,
    WEIGHTS_NAME,
    label_words_dict,
    prompt_template_text,
)


def freeze_lower_bert_layers(plm, num_frozen_layers: int):
    """Đóng băng embedding và num_frozen_layers tầng encoder đầu (0 .. n-1)."""
    for name, param in plm.named_parameters():
        if "embeddings" in name:
            param.requires_grad = False
            continue
        for i in range(num_frozen_layers):
            if f"encoder.layer.{i}." in name:
                param.requires_grad = False
                break


def build_prompt_model(classes=None, device=None):
    if classes is None:
        classes = list(NER_CLASSES)
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", PUBMEDBERT_MODEL)
    freeze_lower_bert_layers(plm, FREEZE_EMBEDDING_AND_FIRST_LAYERS)

    lw = label_words_dict()
    label_words_list = [lw[c] for c in classes]
    template = ManualTemplate(
        tokenizer=tokenizer,
        text=prompt_template_text(),
    )
    verbalizer = ManualVerbalizer(
        classes=classes,
        label_words=label_words_list,
        tokenizer=tokenizer,
    )
    prompt_model = PromptForClassification(
        template=template,
        plm=plm,
        verbalizer=verbalizer,
    )
    if device is not None:
        prompt_model = prompt_model.to(device)
    return prompt_model, template, verbalizer, tokenizer, WrapperClass, classes


def save_checkpoint(
    prompt_model,
    classes,
    out_dir: Optional[Path] = None,
    extra_config: Optional[dict] = None,
):
    out_dir = Path(out_dir or CHECKPOINT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(prompt_model.state_dict(), out_dir / WEIGHTS_NAME)
    with open(out_dir / CLASSES_NAME, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    cfg = {
        "pubmedbert_model": PUBMEDBERT_MODEL,
        "template": prompt_template_text(),
        "label_words": label_words_dict(),
        "ner_classes": classes,
        "freeze_embedding_and_first_layers": FREEZE_EMBEDDING_AND_FIRST_LAYERS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "decoder_max_length": DECODER_MAX_LENGTH,
    }
    if extra_config:
        cfg.update(extra_config)
    with open(out_dir / TRAIN_CONFIG_NAME, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_prompt_model_for_eval(
    ckpt_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
):
    ckpt_dir = Path(ckpt_dir or CHECKPOINT_DIR)
    with open(ckpt_dir / CLASSES_NAME, "r", encoding="utf-8") as f:
        classes = json.load(f)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_model, template, verbalizer, tokenizer, WrapperClass, _ = build_prompt_model(
        classes=classes, device=device
    )
    state = torch.load(ckpt_dir / WEIGHTS_NAME, map_location=device)
    prompt_model.load_state_dict(state, strict=True)
    prompt_model.eval()
    return prompt_model, template, verbalizer, tokenizer, WrapperClass, classes


def make_eval_dataloader(examples, template, tokenizer, WrapperClass, shuffle=False):
    return PromptDataLoader(
        dataset=examples,
        template=template,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=MAX_SEQ_LENGTH,
        decoder_max_length=DECODER_MAX_LENGTH,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="head",
    )


def predict_labels_batch(prompt_model, dataloader, device):
    """Trả về list chỉ số lớp dự đoán theo thứ tự batch."""
    preds = []
    prompt_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            logits = prompt_model(batch)
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    return preds
