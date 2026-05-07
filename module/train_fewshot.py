import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate

# 1. Cấu hình các tham số
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
DATASET_PATH = "data/processed"
OUTPUT_DIR = "./results"

# Danh sách nhãn chuẩn (Phải khớp hoàn toàn với module/prepare_dataset.py)
label_list = [
    "O",
    "B-CHEMICAL", "I-CHEMICAL",
    "B-DISEASE", "I-DISEASE",
    "B-PROTEIN", "I-PROTEIN",
    "B-DNA_RNA", "I-DNA_RNA",
    "B-CELL_TYPE", "I-CELL_TYPE"
]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for i, label in enumerate(label_list)}

# 2. Load Dataset và Tokenizer
dataset = load_from_disk(DATASET_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Chia tập train/val (Vì chỉ có 300 câu, ta lấy 250 train, 50 val để theo dõi)
dataset = dataset.train_test_split(test_size=0.15, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# 3. Khởi tạo Mô hình
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id_to_label,
    label2id=label_to_id
)

# 4. Metric đánh giá (Dùng seqeval cho NER)
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 5. Cấu hình huấn luyện (Cập nhật cho Transformers v5)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",        # Sửa lỗi evaluation_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
    push_to_hub=False,
    report_to="none"
)

# 6. Khởi tạo Trainer với Tokenizer đúng chuẩn
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 7. Bắt đầu huấn luyện
print("[*] Bắt đầu huấn luyện PubMedBERT Few-shot...")
trainer.train()

# 8. Lưu mô hình cuối cùng
os.makedirs("models/pubmedbert-bioner-fewshot", exist_ok=True)
trainer.save_model("models/pubmedbert-bioner-fewshot")
tokenizer.save_pretrained("models/pubmedbert-bioner-fewshot")
print("[Done] Mô hình và Tokenizer đã được lưu tại models/pubmedbert-bioner-fewshot")
