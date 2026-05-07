import json
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

def prepare_huggingface_dataset(input_json, model_checkpoint, output_dir="data/processed"):
    # 1. Load cleaned data
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. Initialize Tokenizer (PubMedBERT)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Define label map (BIO format is standard for NER)
    # 0: O, 1: B-CHEMICAL, 2: I-CHEMICAL, etc.
    label_list = [
        "O",
        "B-CHEMICAL", "I-CHEMICAL",
        "B-DISEASE", "I-DISEASE",
        "B-PROTEIN", "I-PROTEIN",
        "B-DNA_RNA", "I-DNA_RNA",
        "B-CELL_TYPE", "I-CELL_TYPE"
    ]
    label_to_id = {label: i for i, label in enumerate(label_list)}

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_offsets_mapping=True
        )
        
        labels = []
        for i, text in enumerate(examples["text"]):
            offsets = tokenized_inputs["offset_mapping"][i]
            entities = examples["entities"][i]
            
            # Initialize all labels as 'O' (0)
            token_labels = [0] * len(offsets)
            
            for ent in entities:
                start_char = ent["start"]
                end_char = ent["end"]
                label_name = ent["label"]
                
                # Assign B- and I- tags based on character offsets
                first_token = True
                for idx, (start, end) in enumerate(offsets):
                    # Skip special tokens ([CLS], [SEP], [PAD])
                    if start == end == 0:
                        token_labels[idx] = -100 # Ignore in loss
                        continue
                        
                    # Check if token is inside entity span
                    if start >= start_char and end <= end_char:
                        if first_token:
                            token_labels[idx] = label_to_id.get(f"B-{label_name}", 0)
                            first_token = False
                        else:
                            token_labels[idx] = label_to_id.get(f"I-{label_name}", 0)
                
            labels.append(token_labels)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Convert to HF Dataset object
    raw_dataset = Dataset.from_list(data)
    
    # Apply tokenization and alignment
    processed_dataset = raw_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_dataset.column_names
    )
    
    # Create directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed dataset
    processed_dataset.save_to_disk(output_dir)
    print(f"[Done] Dataset prepared and saved to {output_dir}")
    print(f"Label Mapping: {label_to_id}")

if __name__ == "__main__":
    import os
    MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    prepare_huggingface_dataset("final_train_data.json", MODEL_NAME)
