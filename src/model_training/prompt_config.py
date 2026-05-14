"""Hằng số và cấu hình chung cho Prompt-Tuning PubMedBERT (BioNER few-shot)."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

NER_CLASSES = ["GENE", "PROTEIN", "DNA_RNA", "CELL_TYPE", "CHEMICAL", "DISEASE"]

PUBMEDBERT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# Giai đoạn 2: few-shot, chống overfitting (theo kế hoạch)
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
EPOCHS = 4
FREEZE_EMBEDDING_AND_FIRST_LAYERS = 6
MAX_SEQ_LENGTH = 256
DECODER_MAX_LENGTH = 3


def prompt_template_text():
    """
    Khuôn mẫu theo đặc tả: văn bản gốc + cụm thực thể + [MASK].
    OpenPrompt: text_a = câu gốc, meta.entity = span cần phân loại.
    """
    return (
        '{"placeholder":"text_a"} '
        'Trong câu này, cụm từ \'{"meta":"entity"}\' là một {"mask"}.'
    )


def label_words_dict():
    """Verbalizer: ánh xạ [MASK] về 6 nhãn chuẩn qua từ neo."""
    return {
        "GENE": ["gene", "genes", "genetics"],
        "PROTEIN": ["protein", "proteins", "enzyme", "peptide", "antigen"],
        "DNA_RNA": ["dna", "rna", "nucleotide", "transcript"],
        "CELL_TYPE": ["cell", "cells", "tissue"],
        "CHEMICAL": ["chemical", "drug", "substance", "compound", "acid"],
        "DISEASE": ["disease", "illness", "syndrome", "disorder", "fever", "tumor"],
    }


def normalize_silver_entity_label(label_str: str):
    """Chuẩn hóa nhãn từ JSON silver / LLM về một trong NER_CLASSES hoặc None."""
    if not label_str:
        return None
    x = label_str.strip().upper().replace(" ", "_")
    aliases = {
        "DNA": "DNA_RNA",
        "RNA": "DNA_RNA",
        "CELL_LINE": "CELL_TYPE",
        "GENE": "GENE",
    }
    x = aliases.get(x, x)
    if x in NER_CLASSES:
        return x
    return None


CHECKPOINT_DIR = REPO_ROOT / "models" / "prompt_ner"
TRAIN_CONFIG_NAME = "train_config.json"
CLASSES_NAME = "classes.json"
WEIGHTS_NAME = "pytorch_model.bin"
