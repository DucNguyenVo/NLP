"""Đọc CoNLL/IOB2 và ánh xạ nhãn về 6 lớp thống nhất (theo đặc tả dự án)."""
from __future__ import annotations


def parse_conll_sentences(file_path: str, skip_docstart: bool = True):
    """
    Trả về list câu: mỗi phần tử là dict {"tokens": [...], "labels": [...]}.
    """
    sentences = []
    current_tokens: list[str] = []
    current_labels: list[str] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if current_tokens:
                    sentences.append(
                        {"tokens": current_tokens, "labels": current_labels}
                    )
                    current_tokens, current_labels = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            token, label = parts[0], parts[-1]
            if skip_docstart and token == "-DOCSTART-" and label == "O":
                continue
            current_tokens.append(token)
            current_labels.append(label)

    if current_tokens:
        sentences.append({"tokens": current_tokens, "labels": current_labels})
    return sentences


def bio_label_to_unified_class(raw_label: str, source: str) -> str | None:
    """Ánh xạ nhãn IOB gốc (hậu tố) về một trong 6 lớp hoặc None nếu không dùng."""
    raw_label = raw_label.strip()
    if raw_label == "O" or "-" not in raw_label:
        return None
    _, suff = raw_label.split("-", 1)
    suff_l = suff.lower()
    src = source.replace(" ", "_")

    if src == "BC5CDR":
        if "chemical" in suff_l:
            return "CHEMICAL"
        if "disease" in suff_l:
            return "DISEASE"
    elif src == "NCBI_Disease":
        if "disease" in suff_l:
            return "DISEASE"
    elif src == "JNLPBA":
        if suff_l in ("dna", "rna"):
            return "DNA_RNA"
        if suff_l == "protein":
            return "PROTEIN"
        if suff_l == "cell_type":
            return "CELL_TYPE"
        if suff_l == "cell_line":
            return "CELL_TYPE"
    return None


def extract_entity_spans(tokens: list[str], labels: list[str], source: str):
    """
    Trích các span thực thể (chỉ số token bao hàm) và lớp thống nhất (IOB2).
    """
    spans: list[tuple[int, int, str]] = []
    cur_start = cur_end = cur_cls = None

    for i, lab in enumerate(labels):
        if lab == "O" or "-" not in lab:
            if cur_start is not None:
                spans.append((cur_start, cur_end, cur_cls))
                cur_start = cur_end = cur_cls = None
            continue

        pref = lab.split("-", 1)[0].upper()
        u = bio_label_to_unified_class(lab, source)
        if u is None:
            if cur_start is not None:
                spans.append((cur_start, cur_end, cur_cls))
                cur_start = cur_end = cur_cls = None
            continue

        if pref == "B":
            if cur_start is not None:
                spans.append((cur_start, cur_end, cur_cls))
            cur_start, cur_end, cur_cls = i, i, u
        elif pref == "I":
            if cur_cls is not None and u == cur_cls:
                cur_end = i
            else:
                if cur_start is not None:
                    spans.append((cur_start, cur_end, cur_cls))
                cur_start, cur_end, cur_cls = i, i, u

    if cur_start is not None:
        spans.append((cur_start, cur_end, cur_cls))
    return spans


def tags_for_sentence(num_tokens: int, spans: list[tuple[int, int, str]]):
    """Sinh chuỗi nhãn IOB2 (token-level) từ danh sách span đã chuẩn hóa."""
    tags = ["O"] * num_tokens
    for start, end, cls in spans:
        tags[start] = f"B-{cls}"
        for k in range(start + 1, end + 1):
            tags[k] = f"I-{cls}"
    return tags
