# NLP - Few-shot Biomedical NER Pipeline

Du an nay xay dung mot quy trinh tao du lieu few-shot va danh gia chat luong gan nhan thuc the y sinh hoc (BioNER) voi LLM.

## Muc tieu

- Lay mau du lieu tu cac tap train CoNLL (BC5CDR, NCBI_Disease, JNLPBA).
- Tao:
  - tap du lieu tho khong nhan cho LLM (`raw_seed_data.json`),
  - tap ground truth de doi soat (`ground_truth_300.json`).
- Goi Gemini de gan nhan da lop va luu ket qua (`few_shot_train_multilabel.json`).
- Danh gia ket qua bang cac chi so Precision / Recall / F1.

## Cau truc thu muc

```text
NLP/
├── data/
│   ├── raw/                 # Dữ liệu gốc (BC5CDR, NCBI, JNLPBA .tsv)
│   ├── processed/           # Dữ liệu đã xử lý (raw_seed_data.json, ground_truth_300.json)
│   └── silver_labels/       # Dữ liệu do LLM gán nhãn (few_shot_train_multilabel.json)
├── src/                     
│   ├── data_prep/           # Xử lý dữ liệu ban đầu
│   │   └── preprocess.py
│   ├── llm_annotate/        # Gọi API Gemini
│   │   └── annotate.py
│   └── evaluation/          # File đánh giá
│       ├── validate.py
│       └── debug_labels.py
├── .env
└── README.md
```

## Cac thanh phan chinh

### 1) `src/data_prep/preprocess.py`

- Doc file CoNLL (`train.tsv`) va tach token/label.
- Chuan hoa label `DISEASE` ve dang BIO thong nhat.
- Trich xuat entities tu nhan BIO.
- Lay mau `k_shot` cho moi dataset (mac dinh `k_shot=100`).
- Xuat:
  - `data/processed/raw_seed_data.json`: dung de gui LLM gan nhan,
  - `data/processed/ground_truth_300.json`: dung de danh gia doi soat.

> Mac dinh 3 nguon x 100 mau = 300 cau.

### 2) `src/llm_annotate/annotate.py`

- Doc `data/processed/raw_seed_data.json`.
- Goi Gemini theo batch (`BATCH_SIZE=10`) de trich xuat entity cac nhan:
  - `GENE`, `PROTEIN`, `DNA_RNA`, `CELL_TYPE`, `CHEMICAL`, `DISEASE`.
- Co retry 3 lan neu loi, va luu checkpoint sau moi batch.
- Xuat ket qua ra `data/silver_labels/few_shot_train_multilabel.json`.

### 3) `src/evaluation/validate.py`

- Doi soat ket qua Gemini voi ground truth.
- Co mapping chuan hoa nhan de giam sai lech giua cac bo du lieu:
  - vi du `GENE -> PROTEIN`, `DNA/RNA -> DNA_RNA`, `CELL_LINE -> CELL_TYPE`.
- So khop fuzzy theo cap `(word, label)` de xu ly truong hop span dai/ngan khac nhau.
- In ra `TP`, `FP`, `FN` va `Precision`, `Recall`, `F1`.

### 4) `src/evaluation/debug_labels.py`

- Ban danh gia bo sung voi chuan hoa text (bo dau cau, viet thuong).
- So sanh hai chieu (Gemini -> GT va GT -> Gemini) de phan tich thua/sot.

## Yeu cau moi truong

- Python 3.9+ (khuyen nghi).
- Thu vien:

```bash
pip install google-genai python-dotenv
```

## Cach chay

Chay cac lenh tai thu muc goc `NLP`.

### Buoc 1: Tao seed data va ground truth

```bash
python src/data_prep/preprocess.py
```

Ket qua:

- `data/processed/raw_seed_data.json`
- `data/processed/ground_truth_300.json`

### Buoc 2: Gan nhan bang Gemini

```bash
python src/llm_annotate/annotate.py
```

Ket qua:

- `data/silver_labels/few_shot_train_multilabel.json`

### Buoc 3: Danh gia

```bash
python src/evaluation/validate.py
```

Hoac dung script debug:

```bash
python module/debug_labels.py
```

## Dinh dang du lieu

Moi mau du lieu co dang:

```json
{
  "id": 260,
  "source": "JNLPBA",
  "text": "This work aims at identifying ...",
  "entities": [
    {
      "word": "human immunodeficiency virus",
      "label": "DISEASE",
      "start": 62,
      "end": 90
    }
  ]
}
```

## Luu y quan trong ve API key

Hien tai `module/annotate.py` dang de API key hard-code trong ma nguon. Nen doi sang bien moi truong de an toan.

Goi y:

1. Dat bien moi truong `GOOGLE_API_KEY`.
2. Doc key trong code bang `os.getenv("GOOGLE_API_KEY")`.
3. Khong commit API key that vao git.

## Mo rong de xuat

- Them `requirements.txt` de khoa version thu vien.
- Them script benchmark theo tung source (BC5CDR / NCBI Disease / JNLPBA).
- Them strict evaluation theo span exact-match ben canh fuzzy-match.
- Them log chi tiet cac truong hop FP/FN de phan tich loi nhan.
