import os
import random
import json

class BioNERPreprocessor:
    def __init__(self, k_shot=100):
        self.k_shot = k_shot
        self.raw_data = []         # Dữ liệu thô cho LLM
        self.ground_truth_data = [] # Dữ liệu nhãn gốc để đối soát

    def parse_conll(self, file_path, source_name):
        """
        Đọc file CoNLL và bóc tách đồng thời token và entities.
        """
        sentences = []
        print(f"[*] Đang xử lý file: {file_path} (Source: {source_name})")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                current_tokens = []
                current_labels = []
                
                for line in f:
                    line = line.strip()
                    if not line:
                        if current_tokens:
                            sentences.append({
                                "tokens": current_tokens,
                                "labels": current_labels
                            })
                            current_tokens, current_labels = [], []
                        continue
                    
                    parts = line.split()
                    token = parts[0]
                    label = parts[-1] # Luôn lấy cột cuối cùng làm nhãn
                    if len(parts) >= 2:
                        current_tokens.append(parts[0])
                        # Chuẩn hóa nhãn DISEASE
                        label = parts[-1]
                        if "DISEASE" in label.upper():
                            label = label.split("-")[0] + "-DISEASE" if "-" in label else "B-DISEASE"
                        current_labels.append(label)
                
                if current_tokens:
                    sentences.append({"tokens": current_tokens, "labels": current_labels})
            
            print(f"[OK] Đã đọc {len(sentences)} câu từ {source_name}.")
            return sentences
        except Exception as e:
            print(f"[Error] Lỗi: {str(e)}")
            return []

    def extract_entities_from_bio(self, tokens, labels):
        """
        Hàm chuyển đổi chuỗi nhãn BIO thành danh sách thực thể JSON.
        """
        entities = []
        current_ent = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith("B-"):
                if current_ent: entities.append(current_ent)
                current_ent = {"word": token, "label": label[2:], "start_idx": i}
            elif label.startswith("I-") and current_ent and label[2:] == current_ent["label"]:
                current_ent["word"] += " " + token
            else:
                if current_ent:
                    entities.append(current_ent)
                    current_ent = None
        if current_ent: entities.append(current_ent)
        return entities

    def process(self, dataset_configs):
        global_id = 1
        for config in dataset_configs:
            all_data = self.parse_conll(config['path'], config['name'])
            
            if len(all_data) < self.k_shot:
                sampled = all_data
            else:
                sampled = random.sample(all_data, self.k_shot)
            
            for item in sampled:
                text = " ".join(item['tokens'])
                # 1. Lưu vào tập Raw (Không nhãn)
                self.raw_data.append({
                    "id": global_id,
                    "source": config['name'],
                    "text": text
                })
                # 2. Lưu vào tập Ground Truth (Có nhãn thực thể)
                entities = self.extract_entities_from_bio(item['tokens'], item['labels'])
                self.ground_truth_data.append({
                    "id": global_id,
                    "source": config['name'],
                    "text": text,
                    "entities": entities
                })
                global_id += 1

    def save_results(self):
        # Lưu file cho LLM
        with open("data/processed/raw_seed_data.json", 'w', encoding='utf-8') as f:
            json.dump(self.raw_data, f, ensure_ascii=False, indent=4)
        
        # Lưu file đối soát
        with open("data/processed/ground_truth_300.json", 'w', encoding='utf-8') as f:
            json.dump(self.ground_truth_data, f, ensure_ascii=False, indent=4)
            
        print(f"\n[Done] Đã xuất {len(self.raw_data)} câu.")
        print("- File cho LLM: data/processed/raw_seed_data.json")
        print("- File đối soát: data/processed/ground_truth_300.json")

if __name__ == "__main__":
    random.seed(42)
    datasets = [
        {"path": "data/raw/BC5CDR/train.tsv", "name": "BC5CDR"},
        {"path": "data/raw/NCBI_Disease/train.tsv", "name": "NCBI Disease"},
        {"path": "data/raw/JNLPBA/train.tsv", "name": "JNLPBA"}
    ]
    preprocessor = BioNERPreprocessor(k_shot=100)
    preprocessor.process(datasets)
    preprocessor.save_results()