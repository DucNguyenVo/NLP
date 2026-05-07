import json

def normalize_label(label):
    """Chuẩn hóa nhãn về một danh mục chung nhất."""
    label = label.upper().strip()
    mapping = {
        'CHEMICAL': 'CHEMICAL',
        'DISEASE': 'DISEASE',
        'PROTEIN': 'PROTEIN',
        'GENE': 'PROTEIN',  # Thường được gộp trong nhiều bộ dữ liệu
        'DNA': 'DNA_RNA',
        'RNA': 'DNA_RNA',
        'DNA_RNA': 'DNA_RNA',
        'CELL_LINE': 'CELL_TYPE',
        'CELL_TYPE': 'CELL_TYPE'
    }
    return mapping.get(label, label)

def normalize_word(word):
    """Xóa dấu cách thừa và đưa về viết thường."""
    return " ".join(word.lower().split()).strip(",. ")

def evaluate_professional(gt_json, gm_json):
    with open(gt_json, 'r', encoding='utf-8') as f:
        gt_data = {item['id']: item for item in json.load(f)}
    with open(gm_json, 'r', encoding='utf-8') as f:
        gm_data = json.load(f)

    tp, fp, fn = 0, 0, 0
    
    print(f"[*] Đang đối soát {len(gm_data)} câu với logic Mapping chuyên sâu...")

    for gm_item in gm_data:
        item_id = gm_item['id']
        if item_id not in gt_data: continue
        
        # 1. Chuẩn hóa tập thực thể từ Ground Truth
        gt_ents = set([
            (normalize_word(e['word']), normalize_label(e['label'])) 
            for e in gt_data[item_id]['entities']
        ])
        
        # 2. Chuẩn hóa tập thực thể từ Gemini
        gm_ents = set([
            (normalize_word(e['word']), normalize_label(e['label'])) 
            for e in gm_item['entities']
        ])

        # 3. Tính toán Precision/Recall
        for gm_e in gm_ents:
            # So sánh fuzzy: nếu từ của Gemini nằm trong từ của GT hoặc ngược lại
            # Giúp xử lý lỗi "headache" vs "relief of headache"
            is_match = False
            gt_set_list = list(gt_ents)
            for gt_e in gt_set_list:
                if (gm_e[0] in gt_e[0] or gt_e[0] in gm_e[0]) and gm_e[1] == gt_e[1]:
                    is_match = True
                    break
            
            if is_match:
                tp += 1
            else:
                fp += 1
        
        for gt_e in gt_ents:
            is_found = False
            for gm_e in list(gm_ents):
                if (gm_e[0] in gt_e[0] or gt_e[0] in gm_e[0]) and gm_e[1] == gt_e[1]:
                    is_found = True
                    break
            if not is_found:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n" + "="*40)
    print(f"{'METRIC':<15} | {'VALUE':<10}")
    print("-" * 40)
    print(f"{'Precision':<15} | {precision:.2%}")
    print(f"{'Recall':<15} | {recall:.2%}")
    print(f"{'F1-Score':<15} | {f1:.2%}")
    print("-" * 40)
    print(f"TP: {tp} | FP: {fp} | FN: {fn}")
    print("="*40)

if __name__ == "__main__":
    evaluate_professional("ground_truth_300.json", "few_shot_train_multilabel.json")