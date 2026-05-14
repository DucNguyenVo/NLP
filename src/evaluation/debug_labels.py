import json
import re

def normalize(text):
    """Chuẩn hóa văn bản: viết thường, bỏ dấu câu, xóa khoảng trắng thừa."""
    if not text: return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

def evaluate_comprehensive(gt_json, gm_json):
    with open(gt_json, 'r', encoding='utf-8') as f:
        gt_data = {item['id']: item for item in json.load(f)}
    with open(gm_json, 'r', encoding='utf-8') as f:
        gm_data = json.load(f)

    tp = 0  # True Positives
    fp = 0  # False Positives (Gemini thừa/sai)
    fn = 0  # False Negatives (Gemini bỏ sót)

    print(f"[*] Đang thực hiện so sánh đối chiếu hai chiều...")

    for gm_item in gm_data:
        item_id = gm_item['id']
        if item_id not in gt_data: continue
        
        # Chuẩn hóa nhãn và từ của cả 2 bên về viết thường
        gt_ents = [(normalize(e['word']), e['label'].lower()) for e in gt_data[item_id]['entities']]
        gm_ents = [(normalize(e['word']), e['label'].lower()) for e in gm_item['entities']]

        # --- Kiểm tra chiều XUÔI (Gemini vs GT) ---
        matched_gm_indices = set()
        matched_gt_indices = set()

        for i, (gm_w, gm_l) in enumerate(gm_ents):
            found = False
            for j, (gt_w, gt_l) in enumerate(gt_ents):
                # Khớp nếu: Cùng nhãn VÀ (Từ này nằm trong từ kia)
                if gm_l == gt_l and (gm_w in gt_w or gt_w in gm_w):
                    found = True
                    matched_gt_indices.add(j)
                    break
            if found:
                tp += 1
                matched_gm_indices.add(i)
            else:
                fp += 1

        # --- Kiểm tra chiều NGƯỢC (Những gì GT có mà Gemini không thấy) ---
        for j in range(len(gt_ents)):
            if j not in matched_gt_indices:
                fn += 1

    # Tính toán
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*40)
    print(f"{'CHỈ SỐ ĐÁNH GIÁ':^40}")
    print("="*40)
    print(f"1. Độ chính xác (Precision): {precision:.2%}")
    print(f"   (Gemini gán nhãn đúng bao nhiêu %)")
    print(f"\n2. Độ bao phủ (Recall):      {recall:.2%}")
    print(f"   (Gemini tìm được bao nhiêu % của GT)")
    print(f"\n3. F1-Score:                 {f1:.2%}")
    print("-" * 40)
    print(f"Số thực thể khớp đúng:  {tp}")
    print(f"Số thực thể thừa/sai:   {fp}")
    print(f"Số thực thể bị bỏ sót:  {fn}")
    print("="*40)

if __name__ == "__main__":
    evaluate_comprehensive("data/processed/ground_truth_300.json", "data/silver_labels/few_shot_train_multilabel.json")