import json
import os

def normalize_label(label):
    """Chuẩn hóa nhãn về danh mục chung."""
    label = label.upper().strip()
    mapping = {
        'CHEMICAL': 'CHEMICAL',
        'DISEASE': 'DISEASE',
        'PROTEIN': 'PROTEIN',
        'GENE': 'PROTEIN',
        'DNA': 'DNA_RNA',
        'RNA': 'DNA_RNA',
        'DNA_RNA': 'DNA_RNA',
        'CELL_LINE': 'CELL_TYPE',
        'CELL_TYPE': 'CELL_TYPE'
    }
    return mapping.get(label, label)

def merge_and_clean_data(gt_json, gm_json, output_json):
    if not os.path.exists(gt_json) or not os.path.exists(gm_json):
        print("[!] Không tìm thấy file input.")
        return

    with open(gt_json, 'r', encoding='utf-8') as f:
        gt_data = {item['id']: item for item in json.load(f)}
    with open(gm_json, 'r', encoding='utf-8') as f:
        gm_data = json.load(f)

    final_dataset = []
    print(f"[*] Đang thực hiện trộn dữ liệu (Smart Merging)...")

    for gm_item in gm_data:
        item_id = gm_item['id']
        text = gm_item['text']
        source = gm_item['source']
        
        # Lấy thực thể từ cả 2 nguồn
        gt_ents = gt_data.get(item_id, {}).get('entities', [])
        gm_ents = gm_item.get('entities', [])

        # Logic: 
        # 1. Ưu tiên tuyệt đối Ground Truth (vì đây là nhãn chuẩn)
        # 2. Chỉ lấy thêm từ Gemini nếu thực thể đó KHÔNG trùng lặp với GT 
        #    và NHÃN phải phù hợp với nguồn dữ liệu (Source) để tránh nhiễu chéo.
        
        merged_entities = []
        existing_spans = []

        # Thêm toàn bộ GT vào trước
        for e in gt_ents:
            merged_entities.append({
                "word": e['word'],
                "label": normalize_label(e['label']),
                "start": e.get('start_idx', e.get('start')), # Handle different keys
                "end": e.get('end_idx', e.get('end'))
            })
            existing_spans.append((e.get('start_idx', e.get('start')), e.get('end_idx', e.get('end'))))

        # Thêm từ Gemini nếu hợp lệ
        for ge in gm_ents:
            g_start = ge.get('start')
            g_end = ge.get('end')
            g_label = normalize_label(ge['label'])
            
            # Kiểm tra xem có bị đè (overlap) với GT không
            is_overlap = False
            for s, e in existing_spans:
                if not (g_end <= s or g_start >= e):
                    is_overlap = True
                    break
            
            if is_overlap: continue

            # Kiểm tra tính phù hợp với Source (Lọc nhiễu chéo)
            is_valid_source = False
            if source == "BC5CDR" and g_label in ["CHEMICAL", "DISEASE"]:
                is_valid_source = True
            elif source == "NCBI Disease" and g_label == "DISEASE":
                is_valid_source = True
            elif source == "JNLPBA" and g_label in ["PROTEIN", "DNA_RNA", "CELL_TYPE"]:
                is_valid_source = True
            
            if is_valid_source:
                merged_entities.append({
                    "word": ge['word'],
                    "label": g_label,
                    "start": g_start,
                    "end": g_end
                })

        final_dataset.append({
            "id": item_id,
            "source": source,
            "text": text,
            "entities": merged_entities
        })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)
    
    print(f"[Done] Đã tạo tập dữ liệu huấn luyện sạch: {output_json}")

if __name__ == "__main__":
    merge_and_clean_data("ground_truth_300.json", "few_shot_train_multilabel.json", "final_train_data.json")
