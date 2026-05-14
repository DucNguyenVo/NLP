import json
import time
import os
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()
# --- Cấu hình API ---
API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# Dựa trên danh sách models/ của bạn, đây là định danh Lite chính xác nhất
MODEL_ID = "gemini-flash-lite-latest" 
BATCH_SIZE = 10

class BioNERAnnotatorLite:
    def __init__(self):
        self.system_instruction = """
        Role: Biomedical Expert.
        Task: Extract entities (GENE, PROTEIN, DNA_RNA, CELL_TYPE, CHEMICAL, DISEASE).
        Return JSON format: {"results": [{"id": ..., "entities": [{"word": "...", "label": "...", "start": 0, "end": 0}]}]}
        
        Examples:
        Input: "ID 1: Aspirin reduces fever and headache."
        Output: {"results": [{"id": 1, "entities": [{"word": "Aspirin", "label": "CHEMICAL"}, {"word": "fever", "label": "DISEASE"}, {"word": "headache", "label": "DISEASE"}]}]}
        
        Input: "ID 2: the CD44 surface antigen"
        Output: {"results": [{"id": 2, "entities": [{"word": "CD44 surface antigen", "label": "PROTEIN"}]}]}
        """

    def process_batch(self, batch_items):
        batch_input = "\n".join([f"ID {item['id']}: {item['text']}" for item in batch_items])
        
        try:
            # Lưu ý: SDK v2 sử dụng model=MODEL_ID (không có prefix models/)
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=f"Extract entities for this list:\n{batch_input}",
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    response_mime_type="application/json",
                )
            )
            if response.text:
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                elif text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                return json.loads(text).get("results", [])
        except Exception as e:
            print(f"\n[!] Error during API call: {str(e)}")
            return None

    def run(self, input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_results = []
        total = len(data)
        print(f"[*] Đang sử dụng Model: {MODEL_ID}")
        print(f"[*] Chế độ Batch: {BATCH_SIZE} câu/lượt.")

        for i in range(0, total, BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            print(f"[*] Tiến độ: {min(i + BATCH_SIZE, total)}/{total}", end='\r')
            
            # Retry logic
            for attempt in range(3):
                res_list = self.process_batch(batch)
                if res_list:
                    for res_item, original in zip(res_list, batch):
                        all_results.append({
                            "id": original['id'],
                            "source": original['source'],
                            "text": original['text'],
                            "entities": res_item.get('entities', [])
                        })
                    break
                print(f"\n[!] Thử lại lượt {attempt+1} sau 10s...")
                time.sleep(10)
            
            # Nghỉ 6 giây giữa các lượt (đáp ứng 10 RPM của bản Lite)
            time.sleep(6.0)

            # Checkpoint mỗi lượt
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)

        print(f"\n[Done] Hoàn thành 300 câu! File: {output_file}")

if __name__ == "__main__":
    annotator = BioNERAnnotatorLite()
    annotator.run("data/processed/raw_seed_data.json", "data/silver_labels/few_shot_train_multilabel.json")