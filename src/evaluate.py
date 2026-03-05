import os
import time
import csv
import json
import numpy as np
from datetime import datetime
from trulens.core import Tru
from trulens.core import Feedback
from trulens.core import Select
from trulens.providers.google import GoogleProvider

from src.main import AORUSAssistant
from config import Config

# --- 1. TruLens 配置 ---
tru = Tru()
# 這裡建議使用 OpenAI 作為評審 (Judge)，效果最穩。如果是純離線，可換成本地 HuggingFace 模型。
provider = GoogleProvider(model_engine="gpt-4o") 

# 定義 RAG 三元組反饋
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(Select.Record_calls.retriever.retrieve.collect()) # 假設 retriever 的輸出
    .on_output()
)
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input()
    .on_output()
)
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on(Select.Record_calls.retriever.retrieve.collect())
)

class Evaluator:
    def __init__(self):
        self.bot = AORUSAssistant()
        self.input_csv = os.path.join(Config.TEST_DATA_PATH, "aorus_test_cases.csv")
        self.output_csv = os.path.join(Config.TEST_DATA_PATH, "aorus_test_results.csv")

    def run_eval(self):
        if not os.path.exists(self.input_csv):
            print(f"Can't find: {self.input_csv}")
            return

        test_data = []
        with open(self.input_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            test_data = list(reader)

        results = []
        print(f"Start evaluating {len(test_data)} test cases...")

        for row in test_data:
            query = row["Question"]
            
            # --- 效能計時開始 ---
            start_time = time.time()
            ttft = 0.0
            token_count = 0
            full_answer = ""
            
            # 調用機器人的串流生成
            stream = self.bot.generate_stream(query)
            
            for i, chunk in enumerate(stream):
                if i == 0:
                    ttft = time.time() - start_time  # 第一個 Token 出現的時間
                
                full_answer += chunk
                # 簡單估計 token 數 (中文 1 字 ~ 1 token, 英文由空白切分)
                token_count += len(chunk) 
            
            end_time = time.time()
            total_duration = end_time - start_time
            tps = token_count / total_duration if total_duration > 0 else 0
            
            # --- 記錄結果 ---
            row["Actual_Answer"] = full_answer
            row["TTFT(s)"] = round(ttft, 3)
            row["TPS"] = round(tps, 2)
            row["Total_Time(s)"] = round(total_duration, 2)
            
            print(f"✅ ID:{row['ID']} | TTFT: {row['TTFT(s)']}s | TPS: {row['TPS']}")
            results.append(row)

        # 儲存 CSV 報告
        self._save_report(results)
        
        # --- 啟動 TruLens Dashboard ---
        # 註：這裡通常會將 bot 封裝進 TruLlama 以自動記錄，
        # 但因為我們用了自定義的串流與 llama-cpp，這裡我們跑完後手動啟動 Dashboard 檢視。
        print("\n✨ Test completed！Start TruLens Dashboard (default port 8501)...")
        tru.run_dashboard()

    def _save_report(self, results):
        fieldnames = results[0].keys()
        with open(self.output_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Detailed saved at: {self.output_csv}")

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_eval()