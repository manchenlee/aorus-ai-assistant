import os
import time
import json
import csv
import gc 
from trulens.core import Tru, Metric, Selector
from trulens.apps.virtual import VirtualApp, VirtualRecord
from trulens.core.feedback.provider import Provider
from llama_cpp import Llama

from src.main import AORUSAssistant
from config import Config
from src.eval_prompts import (
    PROMETHEUS_CORRECTNESS_TEMPLATE,
    PROMETHEUS_RELEVANCY_TEMPLATE,
    PROMETHEUS_FAITHFULNESS_TEMPLATE
)

# ==========================================
# ⚖️ 自定義本地裁判 Provider (M-Prometheus)
# ==========================================
import re
from typing import Tuple, Dict
from trulens.apps.custom import instrument
from trulens.apps.custom import TruCustomApp

# 假設你已經將前面的 Prometheus 模板存放在 prompts.py 或這裡直接定義
# from prompts import PROMETHEUS_CORRECTNESS_TEMPLATE, PROMETHEUS_RELEVANCY_TEMPLATE, PROMETHEUS_FAITHFULNESS_TEMPLATE

class LlamaCppJudge(Provider):
    # 讓 Pydantic 允許包含非標準型別 (Llama) 的屬性
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, model_path, n_gpu_layers=25, n_ctx=4096):
        super().__init__()
        print(f"Loading local model {model_path} ...")
        object.__setattr__(self, 'llm', Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False
        ))

    def _create_chat_completion(self, prompt: str = None, messages: list = None, **kwargs) -> str:
        """核心推論邏輯，可接收 prompt 字串或 messages 陣列"""
        if prompt:
            messages = [{"role": "user", "content": prompt}]
            
        # 預設參數：評分需要穩定的輸出 (temperature=0.0)，且需要足夠長度寫 COT (max_tokens=512)
        kwargs.setdefault("temperature", 0.0)
        kwargs.setdefault("max_tokens", 512)

        res = self.llm.create_chat_completion(messages=messages, **kwargs)
        return res["choices"][0]["message"]["content"]

    def _extract_score_and_reasons(self, raw_response: str) -> Tuple[float, Dict]:
        """解析 Prometheus 的輸出格式並轉換為 TruLens 的 0.0 - 1.0 分數"""
        # 尋找輸出中類似 "[RESULT] 5" 或 "[RESULT] 4" 的段落
        match = re.search(r"\[RESULT\]\s*(\d+)", raw_response)
        if match:
            score_1_to_5 = int(match.group(1))
            # TruLens 儀表板預期分數是 0.0 ~ 1.0
            # 將 1-5 分映射至 0.0-1.0 (1=0.0, 2=0.25, 3=0.5, 4=0.75, 5=1.0)
            score_0_to_1 = (score_1_to_5 - 1) / 4.0 
        else:
            # 解析失敗時給 0 分，保留 raw_response 以便除錯
            score_0_to_1 = 0.0 
            
        return float(score_0_to_1), {"reasons": raw_response}

    # --- 以下為 TruLens 評估指標實作 ---

    def relevance_with_cot_reasons(self, query_str: str, response: str) -> Tuple[float, Dict]:
        """指標 1：評估回答與使用者問題的相關性"""
        eval_prompt = PROMETHEUS_RELEVANCY_TEMPLATE.format(
            query=query_str,
            generated_answer=response
        )
        raw_output = self._create_chat_completion(prompt=eval_prompt)
        return self._extract_score_and_reasons(raw_output)

    def correctness_with_cot_reasons(self, query: str, response: str, reference_answer: str) -> Tuple[float, Dict]:
        """指標 2：評估回答是否正確 (需與 Ground Truth 比較)"""
        eval_prompt = PROMETHEUS_CORRECTNESS_TEMPLATE.format(
            query=query,
            generated_answer=response,
            reference_answer=reference_answer
        )
        raw_output = self._create_chat_completion(prompt=eval_prompt)
        return self._extract_score_and_reasons(raw_output)

    def faithfulness_with_cot_reasons(self, query_str: str, response: str) -> Tuple[float, Dict]:
        """指標 3：評估回答是否忠於檢索到的內容 (有無幻覺)"""
        eval_prompt = PROMETHEUS_FAITHFULNESS_TEMPLATE.format(
            context=query_str,
            generated_answer=response
        )
        raw_output = self._create_chat_completion(prompt=eval_prompt)
        return self._extract_score_and_reasons(raw_output)

class OfflineEvaluator:
    @instrument
    def run_eval(self, query: str, context: str, reference_answer: str) -> str:
        # 這個函數被呼叫時，OTEL 就會錄下這三個參數
        return self.current_answer

# ==========================================
# 🟢 階段一：生成與效能測試
# ==========================================
def run_generation_stage(input_csv, output_jsonl):
    print("\n" + "="*40)
    print("Stage 1: Start generate and performance test")
    print("="*40)
    
    if not os.path.exists(input_csv):
        print(f"Can't find test cases: {input_csv}")
        return False

    bot = AORUSAssistant()
    
    with open(input_csv, "r", encoding="utf-8-sig") as f:
        test_data = list(csv.DictReader(f))

    with open(output_jsonl, "w", encoding="utf-8") as out_file:
        for row in test_data:
            query = row["Question"]

            start_time = time.time()
            ttft = 0.0
            token_count = 0
            full_answer = ""
            
            stream = bot.generate_stream(query)
            for i, chunk in enumerate(stream):
                if i == 0:
                    ttft = time.time() - start_time
                full_answer += chunk
                token_count += len(chunk) 
            
            total_duration = time.time() - start_time
            tps = token_count / total_duration if total_duration > 0 else 0

            context = bot.last_context
            
            print(f"Completed. ID:{row.get('ID', '?')} | TTFT: {ttft:.3f}s | TPS: {tps:.1f}")
            
            record = {
                "id": row.get("ID", ""),
                "query": query,
                "context": context,
                "answer": full_answer,
                "ttft": round(ttft, 3),
                "tps": round(tps, 2),
                "total_time": round(total_duration, 2)
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Stage 1 completed. File saved: {output_jsonl}")
    
    # 🧹 物理級清空 VRAM 的關鍵步驟！
    print("Releasing VRAM...")
    del bot  # 刪除模型實例
    gc.collect()  # 強制回收記憶體
    time.sleep(2) # 稍微停頓讓 OS 確實釋放資源
    
    return True

# ==========================================
# 🔵 階段二：TruLens 裁判評分
# ==========================================
def run_evaluation_stage(input_jsonl):

    evaluator = OfflineEvaluator()

    print("\n" + "="*40)
    print("Stage 2: Start Trulens Offline Evaluation.")
    print("="*40)
    
    if not os.path.exists(input_jsonl):
        print(f"Can't find test_result file: {input_jsonl}")
        return

    tru = Tru()
    judge_provider = LlamaCppJudge(Config.JUDGE_MODEL_FILE, 24, 4096)

    f_relevance = Metric(
        name="Relevance",
        implementation=judge_provider.relevance_with_cot_reasons,
        selectors={
            "query_str": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        }
    )
    f_correctness = Metric(
        name="Correctness",
        implementation=judge_provider.correctness_with_cot_reasons,
        selectors={
            "query": Selector.select_record_input(),
            "response": Selector.select_record_output(),
            "reference_answer": Selector(function_attribute="record.app.run_eval.args.reference_answer")
        }
    )
    f_faithfulness = Metric(
        name="Faithfulness",
        implementation=judge_provider.faithfulness_with_cot_reasons,
        selectors={
            "query_str": Selector(function_attribute="record.app.run_eval.args.context"),
            "response": Selector.select_record_output()
        }
    )

    #virtual_app = VirtualApp()
    tru_recorder = TruCustomApp(
        evaluator,
        app_id="AORUS_Assistant_Eval_v4",
        feedbacks=[f_relevance, f_correctness, f_faithfulness]
    )
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            evaluator.current_answer = row["answer"]
            
            # 🌟 透過啟動 recorder 來執行函數，這會自動產生 record，不需要 add_record
            with tru_recorder as recording:
                evaluator.run_eval(
                    query=row["query"],
                    context=row.get("context", ""),
                    reference_answer=row.get("reference_answer", "")
                )
            print(f"Complete ID: {row.get('id', 'Unknown')}")
    
    #tru.get_leaderboard()
    tru.run_dashboard()
    records_df, feedback_cols = tru.get_records_and_feedback(app_name=["AORUS_Assistant_Eval_v4"])
    records_df.to_excel("trulens_records_detail.xlsx", index=False)
    print(f"File saved: {feedback_cols}")
    print("\nTest Completed! Start TruLens Dashboard...")
    

# ==========================================
# 🚀 程式進入點
# ==========================================
if __name__ == "__main__":
    input_csv = os.path.join(Config.TEST_DATA_PATH, "aorus_test_cases.csv")
    output_jsonl = os.path.join(Config.TEST_DATA_PATH, "aorus_test_results.jsonl")

    # 1. 跑生成階段
    if os.path.exists(output_jsonl):
        success = True
    else:
        success = run_generation_stage(input_csv, output_jsonl)
    
    # 2. 如果生成成功，接著跑評估階段
    if success:
        run_evaluation_stage(output_jsonl)