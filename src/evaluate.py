import os
import time
import json
import csv
import gc
import re
from llama_cpp import Llama

from src.main import AORUSAssistant
from config import Config
from src.eval_prompts import (
    PROMETHEUS_CORRECTNESS_TEMPLATE,
    PROMETHEUS_RELEVANCY_TEMPLATE,
    PROMETHEUS_FAITHFULNESS_TEMPLATE
)
from src.utils import validate_query
import asyncio
from deepeval.models.base_model import DeepEvalBaseLLM

class PrometheusJudge(DeepEvalBaseLLM):
    def __init__(self, model_path: str):
        self.model = Llama(
            model_path=model_path, 
            n_ctx=8192,
            n_gpu_layers=-1,
            verbose=False
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        llm = self.load_model()
        # Prometheus 需要輸出評分與推理過程，max_tokens 不能太少
        response = llm(prompt, max_tokens=512, stop=["</s>"])
        return response["choices"][0]["text"].strip()

    async def a_generate(self, prompt: str) -> str:
        return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self):
        return "M-Prometheus-7B-GGUF"
    
def parse_prometheus_result(raw_output):
    """
    解析 Prometheus 格式: Feedback: ... [RESULT] 5 (或 YES/NO)
    """
    try:
        # 使用正則表達式抓取 [RESULT] 後面的內容
        match = re.search(r"\[RESULT\]\s*(.*)", raw_output)
        if match:
            result = match.group(1).strip()
            # 轉換 Correctness 的 1-5 分到 0.0-1.0
            if result.isdigit():
                score = int(result)
                return (score - 1) / 4.0
            # 轉換 YES/NO 到 1.0/0.0
            if "YES" in result.upper():
                return 1.0
            if "NO" in result.upper():
                return 0.0
        return 0.0
    except Exception as e:
        print(f"解析失敗: {e} | 原始輸出: {raw_output}")
        return 0.0

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
            
            check, query = validate_query(query)
            if not check:
                print(query)
                continue
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
                "actual_answer": full_answer,
                "expected_answer": row["Expected_Answer"],
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
# 🔵 階段二：DeepEval 裁判評分
# ==========================================
def run_evaluation_stage(input_jsonl):

    print("\n" + "="*40)
    print("Stage 2: Start Trulens Offline Evaluation.")
    print("="*40)
    
    if not os.path.exists(input_jsonl):
        print(f"Can't find test_result file: {input_jsonl}")
        return

    output_csv = input_jsonl.replace(".jsonl", "_eval.csv")
    judge_llm = PrometheusJudge(model_path=Config.JUDGE_MODEL_FILE)
    
    results = []

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            query = row["query"]
            context = row.get("context", "")
            actual_answer = row["actual_answer"]
            expected_answer = row["expected_answer"]

            print(f"Processing ID: {row.get('id', i+1)}...", end="\r")

            # 1. Correctness (1-5分)
            prompt_cor = PROMETHEUS_CORRECTNESS_TEMPLATE.format(
                query=query,
                generated_answer=actual_answer,
                reference_answer=expected_answer
            )
            score_cor = parse_prometheus_result(judge_llm.generate(prompt_cor))

            # 2. Relevancy (YES/NO)
            # 這裡 query_str 通常包含問題與回答的配對
            prompt_rel = PROMETHEUS_RELEVANCY_TEMPLATE.format(
                query_str=f"Question: {query}\nResponse: {actual_answer}",
                context_str=context
            )
            score_rel = parse_prometheus_result(judge_llm.generate(prompt_rel))

            # 3. Faithfulness (YES/NO)
            # 這裡 query_str 指的是「待驗證的事實資訊」，通常就是 actual_answer
            prompt_fai = PROMETHEUS_FAITHFULNESS_TEMPLATE.format(
                query_str=actual_answer,
                context_str=context
            )
            score_fai = parse_prometheus_result(judge_llm.generate(prompt_fai))

            # 紀錄結果
            eval_row = {
                "id": row.get("id", i+1),
                "query": query,
                "correctness": score_cor,
                "relevancy": score_rel,
                "faithfulness": score_fai,
                "avg_score": (score_cor + score_rel + score_fai) / 3.0
            }
            print(row)
            results.append(eval_row)

    # 將評估結果輸出成 CSV
    keys = results[0].keys()
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

    print(f"\nStage 2 completed. File saved: {output_csv}")
    

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