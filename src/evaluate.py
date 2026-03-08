import argparse
import os
import time
import json
import csv
import gc
import re
import math
from src.vram_monitor import VRAMMonitor
from deepeval.models.base_model import DeepEvalBaseLLM
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
from config import Config
from src.main import AORUSAssistant
from config import Config
from src.eval_prompts import (
    PROMETHEUS_CORRECTNESS_TEMPLATE,
    PROMETHEUS_RELEVANCY_TEMPLATE,
    PROMETHEUS_FAITHFULNESS_TEMPLATE
)
from src.utils import validate_query
import asyncio
import spacy

os.environ["HF_TOKEN"] = Config.HF_TOKEN
model_embed = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model_nli = CrossEncoder('MoritzLaurer/mDeBERTa-v3-base-mnli-xnli')
model_rel = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

monitor = VRAMMonitor()
monitor.start()

from llama_cpp import Llama

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
        response = llm(prompt, max_tokens=512, stop=["</s>"])
        return response["choices"][0]["text"].strip()

    async def a_generate(self, prompt: str) -> str:
        return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self):
        return "M-Prometheus-7B-GGUF"
    
def parse_prometheus_result(raw_output):
    """
    解析 Prometheus 格式，並回傳 (score, feedback)
    """
    score = 0.0
    feedback = "Parsing failed"
    
    try:
        # 1. 先處理 Feedback：抓取 [RESULT] 之前的所有內容
        if "[RESULT]" in raw_output:
            feedback_part = raw_output.split("[RESULT]")[0]
            feedback = feedback_part.replace("Feedback:", "").strip()
        else:
            feedback = raw_output.strip()

        # 2. 使用你原本最穩定的正則邏輯抓取 [RESULT] 後面的內容
        match = re.search(r"\[RESULT\]\s*(.*)", raw_output)
        if match:
            result = match.group(1).strip()
            
            if result.isdigit():
                raw_score = int(result)
                score = (raw_score - 1) / 4.0
                return score, feedback
                
            if "YES" in result.upper():
                score = 1.0
                return score, feedback
            if "NO" in result.upper():
                score = 0.0
                return score, feedback
        return score, feedback

    except Exception as e:
        print(f"Paring failed: {e} | Original output: {raw_output}")
        return 0.0, "Parsing Error"

def get_nli_entailment_score(context, actual_answer):
    """計算 Faithfulness (NLI 模型判定為 Entailment 的機率)"""
    #print(actual_answer)
    if len(context) <= 800:
        scores = model_nli.predict([(context, actual_answer)])
        probs = softmax(scores, axis=1)[0]
        return float(probs[0]) 

    window_size = 800
    overlap = 400
    max_entailment_score = 0.0

    for i in range(0, len(context), window_size - overlap):
        chunk = context[i : i + window_size]
        
        if len(chunk) < 50: 
            break
            
        scores = model_nli.predict([(chunk, actual_answer)])
        probs = softmax(scores, axis=1)[0]
        entailment_prob = float(probs[0]) 
        if entailment_prob > max_entailment_score:
            max_entailment_score = entailment_prob
            
        if max_entailment_score > 0.9:
            break
            
    return max_entailment_score

def get_technical_entities(text):
    """
    結合 spaCy 與 Regex，精準抓出中英混排中的技術規格
    """
    nlp = spacy.load("zh_core_web_trf")
    doc = nlp(text)
    entities = set()
    
    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "ORG", "QUANTITY", "CARDINAL"]:
            entities.add(ent.text.lower().strip())
            
    regex_patterns = [
        r'[a-zA-Z]+\s?\d+[a-zA-Z]*', # 抓 RTX 5090, i9-14900, 16GB
        r'\d+\.\d+\s?[gG][hH]z',      # 抓 5.4GHz
    ]
    
    for pattern in regex_patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            entities.add(m.lower().strip())
            
    return entities

def get_entity_overlap_score(context, actual_answer):
    answer_entities = get_technical_entities(actual_answer)
    
    if not answer_entities:
        return 1.0
        
    context_lower = context.lower()
    match_count = sum(1 for ent in answer_entities if ent in context_lower)
    
    return match_count / len(answer_entities)

def get_cross_encoder_relevance(query, answer):
    """計算 Relevance (使用 MS-MARCO 檢索模型)"""
    # ms-marco 模型通常直接輸出一個代表相關性的原始分數 (Logit)
    # 分數越高越相關，通常不一定是 0~1，但可以用作相對評估
    score = model_rel.predict([(query, answer)])
    #print(score)
    normalized_score = 1 / (1 + math.exp(-score))
    # print('cross encoder score:', relevance_score)
    return float(normalized_score)

# ==========================================
# 階段一：生成與效能測試
# ==========================================
def run_generation_stage(input_csv, output_jsonl, log):
    print("\n" + "="*40)
    print("Stage 1: Start generate and performance test")
    print("="*40)
    
    if not os.path.exists(input_csv):
        print(f"Can't find test cases: {input_csv}")
        return False

    bot = AORUSAssistant()
    
    with open(input_csv, "r", encoding="utf-8-sig") as f:
        test_data = list(csv.DictReader(f))

    output_csv_stage1 = output_jsonl.replace(".jsonl", ".csv")
    #test_data = random.sample(test_data, 5) # 測試用 testing

    with open(output_jsonl, "w", encoding="utf-8") as out_jsonl, \
         open(output_csv_stage1, "w", newline="", encoding="utf-8-sig") as out_csv:
        
        csv_writer = csv.writer(out_csv)
        csv_writer.writerow(["id", "query", "expected_answer", "actual_answer", "ttft", "tps", "total_time"])

        total_ttft = 0.0
        total_tps = 0.0
        record_count = 0

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
            
            total_ttft += ttft
            total_tps += tps
            record_count += 1

            # 寫入 JSONL
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
            out_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 需求 1: 寫入對照用 CSV (不含 context)
            csv_writer.writerow([
                record["id"], 
                record["query"], 
                record["expected_answer"], 
                record["actual_answer"], 
                record["ttft"], 
                record["tps"], 
                record["total_time"]
            ])

    avg_ttft = total_ttft / record_count if record_count > 0 else 0.0
    avg_tps = total_tps / record_count if record_count > 0 else 0.0

    print(f"Stage 1 completed. Files saved:\n- {output_jsonl}\n- {output_csv_stage1}")
    with open(log, "a", encoding="utf-8") as f:
        print("=" * 50, file=f)
        print(f"Performance Summary (Average of {record_count} queries):", file=f)
        print("=" * 50, file=f)
        print(f"   Avg TTFT: {avg_ttft:.3f} seconds", file=f)
        print(f"   Avg TPS : {avg_tps:.2f} tokens/second", file=f)
        print("=" * 50, file=f)

    # 🧹 物理級清空 VRAM 的關鍵步驟！
    print("Releasing VRAM...")
    del bot  # 刪除模型實例
    gc.collect()  # 強制回收記憶體
    time.sleep(2) # 稍微停頓讓 OS 確實釋放資源
    
    return True

# ==========================================
# 階段二：DeepEval測試
# ==========================================
def run_evaluation_stage_llm(input_jsonl, log):
    print("\n" + "="*40)
    print("Stage 2: Start Prometheus Offline Evaluation.")
    print("="*40)
    
    if not os.path.exists(input_jsonl):
        print(f"Can't find test_result file: {input_jsonl}")
        return

    output_eval_csv = input_jsonl.replace(".jsonl", "_eval_llm.csv")
    output_feedback_csv = input_jsonl.replace(".jsonl", "_eval_llm_feedback.csv")
    
    judge_llm = PrometheusJudge(model_path=Config.JUDGE_MODEL_FILE)
    
    eval_results = []
    feedback_results = []

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f)):
            row = json.loads(line)
            q_id = row.get("id", i+1)
            query = row["query"]
            context = row.get("context", "")
            actual_answer = row["actual_answer"]
            expected_answer = row["expected_answer"]

            print(f"Processing ID: {q_id}...", end="\r")

            # 1. Correctness (1-5分) -> 💡 所有題目都要測
            prompt_cor = PROMETHEUS_CORRECTNESS_TEMPLATE.format(
                generated_answer=actual_answer,
                reference_answer=expected_answer
            )
            score_cor, fb_cor = parse_prometheus_result(judge_llm.generate(prompt_cor))

            if i < Config.NUM_OF_NORMAL_Q:
                # 2. Relevancy (YES/NO)
                prompt_rel = PROMETHEUS_RELEVANCY_TEMPLATE.format(
                    query_str=f"Question: {query}\nResponse: {actual_answer}",
                    context_str=context
                )
                score_rel, fb_rel = parse_prometheus_result(judge_llm.generate(prompt_rel))

                # 3. Faithfulness (YES/NO)
                prompt_fai = PROMETHEUS_FAITHFULNESS_TEMPLATE.format(
                    query_str=actual_answer,
                    context_str=context
                )
                score_fai, fb_fai = parse_prometheus_result(judge_llm.generate(prompt_fai))
                
                avg_score = round((score_cor + score_rel + score_fai) / 3.0, 2)
                
            else:
                score_rel = -1.0
                fb_rel = "N/A"
                
                score_fai = -1.0
                fb_fai = "N/A"
                
                avg_score = round(score_cor, 2)

            # 紀錄分數結果
            eval_results.append({
                "id": q_id,
                "query": query,
                "correctness": score_cor,
                "relevancy": score_rel,
                "faithfulness": score_fai,
                "avg_score": avg_score
            })

            # 紀錄 Feedback 結果
            feedback_results.append({
                "id": q_id,
                "query": query,
                "correctness_feedback": fb_cor,
                "relevancy_feedback": fb_rel,
                "faithfulness_feedback": fb_fai
            })

    # 將評估分數輸出成 CSV
    with open(output_eval_csv, "w", newline="", encoding="utf-8-sig") as f:
        dict_writer = csv.DictWriter(f, fieldnames=eval_results[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(eval_results)

    # 將 Feedback 輸出成獨立的 CSV
    with open(output_feedback_csv, "w", newline="", encoding="utf-8-sig") as f:
        dict_writer = csv.DictWriter(f, fieldnames=feedback_results[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(feedback_results)

    print(f"\nStage 2 (LLM) completed. Files saved:\n- {output_eval_csv}\n- {output_feedback_csv}")

def run_evaluation_stage_nonllm(input_jsonl, log):
    print("\n" + "="*40)
    print("Stage 2: Start non-LLM Offline Evaluation (via HF API).")
    print("="*40)
    
    if not os.path.exists(input_jsonl):
        print(f"Can't find test_result file: {input_jsonl}")
        return
    
    output_eval_csv = input_jsonl.replace(".jsonl", "_eval_nonllm.csv")
    eval_results = []
    
    # 初始化 ROUGE 評分器
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    with open(input_jsonl, "r", encoding="utf-8") as f:
        # 計算總行數供 tqdm 顯示進度
        lines = f.readlines()
        
        for i, line in tqdm(enumerate(lines), total=len(lines)):
            row = json.loads(line)
            q_id = row.get("id", i+1)
            query = row["query"]
            context = row.get("context", "")
            actual_answer = row["actual_answer"]
            expected_answer = row["expected_answer"]
            
            # 初始化該筆資料的預設指標值
            metrics = {
                "id": q_id,
                "query": query,
                "correctness_rougeL": None,
                "correctness_semantic_sim": None,
                "faithfulness_nli": None,
                "faithfulness_entity_overlap": None,
                "relevance_cross_encoder": None,
                "robustness_semantic_sim": None
            }

            # ----------------------------------------
            # 正常狀況題
            # ----------------------------------------
            if i < Config.NUM_OF_NORMAL_Q:
                # 1. Correctness: ROUGE-L
                rouge_scores = scorer.score(expected_answer, actual_answer)
                metrics["correctness_rougeL"] = round(rouge_scores['rougeL'].fmeasure, 4)
                
                # 2. Correctness: Semantic Similarity
                embeddings = model_embed.encode([actual_answer, expected_answer])
                sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                metrics["correctness_semantic_sim"] = round(float(sim), 4)
                
                # 3. Faithfulness: NLI 分類
                # 若 Context 太長，可擷取前段避免 API Token 超載
                truncated_context = context[:2000] if context else ""
                metrics["faithfulness_nli"] = round(get_nli_entailment_score(truncated_context, actual_answer), 4)

                metrics["faithfulness_entity_overlap"] = get_entity_overlap_score(context, actual_answer)
                
                # 4. Relevance: Cross-Encoder
                metrics["relevance_cross_encoder"] = round(get_cross_encoder_relevance(query, actual_answer), 4)
                
            # ----------------------------------------
            # 異常狀況題
            # ----------------------------------------
            else:
                # Robustness: Semantic Similarity 與標準拒答相比
                embeddings = model_embed.encode([actual_answer, expected_answer])
                sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                metrics["robustness_semantic_sim"] = round(float(sim), 4)
            #print(metrics)
            eval_results.append(metrics)

    # 將結果輸出為單一 CSV
    df = pd.DataFrame(eval_results)
    df.to_csv(output_eval_csv, index=False, encoding="utf-8-sig")

    df_normal = df.head(50)
    avg_correctness_rouge = df_normal["correctness_rougeL"].mean()
    avg_correctness_sim = df_normal["correctness_semantic_sim"].mean()
    avg_faithfulness = df_normal["faithfulness_nli"].mean()
    avg_faithfulness_entity_overlap = df_normal["faithfulness_entity_overlap"].mean()
    avg_relevance = df_normal["relevance_cross_encoder"].mean()

    df_robust = df.tail(50)
    avg_robustness = df_robust["robustness_semantic_sim"].mean()

    summary_data = {
        "Metric Category": [
            "Correctness (ROUGE-L)", 
            "Correctness (Semantic Sim)", 
            "Faithfulness (NLI)", 
            "Faithfulness (Entity Overlap)",
            "Relevance (Cross-Encoder)", 
            "Robustness (Refusal)"
        ],
        "Target": ["Normal", "Normal", "Normal", "Normal", "Normal", "Abnormal"],
        "Average Score": [
            avg_correctness_rouge, 
            avg_correctness_sim, 
            avg_faithfulness,
            avg_faithfulness_entity_overlap, 
            avg_relevance, 
            avg_robustness
        ],
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    with open(log, "a", encoding="utf-8") as f:
        print("\n" + "="*50, file=f)
        print("Evaluation Summary Report", file=f)
        print("="*50, file=f)
        print(summary_df.to_string(index=False), file=f)
        print("="*50, file=f)


    print(f"\nStage 2 (non-LLM) completed. Files saved:\n- {output_eval_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AORUS RAG Evaluation tools")

    parser.add_argument(
        "--stage", 
        choices=["all", "st1", "st2"], 
        default="all",
        help="'all', 'st1', 'st2'"
    )

    parser.add_argument(
            "--eval_mode", 
            type=str, 
            choices=["llm", "nonllm"], 
            default="nonllm",
            help="'llm' or 'nonllm'"
    )

    parser.add_argument(
        "--res", 
        type=str,
        help=""
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    input_csv = os.path.join(Config.TEST_DATA_PATH, "test_cases.csv")
    output_jsonl = os.path.join(Config.TEST_DATA_PATH, f"results_{timestamp}.jsonl")
    log = os.path.join(Config.TEST_DATA_PATH, f"log_{timestamp}.txt")

    a = parser.parse_args()

    if a.stage in ["all", "st1"]:
        try:
            run_generation_stage(input_csv, output_jsonl, log)
        finally:
            monitor.stop(log)
        
    if a.stage in ["all", "st2"]:
        if a.stage == "st2":
            assert a.res != ""
            output_jsonl = os.path.join(Config.TEST_DATA_PATH, a.res)
            temp = re.split('[_.]', a.res)
            log_name = 'log_' + temp[1] + temp[2] + '.txt'
            print(log_name)
            log = os.path.join(Config.TEST_DATA_PATH, log_name)
        if a.eval_mode == 'llm':
            run_evaluation_stage_llm(output_jsonl, log)
        elif a.eval_mode == 'nonllm':
            run_evaluation_stage_nonllm(output_jsonl, log)
