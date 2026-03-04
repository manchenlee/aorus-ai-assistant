import csv
import os
import time
from llama_cpp import Llama
from retriever import AorusRetriever
import opencc

class AORUSChatbot:
    def __init__(self, model_path="models/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf"):
        # 1. 初始化 Retriever
        self.retriever = AorusRetriever()
        
        # 2. 初始化 llama.cpp 模型
        print(f"Loading llama.cpp: {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=8192,            # 上下文窗口
            n_gpu_layers=-1,       # -1 代表將所有層放入 GPU (4GB VRAM 跑 3B 沒問題)
            verbose=False          # 關閉詳細日誌讓輸出乾淨一點
        )
        
        # 3. 讀取動態免責聲明素材
        self.sys_context = self._load_system_context()

    def _load_system_context(self):
        file_path = "data/warning_context.txt"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return "（暫無原廠免責聲明）"

    def generate_answer(self, user_query):
        # A. 檢索 RAG 內容
        related_chunks = self.retriever.retrieve(user_query, k=4)
        context_text = "\n".join(related_chunks)

        print(f"\n[Debug] Chunk catch:\n{context_text}\n")

        # B. 組合雙語支援的 System Prompt
        system_prompt = f"""You are a professional, helpful, and human-like AORUS customer support assistant.
Answer the user's question based ONLY on the <Knowledge_Base> below.

<Knowledge_Base>
{context_text}

[Official Disclaimers]:
{self.sys_context}
</Knowledge_Base>
"""

        user_query = f"""{user_query}

You MUST strictly adhere to the following output format (extract data to draft first, then answer):

<Draft>
(Extract ONLY the specification data directly relevant to the question.)
</Draft>
<Answer>
(Provide the final answer in a natural, conversational tone based ONLY on your Draft. If the Draft says "No Data", politely reply that this information is not provided in the hardware specifications.)
</Answer>

CRITICAL RULES:
1. Language Matching: You MUST write the <Answer> in the EXACT SAME LANGUAGE as the user's query. (English to English, Traditional Chinese (zh-TW) to Traditional Chinese (zh-TW)).
2. Disclaimers: If the question is about "Weight", "Dimensions", "GPU/Graphics" or "Storage", explicitly mention the relevant disclaimer gracefully at the end of your <Answer>.
"""
        # C. 使用 llama.cpp 的 create_chat_completion (支援 Chat Template)
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        
        full_reply = response["choices"][0]["message"]["content"]

        print(full_reply)

        # 簡單的字串切割，只取 <Answer> 裡面的內容
        if "<Answer>" in full_reply:
            answer_text = full_reply.split("<Answer>")[1]
            final_output = answer_text.replace("</Answer>", "").strip()
        else:
            final_output = full_reply.strip()
            
        converter = opencc.OpenCC('s2twp')
        final_output = converter.convert(final_output)
        #print(final_output)
        return final_output

# --- 測試執行 ---
if __name__ == "__main__":
    bot = AORUSChatbot() 
    input_csv = "data/test/test_cases.csv"
    output_csv = "data/test/test_results.csv"

    print(f"\nStart Testing! Loading {input_csv} ...")

    # 1. 讀取測試題目
    test_data = []
    if not os.path.exists(input_csv):
        print(f"Can't find {input_csv}!")
        exit()
        
    # 使用 utf-8-sig 確保 Excel 開啟時中文不會亂碼
    with open(input_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_data.append(row)

    results = []
    
    # 2. 逐題進行測試
    for index, row in enumerate(test_data, start=1):
        query = row["Question"]
        print(f"\n[{index}/{len(test_data)}] User query: {query}")
        
        # 記錄推論時間，這對評估模型效能很有幫助
        start_time = time.time()
        actual_answer = bot.generate_answer(query)
        end_time = time.time()
        
        time_taken = round(end_time - start_time, 2)
        print(f"Time: {time_taken} second")
        # 只印出前 60 個字作為預覽，避免畫面被長篇大論洗版
        preview_text = actual_answer[:60].replace('\n', ' ')
        print(f"Assistant: {preview_text}...") 
        
        # 將結果與時間記錄存入字典
        row["Actual_Answer"] = actual_answer
        row["Time_Taken(s)"] = time_taken
        results.append(row)

    # 3. 將結果輸出成新的 CSV 報告
    print(f"\n💾 Testing finished! Output {output_csv} ...")
    
    # 定義輸出的欄位 (包含原本的欄位加上我們新增的 Actual_Answer 和 Time_Taken)
    fieldnames = ["ID", "Category", "Language", "Question", "Expected_Answer", "Actual_Answer", "Time_Taken(s)"]
    
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Output {output_csv}.")