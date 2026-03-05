import os
import opencc
from llama_cpp import Llama
from retriever import AorusRetriever

class AORUSChatbot:
    def __init__(self, model_path="models/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf"):
        # 1. 初始化 Retriever
        self.retriever = AorusRetriever()
        self.converter = opencc.OpenCC('s2twp')

        # 2. 初始化 llama.cpp 模型
        print(f"Loading llama.cpp: {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=8192,            # 上下文窗口
            n_gpu_layers=-1,       # -1 代表將所有層放入 GPU
            verbose=False          # 關閉詳細日誌
        )
        
        # 3. 讀取動態免責聲明素材
        self.sys_context = self._load_system_context()

    def _load_system_context(self):
        file_path = "data/warning_context.txt"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return "（暫無原廠免責聲明）"

    def generate_stream(self, user_query):
        """核心推論邏輯，使用 yield 回傳串流文字"""
        # A. 檢索 RAG 內容
        related_chunks = self.retriever.retrieve(user_query, k=4)
        context_text = "\n".join(related_chunks)

        # B. 組合 Prompt
        system_prompt = f"""You are a professional, helpful, and human-like AORUS customer support assistant.
Answer the user's question based ONLY on the <Knowledge_Base> below.

<Knowledge_Base>
{context_text}
{self.sys_context}
</Knowledge_Base>
"""

        user_query_prompt = f"""[User Query] 
        {user_query}
[INSTRUCTION]
Please strictly adhere to the following output format (Extract data to draft first, then answer).
<Draft>
(Only extract specifications that are DIRECTLY relevant to the user's specific question. Do not include unrelated hardware categories. Use bullet points. MAX 5 LINES. If the information is missing from the Knowledge Base, write exactly "No Data". Do NOT copy unrelated specs.)
</Draft>
<Answer>
(Provide a natural, conversational response in the EXACT SAME LANGUAGE as [User Query]. Do not provide unrelated hardware specs. NEVER contradict yourself mathematically. For Yes/No questions (e.g., 'Is it...', 'Does it have...'), always start your answer with a clear 'Yes' or 'No' (是的 / 不是). If the Draft says "No Data", politely state that the specifications do not provide this information.)
</Answer>
"""
        # C. 使用 llama.cpp 生成，並確保 stream=True
        response_stream = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query_prompt}
            ],
            temperature=0.1,
            max_tokens=1024,
            stream=True
        )
        # 🌟 D. 狀態機與滑動視窗初始化
        is_answering = False
        full_raw_text = ""
        buffer = ""
        
        # 滑動視窗參數：你可以自己微調這兩個數字來感受打字速度
        window_size = 8      # 當籃子累積到幾個字時，觸發輸出
        lookahead_size = 4   # 每次輸出時，強制把最後幾個字「扣留」下來當作下次的上下文

        for chunk in response_stream:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                text_piece = delta["content"]
                full_raw_text += text_piece
                
                # 階段 1：隱藏 Draft，尋找 Answer 標籤
                if not is_answering:
                    if "<Answer>" in full_raw_text:
                        is_answering = True
                        buffer = full_raw_text.split("<Answer>")[1].lstrip("\n")
                
                # 階段 2：進入 Answer 區塊，啟用滑動視窗流暢輸出
                else:
                    buffer += text_piece
                    
                    # --- 1. 標籤攔截器 ---
                    if "<" in buffer:
                        if "</Answer>" in buffer:
                            final_chunk = buffer.split("</Answer>")[0]
                            if final_chunk: # 把剩下的字一次轉完吐出
                                yield self.converter.convert(final_chunk)
                            break # 直接結束串流
                            
                        # 如果 '<' 後面跟了超過 10 個字還沒變成 </Answer>
                        # 代表只是單純的符號 (如：溫度 < 90度)，解除警報！
                        elif len(buffer) - buffer.find("<") > 10:
                            pass # 繼續往下走滑動視窗邏輯
                        else:
                            # 疑似是標籤結尾，先憋住不輸出！
                            continue 
                            
                    # --- 2. 滑動視窗轉換引擎 ---
                    # 只要緩衝區超過設定的大小，就切出一塊安全的字串丟出去
                    if len(buffer) >= window_size:
                        # 切出「要轉換輸出的部分」
                        safe_chunk = buffer[:-lookahead_size]
                        yield self.converter.convert(safe_chunk)
                        
                        # 把 buffer 更新為剩下的「保留區」，留給下一個 Token 做上下文
                        buffer = buffer[-lookahead_size:]
                        
        # 🌟 E. 防呆機制與殘留字元清理
        if not is_answering:
            yield self.converter.convert(full_raw_text.strip())
        elif buffer and "</Answer>" not in buffer:
            # 迴圈結束後，把扣留在保留區的最後幾個字吐出來
            yield self.converter.convert(buffer.strip())

# --- 單機簡單測試區 ---
if __name__ == "__main__":
    bot = AORUSChatbot()
    print("\n✅ AORUS AI Assistant is Online! (type \'exit\' to exit.)")
    while True:
        query = input("\n[User]: ")
        if query.lower() == 'exit':
            break
        
        print("[Assistant]: ", end="", flush=True)
        for text_piece in bot.generate_stream(query):
            print(text_piece, end="", flush=True)
        print()