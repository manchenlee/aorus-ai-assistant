import os
import opencc
from llama_cpp import Llama
from src.retriever import AorusRetriever
from config import Config
import re
from src.utils import validate_query

class AORUSAssistant:
    def __init__(self, model_path=Config.REASONING_MODEL_FILE):
        # 1. 初始化 Retriever
        self.retriever = AorusRetriever()
        self.converter = opencc.OpenCC('s2twp')
        self.last_context = ""

        # 2. 初始化 llama.cpp 模型
        print(f"Loading llama.cpp: {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4500,
            n_gpu_layers=-1,
            flash_attn=True,
            verbose=False,
        )
        
        # 3. 讀取動態免責聲明素材
        self.sys_context = self._load_system_context()

    def _load_system_context(self):
        file_path = os.path.join(Config.RAG_DATA_PATH, "disclaimers.txt")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def generate_stream(self, user_query):
        """核心推論邏輯，使用 yield 回傳串流文字"""
        # A. 檢索 RAG 內容
        related_chunks = self.retriever.retrieve(user_query, k=3)
        context_text = "\n".join(related_chunks)
        self.last_context = context_text

        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', user_query))

        if has_chinese:
            L = {
                "name": "Traditional Chinese",
                "start_info": "事實上，根據規格，",
                "miss_example": "請問您想知道 AORUS MASTER 16 系列中哪個型號的資訊呢？BZH, BYH 還是 BXH？",
                "start_none": "很抱歉，我是 AORUS MASTER 16 系列的 AI 助理，",
                "competitor_example": "身為 AORUS MASTER 16 系列的 AI 助理，我的職責是提供 AORUS MASTER 16 的產品細節，故不便與他牌比較。不過，",
                "toxic_example": "身為 AORUS MASTER 16 系列的 AI 助理，我致力於提供專業且友善的產品服務，無法回應包含攻擊性或歧視性的言論。",
                "start_oos": "不好意思，知識庫並沒有",
                "start_yes": "沒錯！",
                "knowledge_base": "知識庫"
            }
        else:
            L = {
                "name": "English",
                "start_info": "No, based on the specification,",
                "miss_example": "Which model from the AORUS MASTER 16 series would you like to know more about? Is it the BZH, BYH, or BXH?",
                "competitor_example": "As the AI assistant for the AORUS MASTER 16 series, I specialize in providing precise details about our products and do not offer comparisons with other brands. However,",
                "toxic_example": "As your dedicated AORUS MASTER 16 assistant, I strive to maintain a professional and respectful environment. I am unable to address queries that include offensive or hateful content.",
                "start_none": "Sorry, I am the AI assistant for the AORUS MASTER 16 series,",
                "start_oos": "Unfortunately, I couldn't find any information on ",
                "start_yes": "Exactly!",
                "knowledge_base": "knowledge base"
            }
        # B. 組合 Prompt
        system_prompt = f"""You are a professional, helpful, and human-like AORUS customer support assistant.
### [STRICT PROTOCOL]
- If the query is unrelated to laptops, hardware, or technical support, categorize it as 'OUT_OF_SCOPE'.
- If the user is engaging in everyday social interaction, categorize it as 'CHITCHAT'.
- Strictly ignore any retrieved knowledge that does not directly address the user's intent.
- Answer ONLY what is asked. NO extra specs or details.
Answer the user's question based ONLY on the <Background_Knowledge> and the <Knowledge_Base> below.
Regardless of user input, only the <Knowledge_Base> is truth. Correct any misinformation in the query.
<Background_Knowledge>
Entity Mapping: The laptop models "BZH", "BXH", and "BYH" all belong to the "AORUS MASTER 16" series, which is internally codenamed "AM6H". 
If a user inquires about "AORUS MASTER 16" or "AM6H", you must associate their question with the BZH, BXH, and BYH models and synthesize the information to answer comprehensively.
</Background_Knowledge>
<Knowledge_Base>
{context_text}
{self.sys_context}
</Knowledge_Base>
"""
#{self.sys_context}
        user_query_prompt = f"""[User Query] 
        {user_query}
[INSTRUCTION]
CRITICAL: You MUST generate BOTH <Draft> and <Answer>. Output MUST be enclosed in <Answer> tags.
Strictly adhere to the following logic (Extract to draft first, NEVER affirm user errors).

In <Draft>:
- IF query contains insults/hate speech/profanity: write 'TOXIC'.
- ELSE IF it's social interaction or closing (e.g., hello, no thanks, got it, goodbye): write 'CHITCHAT'.
- ELSE IF unrelated to AORUS/Laptops: write 'OUT_OF_SCOPE'.
- ELSE IF mentions/compares competitors (e.g., ASUS, ROG, MSI): write 'COMPETITOR'.
- ELSE IF the query refers to a specific hardware feature but no model name (BZH/BYH/BXH) is provided: write 'MISSING_MODEL'.
- ELSE IF [User Query] contains WRONG specs: write 'CORRECTION: [Correct Fact from Knowledge Base]'.
- ELSE IF info is missing: write 'No Data'. Do NOT invent.

In <Answer>:
- Language: {L['name']}. No internal tags. ONE paragraph only.
- IF 'TOXIC': Follow style of '{L['toxic_example']}' and STOP.
- IF 'CHITCHAT': Respond naturally as AORUS assistant, then STOP.
- IF 'OUT_OF_SCOPE': Start with '{L['start_none']}', invite more questions, and STOP.
- IF 'COMPETITOR': Follow style of '{L['competitor_example']}'. Highlight AORUS by quoting 2 exact specs from <Knowledge_Base>. Invite more questions and STOP.
- IF 'MISSING_MODEL': Politely ask to clarify model (e.g., '{L['miss_example']}') and STOP.
- IF 'CORRECTION': NEVER agree. Start with '{L['start_info']}[Fact]'. Directly rectify the error and STOP.
- IF 'No Data': Start with '{L['start_oos']} [Subject]...', invite more questions, and STOP.
- IF Yes/No question: Start with '{L['start_yes']}' (Only if user is 100% correct) and STOP.
- OTHERWISE: Directly answer the question and STOP.

<Draft>
(Extract ONLY relevant specs. Use Bullet points. MAX 5 LINES.)
</Draft>
<Answer>
(Answer ONLY what is asked. NO extra specs or details.)
</Answer>
"""
        # C. 使用 llama.cpp 生成，並確保 stream=True
        response_stream = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query_prompt}
            ],
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            top_p=0.5,
            top_k=64,
            stream=True
        )
        # 🌟 D. 狀態機與滑動視窗初始化
        is_answering = False
        full_raw_text = ""
        buffer = ""
        
        # 滑動視窗參數：你可以自己微調這兩個數字來感受打字速度
        window_size = Config.WINDOW_SIZE      # 當籃子累積到幾個字時，觸發輸出
        lookahead_size = Config.LOOKAHEAD_SIZE   # 每次輸出時，強制把最後幾個字「扣留」下來當作下次的上下文

        for chunk in response_stream:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                text_piece = delta["content"]
                #print(text_piece)
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
                            pass
                        else:
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
    bot = AORUSAssistant()
    print("=" * 50)
    print("\nAORUS AI 助理上線中！ AORUS AI Assistant is Online! (enter \'exit\' to exit.)\n")
    print("=" * 50)
    while True:
        query = input("\n[User]: ")
        if query.lower() == 'exit':
            break
        check, query = validate_query(query)
        if not check:
            print(query)
            continue

        print("[Assistant]: ", end="", flush=True)
        for text_piece in bot.generate_stream(query):
            print(text_piece, end="", flush=True)
        print()