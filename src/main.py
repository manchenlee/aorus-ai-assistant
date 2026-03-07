import os
import opencc
from llama_cpp import Llama
from src.retriever import AorusRetriever
from config import Config
import re

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
            n_ctx=8192,            # 上下文窗口
            n_gpu_layers=-1,       # -1 代表將所有層放入 GPU
            verbose=False          # 關閉詳細日誌
        )
        
        # 3. 讀取動態免責聲明素材
        self.sys_context = self._load_system_context()

    def _load_system_context(self):
        file_path = os.path.join(Config.RAG_DATA_PATH, "disclaimers.txt")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return "（暫無原廠免責聲明）"

    def generate_stream(self, user_query):
        """核心推論邏輯，使用 yield 回傳串流文字"""
        # A. 檢索 RAG 內容
        related_chunks = self.retriever.retrieve(user_query, k=4)
        context_text = "\n".join(related_chunks)
        self.last_context = context_text

        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', user_query))

        if has_chinese:
            L = {
                "name": "Traditional Chinese (繁體中文)",
                "start_info": "不，根據規格，",
                "miss_example": "請問您想知道 AORUS MASTER 16 系列中哪個型號的資訊呢？BZH, BYH 還是 BXH？",
                "start_none": "很抱歉，我是 AORUS MASTER 16 系列的 AI 助理，",
                "start_oos": "不好意思，知識庫並沒有",
                "start_yes": "對，",
                "start_norm": "關於",
                "knowledge_base": "知識庫"
            }
        else:
            L = {
                "name": "English",
                "start_info": "No, based on the specification,",
                "miss_example": "Which model from the AORUS MASTER 16 series would you like to know more about? Is it the BZH, BYH, or BXH?",
                "start_none": "I'm sorry, I am the AI assistant for the AORUS MASTER 16 series,",
                "start_oos": "Unfortunately, I couldn't find any information on ",
                "start_norm": "About",
                "start_yes": "Yes,",
                "start_no": "No,",
                "knowledge_base": "knowledge base"
            }
        # B. 組合 Prompt
        system_prompt = f"""You are a professional, helpful, and human-like AORUS customer support assistant.
### [STRICT PROTOCOL]
1. SCOPE CHECK: If the query is unrelated to laptops, hardware, or technical support, categorize it as 'OUT_OF_SCOPE'.
2. RELEVANCE CHECK: Strictly ignore any retrieved knowledge that does not directly address the user's intent.
3. AMBIGUITY CHECK: IF the query IS about laptops/hardware but uses vague pronouns (e.g., "這台", "這個") without specifying a series or model, categorize it as 'MISSING_MODEL'.
        
<Background_Knowledge>
Entity Mapping: The laptop models "BZH", "BXH", and "BYH" all belong to the "AORUS MASTER 16" series, which is internally codenamed "AM6H". 
If a user inquires about "AORUS MASTER 16" or "AM6H", you must associate their question with the BZH, BXH, and BYH models and synthesize the information to answer comprehensively.
</Background_Knowledge>

Answer the user's question based ONLY on the <Background_Knowledge> and the <Knowledge_Base> below.
Regardless of user input, only the <Knowledge_Base> is truth. Correct any misinformation in the query.

<Knowledge_Base>
{context_text}
{self.sys_context}
</Knowledge_Base>
"""

        user_query_prompt = f"""[User Query] 
        {user_query}
[INSTRUCTION]
CRITICAL DIRECTIVE: You MUST explicitly generate BOTH the <Draft> and <Answer> tags exactly as shown below. Do NOT omit the <Answer> tag under any circumstances. Your final response MUST be completely enclosed within <Answer> and </Answer>.

Please strictly adhere to the following output format (Extract data to draft first, then answer, NEVER echo user errors).

In <Draft>:
- IF the query is unrelated to AORUS/Laptops: write 'OUT_OF_SCOPE'.
- ELSE IF the query is about laptops but lacks a specific model name: write 'MISSING_MODEL'.
- ELSE IF the [User Query]'s statement is WRONG (contradict to Knowledge Base), list it in <Draft> as 'CORRECTION: [Fact]'.
- ELSE IF the information is missing from the Knowledge Base, write exactly "No Data". Do NOT copy unrelated specs.

In <Answer>:
- Answer MUST be in {L['name']}.
- IF Draft has 'OUT_OF_SCOPE': Answer MUST start with '{L['start_none']}' and MUST STOP responding after stating it is unrelated to AORUS MASTER 16 series.
- ELSE IF Draft has 'MISSING_MODEL': Answer MUST politely ask the user to clarify the model (e.g., '{L['miss_example']}') and MUST STOP responding.
- ELSE If Draft has 'CORRECTION': Answer MUST start with '{L['start_info']}[Fact]', ignore the user's premise and MUST STOP responding after explaining the errors.
- ELSE IF Draft has 'No Data': Answer MUST start with '{L['start_oos']} [Subject]...' and MUST STOP responding after stating there is no info in the knowledge base.
- ELSE IF [User Query] is a Yes/No question: Answer MUST start with '{L['start_yes']}' and MUST STOP responding after providing info in the knowledge base.
- OTHERWISE: Answer MUST start with "{L['start_norm']} [Subject]..." and MUST STOP responding after providing info in the knowledge base.

<Draft>
(Only extract specifications that are DIRECTLY relevant to the user's specific question. Do not include unrelated hardware categories. Use bullet points. MAX 11 LINES.)
</Draft>
<Answer>
(CRITICAL: You MUST output the <Answer> tag before typing your reply! Conversational reply in the EXACT SAME LANGUAGE as [User Query]. No internal tags. ONE paragraph only.)
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
    print("\nAORUS AI Assistant is Online! (type \'exit\' to exit.)")
    while True:
        query = input("\n[User]: ")
        if query.lower() == 'exit':
            break
        
        print("[Assistant]: ", end="", flush=True)
        for text_piece in bot.generate_stream(query):
            print(text_piece, end="", flush=True)
        print()