import os
from dotenv import load_dotenv

# 自動尋找並載入專案根目錄的 .env 檔案
load_dotenv()

class Config:
    # ==========================================
    # 📂 Paths (路徑與模型識別碼)
    # ==========================================
    # 推論模型 (Reasoning Model)
    REASONING_MODEL_REPO_ID = os.getenv("REASONING_MODEL_REPO_ID", "Qwen/Qwen2.5-3B-Instruct-GGUF")
    REASONING_MODEL_FILE = os.getenv("REASONING_MODEL_FILE", "./models/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf")
    REASONING_MODEL_PATH = os.getenv("REASONING_MODEL_PATH", "./models/qwen2.5-3b/")
    
    # 嵌入模型 (Embedding Model)
    EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "./models/embedding/")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

    # 評審模型
    JUDGE_MODEL_REPO_ID = os.getenv("JUDGE_MODEL_REPO_ID", "divish/M-Prometheus-7B-Q4_K_M-GGUF")
    JUDGE_MODEL_FILE = os.getenv("JUDGE_MODEL_FILE", "./models/judge/M-Prometheus-7B-Q4_K_M.gguf")
    JUDGE_MODEL_PATH = os.getenv("JUDGE_MODEL_PATH", "./models/judge/")
    
    # 專案資料路徑 (Data Paths)
    RAG_DATA_PATH = os.getenv("RAG_DATA_PATH", "./data/rag")
    TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", "./data/test")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")

    # ==========================================
    # ⚙️ Model Config (模型生成參數)
    # ==========================================
    # 注意：這裡必須轉成 float 和 int，否則推論時會報錯
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

    # ==========================================
    # 🌊 Stream Config (串流輸出設定)
    # ==========================================
    # 滑動視窗參數，控制打字機流暢度與繁簡轉換的安全性
    WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "8"))
    LOOKAHEAD_SIZE = int(os.getenv("LOOKAHEAD_SIZE", "4"))
    MAX_CHARS = int(os.getenv("MAX_CHARS", "300"))
    NUM_OF_NORMAL_Q = int(os.getenv("NUM_OF_NORMAL_Q"))