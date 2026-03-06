import os
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from config import Config

def setup():
    # 確保所有需要的基礎資料夾都存在
    os.makedirs(Config.REASONING_MODEL_PATH, exist_ok=True)
    os.makedirs(Config.EMBEDDING_MODEL_PATH, exist_ok=True)
    os.makedirs(Config.JUDGE_MODEL_PATH, exist_ok=True)

    print("Start Downloading...")

    # ==========================================
    # 1. 下載 Reasoning 模型 (AORUS 生成用)
    # ==========================================
    reasoning_model_name = os.path.basename(Config.REASONING_MODEL_FILE)
    if os.path.exists(Config.REASONING_MODEL_FILE):
        print(f"\n[1/3] {reasoning_model_name} exists. skip downloading.")
    else:
        print(f"\n[1/3] Downloading {reasoning_model_name}...")
        hf_hub_download(
            repo_id=Config.REASONING_MODEL_REPO_ID,
            filename=reasoning_model_name,
            local_dir=Config.REASONING_MODEL_PATH
        )

    # ==========================================
    # 2. 下載 Judge 模型 (TruLens 裁判用)
    # ==========================================
    judge_model_name = os.path.basename(Config.JUDGE_MODEL_FILE)
    if os.path.exists(Config.JUDGE_MODEL_FILE):
        print(f"\n[2/3] {judge_model_name} exists. skip downloading.")
    else:
        print(f"\n[2/3] Downloading Judge Model ({judge_model_name})...")
        hf_hub_download(
            repo_id=Config.JUDGE_MODEL_REPO_ID,
            filename=judge_model_name,
            local_dir=Config.JUDGE_MODEL_PATH
        )

    # ==========================================
    # 3. 下載 Embedding 模型 (RAG 檢索用)
    # ==========================================
    print("\n[3/3] Checking Multilingual MiniLM...")
    # 檢查目錄下是否有 config.json，這是判斷 SentenceTransformer 是否下載完整的最穩方式
    embedding_config_path = os.path.join(Config.EMBEDDING_MODEL_PATH, "config.json")
    
    if os.path.exists(embedding_config_path):
        print("Embedding exists. skip downloading.")
    else:
        print("Downloading Multilingual MiniLM...")
        model_name = Config.EMBEDDING_MODEL_NAME
        model = SentenceTransformer(model_name)
        model.save(Config.EMBEDDING_MODEL_PATH)

    print("\nCompleted.")
    print("Path: ./models/")

if __name__ == "__main__":
    setup()