import os
import argparse
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from config import Config

def setup(download_judge=False):
    # 基礎資料夾確保存在
    os.makedirs(Config.REASONING_MODEL_PATH, exist_ok=True)
    os.makedirs(Config.EMBEDDING_MODEL_PATH, exist_ok=True)

    print("="*40)
    print("🚀 AORUS Assistant Setup Process")
    print("="*40)

    # ==========================================
    # 1. 下載 Reasoning 模型 (AORUS 生成用)
    # ==========================================
    reasoning_model_name = os.path.basename(Config.REASONING_MODEL_FILE)
    if os.path.exists(Config.REASONING_MODEL_FILE):
        print(f"✅ [1/3] Reasoning Model: {reasoning_model_name} exists.")
    else:
        print(f"📥 [1/3] Downloading Reasoning Model ({reasoning_model_name})...")
        hf_hub_download(
            repo_id=Config.REASONING_MODEL_REPO_ID,
            filename=reasoning_model_name,
            local_dir=Config.REASONING_MODEL_PATH
        )

    # ==========================================
    # 2. 下載 Judge 模型 (DeepEval 裁判用)
    # ==========================================
    if download_judge:
        os.makedirs(Config.JUDGE_MODEL_PATH, exist_ok=True)
        judge_model_name = os.path.basename(Config.JUDGE_MODEL_FILE)
        if os.path.exists(Config.JUDGE_MODEL_FILE):
            print(f"✅ [2/3] Judge Model: {judge_model_name} exists.")
        else:
            print(f"📥 [2/3] Downloading Judge Model ({judge_model_name})...")
            hf_hub_download(
                repo_id=Config.JUDGE_MODEL_REPO_ID,
                filename=judge_model_name,
                local_dir=Config.JUDGE_MODEL_PATH
            )
    else:
        print("⏭️  [2/3] Judge Model: Download skipped (Use --download_judge to enable).")

    # ==========================================
    # 3. 下載 Embedding 模型 (RAG 檢索用)
    # ==========================================
    embedding_config_path = os.path.join(Config.EMBEDDING_MODEL_PATH, "config.json")
    if os.path.exists(embedding_config_path):
        print("✅ [3/3] Embedding Model: Already exists.")
    else:
        print("📥 [3/3] Downloading Multilingual MiniLM...")
        model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        model.save(Config.EMBEDDING_MODEL_PATH)

    print("\n" + "="*40)
    print("✨ Setup Completed.")
    print(f"Models location: {os.path.abspath('./models/')}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AORUS Assistant Model Downloader")
    # 加入開關：預設是 False
    parser.add_argument(
        "--download_judge", 
        action="store_true", 
        help="Download the heavy LLM judge model (default: skip)"
    )
    args = parser.parse_args()

    setup(download_judge=args.download_judge)