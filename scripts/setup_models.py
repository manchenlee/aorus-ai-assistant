import os
from huggingface_hub import hf_hub_download, snapshot_download
from sentence_transformers import SentenceTransformer
from config import Config

def setup():
    os.makedirs(Config.REASONING_MODEL_PATH, exist_ok=True)
    os.makedirs(Config.EMBEDDING_MODEL_PATH, exist_ok=True)

    print("Start Downloading...")

    reasoning_model_name = os.path.basename(Config.REASONING_MODEL_FILE)
    print(f"\n[1/2] Downloading {reasoning_model_name}...")
    hf_hub_download(
        repo_id=Config.REASONING_MODEL_REPO_ID,
        filename=reasoning_model_name,
        local_dir=Config.REASONING_MODEL_FILE
    )

    # 2. 下載 Embedding 模型 (整個資料夾)
    print("\n[2/2] Downloading Multilingual MiniLM...")
    # 方法：先加載到記憶體再儲存，或是用 snapshot_download
    model_name = Config.EMBEDDING_MODEL_NAME
    model = SentenceTransformer(model_name)
    model.save(Config.EMBEDDING_MODEL_PATH)

    print("\nCompleted.")
    print(f"Path: ./models/")

if __name__ == "__main__":
    setup()