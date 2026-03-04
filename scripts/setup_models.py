import os
from huggingface_hub import hf_hub_download, snapshot_download
from sentence_transformers import SentenceTransformer

def setup():
    base_dir = "models"
    os.makedirs(f"{base_dir}/qwen2.5-3b", exist_ok=True)
    os.makedirs(f"{base_dir}/embedding", exist_ok=True)

    print("Start Downloading...")

    # 1. 下載 Qwen LLM (單一 GGUF 檔案)
    # 我們選擇 Q4_K_M 量化版本，兼顧性能與 4GB VRAM 限制
    print("\n[1/2] Downloading Qwen2.5-3B-Instruct GGUF...")
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        local_dir=f"{base_dir}/qwen2.5-3b"
    )

    # 2. 下載 Embedding 模型 (整個資料夾)
    print("\n[2/2] Downloading Multilingual MiniLM...")
    # 方法：先加載到記憶體再儲存，或是用 snapshot_download
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer(model_name)
    model.save(f"{base_dir}/embedding")

    print("\nCompleted.")
    print(f"Path: ./{base_dir}/")

if __name__ == "__main__":
    setup()