from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import os

spec_mapping = {
    "作業系統": "作業系統 (OS/Operating System)",
    "中央處理器": "中央處理器 (CPU/Processor)",
    "顯示晶片": "顯示晶片 (GPU/Graphics)",
    "顯示器": "顯示器 (Display/Screen/Resolution)",
    "記憶體": "記憶體 (RAM/Memory)",
    "儲存裝置": "儲存裝置 (SSD/Storage/Hard Drive)",
    "鍵盤種類": "鍵盤種類 (Keyboard/Backlight)",
    "連接埠": "連接埠 (I/O Ports/Interface/USB/HDMI)",
    "音效": "音效 (Audio/Speakers/Mic)",
    "通訊": "通訊 (Communication/Wi-Fi/Bluetooth/Ethernet)",
    "視訊鏡頭": "視訊鏡頭 (Webcam/Camera)",
    "安全裝置": "安全裝置 (Security/TPM/Fingerprint)",
    "電池": "電池 (Battery/Capacity)",
    "變壓器": "變壓器 (Power Adapter/Wattage)",
    "尺寸": "尺寸 (Dimensions/Size)",
    "重量": "重量 (Weight/Mass)",
    "顏色": "顏色 (Color/Finish)"
}

content_mapping = {
    # 1. 核心品牌
    "Intel": "Intel (英特爾)",
    "NVIDIA": "NVIDIA (輝達)",
    "Windows": "Windows (微軟)",
    "GIGABYTE": "GIGABYTE (技嘉)",
    
    # 2. 視聽與傳輸技術
    "Dolby Atmos": "Dolby Atmos (杜比全景聲)",
    "Dolby Vision": "Dolby Vision (杜比視界)",
    "Thunderbolt": "Thunderbolt (雷電/雷霆介面)",
    
    # 3. 硬體組件俗稱
    "SSD": "SSD (固態硬碟)",
    "WIFI": "Wi-Fi (無線網路)",
    "Bluetooth": "Bluetooth (藍牙)",
    "Backlit": "Backlit (背光)",
    "Webcam": "Webcam (視訊鏡頭/網路攝影機)",
    "LAN": "LAN (有線網路)",
    "Low Blue Light": "Low Blue Light (低藍光/抗藍光)",
    "DisplayHDR": "DisplayHDR (支援HDR)",
    "OLED": "OLED (OLED面板)",
    "顯示卡": "顯示晶片 (GPU/Graphics)",
    "顯卡": "顯示晶片 (GPU/Graphics)",
    "VRAM": "顯示晶片 (GPU/Graphics) GDDR7", 
    "GDDR7": "顯示晶片 (GPU/Graphics) GDDR7",
    
    # 4. 螢幕解析度
    "2560×1600": "2560×1600 (2K/2.5K解析度)"
}

os.makedirs('models/embedding', exist_ok=True)

class AorusRetriever:
    def __init__(self, model_name='./models/embedding', json_path='data/specs.json', device='cpu'):
        # 1. 載入支援 50+ 語言的跨語言模型
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.chunks = []

        if os.path.exists(json_path):
            print(f"Reading {json_path} and creating FAISS vector index...")
            self.prepare_data(json_path)
            print(f"Created FAISS vector index. Total {len(self.chunks)} chunks.")
        else:
            print(f"Can't find spec_data {json_path}.")

    def prepare_data(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 建立兩個暫存的字典，用來分別組裝「策略一」與「策略二」的資料
        dimension_data = {}  # 用來存裝策略一：{ "顯示晶片": ["BXH: ...", "BYH: ..."] }
        document_data = {}   # 用來存裝策略二：{ "BXH": ["處理器: ...", "顯示晶片: ..."] }

        # 第一階段：解析 JSON 並進行內容清洗 (Preprocessing)
        for model_name, specs in data.items():
            document_data[model_name] = [] # 初始化這台筆電的清單
            
            for key, value in specs.items():
                display_key = spec_mapping.get(key, key)
                
                # 語意增強與俗稱替換
                processed_value = value
                for eng_keyword, chi_keyword in content_mapping.items():
                    if eng_keyword in processed_value:
                        processed_value = processed_value.replace(eng_keyword, chi_keyword)
                
                # --- 收集給策略一 (橫向打包) ---
                if display_key not in dimension_data:
                    dimension_data[display_key] = []
                dimension_data[display_key].append(f"{model_name}: {processed_value}")
                
                # --- 收集給策略二 (整機打包) ---
                document_data[model_name].append(f"{display_key}: {processed_value}")

        # 第二階段：組裝最終的 Chunks

        # 1. 產生策略一的 Chunks (每個規格一個 Chunk，裡面包含所有型號)
        for dim_key, dim_values in dimension_data.items():
            chunk_lines = [f"【{dim_key} 規格比較】"] + dim_values
            chunk_str = "\n".join(chunk_lines)
            self.chunks.append(chunk_str)

        # 2. 產生策略二的 Chunks (每台筆電一個 Chunk，裡面包含所有規格)
        for model, doc_values in document_data.items():
            chunk_lines = [f"【產品完整規格】{model}"] + doc_values
            chunk_str = "\n".join(chunk_lines)
            self.chunks.append(chunk_str)

        # 3. 向量化 (Embedding)
        embeddings = self.model.encode(self.chunks)
        
        # 4. 建立 FAISS 索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def retrieve(self, query, k=3):
        # 將使用者的問題（不論中英）轉為向量，並在資料庫中找最接近的 k 個片段
        for keyword, standard_term in content_mapping.items():
            if keyword in query:
                query = query.replace(keyword, standard_term)
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        return [self.chunks[i] for i in indices[0]]

# 測試用
if __name__ == "__main__":
    retriever = AorusRetriever()
    #retriever.prepare_data("data/specs.json")
    # 測試中英混合提問
    results = retriever.retrieve("AORUS 16 AM6H 的 CPU processor 是哪顆？")
    for r in results:
        print(f"檢索到相關資料: {r}")