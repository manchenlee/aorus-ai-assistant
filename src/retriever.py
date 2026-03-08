from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import os
import jieba
from rank_bm25 import BM25Okapi
import re
from config import Config
from src.utils import validate_query

os.makedirs(Config.EMBEDDING_MODEL_PATH, exist_ok=True)

JSON_PATH = os.path.join(Config.RAG_DATA_PATH,"specs.json")
SYNONYM_PATH = os.path.join(Config.RAG_DATA_PATH,"synonyms.json")

class AorusRetriever:
    def __init__(self, model_name=Config.EMBEDDING_MODEL_PATH, json_path=JSON_PATH, synonym_path=SYNONYM_PATH, device='cpu'):
        # 1. 初始化模型與變數
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.chunks = []
        self.bm25 = None 
        self.synonym_mapping = {}
        self.replace_dict = {}
        self.pattern = None 

        jieba.add_word("BZH", freq=5000)
        jieba.add_word("BYH", freq=5000)
        jieba.add_word("BXH", freq=5000)

        if os.path.exists(synonym_path):
            print(f"Reading synonyms from {synonym_path}...")
            with open(synonym_path, 'r', encoding='utf-8') as f:
                self.synonym_mapping = json.load(f)
                self._build_regex_pattern()

        else:
            print(f"Warning: Can't find synonyms data at {synonym_path}. Proceeding without synonyms.")

        if os.path.exists(json_path):
            print(f"Reading {json_path} and creating FAISS & BM25 indexes...")
            self.prepare_data(json_path)
            print(f"Created Hybrid Vector+BM25 index. Total {len(self.chunks)} chunks.")
        else:
            print(f"Can't find spec_data {json_path}.")

    def _build_regex_pattern(self):
        for standard_term, synonyms in self.synonym_mapping.items():
            self.replace_dict[standard_term] = standard_term
            for syn in synonyms:
                self.replace_dict[syn] = standard_term
                
        sorted_keys = sorted(self.replace_dict.keys(), key=len, reverse=True)
        
        pattern_parts = []
        for k in sorted_keys:
            escaped_k = re.escape(k)
            if re.fullmatch(r'[A-Za-z0-9]+', k):
                pattern_parts.append(r'\b' + escaped_k + r'\b')
            else:
                pattern_parts.append(escaped_k)
                
        self.pattern = re.compile(r'(' + '|'.join(pattern_parts) + r')')

    def normalize_text(self, text):
        if not self.pattern: 
            return text
            
        return self.pattern.sub(lambda m: self.replace_dict[m.group(0)], text)

    def prepare_data(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.chunks = []
        search_chunks = []

        dimension_data = {}

        # ==========================================
        # 第一階段：組裝【產品完整規格】
        # ==========================================
        for model_name, specs in data.items():
            doc_display_lines = [f"型號完整規格 {model_name}"]
            doc_search_lines = [f"型號完整規格 {model_name}"]
            
            for key, value in specs.items():
                orig_v = str(value)
                
                doc_display_lines.append(f"{key}: {orig_v}")
                
                norm_k = self.normalize_text(key)
                norm_v = self.normalize_text(orig_v)
                doc_search_lines.append(f"{norm_k}: {norm_v}")
                
                if key not in dimension_data:
                    dimension_data[key] = {'display': [], 'search': [], 'norm_key': norm_k}
                
                #model_name = model_name.replace("AORUS MASTER 16 ", "")
                dimension_data[key]['display'].append(f"{model_name}: {orig_v}")
                dimension_data[key]['search'].append(f"{model_name}: {norm_v}")
                
            self.chunks.append("\n".join(doc_display_lines))
            search_chunks.append("\n".join(doc_search_lines))

        # ==========================================
        # 第二階段：組裝【維度比較】 (橫向策略)
        # ==========================================
        for key, dim_dict in dimension_data.items():
            disp_chunk = "\n".join([f"{dim_dict['norm_key']}比較"] + dim_dict['display'])
            self.chunks.append(disp_chunk)
            
            srch_chunk = "\n".join([f"{dim_dict['norm_key']}比較"] + dim_dict['search'])
            search_chunks.append(srch_chunk)

        # ==========================================
        # 第三階段：自動化【型號差異分析】超級 Chunk
        # ==========================================
        diff_metrics = []
        
        for key, dim_dict in dimension_data.items():
            unique_values = set(self.normalize_text(str(v)) for v in dim_dict['display'])
            
            if len(unique_values) > 1:
                diff_metrics.append(key)

        # 組裝超級比較 Chunk
        comparison_title = "AORUS MASTER 16 系列所有型號差異總覽"
        comparison_display = [comparison_title]
        comparison_search = [self.normalize_text(comparison_title)]

        if diff_metrics:
            for model_name in data.keys():
                model_diff_info = [f"● 型號: {model_name}"]
                for metric in diff_metrics:
                    # 從原始 data 中抓取該型號對應的差異數值
                    val = data[model_name].get(metric, "N/A")
                    model_diff_info.append(f"  - {metric}: {val}")
                
                info_str = "\n".join(model_diff_info)
                comparison_display.append(info_str)
                comparison_search.append(self.normalize_text(info_str))
        else:
            comparison_display.append("此系列型號之核心規格基本一致。")

        # 加入最終 Chunks
        final_disp_chunk = "\n\n".join(comparison_display)
        final_srch_chunk = "\n\n".join(comparison_search)
        
        self.chunks.append(final_disp_chunk)
        search_chunks.append(final_srch_chunk)

        # ==========================================
        # 第四階段：建立索引
        # ==========================================
        # 1. 建立 FAISS 向量索引
        embeddings = self.model.encode(search_chunks)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

        # 2. 建立 BM25 關鍵字索引
        tokenized_corpus = [list(jieba.cut(chunk.lower())) for chunk in search_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query, k=3, distance_threshold=21):
        # 1. 查詢字串正規化 (讓同義詞對齊知識庫)
        norm_query = self.normalize_text(query)
        total_chunks = len(self.chunks)

        # =====================================
        # 🌟 檢索路線 A：FAISS Vector Search
        # =====================================
        query_embedding = self.model.encode([norm_query])
        # 注意：為了做 RRF，我們需要知道「所有」文件的排名，所以這裡搜全部 (total_chunks)
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), total_chunks)
        
        # 轉換成排名 (indices[0] 已經是由近到遠排序了)
        vector_ranks = {doc_idx: rank + 1 for rank, doc_idx in enumerate(indices[0])}
        faiss_distances = {doc_idx: dist for dist, doc_idx in zip(distances[0], indices[0])}

        # =====================================
        # 🌟 檢索路線 B：BM25 Keyword Search
        # =====================================
        tokenized_query = list(jieba.cut(norm_query.lower()))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 分數由高到低排序，取得排名
        bm25_ranking = np.argsort(bm25_scores)[::-1]
        bm25_ranks = {doc_idx: rank + 1 for rank, doc_idx in enumerate(bm25_ranking)}

        # =====================================
        # 🌟 終極融合：Reciprocal Rank Fusion (RRF)
        # =====================================
        rrf_k = 60 # RRF 演算法平滑常數
        rrf_scores = {}

        for i in range(total_chunks):
            v_rank = vector_ranks[i]
            b_rank = bm25_ranks[i]
            
            # RRF 公式計算
            rrf_score = (1.0 / (rrf_k + b_rank)) + (1.0 / (rrf_k + v_rank))
            rrf_scores[i] = rrf_score

        # 根據 RRF 分數進行最終排序
        final_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        # 回傳 Top K 的 Chunk

        filtered_chunks = []
        for doc_idx, rrf_score in final_ranking:
            chunk_distance = faiss_distances[doc_idx]
            if distance_threshold is not None and chunk_distance > distance_threshold:
                continue
                
            filtered_chunks.append(self.chunks[doc_idx])
            
            if len(filtered_chunks) == k:
                break

        if not filtered_chunks:
            return []
        return [self.chunks[doc_idx] for doc_idx, score in final_ranking[:k]]

# 測試用
if __name__ == "__main__":
    retriever = AorusRetriever()
    
    # 測試同義詞與混合檢索能力
    test_queries = [
        "AORUS 16 BZH 的顯卡是哪張？", # 測試 "顯卡" 同義詞與型號精確度
        "老黃家的 GPU 有幾 GB VRAM？"  # 測試極端同義詞
    ]
    
    for q in test_queries:
        check, q = validate_query(q)
        if not check :
            print(q)
            continue
        print(f"\n[問題]: {q}")
        results = retriever.retrieve(q, k=2)
        for i, r in enumerate(results):
            print(f"-> Top {i+1}: {r[:50]}...") # 只印前50字觀察