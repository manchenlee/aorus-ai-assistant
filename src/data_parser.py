import requests
from bs4 import BeautifulSoup
import json
import os

extracted_disclaimers = set()

def clean_spec_value(val, title):
    if not val:
        return ""
        
    parts = val.split('; ')
    kept_parts = []
    
    for i in range(len(parts)):
        p = parts[i].strip()
        if not p: continue
        
        is_disclaimer = False
        if p.startswith('**'):
            is_disclaimer = True
        elif p.startswith('*') and not p.startswith('* '):
            is_disclaimer = True
        
        if is_disclaimer:
            pall = '; '.join(parts[i:]).strip()
            extracted_disclaimers.add(f"{title}: {pall}")
            break 
        else:
            kept_parts.append(p)
            
    return '; '.join(kept_parts)

def parse_aorus_comparison(url):
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    res = requests.get(url, headers=headers)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, 'html.parser')

    # 1. 抓取三種型號名稱
    # 位於 title
    laptop_title = soup.select_one(".model-base-info-subtitle").get_text(strip=True)
    laptop_names = laptop_title.split(' / ')
    print(laptop_names)

    # 2. 抓取比較項目名稱 (e.g. 中央處理器)
    # 位於 .spec-column -> .multiple-title
    spec_titles = []
    spec_column = soup.select_one(".spec-column")
    if spec_column:
        titles = spec_column.select(".multiple-title")
        spec_titles = [t.get_text(strip=True) for t in titles]
    print(spec_titles)

    # 3. 抓取所有型號的內容
    # swiper-slide 裡面的 spec-item-list span
    all_content_spans = soup.select(".swiper-wrapper .swiper-slide .spec-item-list span")
    
    # 4. 核心邏輯：將一維陣列轉換為結構化 JSON
    # 假設有 N 個型號，每 N 個 value 對應一個 spec_title
    num_specs = len(spec_titles)
    final_data = {name: {} for name in laptop_names}

    for i, laptop_name in enumerate(laptop_names):
        for j, title in enumerate(spec_titles):
            idx = i * num_specs + j
            if idx < len(all_content_spans):
                raw_val = all_content_spans[idx].get_text(separator="; ", strip=True)
                clean_val = clean_spec_value(raw_val, title)
                final_data[laptop_name][title] = clean_val

    notes = soup.select(".warning-note .note-item")
    bottom_footnotes = []

    for note in notes:
        full_text = note.get_text(strip=True)
        segments = [s.strip() for s in full_text.split('*') if s.strip()]
        for seg in segments:
            bottom_footnotes.append(f"* {seg}")
    
    with open("data/warning_context.txt", "w", encoding="utf-8") as f:
        f.write("【系統應答守則：規格特定免責聲明】\n")
        f.write("當你回答以下特定硬體項目時，請務必參考括號內的原廠說明進行回覆：\n")
        
        for disc in sorted(list(extracted_disclaimers)):
            f.write(f"- {disc}\n")
            
        f.write("\n【通用免責聲明】\n")
        for note in bottom_footnotes:
            if note:
                f.write(f"- {note}\n")

    print("Output data/warning_context.txt")

    return final_data

# 執行並儲存
url = "https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp"
data = parse_aorus_comparison(url)
with open("data/specs.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)