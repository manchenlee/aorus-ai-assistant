import os
import json
import csv
from google import genai
from config import Config

def generate_aorus_test_data():

    specs_file_path = os.path.join(Config.RAG_DATA_PATH, "specs.json")
    warning_file_path = os.path.join(Config.RAG_DATA_PATH, "disclaimers.txt")
    
    try:
        with open(specs_file_path, "r", encoding="utf-8") as f:
            specs_content = f.read()
            
        with open(warning_file_path, "r", encoding="utf-8") as f:
            warning_content = f.read()
    except FileNotFoundError as e:
        print(f"Can't find {e.filename}!")
        return
    
    if not Config.GEMINI_API_KEY:
        print("Can't find API key!")
        return

    client = genai.Client(api_key=Config.GEMINI_API_KEY)
    
    print("Call Gemini to generate 100 test cases...")

    prompt = f"""
    你現在是一位專業的 AI 測試工程師（QA Engineer），負責測試客服機器人的極限。
    請幫我生成 100 題針對「AORUS 16 吋電競筆電（如 BZH, BXH, BYH）」的客服測試資料。

    【🚨 絕對嚴格限制：基於真實資料 🚨】
    1. 規格題：請務必「完全根據」下方的【規格表內容】來設計問題與解答，絕對不可自行捏造規格。
    2. 免責/保固題：請務必參考下方的【免責聲明內容】，當遇到重量、尺寸、擴充等陷阱題時，解答中需提及需要觸發特定的聲明。
    3. 背景知識映射：型號 BZH、BYH 和 BXH 皆屬於「AORUS MASTER 16」系列，內部代號為「AM6H」。請在部分題目中「刻意」使用「AORUS MASTER 16」或「AM6H」來提問，測試機器人是否能自動關聯這三款型號並給出綜合解答。

    === 規格表內容開始 ===
    {specs_content}
    === 規格表內容結束 ===

    === 免責聲明內容開始 ===
    {warning_content}
    === 免責聲明內容結束 ===

    【🗣️ 提問口吻與模糊性要求 (非常重要) 🗣️】
    真實的客人不會照著規格表唸！這 100 題中，請保持 20% 的題目是「模糊、口語化、不精準」的俗稱問法。
    - 不要總是問「Memory 規格」，改問「RAM 給多大？」、「記憶體可以自己擴充嗎？」
    - 不要總是問「Storage 容量」，改問「硬碟有多大？」、「可以存很多遊戲嗎？」
    - 不要總是問「I/O Ports」，改問「旁邊有幾個 USB 孔？」、「可以外接雙螢幕嗎？」
    - 稱呼筆電時，多用「AM6H系列」、「BZH」、「AORUS BYH」等可以識別產品的簡稱，少用完整的「AORUS MASTER 16 BZH」。
    - 英文提問也可以口語一點，例如用 "rig"、"specs"、"storage space" 來取代死板的專有名詞。

    【語言比例要求】
    這 100 題必須混合以下三種語言設定，請隨機分配：
    - 純中文 (Traditional Chinese)
    - 純英文 (English)
    - 中英混合 (Chinglish, 例如: "這台的 VRAM 給到幾 G？")

    【題型與比例要求 (共 100 題)】
    請嚴格遵守以下題型與比例配置，並在 Category 欄位填入對應的英文標籤：
    1. What (Fact Extraction) - 20題：精準抓出特定規格數值。例如：「電池充飽大概幾 Wh？」
    2. Yes/No (Positive Verification) - 15題：比對事實並回答 Yes。例如：「這台是 OLED 螢幕對吧？」
    3. Which (Comparison) - 15題：跨規格統整與比較。例如：「AM6H 這幾台，哪一台的顯卡最頂？」
    4. Summarize (Summarization) - 10題：長文本生成。例如：「幫我列出這台所有的插孔。」
    5. Negative Yes/No (Negative Verification) - 15題：勇敢說 No 並給正確答案。例如：「這台記憶體還是舊的 DDR4 嗎？」
    6. Negative Which (Adversarial Reasoning) - 10題：破解錯誤前提。例如：「既然這台沒獨顯，玩 3A 大作會卡嗎？」(糾正:有獨顯)
    7. Disclaimer (Disclaimer) - 5題：觸發保固、尺寸、重量等免責聲明。例如：「自己拆殼加裝 SSD 還算保固內嗎？」
    8. Out of bound (Out-of-Scope) - 10題：超綱或閒聊。例如：「隔壁棚的 ROG 比較好還是你們家好？」

    【輸出格式限制】
    請「嚴格」輸出為一個 JSON Array，裡面包含 100 個 JSON Object。
    每個 Object 必須包含以下 5 個 Key (請完全依照這個命名)：
    - "ID": 數字 (1 到 100)
    - "Category": 題型標籤 (填寫上方括號內的英文，如 Fact Extraction)
    - "Language": 語言類型 (填寫: 純中文 / 純英文 / 中英混合)
    - "Question": 測試問題內容
    - "Expected_Answer": 預期機器人應該回答的標準答案（語言需與問題相符，免責題須符合免責聲明內容，若題目詢問 AM6H 系列則需綜合 BZH/BXH/BYH 回答）

    請直接輸出 JSON，不要包含任何 ```json 的 Markdown 標籤，也不要加上任何解釋廢話。
    """

    try:
        # 呼叫 Gemini 2.5 Flash
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            # 強制要求模型輸出 JSON 格式 (防呆機制)
            config={
                "response_mime_type": "application/json",
            }
        )
        
        # 解析 JSON 資料
        raw_text = response.text.strip()
        test_data = json.loads(raw_text)

        # 確保儲存的路徑存在
        os.makedirs(Config.TEST_DATA_PATH, exist_ok=True)
        # 這次我們存成 CSV 檔案
        output_file = os.path.join(Config.TEST_DATA_PATH, "aorus_test_cases.csv")

        # 將 JSON 資料寫入 CSV
        # 💡 使用 utf-8-sig 讓 Excel 打開時不會變成亂碼
        with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
            # 定義 CSV 的欄位名稱
            fieldnames = ["ID", "Category", "Language", "Question", "Expected_Answer"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # 寫入標頭 (Header)
            writer.writeheader()
            
            # 將 100 筆資料逐行寫入
            for row in test_data:
                # 簡單過濾，確保模型吐出來的 key 是正確的
                filtered_row = {k: row.get(k, "") for k in fieldnames}
                writer.writerow(filtered_row)
            
        print(f"Successfully generate {len(test_data)} test cases!")
        print(f"CSV file saved: {output_file}")

    except json.JSONDecodeError:
        print("Json parsing failed.")
        print("Preview the return value:", response.text[:500])
    except Exception as e:
        print(f"Error while generating data: {e}")

if __name__ == "__main__":
    generate_aorus_test_data()