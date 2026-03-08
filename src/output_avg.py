import csv

def calculate_performance_metrics(file_path):
    ttft_values = []
    tps_values = []

    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            # 讀取 CSV，自動將首行視為 header
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    # 確保欄位名稱與你寫入時的一致 (通常 csv.writer 不帶 header，請確認你的 CSV 是否有標題行)
                    # 如果你的 CSV 沒有標題行，請改用 row[4] 和 row[5] 的方式存取
                    ttft_values.append(float(row['ttft']))
                    tps_values.append(float(row['tps']))
                except (ValueError, KeyError):
                    # 略過可能的標題行重複或空值
                    continue

        if not ttft_values:
            print("No valid data found in CSV.")
            return

        avg_ttft = sum(ttft_values) / len(ttft_values)
        avg_tps = sum(tps_values) / len(tps_values)

        print("-" * 40)
        print(f"📊 CSV Metrics Summary:")
        print(f"   Processed: {len(ttft_values)} rows")
        print(f"   Avg TTFT:  {avg_ttft:.3f} seconds")
        print(f"   Avg TPS:   {avg_tps:.2f} tokens/sec")
        print("-" * 40)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

# 使用方法：替換成你的 CSV 檔名
calculate_performance_metrics("data\\test\\aorus_test_results_20260307_230413.csv")