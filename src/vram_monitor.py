import pynvml
import threading
import time

class VRAMMonitor:
    def __init__(self, device_index=0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        
        # 1. 在初始化時就抓取「當前背景佔用」
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.initial_vram_mb = info.used / (1024 * 1024)
        
        self.peak_vram_mb = self.initial_vram_mb
        self.is_running = False
        self.limit_mb = 4096  # 4GB 專案限制

    def _monitor(self):
        while self.is_running:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            used_mb = info.used / (1024 * 1024)
            if used_mb > self.peak_vram_mb:
                self.peak_vram_mb = used_mb
            time.sleep(0.1)

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        print(f"🔍 VRAM Monitor started... (Baseline: {self.initial_vram_mb:.2f} MB)")

    def stop(self, log):
        self.is_running = False
        time.sleep(0.2) 
        
        net_usage_mb = self.peak_vram_mb - self.initial_vram_mb
        
        with open(log, "a", encoding="utf-8") as f:
            print("\n" + "="*40, file=f)
            print(f"VRAM Usage Report", file=f)
            print(f"   Baseline VRAM (System): {self.initial_vram_mb:.2f} MB", file=f)
            print(f"   Peak VRAM (Total):     {self.peak_vram_mb:.2f} MB", file=f)
            print(f"   Net Project Usage:     {net_usage_mb:.2f} MB", file=f)
        
            if net_usage_mb > self.limit_mb:
                print(f"   WARNING: Project Net Usage exceeded {self.limit_mb} MB!", file=f)
            else:
                print(f"   PASS: Net Usage within {self.limit_mb} MB limit.", file=f)
            print("="*40)
        pynvml.nvmlShutdown()