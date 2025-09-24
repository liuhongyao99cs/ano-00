import threading
import time
import torch
from sklearn.svm import OneClassSVM
import os
import numpy as np

# WiKV semantic coding

class WiKV_Controller:
    def __init__(self, args, shape, dtype=torch.float32, threshold=0.25, device='cpu'):

        self.args = args
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.threshold = threshold

        self.tensor = torch.zeros(shape, dtype=dtype, device=device)

        self.filled_count = 0
        self.total_elements = self.tensor.numel()


        self.lock = threading.Lock()          # 保护 tensor 和 filled_count
        self.stop_event = threading.Event()   # 用于停止填充线程
        self.ready_event = threading.Event()  # 用于 probe 等待阈值
        self.full_event = threading.Event()

    def start_kv_fill(self):
        self.fill_thread = threading.Thread(target=self._fill_worker, daemon=True)
        self.fill_thread.start()

    def _fill_worker(self):
        # =====================
        # KV cache loading process
        # =====================

        idx = 0
        step = 0.05
        total = self.total_elements
        tensor_flat = self.tensor.view(-1)  # 展平，方便逐元素赋值

        while idx < total:

            with self.lock:
                if idx < total:
                    tensor_flat[idx] = idx * 0.01  # 示例赋值逻辑
                    self.filled_count += 1
                    idx += 1

                    # 检查是否达到阈值，触发 ready_event
                    if self.filled_count / self.total_elements >= self.threshold:  # 可设为动态阈值
                        self.threshold += step
                        self.ready_event.set()
        if idx == total:
            self.full_event.set()
            self.ready_event.set()

        print("✅ 填充线程完成或被停止。")

    def probe(self, target_device='cuda:0'):
        
        print(f"🔍 等待填充比例达到 {self.threshold}...")
        if not self.full_event.is_set():
            # 等待达到阈值（阻塞直到 ready_event 被 set）
            self.ready_event.wait()
            self.ready_event.clear()
    
            with self.lock:
                tensor_gpu = self.tensor.to(target_device).clone()
                print(f"🎯 已在比例 {self.filled_count/self.total_elements:.2f} 时锁定并复制到 {target_device}")
        
        return tensor_gpu

            
        

    def get_progress(self):
        """获取当前填充比例（调试用）"""
        with self.lock:
            return self.filled_count / self.total_elements


    def boundary(self):
        # =======================
        # A SVM learn a boundary with full attention
        # =======================
        datasets = ['nqa', 'tqa', 'longchat', 'gov_report', 'hotpotqa']
        k_coverage = []
        entro = []
        for data in datasets:
            for session in range(10):
                file_path = os.path.join(self.args.save_metric_dir, f"{data}/k_top_{session}.pt")
                k_top = torch.load(file_path)
                k_coverage.extend(k_top)
            
            for session in range(10):
                file_path = os.path.join(self.args.save_metric_dir, f"{data}/entro_{session}.pt")
                en = torch.load(file_path)
                entro.extend(k_top)
        print(len(entro))
        print(len(k_coverage))

        data = np.column_stack((k_coverage, entro))
        model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.02)
        model.fit(data)
        print(model)
        '''
        pred = []
        for i in range(len(k_coverage)):
            predx = model.decision_function([[k_coverage[i],entro[i]]])
            pred.append(torch.tensor(predx[0]).to(torch.float16).item())

        print(pred)
        '''