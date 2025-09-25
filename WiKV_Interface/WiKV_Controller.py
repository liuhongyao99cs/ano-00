import threading
import time
import torch
import sys
from sklearn.svm import OneClassSVM
import os
import numpy as np

from src import *

# WiKV semantic coding

class WiKV_Controller:

    def __init__(self, model, tokenizer, args, shape, dtype=torch.float32, threshold=0.25, device='cpu'):

        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.threshold = threshold
        self.tensor = torch.zeros(shape, dtype=dtype, device=device)
        self.step = 0
        self.num_sample = int(10)

        self.filled_count = 0
        self.total_elements = self.tensor.numel()


        self.lock = threading.Lock()          # 保护 tensor 和 filled_count
        self.stop_event = threading.Event()   # 用于停止填充线程
        self.ready_event = threading.Event()  # 用于 probe 等待阈值
        self.full_event = threading.Event()

    def kv_pool_initialize(self, kv):
        # cpu kv pool to handle the streaming data
        self.kv_pool = torch.zeros_like(kv).to('cpu')
        self.total_elements = self.kv_pool.numel()


    def start_kv_fill(self, semantic_seq, bw_trace, kv_gpu, code_size):
        self.fill_thread = threading.Thread(
            target=self._fill_worker, 
            args=(semantic_seq, bw_trace, kv_gpu, code_size),
            daemon=True
        )
        self.fill_thread.start()

    def _fill_worker(self, semantic_seq, bw_trace, kv_gpu, code_size):

        # =====================
        # KV cache loading process
        # =====================

        # semantic_seq is the final order of streaming
        # bw_trace record the throughput of each 0.1s, (Mbps)
        # kv_gpu is the quantized kv on gpu, we use it to fill the self.kv_pool in cpu
        # code_size is the file after encoding (MB)

        idx = 0

        t = 0.1
        total = semantic_seq.shape[0]
        self.filled_count = 0

        while self.filled_count < total:

            st = time.perf_counter()

            with self.lock:
                    
                    # the ratio : streamed by bw / code_size

                    propor = bw_trace[idx] * t / (code_size * 8)
                    streamed_num = int(propor * total)
                    
                    idx_range = slice(self.filled_count, min(self.filled_count + streamed_num, total))
                    indices = semantic_seq[idx_range]
                    print(self.kv_pool.shape)
                    # 将 GPU tensor 的对应部分复制到 CPU，然后赋值给 CPU tensor
                    self.kv_pool[
                        indices[:, 0], :, 
                        indices[:, 1], 
                        indices[:, 2], :
                    ] = kv_gpu[
                        indices[:, 0], :, 
                        indices[:, 1], 
                        indices[:, 2], :
                    ].cpu()  # 关键：添加 .cpu() 将数据从 GPU 移到 CPU
                    
                    #self.kv_pool[semantic_seq[self.filled_count:min(self.filled_count+streamed_num,total),0],:,semantic_seq[self.filled_count:min(self.filled_count+streamed_num,total),1],semantic_seq[self.filled_count:min(self.filled_count+streamed_num,total),1],:] = kv_gpu[semantic_seq[self.filled_count:min(self.filled_count+streamed_num,total),0],:,semantic_seq[self.filled_count:min(self.filled_count+streamed_num,total),1],semantic_seq[self.filled_count:min(self.filled_count+streamed_num,total),1],:]
                    self.filled_count = min(self.filled_count + streamed_num, total)
                    idx += 1

                   
                    if self.filled_count / total >= self.threshold:
                        self.threshold += self.step
                        self.ready_event.set()

            while True:
                end = time.perf_counter()
                elapsed_time = end - st
                
                if elapsed_time >= 0.1:
                    break

        
        if self.filled_count == total:
            self.full_event.set()
            self.ready_event.set()

        print("✅ Fill the KV buffer thread is completed")

    def probe(self, target_device='cuda:0'):
        
        print(f"🔍 等待填充比例达到 {self.threshold}...")
        if not self.full_event.is_set():
            # 等待达到阈值（阻塞直到 ready_event 被 set）
            self.ready_event.wait()
            self.ready_event.clear()
    
            with self.lock:
                tensor_gpu = self.kv_pool.to(target_device).clone()
                self.step = 0.1
                #print(f"🎯 Filling ratio {self.filled_count/self.total_elements:.2f} locked and copy KV cache to {target_device}")
        
        return tensor_gpu

    def get_progress(self):
        """get the proportion in the KV pool"""
        with self.lock:
            return self.filled_count / self.total_elements

    def Metric(self):
        
        # =====================
        # Gather metrics of tokens with full attention
        # =======================


        datasets = ['nqa', 'tqa', 'longchat', 'gov_report', 'hotpotqa']
        for datax in datasets:
            #datax = 'longchat'
            if not os.path.exists(f'/home/hoongyao/data/test_data/{datax}.jsonl'):
                    print("Load test data first...")
                    sys.exit(1)
            
            data = load_testcases(f'/home/hoongyao/data/test_data/{datax}.jsonl')
            for session_id in range(self.num_sample):
                
                if datax in ['longchat', 'tqa', 'nqa']:
                    input_text = data[session_id]['prompt'] 
                else:
                    input_text = data[session_id]['context']
                    
                inputs_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
                input_ids = inputs_ids['input_ids']
                attention_mask = inputs_ids['attention_mask']
                seq_len = input_ids.shape[1]

                
                if not os.path.exists(f"/home/hoongyao/data/KV_cache/{self.args.model}/{datax}/raw_kv_{session_id}.pt"):
                    print("Compute the KV cache first...")
                    sys.exit(1)

                raw_kv = torch.load(f"/home/hoongyao/data/KV_cache/{self.args.model}/{datax}/raw_kv_{session_id}.pt")

                kv = tensor_to_tuple(raw_kv)
                # generate logit scores through model.generate
                generated = self.model.generate(input_ids, past_key_values = kv, max_new_tokens = 40, return_dict_in_generate=True, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id, attention_mask=attention_mask, output_scores=True)
                prediction = self.tokenizer.decode(generated.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
                #print(prediction)
                
                print(f"Dumping the metrics for data {datax} sample {session_id}...")
                k_top = []
                entro = []
                for k in range(len(generated.scores)):
                    k_top.append(K_coverage(generated.scores[k]).item())
                    entro.append(entropy(generated.scores[k]).item())
                torch.save(k_top, f"{self.args.save_metric_dir}/{datax}/k_top_{session_id}.pt")
                torch.save(entro, f"{self.args.save_metric_dir}/{datax}/entro_{session_id}.pt")
                


    def boundary_predictor(self):

        # =======================
        # A SVM learn a boundary with full attention
        # =======================

        datasets = ['nqa', 'tqa', 'longchat', 'gov_report', 'hotpotqa']
        k_coverage = []
        entro = []
        for data in datasets:
            for session in range(self.num_sample):
                file_path = os.path.join(self.args.save_metric_dir, f"{data}/k_top_{session}.pt")

                if not os.path.exists(file_path):
                    print("Compute the metrics for predictor training first...")
                    sys.exit(1)

                k_top = torch.load(file_path)
                #print(k_top)
                k_coverage.extend(k_top)
            
            for session in range(self.num_sample):
                file_path = os.path.join(self.args.save_metric_dir, f"{data}/entro_{session}.pt")

                if not os.path.exists(file_path):
                    print("Compute the metrics for predictor training first...")
                    sys.exit(1)

                en = torch.load(file_path)
                entro.extend(en)
                # print(en)
        #print(len(entro))
        #print(len(k_coverage))

        data = np.column_stack((k_coverage, entro))
        #print(data[0], data[10])
        model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.15)
        model.fit(data)
        print(f"Attention predictor: {model}")
        self.model = model
        torch.save(k_top, f"{self.args.save_metric_dir}/predictor.pt")
