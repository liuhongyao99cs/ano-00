import os
import sys
import time
import math
import copy
import torch
import threading
import numpy as np
from pathlib import Path
from sklearn.svm import OneClassSVM

from src import *
from transformers.cache_utils import Cache, DynamicCache

# WiKV semantic coding

class WiKV_Controller:

    def __init__(self, model, tokenizer, args, shape, dtype=torch.float32, threshold=0.4, device='cpu'):

        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.threshold = threshold
        self.tensor = torch.zeros(shape, dtype=dtype, device=device)

        self.prev_threshold = 0
        self.step = 0.1
        self.num_sample = int(10)

        self.filled_count = 0
        self.total_elements = self.tensor.numel()


        self.lock = threading.Lock()        
        self.stop_event = threading.Event()  
        self.ready_event = threading.Event() 
        self.full_event = threading.Event()
        self.warm_up = threading.Event()
        self.think_st = threading.Event()
        self.think_end = threading.Event()

    def kv_pool_initialize(self, kv):
        # cpu kv pool to handle the streaming data

        #tmp =  torch.zeros_like(kv).to("cuda:0")
        #self.kv_pool = tensor_to_past_key_values(tmp)
        #del tmp
        # buffer is on cpu memory
        self.kv_pool = torch.zeros_like(kv).to("cuda:0")    # ((torch.rand_like(kv) - 0.5) * 0.1).to('cuda:0')
        print(f"kv_pool size: {self.kv_pool.shape}")

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

        # wait for the warm-up finish
        self.warm_up.wait()

        while self.filled_count < total:

            st = time.perf_counter()

            with self.lock:
                    # the ratio : streamed by bw / code_size
                    start = time.perf_counter()
                    propor = bw_trace[idx] * t / (code_size * 8)
                    streamed_num = int(propor * total)
                    
                    idx_range = slice(self.filled_count, min(self.filled_count + streamed_num, total))
                    indices = semantic_seq[idx_range]
                    # print(self.kv_pool.shape)
                    # fill in the kv pool in cpu

                    self.kv_pool[
                        indices[:, 0], :,
                        indices[:, 1], 
                        indices[:, 2], :
                    ] = kv_gpu[
                        indices[:, 0], :,
                        indices[:, 1], 
                        indices[:, 2], :
                    ].to("cuda:0")
                    
                    self.filled_count = min(self.filled_count + streamed_num, total)
                    idx += 1
                    end = time.perf_counter()
                    elapsed_time = end - start
                    #print(f"Write kv pool time: {elapsed_time:.4f}s")
                   
                    if self.filled_count / total >= self.threshold:
                        self.threshold += self.step
                        self.ready_event.set()


            elapsed = time.perf_counter() - st
            #print(elapsed)
            if elapsed < t:
                time.sleep(t - elapsed)

        with self.lock:
            if self.filled_count == total:
                self.threshold = 1
                self.full_event.set()
                self.ready_event.set()

        #print("✅ Fill the KV buffer thread is completed")

    def _fill_worker_fast(self, semantic_seq, bw_trace, kv_gpu, code_size):

        # =====================
        # KV cache loading process
        # =====================

        # semantic_seq is the final order of streaming
        # bw_trace record the throughput of each 0.1s, (Mbps)
        # kv_gpu is the quantized kv on gpu, we use it to fill the self.kv_pool in cpu
        # code_size is the file after encoding (MB)

        os.nice(10)
        idx = 0

        t = 0.1
        total = semantic_seq.shape[0]
        self.filled_count = 0

        # wait for the warm-up finish
        self.warm_up.wait()

        while True:
            st = time.perf_counter()

            elapsed = time.perf_counter() - st
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)

        with self.lock:
            if self.filled_count == total:
                self.threshold = 1
                self.full_event.set()
                self.ready_event.set()


    def probe(self, kv_pace, target_device='cuda:0'):
        
        #print(f"🔍 Wait for the streaming KV proportion: {self.threshold}...")
        if not self.full_event.is_set():
            # wait for the ready event in KV fill thread
            self.ready_event.wait()
            self.ready_event.clear()
        
        # Lock the KV pool to gather KV cache to gpu

        with self.lock:
            kv =  self.kv_pool.to(target_device)
            #kv_tensor = self.kv_pool.to(target_device).clone()

        kv_pace1 = DynamicCache()
        for i in range(len(kv_pace)):
            tmp = kv[i,0]
            tmp = tmp.unsqueeze(0)
            kv_pace[i][0][:,:,:tmp.shape[2],:] = tmp
            #kv_pace1[i][0][:,:,:tmp.shape[2],:] = tmp[:,:,:,:]
            tmp = kv[i,1]
            tmp=tmp.unsqueeze(0)
            kv_pace[i][1][:,:,:tmp.shape[2],:] = tmp
            #kv_pace1.update(kv_pace[i][0].clone(), kv_pace[i][0].clone(),i)
            #kv_pace1[i][1][:,:,:tmp.shape[2],:] = tmp[:,:,:,:]
            #kv_pace[:,:,:,:self.kv_pool.shape[3],:] = self.kv_pool.to(target_device)
            #print(kv_pace[i][1].shape[2])

        # kv_pace = tensor_to_tuple(kv_pace)
        del kv, tmp

        return kv_pace, kv_pace1

    def probe_tuple(self, kv_pace, semantic_seq, target_device='cuda:0'):
        
        total = semantic_seq.shape[0]
        start = round(self.prev_threshold * total)
        if not self.full_event.is_set():
            # wait for the ready event in KV fill thread
            self.ready_event.wait()
            self.ready_event.clear()
        
        # Lock the KV pool to gather KV cache to gpu

        with self.lock:

            tmp = self.kv_pool.to(target_device).contiguous()

            end = min(round(self.threshold * total),total)
            idx_range = slice(start, end)
            indices = semantic_seq[idx_range]
            self.prev_threshold = self.threshold
            print(f"Data copy start: {start} and end {end}")
            #kv_tensor = self.kv_pool.to(target_device).clone()
        
        tmp = tmp.unsqueeze(2)
        start = time.perf_counter()
        print(kv_pace[0][0])
        '''
        for k in range(len(kv_pace)):
            kv_pace[k][0][:, indices[:,1], indices[:,2], :].copy_(tmp[k, 0, :, indices[:,1], indices[:,2], :])
            kv_pace[k][1][:, indices[:,1], indices[:,2], :].copy_(tmp[k, 1, :, indices[:,1], indices[:,2], :])
        '''
        for k in range(len(kv_pace)):
            for j in range(8):
                kv_pace[k][0][:, j, :tmp.shape[4], :].copy_(tmp[k, 0, :, j, :, :]).to(target_device).contiguous()
                kv_pace[k][1][:, j, :tmp.shape[4], :].copy_(tmp[k, 1, :, j, :, :]).to(target_device).contiguous()

        print(kv_pace[0][0])

        end = time.perf_counter()
        elapsed_time = end - start
        print(f"KV cache data copy: {elapsed_time:.4f}s")
                

        del tmp

        return kv_pace

    def get_progress(self):
        """get the proportion in the KV pool"""
        with self.lock:
            return self.filled_count / self.total_elements
    

    def Metric(self, args):
        
        # =====================
        # Gather metrics of tokens with full attention
        # =======================

        datasets = ['nqa', 'tqa', 'longchat', 'gov_report', 'hotpotqa']
        for datax in datasets:
            #datax = 'longchat'
            data_parent_root = Path(args.path_to_context).parent
            if not os.path.exists(f"{data_parent_root}/{datax}.jsonl"):
                    print("Load test data first...")
                    sys.exit(1)

            data = load_testcases(f'{data_parent_root}/{datax}.jsonl')
            for session_id in range(self.num_sample):
                
                if datax in ['longchat', 'tqa', 'nqa']:
                    input_text = data[session_id]['prompt'] 
                elif datax in ['hotpotqa']:
                    input_text = data[session_id]['context'] + "Based on given passages, answer the question: " + data[session_id]['input']
                else:
                    input_text = data[session_id]['context'] + "Summarize the given context in 250 tokens."
            
                    
                inputs_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
                input_ids = inputs_ids['input_ids']
                attention_mask = inputs_ids['attention_mask']
                seq_len = input_ids.shape[1]

                kv_parent_root = Path(args.save_kv_dir).parent
                if not os.path.exists(f"{kv_parent_root}/{self.args.model}/{datax}/raw_kv_{session_id}.pt"):
                    print("Compute the KV cache first...")
                    sys.exit(1)

                raw_kv = torch.load(f"{kv_parent_root}/{self.args.model}/{datax}/raw_kv_{session_id}.pt")
                
                kv = tensor_to_tuple(raw_kv)
                del raw_kv
                # generate logit scores through model.generate
                generated = self.model.generate(input_ids, past_key_values = kv, max_new_tokens = 100, return_dict_in_generate=True, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id, attention_mask=attention_mask, output_scores=True, output_attentions=False)
                prediction = self.tokenizer.decode(generated.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
                #print(prediction)
                del kv
                print(f"Dumping the metrics for data {datax} sample {session_id}...")
                if not os.path.exists(self.args.save_metric_dir):
                    os.makedirs(self.args.save_metric_dir, exist_ok=True)
                if not os.path.exists(f"{self.args.save_metric_dir}/{datax}"):
                    os.makedirs(f"{self.args.save_metric_dir}/{datax}", exist_ok=True)
                k_top = []
                entro = []
                for k in range(len(generated.scores)):
                    k_top.append(K_coverage(generated.scores[k]).item())
                    entro.append(entropy(generated.scores[k]).item())
                torch.save(k_top, f"{self.args.save_metric_dir}/{datax}/k_top_{session_id}.pt")
                torch.save(entro, f"{self.args.save_metric_dir}/{datax}/entro_{session_id}.pt")
                del generated

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
        model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.06)
        model.fit(data)
        print(f"Attention predictor: {model}")
        self.model = model
        torch.save(k_top, f"{self.args.save_metric_dir}/predictor.pt")

    def dot_loading_thread(self):

        while self.think_st.is_set():
            if not self.think_end.is_set():
                sys.stdout.write('.')
                sys.stdout.flush()
                time.sleep(0.1)
            time.sleep(0.01)

    def start_loading_animation(self):
        self.load_thread = threading.Thread(
            target=self.dot_loading_thread, 
            daemon=True
        )
        self.load_thread.start()
    

    def pace_decode(self, kv_tuple, input_idx, attention_maskx, model, tokenizer, ttft_ddl, per_token_ddl, max_new_tokens):

        # ===================
        # pace decoding: wait sufficient KV cache buffer
        # kv_tuple: KV cache used in model.generate that progressively updated
        # input_idx and attention_maskx is the decoding sequence, we add new token to this seqs
        # model, tokenizer: the LLM loaded through transformers
        # ttft_ddl is set 1.2s (based on your requirement) and per_token_ddl is 100 ms
        # max_new_tokens
        # ===================

        BOLD = '\033[1m'
        YELLOW = '\033[93m'
        RESET = '\033[0m'
        UNDERLINE = '\033[4m' 
        TALIC = '\033[3m'
        BRIGHT_BLACK = '\033[90m'    # 灰色
        BRIGHT_RED = '\033[91m'
        BRIGHT_GREEN = '\033[92m'
        BRIGHT_YELLOW = '\033[93m'
        BRIGHT_BLUE = '\033[94m'
        BRIGHT_MAGENTA = '\033[95m'
        BRIGHT_CYAN = '\033[96m'
        BRIGHT_WHITE = '\033[97m'

        print(f"{BOLD}{YELLOW}WiKV:\nThinking", end="", flush=True)
        self.think_st.set()
        self.start_loading_animation()
        idx = 0

        for k in range(max_new_tokens):

            # max time ddl for 1st token or next tokens
            if (k == 0 ):
                ddl = ttft_ddl
                with torch.no_grad():
                    generated = model.generate(
                        input_idx, 
                        attention_mask = attention_maskx,
                        past_key_values=kv_tuple, 
                        max_new_tokens = 1, 
                        eos_token_id=tokenizer.eos_token_id, 
                        pad_token_id=tokenizer.eos_token_id, 
                        return_dict_in_generate=True, 
                        output_scores=True
                    )
                m1 = K_coverage(generated.scores[0]).item()
                m2 = entropy(generated.scores[0]).item()
                data = torch.tensor([m1, m2]).unsqueeze(0)  # shape: [1, 2]
                decide = self.model.decision_function(data)[0]
                startx = time.perf_counter()
                del generated
                self.warm_up.set()
            else:
                ddl = per_token_ddl

            token_st = time.perf_counter()
            flag = True
    
            # if KV cache is not fully streamed
            #if not self.full_event.is_set():
            while (not self.full_event.is_set()):
                    flag = False
                    start = time.perf_counter()
                    kv_tuple, _ = self.probe(kv_tuple, target_device='cuda:0')
                    end = time.perf_counter()
                    elapsed_time = end - start
                    #print(f"Prepare {self.threshold*100}% KV CACHE for token {k}: {elapsed_time:.4f}s")
                    #del kv_pace
                    start = time.perf_counter()
                    kv_tmp = copy.deepcopy(kv_tuple)
                    with torch.no_grad():
                        generated = model.generate(
                            input_idx, 
                            attention_mask = attention_maskx,
                            past_key_values=kv_tmp, 
                            max_new_tokens = 1, 
                            eos_token_id=tokenizer.eos_token_id, 
                            pad_token_id=tokenizer.eos_token_id, 
                            return_dict_in_generate=True, 
                            output_scores=True
                        )
                    #kv_tuple = kv_tuple1
                    # del kv_tuple1
                    end = time.perf_counter()
                    elapsed_time = end - start
                    #print(f"Decode token {k}: {elapsed_time:.4f}s")

                    # check the confidence 
                    start = time.perf_counter()
                    m1 = K_coverage(generated.scores[0]).item()
                    m2 = entropy(generated.scores[0]).item()
                    data = torch.tensor([m1, m2]).unsqueeze(0) #.to("cuda:0")  # shape: [1, 2]
                    decide = self.model.decision_function(data)[0]
                    end = time.perf_counter()
                    elapsed_time = end - start

                    del kv_tmp
                    #print(f"COnfidence check for token {k}: {elapsed_time:.4f}s")
                    #print(f"Metric decide: {decide} score")

                    if (decide < 1e-3) and ((time.perf_counter() - token_st) < ddl) :
                        self.step = 0.25/ ( 1 + 10 * math.e ** (-decide / 20))
                        del generated
                        #print("not enough")
                        continue
                    else:
                        if k == 0:
                            self.think_end.is_set()
                            self.think_st.clear()
                            print(f"{RESET}\n")
                        end = time.perf_counter()
                        if k == 0 : 
                            ttft = end - token_st
                        
                        token = tokenizer.decode(generated.sequences[0][-1], skip_special_tokens=True)
                        print(f"{BOLD}{BRIGHT_WHITE}{UNDERLINE}{TALIC}{token}", end="", flush=True)
                        input_idx = (generated.sequences[0]).unsqueeze(0)
                        new_token = torch.tensor([[1]], device=attention_maskx.device)
                        attention_maskx = torch.cat([attention_maskx, new_token], dim=1)
                        kv_tuple = generated['past_key_values']
                               
                        del generated  
                        self.step = 0.08
                        break
            
            # all KV cache is streamed
            if flag and self.full_event.is_set():
                if idx == 0:
                    idx += 1
                    kv_tuple, _ = self.probe(kv_tuple, target_device='cuda:0')
                with torch.no_grad():
                    
                    generated = model.generate(
                        input_idx, 
                        attention_mask = attention_maskx,
                        past_key_values=kv_tuple, 
                        max_new_tokens = 1, 
                        return_dict_in_generate=True, 
                        eos_token_id=tokenizer.eos_token_id, 
                        pad_token_id=tokenizer.eos_token_id, 
                        output_scores=False
                    )

                input_idx = (generated.sequences[0]).unsqueeze(0)
                new_token = torch.tensor([[1]], device=attention_maskx.device)
                attention_maskx = torch.cat([attention_maskx, new_token], dim=1)
                kv_tuple = generated['past_key_values']
                token = tokenizer.decode(generated.sequences[0][-1], skip_special_tokens=True)
                print(f"{BOLD}{BRIGHT_WHITE}{UNDERLINE}{TALIC}{token}", end="", flush=True)
                del generated
        
        end = time.perf_counter()
        latency = end - startx
        print(f"{RESET}\n")

        return ttft, latency

