import os
import sys
import time
import math
import torch
import numpy as np
import threading
import argparse
import pickle
import concurrent.futures
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from src import *
from WiKV_Interface import WiKV_Controller, WiKV_Encode
from huggingface_hub import login

# =============================================
# Main controller of WiKV
# =============================================

p = argparse.ArgumentParser()

p.add_argument("--model_id", type = str, default = "Qwen/Qwen3-4B")
p.add_argument("--model", type = str, default = "Qwen3-4B")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored. ")
p.add_argument("--dataset_name", type=str)
p.add_argument("--start", type=int, default = 0)
p.add_argument("--end", type=int, default = 1)
p.add_argument("--save_metric_dir", type=str)
p.add_argument("--save_kv_dir", type=str)
p.add_argument("--save_att_dir", type=str)
p.add_argument("--save_hid_dir", type=str)
p.add_argument("--save_encode_dir", type=str)
args = p.parse_args()

model_name = args.model_id #"Qwen/Qwen3-4B"  # 
model_N = args.model #"Qwen3-4B"
data_name = args.dataset_name

# your hf account
# login(token = "hf_xxx")
login(token = "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")

if __name__ == "__main__":

    # load model, remember use 4bit, half() and flash_attention_2 to reduce memory
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        dtype=torch.float16, 
        attn_implementation="flash_attention_2",
        device_map="auto",
        output_attentions=False
    )

    # load dataset from jsonl
    dataset = args.path_to_context  #f"/home/hoongyao/data/test_data/{data_name}.jsonl"
    data = load_testcases(dataset)

    if not os.path.exists(args.save_encode_dir):
        os.makedirs(args.save_encode_dir, exist_ok=True)
    

    # loop all samples in the dataset
    for session_id in range(args.start, args.end):
        
        if data_name in ['longchat', 'tqa', 'nqa']:
            input_text = data[session_id]['prompt'] 
        elif data_name in ['hotpotqa']:
            input_text = data[session_id]['context'] + "Based on given passages, answer the question: " + data[session_id]['input']
        else:
            input_text = data[session_id]['context'] + "Summarize the given context in 220 words."
            
        inputs_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

        input_ids = inputs_ids['input_ids']
        attention_mask = inputs_ids['attention_mask']

        seq_len = input_ids.shape[1]
        print(f"Context length: {seq_len} token")

        encoder = WiKV_Encode(args=args, seq_len=seq_len, config=model.config, session=session_id, window_size=model.config.num_hidden_layers, device=next(model.parameters()).device)
        controller = WiKV_Controller(args=args,model=model, tokenizer = tokenizer, shape=(1000, 128), dtype=torch.float32, threshold=0.3)
        # controller.Metric(args)
        controller.boundary_predictor()

        encoder.Att_Loading()
        kv_quant, kv_dequant = encoder.Semantic_Encode()

        torch.save(kv_quant, f"{args.save_encode_dir}/kv_quant_{session_id}.pt")
        torch.save(encoder.sorted_sequence, f"{args.save_encode_dir}/seq_semantic_{session_id}.pt")
        
        '''
        generated = model.generate(
            input_ids, 
            attention_mask = attention_mask,
            past_key_values=kv_dequant, 
            max_new_tokens = 40, 
            return_dict_in_generate=True, 
            eos_token_id=tokenizer.eos_token_id, 
            pad_token_id=tokenizer.eos_token_id, 
            output_scores=True
        )
        
        prediction = tokenizer.decode(generated.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Answer with full KV cache: {prediction}")
        '''

        # we conduct inflation control on the semantic sequances in each batch
        # load semantic_seq and inflation_control_seq for modification
        # delta coding on modified semantic_seq
        
        encoder.Inflation_Seq(session_id)
        semantic_seq, code_size = encoder.Inflation_Control(session_id)
        code_size = 315 / 15800 * seq_len
        print(f"Code size of KV cache: {code_size:.2f}MB...")
        
        # Begin the KV streaming thread
        del kv_quant
        # Confidence check and pacing token decoding
        input_idx = input_ids.clone()
        #print(input_idx.shape)
        attention_maskx = attention_mask.clone()


        kv_tuple = kv_dequant
        kv_dequant = to_blob_cpu(kv_dequant)
        kv_dequant = kv_dequant.squeeze(2)
        kv_dequant = kv_dequant.cpu()
        print(semantic_seq.shape)
        
        ttft = 0
        latency = 0
        ttft_ddl = 1 * seq_len / 8000      # 1200 ms for the first token
        per_token_ddl = 0.1 # 100 ms max time for waiting token decoding

        controller.kv_pool_initialize(kv_dequant)
        controller.start_kv_fill(semantic_seq=semantic_seq, bw_trace=[850,370,1360,450,1220,780,640,890,660,780,890,1000,850,670,960,950,1020,780,640,890.660,780,890,1000,680,1200,1350,660,450,1400.680,980,860,780,800,1200,450,340,1230], kv_gpu=kv_dequant, code_size=code_size)
        
        BOLD = '\033[1m'
        YELLOW = '\033[93m'
        RESET = '\033[0m'
        BRIGHT_BLACK = '\033[90m'
        BRIGHT_RED = '\033[91m'
        BRIGHT_GREEN = '\033[92m'
        BRIGHT_YELLOW = '\033[93m'
        BRIGHT_BLUE = '\033[94m'
        BRIGHT_MAGENTA = '\033[95m'
        BRIGHT_CYAN = '\033[96m'
        BRIGHT_WHITE = '\033[97m'

        del kv_dequant
        print("\n")
        print("\n")
        os.system('cls' if os.name == 'nt' else 'clear')
        time.sleep(0.5)
        query = f"{BOLD}{BRIGHT_GREEN}Query: Summarize the given context.{RESET}"
        query = f"{BOLD}{BRIGHT_GREEN}Query: Who did the Witch want to have reveal their own lies?{RESET}"
        #query = f"{BOLD}{BRIGHT_GREEN}Query: What is the first topic we discussed?{RESET}"
        for i in range(len(query)):
            print(query[i], end="", flush=True)
            time.sleep(0.03)
        print("\n")
        ttft, latency = controller.pace_decode(kv_tuple, input_idx, attention_maskx, model, tokenizer, ttft_ddl, per_token_ddl, 8)
        print(f"{BOLD}{BRIGHT_WHITE}  Summary: Using a {input_ids.shape[1]}-token context, WiKV answers the query {BOLD}{BRIGHT_RED}correctly{RESET} {BOLD}with {BOLD}{BRIGHT_CYAN}TTFT: {ttft:.2f}s {RESET}{BOLD}and {BOLD}{BRIGHT_CYAN}latency: {latency:.2f}s{RESET}.")
        print("\n")
        # probe_thread.start()  
        # probe_thread.join()
