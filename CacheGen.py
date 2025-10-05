import sys
import time
import math
import torch
import random
import argparse
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from src.utils import *
from huggingface_hub import login

# =============================================
# Demo of the CacheGen performance with a wireless bandwidth trace
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

model_name = args.model_id
model_N = args.model
data_name = args.dataset_name

# your hf account
# login(token = "hf_xxx")
login(token = "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")

# load model, remember use 4bit, half() and flash_attention_2 to reduce memory
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    dtype=torch.float16, 
    attn_implementation="flash_attention_2",
    device_map="auto",
    #output_attentions=False
)

# process dataset, assume we are testing 40K tokens
dataset = args.path_to_context 
data = load_testcases(dataset)

for session_id in range(args.start, args.end):
    
    if data_name in ['longchat', 'tqa', 'nqa']:
        input_text = data[session_id]['prompt'] 
    elif data_name in ['hotpotqa']:
        input_text = data[session_id]['context'] + "Based on given passages, answer the question: " + data[session_id]['input']
    else:
        input_text = data[session_id]['context'] + "Summarize the given context in 250 tokens."

    inputs_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    input_ids = inputs_ids['input_ids']
    attention_mask = inputs_ids['attention_mask']
    seq_len = input_ids.shape[1]

    # load the KV cache
    file_path = os.path.join(args.save_kv_dir, f"raw_kv_{session_id}.pt")
    if not os.path.exists(file_path):
        print("Compute the KV cache for the session...")
        sys.exit(1)

    kv = torch.load(file_path)

    # ============================
    # CacheGen parameter define

    # bitrate level
    bin_list = [[14,10,6,4,4], [18,16,10,4,4], [22,18,14,10,12], [24,20,16,12,12], [30,24,20,16,16]]
    
    layer_group = 9
    code_size = 170 / 8000 * seq_len

    # ttft, chunk level, bw_pred and real trace
    ttft_cachegen = 2.2
    chunk_num = 22
    chunk_len = math.ceil(seq_len / chunk_num)
    chunk_size = code_size / chunk_num
    bw_trace = [850,370,1360,450,1220,780,340,1190,260,1180,690,1200,1250,270,960,950,1020,780,1040,190.960,1380,290,1000,680,1200,1350,660,450,1400.680,980,860,780,800,1200,450,340,1230]
    bw_pred = [val*random.uniform(0.8,1.1) for val in bw_trace]

    # bw adaption: select coding level for each chunk
    level = [1,2,3,4,5]
    code_size_level = [code_size * 0.3, code_size * 0.6, code_size * 1, code_size * 1.15, code_size * 1.3]
    chunk_level = []
    code_size_cachegen = 0
    for i in range(chunk_num):
        # find coding level
        for j in range(len(level), 0, -1):
            if(bw_pred[i] * 0.1 / ( code_size_level[j-1] * 8 / chunk_num) >= 1 ):
                chunk_level.append(j)
                code_size_cachegen += code_size_level[j-1] / chunk_num
                break
            elif (j==1):
                chunk_level.append(1)
                code_size_cachegen += code_size_level[j-1] / chunk_num
                break
    print(chunk_level)
    chunk_level.append(5)


    # re-organize the KV cache based on coding level
    kv_cachegen = torch.zeros_like(kv)  
    code_size

    for i in range(chunk_num):
        start = i * chunk_len
        end = min((i + 1) * chunk_len, kv.shape[3])  
        kv_chunk = kv[:, :, :, start:end, :].clone()
        kv_quant, max_q = layer_quantization(kv_chunk, bin_list[chunk_level[i]-1], layer_group)
        kv_dequant = layer_dequantize(kv_quant, max_q, bin_list[chunk_level[i]-1], layer_group)
        kv_dequant = to_blob(kv_dequant)
        kv_dequant = kv_dequant.squeeze(2)  
        kv_cachegen[:, :,:, start:end, :] = kv_dequant
    kv_cachegen = tensor_to_tuple(kv_cachegen)


    # cachegen decoding 
    # 
    MAX_NEW_TOKENS = 250
    os.system('cls' if os.name == 'nt' else 'clear')
    time.sleep(0.5)
    query = "Query: summarize the given context."
    for i in range(len(query)):
        print(query[i], end="", flush=True)
        time.sleep(0.03)
    print("\n")
    print("CacheGen: ")
    
    start_time = time.time()

    # wait for transferring all KV cache
    idx = 0
    streamed_data = 0
    while (True):
        if (streamed_data >= code_size_cachegen * 8):
            streamed_data = 0
            break
        else:
            streamed_data += bw_trace[idx] * 0.1
            time.sleep(0.1)
            idx += 1

    for i in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            generated = model.generate(
                input_ids, 
                max_new_tokens = 1,
                past_key_values = kv_cachegen,
                attention_mask=attention_mask,
                #output_scores=True, 
                return_dict_in_generate=True
            )
            kv_cachegen = generated['past_key_values']
            input_ids = (generated.sequences[0]).unsqueeze(0)
            new_token = torch.tensor([[1]], device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, new_token], dim=1)
            token = tokenizer.decode(generated.sequences[0][-1], skip_special_tokens=True)
            print(token, end="", flush=True)
    
        if (i == 0):
            end_time = time.time()
            ttft = end_time - start_time
    print("\n")
    end_time = time.time()
    latency = end_time - start_time
    print(f"CacheGen processes a {seq_len}-token context with ttft: {ttft:.2f}s and latency: {latency:.2f}s")
    print("\n")
    print("\n")
