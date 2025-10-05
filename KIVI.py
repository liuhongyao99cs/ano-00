import torch
import torch.nn.functional as F
import time
import argparse
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from src.utils import *
from huggingface_hub import login

# =============================================
# Test the one-time KV cache transfer time of KIVI baseline
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
dataset = args.path_to_context  #f"/home/hoongyao/data/test_data/{data_name}.jsonl"
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

    # layer-wise quantization
    bin_list = [22,18,14,10,12]
    layer_group = 9
    kv_quant, max_q = layer_quantization(kv, bin_list, layer_group)
    kv_dequant = layer_dequantize(kv_quant, max_q, bin_list, layer_group)
    code_size = 170 / 8000 * seq_len

    # bw trace for KV streaming
    bw_trace = [850,370,1360,450,1220,780,640,890,660,780,690,1200,1250,270,960,950,1020,780,1040,490.660,1380,290,1000,680,1200,1350,660,450,1400.680,980,860,780,800,1200,450,340,1230]

    # KIVI pace decoding
    MAX_NEW_TOKENS = 250
    os.system('cls' if os.name == 'nt' else 'clear')
    time.sleep(0.5)
    query = "Query: summarize the given context."
    for i in range(len(query)):
        print(query[i], end="", flush=True)
        time.sleep(0.03)
    print("\n")
    print("KIVI: ")
    
    start_time = time.time()

    # wait for one-time KV cache transfer
    idx = 0
    streamed_data = 0
    while (True):
        if (streamed_data >= code_size * 8):
            streamed_data = 0
            break
        else:
            streamed_data += bw_trace[idx] * 0.1
            time.sleep(0.1)
            idx += 1
    #print(idx)

    for i in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            generated = model.generate(
                input_ids, 
                max_new_tokens = 1,
                past_key_values = kv_dequant,
                attention_mask=attention_mask,
                #output_scores=True, 
                return_dict_in_generate=True
            )
            kv_dequant = generated['past_key_values']
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
    print(f"KIVI processes a {seq_len}-token context with ttft: {ttft:.2f}s and latency: {latency:.2f}s")
    print("\n")
    print("\n")
