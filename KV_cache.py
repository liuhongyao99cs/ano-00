import torch
import torch.nn.functional as F
import time
import argparse
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from src.utils import *
from huggingface_hub import login

# =============================================
# Obtain the KV cache from the contexts
# =============================================
p = argparse.ArgumentParser()
p.add_argument("--model_id", type = str, default = "Qwen/Qwen3-4B")
p.add_argument("--model", type = str, default = "Qwen3-4B")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored. ")
p.add_argument("--dataset_name", type=str)
p.add_argument("--start", type=int, default = 0)
p.add_argument("--end", type=int, default = 1)
p.add_argument("--save_dir", type=str)
args = p.parse_args()

model_name = args.model_id #"Qwen/Qwen3-4B"  # 
model_N = args.model #"Qwen3-4B"
data_name = args.dataset_name

# your hf account
# login(token = "hf_xxx")
if __name__ == "__main__":
    # Check if save_dir exists
    login(token = "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")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

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
    else:
        input_text = data[session_id]['context']
        
    inputs_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_ids = inputs_ids['input_ids']
    attention_mask = inputs_ids['attention_mask']
    print(f"Saving the KV cache for doc {session_id}...")

    with torch.no_grad():
        generated = model.generate(
            input_ids, 
            max_new_tokens = 1,
            attention_mask=attention_mask,
            return_dict_in_generate=True
        )
    
    # since 1 new token, we delete the last token's KV cache and then store
    kv = generated['past_key_values']
    kv = list(kv)
    key_value = []
    for i in range(len(kv)):
        kv[i] = list(kv[i])
        kv[i][0] = kv[i][0][:, :, :-1][0]
        kv[i][1] = kv[i][1][:, :, :-1][0]
        kv[i] = tuple(kv[i])
    kv = tuple(kv)
    kv_tensor = to_blob(kv)
    
    torch.save(kv_tensor, f"{args.save_dir}/raw_kv_{session_id}.pt")
    if session_id == 0:
        pickle.dump(kv, open(f"{args.save_dir}/raw_kv_{session_id}.pkl", "wb"))
