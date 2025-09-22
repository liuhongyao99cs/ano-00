import torch
import torch.nn.functional as F
import time
import argparse
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from src.utils import *
from huggingface_hub import login

# =============================================
# Test the prefill re-computation time of GPU
# =============================================
p = argparse.ArgumentParser()
p.add_argument("--model_id", type = str, default = "Qwen/Qwen3-4B")
p.add_argument("--model", type = str, default = "Qwen3-4B")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored. ")
p.add_argument("--dataset_name", type=str)
args = p.parse_args()

model_name = args.model_id #"Qwen/Qwen3-4B"  # 
model_N = args.model #"Qwen3-4B"
data_name = args.dataset_name

# your hf account
# login(token = "hf_xxx")

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

for session_id in range(1):
    
    if data_name == 'longchat':
        input_text = data[session_id]['prompt']
    else:
        input_text = data[session_id]['context']
    input_text = input_text * 2
    inputs_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    input_ids = inputs_ids['input_ids'][:,0:40000]
    attention_mask = inputs_ids['attention_mask'][:,0:40000]
    
    print(input_ids.shape)
    '''with torch.no_grad():
        outputs = model(
            input_ids=input_ids
        )
    '''
    start_time = time.time()
    with torch.no_grad():
        model.generate(
            input_ids, 
            max_new_tokens = 1,
            attention_mask=attention_mask,
            #output_hidden_states=True, 
            #output_scores=True, 
            #return_dict_in_generate=True
        )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Prefill time: {elapsed_time:.2f} s for a context with {input_ids.shape[1]} tokens.. \n")
    
