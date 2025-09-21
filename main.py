import torch
import torch.nn.functional as F
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from src.utils import *
from huggingface_hub import login

model_name = "Qwen/Qwen3-4B"  # 
model_N = "Qwen3-4B"
data_name = "Academic"

# ========================
# Step 0: 加载模型和 tokenizer
# ========================

login(token = "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")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    dtype=torch.float16, 
    attn_implementation="flash_attention_2",
    device_map="auto",
    #output_attentions=False
)

dataset = f"/home/hoongyao/data/test_data/{data_name}.jsonl"
data = load_testcases(dataset)
for session_id in range(data.shape[0]):
    
    if data_name == 'longchat':
        input_text = data[session_id]['prompt']
    else:
        input_text = data[session_id]['context']
    input_text = input_text * 2
    inputs_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    #input_ids = inputs_ids['input_ids'][:,0:20000]
    #attention_mask = inputs_ids['attention_mask'][:,0:20000]
    
    print(input_ids.shape)
    '''with torch.no_grad():
        outputs = model(
            input_ids=input_ids
        )
    '''
    
    att_extract_(
        model=model, 
        model_name=model_N, # "Qwen3-8b"
        data_name=data_name,  # "longchat"
        session_id=session_id, # 5-th sample
        input_ids = input_ids,
        layer_id=36
    )
    
    '''
    start_time = time.time()
    with torch.no_grad():
        generated = model.generate(
            input_ids, 
            max_new_tokens = 1,
            attention_mask=attention_mask,
            output_hidden_states=True, 
            output_scores=True, 
            return_dict_in_generate=True
        )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Prefill time: {elapsed_time:.2f} s for a context with {input_ids.shape[1]} tokens.. \n")
    '''
