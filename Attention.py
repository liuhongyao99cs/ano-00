import torch
import torch.nn.functional as F

import time
import argparse
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.qwen3.modeling_qwen3 import repeat_kv, apply_rotary_pos_emb
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
p.add_argument("--save_att_dir", type=str)
p.add_argument("--save_hid_dir", type=str)

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

    # process dataset, assume we are testing 40K tokens
    dataset = args.path_to_context  #f"/home/hoongyao/data/test_data/{data_name}.jsonl"
    data = load_testcases(dataset)

for session_id in range(args.end-args.start):
    
    if data_name == 'longchat':
        input_text = data[session_id]['prompt'] + "Repeat the context."
    else:
        input_text = data[session_id]['context'] + "Repeat the context."
        
    inputs_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_ids = inputs_ids['input_ids']
    attention_mask = inputs_ids['attention_mask']
    
    # check the context length
    print(input_ids.shape)
    
    if not os.path.exists(args.save_hid_dir):
        os.makedirs(args.save_hid_dir, exist_ok=True)
    
    # if you have generated hidden_states data
    if not os.path.exists(os.path.join(args.save_hid_dir, f"hidden_s{session_id}_l{model.config.num_hidden_layers-1}.pt")):
        hidden_extract_(
            model=model,        # predefined mdoel
            model_name=model_N, # "Qwen3-8b"
            data_name=data_name,  # "longchat"
            session_id=session_id, # 5-th sample
            save_dir=args.save_hid_dir,
            input_ids = input_ids,
        )
    
    '''
    if not os.path.exists(args.save_att_dir):
        os.makedirs(args.save_att_dir, exist_ok=True)
    if not os.path.exists(os.path.join(args.save_att_dir, f"attn_s{session_id}_l{model.config.num_hidden_layers-1}.pt")):
    
        embed_tokens = model.model.embed_tokens
        for layer_id in range(model.config.num_hidden_layers):
            if (layer_id == 0):
                hidden_states = embed_tokens(input_ids)
            else:
                hidden_path = os.path.join(args.save_hid_dir,f"hidden_s{0}_l{layer_id-1}.pt")
                if os.path.exists(hidden_path):
                    hidden_states = torch.load(hidden_path)
        
            hidden_states = hidden_states.detach().cpu()
            position_ids = torch.arange(0,input_ids.shape[1]).unsqueeze(0)
            
            rotary = model.model.rotary_emb
            layer_norm = model.model.layers[layer_id].input_layernorm.cpu()
            q_proj = model.model.layers[layer_id].self_attn.q_proj.cpu()
            k_proj = model.model.layers[layer_id].self_attn.k_proj.cpu()
            q_norm = model.model.layers[layer_id].self_attn.q_norm.cpu()
            k_norm = model.model.layers[layer_id].self_attn.k_norm.cpu()
            
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, model.config.head_dim)
            query_states = q_norm(q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = k_norm(k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

            position_embeddings = rotary(hidden_states, position_ids)
            hidden_states = layer_norm(hidden_states)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            key_states = repeat_kv(key_states, model.config.num_attention_heads//model.config.num_key_value_heads)
            
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * (model.config.head_dim**-0.5)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float16)
            attn_weights = F.dropout(attn_weights, p=0, training=False)
            
            attn_weights = attn_weights.unsqueeze(0)
            torch.save(attn_weights,f"{args.save_att_dir}/attn_s{session_id}_l{layer_id}.pt")
    '''

    atten_extract_(model, input_ids, args, session_id=0)

