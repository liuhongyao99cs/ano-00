from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import repeat_kv, apply_rotary_pos_emb
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import matplotlib.pyplot as plt
from typing import Callable, Optional, Union
from datasets import load_dataset
import torch
import json
import os
import torch.nn.functional as F
from contextlib import nullcontext

def load_testcases(test_file):
    with open(test_file, 'r', encoding='utf-8') as json_file:
        json_list = list(json_file)

    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)

    return test_cases

'''
def att_extract_(
    model: PreTrainedModel, 
    model_name, # "Qwen3-8b"
    data_name,  # "longchat"
    session_id, # 5-th sample
    layer_id: int,
    input_ids: torch.LongTensor,
    use_cache: Optional[bool] = False,
    past_key_values: Optional[Cache] = None,
) -> torch.tensor:
    
    ## extract modules in model
    embed_tokens = model.model.embed_tokens
    rotary_emb = model.model.rotary_emb

    inputs_embeds = embed_tokens(input_ids)

    ## decoding parameters define
    cache_position = None
    position_ids = None
    attention_mask = None
    
    if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=model.config)
        
    if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
    
    if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
    
    if not isinstance(causal_mask_mapping := attention_mask, dict):
                # Prepare mask arguments
                mask_kwargs = {
                    "config": model.config,
                    "input_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                    "position_ids": position_ids,
                }
                # Create the masks
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }

    hidden_states = inputs_embeds
    position_embeddings = rotary_emb(hidden_states, position_ids)
    
    dir_path = f"Attention_weights/{model_name}/{data_name}/session_{session_id}/"

    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)  # exist_ok=True 确保如果目录已存在不会报错

    # 定义文件路径
    file_path = os.path.join(dir_path, f"hidden_state_{0}.pt")

    # 保存 hidden_states 到文件
    torch.save(hidden_states, file_path)
    
    ## compute attentions in each layer    
    for i in range(layer_id): #(model.config['num_hidden_layers']):
        layer = model.model.layers[i]
        self_att = layer.self_attn
        rms_norm = layer.input_layernorm  
        post_attention_layernorm = layer.post_attention_layernorm
        mlp = layer.mlp

        # define residual and normlize hiiden states
        residual = hidden_states
        hidden_states = rms_norm(hidden_states)
        
        # self-atten
        hidden_states, _ = self_att(
            hidden_states=hidden_states,
            attention_mask=causal_mask_mapping["full_attention"],
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            position_embeddings=position_embeddings
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = post_attention_layernorm(hidden_states)
        hidden_states = mlp(hidden_states)
        hidden_states = residual + hidden_states
        #attention_weight = attention_weights.detach().cpu()
        #del attention_weights
        #attn_head = attention_weight[0]
        #fig, ax = plt.subplots(figsize=(8, 6))
        #im = ax.imshow(attn_head[0,1:,:].numpy(), cmap='viridis', aspect='auto')
        # torch.save(f"/Attention_weights/{model_name}/{data_name}/session_{session_id}/hidden_state_{i}.pt",hidden_states)
        
        # 定义目录路径
        dir_path = f"Attention_weights/{model_name}/{data_name}/session_{session_id}/"

        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)  # exist_ok=True 确保如果目录已存在不会报错

        # 定义文件路径
        file_path = os.path.join(dir_path, f"hidden_state_{i+1}.pt")

        # 保存 hidden_states 到文件
        torch.save(hidden_states, file_path)
'''
        

def hidden_extract_(
    model: PreTrainedModel, 
    model_name, 
    data_name, 
    session_id, 
    save_dir: str,
    input_ids: torch.LongTensor,
    use_cache: Optional[bool] = False,
    past_key_values: Optional[Cache] = None,
) :
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,  
            return_dict=True,
            use_cache=use_cache,
            past_key_values=past_key_values
        )
        
    for i, hidden_state in enumerate(outputs.hidden_states):
        file_path = os.path.join(save_dir, f"hidden_s{session_id}_l{i}.pt")
        torch.save(hidden_state.cpu(), file_path)
        

def to_blob(kv_tuples):
    """ Transform a list of tuples of key and value tensors to a single tensor
    """
    return torch.stack([torch.stack(inner_tuple, dim=0).to("cuda:0") for inner_tuple in kv_tuples], dim=0)

def layer_atten_extract_(model, input_ids, layer_id, args, session_id=0):
    """
    calculate the attention weights of layer_id based on the hidden_states and attention params from pretrained model
    """

    with torch.no_grad():

        device = next(model.parameters()).device
        
        # load the hidden as input
        if layer_id == 0:

            hidden_states = model.model.embed_tokens(input_ids)
        else:

            hidden_path = os.path.join(args.save_hid_dir, f"hidden_s{session_id}_l{layer_id-1}.pt")
            if os.path.exists(hidden_path):
                hidden_states = torch.load(hidden_path, map_location=device)
            else:
                raise FileNotFoundError(f"hidden file {hidden_path} not exists...")
        

        hidden_states = hidden_states.to(device)
        
        position_ids = torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0)
        
        # load model layer
        layer = model.model.layers[layer_id]
        rotary = model.model.rotary_emb
        layer_norm = layer.input_layernorm
        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        q_norm = layer.self_attn.q_norm
        k_norm = layer.self_attn.k_norm
        

        hidden_states = layer_norm(hidden_states)
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, model.config.head_dim)
        

        query_states = q_proj(hidden_states).view(hidden_shape)
        query_states = q_norm(query_states).transpose(1, 2)
        
        key_states = k_proj(hidden_states).view(hidden_shape)
        key_states = k_norm(key_states).transpose(1, 2)
        

        del hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        position_embeddings = rotary(key_states, position_ids) 
        cos, sin = position_embeddings
        
        # rope process
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # only remain the final query attentions (last 100 tokens as query)
        query_states = query_states[:,:,-101:-1,:]


        key_states = repeat_kv(key_states, model.config.num_attention_heads // model.config.num_key_value_heads)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * (model.config.head_dim ** -0.5)
        
        # control the cuda memory
        del query_states, key_states, cos, sin, position_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        

        #attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float16)
        #attn_weights = F.dropout(attn_weights, p=0, training=False)
        
        attn_weights = attn_weights.squeeze(0)
        attn_weights = attn_weights.view(model.config.num_attention_heads // model.config.num_key_value_heads, model.config.num_key_value_heads, *attn_weights.shape[1:])
        attn_weights = attn_weights.sum(dim=0)
        print(attn_weights.shape)
        attn_path = os.path.join(args.save_att_dir, f"attn_s{session_id}_l{layer_id}.pt")
        torch.save(attn_weights, attn_path)
        

        del attn_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def atten_extract_(model, input_ids, args, session_id=0):
    """
    process all layers in the model
    """
    os.makedirs(args.save_att_dir, exist_ok=True)
    os.makedirs(args.save_hid_dir, exist_ok=True)
    
    for layer_id in range(model.config.num_hidden_layers):
        print(f"Compute the attention weights of {layer_id}-layer...\n")
        layer_atten_extract_(model, input_ids, layer_id, args, session_id)
        
       
