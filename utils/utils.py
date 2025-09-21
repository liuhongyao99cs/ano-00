from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import matplotlib.pyplot as plt
from typing import Callable, Optional, Union
from datasets import load_dataset
import torch
import json
import os

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
        

def att_extract_(
    model: PreTrainedModel, 
    model_name, 
    data_name, 
    session_id, 
    layer_id: int,
    input_ids: torch.LongTensor,
    use_cache: Optional[bool] = False,
    past_key_values: Optional[Cache] = None,
) :
    
    # 提前创建目录
    dir_path = f"Attention_weights/{model_name}/{data_name}/session_{session_id}/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    with torch.no_grad():
        # 使用模型的前向传播获取所有隐藏状态
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,  # 获取所有隐藏状态
            return_dict=True,
            use_cache=use_cache,
            past_key_values=past_key_values
        )
        
        # 保存所有隐藏状态
    for i, hidden_state in enumerate(outputs.hidden_states):
        file_path = os.path.join(dir_path, f"hidden_state_{i}.pt")
        torch.save(hidden_state.cpu(), file_path)

