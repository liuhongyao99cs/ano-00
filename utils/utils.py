from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import matplotlib.pyplot as plt
from typing import Callable, Optional, Union
from datasets import load_dataset

def load_testcases(test_file):
    with open(test_file, 'r', encoding='utf-8') as json_file:
        json_list = list(json_file)

    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)

    return test_cases

def att_extract_(
    model: PreTrainedModel, 
    layer_id: int,
    input_ids: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = False,
    past_key_values: Optional[Cache] = None,
) -> torch.tensor:
    
    ## extract modules in model
    embed_tokens = model.model.embed_tokens
    rotary_emb = model.model.rotary_emb

    inputs_embeds = embed_tokens(input_ids)

    ## decoding parameters define
    use_cache = None
    past_key_values = None
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
        hidden_states, attention_weights = self_att(
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
        attention_weight = attention_weights.detach().cpu()
        del attention_weights
        attn_head = attention_weight[0]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(attn_head[0,1:,:].numpy(), cmap='viridis', aspect='auto')

    return attention_weight