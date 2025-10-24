import os
import copy
import json
import heapq
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from collections import Counter
from typing import Callable, Optional, Union

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import repeat_kv, apply_rotary_pos_emb
#from transformers.models.mistral.modeling_mistral import repeat_kv, apply_rotary_pos_emb
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# ====================================
# load dataset from json file
# ====================================

def load_testcases(test_file):
    with open(test_file, 'r', encoding='utf-8') as json_file:
        json_list = list(json_file)

    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)

    return test_cases

# ====================================
# KV cache load in tensor, transfer to tuple for inference
# ===================================

def tensor_to_tuple(kv):
    """ Convert a tensor to a list of tuples
    Input tensor's shape should be (num_layers, 2, num_heads, seq_len, heads_dim)
    """
    new_kv = []
    for i in range(len(kv)):
        new_kv.append((kv[i][0].unsqueeze(0), 
                       kv[i][1].unsqueeze(0)))
    return tuple(new_kv)

def tensor_to_past_key_values(kv_tensor):
    """
    Convert [num_layers, 2, num_heads, seq_len, head_dim] 
    to transformers.DynamicCache (compatible with Qwen3).
    
    Args:
        kv_tensor: torch.Tensor of shape [L, 2, H, S, D]
    
    Returns:
        cache: DynamicCache
    """
    num_layers = kv_tensor.shape[0]
    
    keys = kv_tensor[:, 0, :, :, :]  # [36, 8, seq, 128]
    vals = kv_tensor[:, 1, :, :, :]  # [36, 8, seq, 128]
    
    keys = keys.unsqueeze(1)
    vals = vals.unsqueeze(1)
    
    cache = DynamicCache()
    
    for i in range(num_layers):
        cache.update(keys[i], vals[i], layer_idx=i)
    
    return cache

def to_blob(kv_tuples):
    """ Transform a list of tuples of key and value tensors to a single tensor
    """
    return torch.stack([torch.stack(inner_tuple, dim=0).to("cuda:0") for inner_tuple in kv_tuples], dim=0)

def to_blob_cpu(kv_tuples):
    """ Transform a list of tuples of key and value tensors to a single tensor
    """
    return torch.stack([torch.stack(inner_tuple, dim=0).to("cpu") for inner_tuple in kv_tuples], dim=0)

# ==================================
# Delta encode / decode
# ==================================

def delta_encode(tensor_2d):
    """
    Delta encoding between kv vectors (PyTorch version)
    
    Args:
        tensor_2d: 2D torch.Tensor of shape (n_samples, n_features)
    
    Returns:
        flat_deltas: 1D torch.Tensor (flattened delta-encoded data)
        first_sample: torch.Tensor, the first row of original tensor
    """
    # Ensure input is a tensor
    if not isinstance(tensor_2d, torch.Tensor):
        tensor_2d = torch.as_tensor(tensor_2d)
    
    if tensor_2d.dim() != 2:
        raise ValueError("Input must be 2D tensor")
    
    n_samples, n_features = tensor_2d.shape
    if n_samples == 0:
        return torch.tensor([]).to(tensor_2d.dtype), torch.tensor([]).to(tensor_2d.dtype)
    
    # Create deltas tensor
    deltas = torch.empty_like(tensor_2d)
    deltas[0] = tensor_2d[0]
    
    if n_samples > 1:
        deltas[1:] = tensor_2d[1:] - tensor_2d[:-1]
    
    flat_deltas = deltas.flatten()
    first_sample = tensor_2d[0].clone()
    
    return flat_deltas, first_sample


def delta_decode(flat_deltas, first_sample, n_samples):
    """
    Decode 1D delta-encoded data back to 2D tensor (PyTorch version)
    
    Args:
        flat_deltas: 1D torch.Tensor
        first_sample: torch.Tensor, shape (n_features,)
        n_samples: int, number of original rows
    
    Returns:
        original: 2D torch.Tensor of shape (n_samples, n_features)
    """
    if not isinstance(flat_deltas, torch.Tensor):
        flat_deltas = torch.as_tensor(flat_deltas)
    if not isinstance(first_sample, torch.Tensor):
        first_sample = torch.as_tensor(first_sample)
    
    n_features = first_sample.numel()
    expected_length = n_samples * n_features
    if flat_deltas.numel() != expected_length:
        raise ValueError(f"flat_deltas length mismatch: got {flat_deltas.numel()}, expected {expected_length}")
    
    # Reshape to 2D
    deltas = flat_deltas.reshape(n_samples, n_features)
    
    # Reconstruct original
    original = torch.empty_like(deltas)
    original[0] = first_sample
    
    if n_samples > 1:
        # Cumulative sum of deltas[1:], then add first_sample
        original[1:] = torch.cumsum(deltas[1:], dim=0) + first_sample
    
    return original

# ===================================
# Huffman ecoding 
# ===================================
class HuffmanCodec:
    def __init__(self):
        self.codebook = {}      # symbol -> code (string of '0'/'1')
        self.reverse_codebook = {}  # code -> symbol
    
    def build_codebook(self, symbols):
        if len(symbols) == 0:
            return
        
        # 
        freq = Counter(symbols)
        
        # if only one symbol
        if len(freq) == 1:
            symbol = next(iter(freq))
            self.codebook = {symbol: '0'}
            self.reverse_codebook = {'0': symbol}
            return
        
        # construct Huffman tree
        heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
        #construct Huffman code
        self.codebook = dict(heapq.heappop(heap)[1:])
        self.reverse_codebook = {code: symbol for symbol, code in self.codebook.items()}
    
    def encode(self, symbols):
        """symbol sequence into """
        if not self.codebook:
            raise ValueError("Codebook not built. Call build_codebook() first.")
        
        return ''.join(self.codebook[symbol] for symbol in symbols)
    
    def decode(self, encoded_bits):
        """covert bitstream into symbol seq"""
        if not self.reverse_codebook:
            raise ValueError("Codebook not built.")
        
        decoded = []
        current_code = ""
        
        for bit in encoded_bits:
            current_code += bit
            if current_code in self.reverse_codebook:
                decoded.append(self.reverse_codebook[current_code])
                current_code = ""
        
        if current_code:
            raise ValueError("Invalid encoded data: incomplete code at end")
        
        return decoded
    
    def save_codebook(self, filepath):
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.codebook, f)
    
    def load_codebook(self, filepath):
        
        with open(filepath, 'rb') as f:
            self.codebook = pickle.load(f)
        self.reverse_codebook = {code: symbol for symbol, code in self.codebook.items()}

def bits_to_bytes(bit_string):
    # pad the bit_string to 8*i
    if len(bit_string) % 8 != 0:
        # pad 0 to the last 
        bit_string = bit_string.ljust((len(bit_string) + 7) // 8 * 8, '0')
    
    # transfer 8bit to 1 byte
    byte_array = bytearray()
    for i in range(0, len(bit_string), 8):
        byte_str = bit_string[i:i+8]
        byte_val = int(byte_str, 2) 
        byte_array.append(byte_val)
    
    return bytes(byte_array)

# ==================================
# quantization: layer wise quantization
# ==================================

def layer_quantization(kv, bin, N):
    """ 
    Layer-wise quantize the key value tensors into tuple of key and value tensors
    bin is 2^bit 
    max_tensors is the scalable mark 
    N is the layer number that shares the same quantization bins
    """
    channels = kv.shape[-1] * kv.shape[-3]
    max_tensors = None
    for i in range(len(kv)):
        key = kv[i][0]
        value = kv[i][1]
        key = key.permute((1, 0, 2)).reshape(kv.shape[-2], channels)
        value = value.permute((1, 0, 2)).reshape(value.shape[-2], channels)

        bins = bin[i//N]
        
        key, maxk = torch_quant(bins, key)
        value, maxv = torch_quant(bins, value)
        quant_key = key.reshape(kv[i][0].shape[-2], kv[i][0].shape[-3], kv[i][0].shape[-1]).permute((1, 0, 2))
        quant_value = value.reshape(kv[i][1].shape[-2], kv[i][1].shape[-3], kv[i][1].shape[-1]).permute((1, 0, 2))
        kv[i][0] = quant_key
        kv[i][1] = quant_value
        concated_max = torch.cat((maxk.unsqueeze(0), maxv.unsqueeze(0)), dim=0)

        if max_tensors is None:
            max_tensors = concated_max.unsqueeze(0)
        else:
            max_tensors = torch.cat((max_tensors, concated_max.unsqueeze(0)), dim=0)
        
    return kv.to(torch.int8), max_tensors

def torch_quant(bins: int, qA: torch.Tensor):
    """
    Quantize a float tensor to fixed number of bins

    Input:
        bins: number of bins
        qA: the input tensor

    Returns:
        xq: the quantized tensor, in float32
        max1: the maximum value of the tensor
    """
    MAX = bins // 2 - 1
    C = MAX
    max1 = torch.amax(torch.abs(qA), dim=-1, keepdim=True)
    xq = torch.round(qA * (C / max1)).to(torch.int8)
    
    x = (xq / C * max1).to(torch.float16)
    
    return xq, max1


def torch_dequant(bins: int, xq: torch.Tensor, max1: torch.Tensor):
    """
    Dequantize a quantized tensor

    Input:
        bins: number of bins
        xq: the quantized tensor
        max1: the maximum value of the tensor

    Returns:
        x: the dequantized tensor
    """
    MAX = bins // 2 - 1
    C = MAX
    x = (xq / C * max1).to(torch.float16)
    return x

def layer_dequantize(kv, max_tensors, bin, N):
    """
    bin is 2^bit 
    max_tensors is the scalable mark 
    N is the layer number that shares the same quantization bins
    """
    channels = kv.shape[-1] * kv.shape[-3]
    kv = kv.to(torch.float16)
    for i in range(len(kv)):
        key = kv[i][0]
        value = kv[i][1]
        key = key.permute((1, 0, 2)).reshape(kv.shape[-2], channels)
        value = value.permute((1, 0, 2)).reshape(value.shape[-2], channels)

        bins = bin[i//N]

        dequant_k = torch_dequant(bins, key, max_tensors[i][0])
        dequant_v = torch_dequant(bins, value, max_tensors[i][1])
        dequant_key = dequant_k.reshape(kv[i][0].shape[-2], kv[i][0].shape[-3], kv[i][0].shape[-1]).permute((1, 0, 2))
        dequant_value = dequant_v.reshape(kv[i][1].shape[-2], kv[i][1].shape[-3], kv[i][1].shape[-1]).permute((1, 0, 2))
        kv[i][0] = dequant_key
        kv[i][1] = dequant_value

    return tensor_to_tuple(kv)

# ===================================
# compute metrics to judge attention accumulation
# ===================================

def K_coverage(scores,temp=1, K=50):
    scores = scores.squeeze(0)
    v = F.softmax(scores/temp,dim=-1)
    v_k, ind_k = torch.topk(v, K)
    ratio = torch.sum(v_k) / torch.sum(v)
    
    return ratio

def entropy(scores,temp=1, K=100):
    scores = scores.squeeze(0)
    v = F.softmax(scores/temp,dim=-1)
    v_k, ind_k = torch.topk(v, K)
    v_k = torch.clamp(v_k, min=1e-10)
    v_k = v_k / torch.sum(v_k)
    entropyx = -torch.sum(v_k * torch.log(v_k))
    
    return entropyx

# =================================
# Inflation control tool fuction
# =================================

def constrained_two_opt(initial_solution, distance_matrix, max_deviation, max_iter = 10, improve_threshold=1e-6):
    
    """
    position constrained 2-opt ALGORITHM

    Args:
        initial_solution: initial list e.g. [0, 2, 1, 3, 4]
        distance_matrix: distance matrix (N*N)
        max_deviation: MAX POS DEVIATION-> d
        improve_threshold: IMPROVEMENT THRESHOLD

    Returns:
        best_solution: NEW PATH
        best_distance: NEW DIST
    """

    # map initial nodes 
    original_positions = {node: idx for idx, node in enumerate(initial_solution)}
    n = len(initial_solution)

    def calculate_total_distance(path):
        
        total = 0.0
        for i in range(1,n):
            total += distance_matrix[path[i-1], path[i]]
        return total

    def is_valid_swap(i, j, seq):
        """
        whether constraint is meeted
        """
        
        original_pos = [seq[i], seq[j]]
        new_pos = [j,i]

        if (abs(new_pos[0]-original_pos[0])>max_deviation) or (abs(new_pos[1]-original_pos[1])>max_deviation):
            return False
            
        return True

    seq = copy.deepcopy(initial_solution)
    n = len(seq)

    best_solution = copy.deepcopy(seq)
    best_distance = calculate_total_distance(best_solution)
    
    for iter in range(max_iter):
        improved = False
        best_swap = None
        best_new_distance = best_distance

        for i in range(0,n,2):
            for j in range(i+1,n):
                if not is_valid_swap(i,j,best_solution):
                    continue

                seq[i], seq[j] = seq[j], seq[i]
                new_dist = calculate_total_distance(seq)

                if new_dist < best_new_distance:
                    best_new_distance = new_dist
                    best_swap = (i, j)
                    improved = True
                
                seq[i], seq[j] = seq[j], seq[i]
        
        if not improved:
            break

        i, j = best_swap
        seq[i], seq[j] = seq[j], seq[i]
        best_distance = best_new_distance
        best_solution = copy.deepcopy(seq)

    return best_solution, best_distance

        

def hidden_extract_(
    model: PreTrainedModel, 
    model_name, 
    data_name, 
    session_id, 
    attention_mask,
    save_dir: str,
    input_ids: torch.LongTensor,
    use_cache: Optional[bool] = False,
    past_key_values: Optional[Cache] = None,
) :

    device = next(model.parameters()).device
    hidden_states = model.model.embed_tokens(input_ids)
    position_ids = torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0)
        
    use_cache = None
    past_key_values = None
    cache_position = None
    
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=model.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + model.model.embed_tokens(input_ids).shape[1], device=model.model.embed_tokens(input_ids).device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    if model.config.model_type == 'qwen3':
        mask_kwargs = {
                "config": model.config,
                "input_embeds": model.model.embed_tokens(input_ids),
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
        causal_mask = {
                "full_attention": create_causal_mask(**mask_kwargs),
        }
    elif model.config.model_type == 'mistral':
        mask_function = create_causal_mask if model.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=model.config,
            input_embeds=model.model.embed_tokens(input_ids),
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
    elif model.config.model_type == 'llama':
        causal_mask = create_causal_mask(
            config=model.config,
            input_embeds=model.model.embed_tokens(input_ids),
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

    layer = model.model.layers[0]
    rotary = model.model.rotary_emb
    layer_norm = layer.input_layernorm
    q_proj = layer.self_attn.q_proj
    k_proj = layer.self_attn.k_proj
    if model.config.model_type == 'qwen3':
        q_norm = layer.self_attn.q_norm
        k_norm = layer.self_attn.k_norm
    

    hidden_states = layer_norm(hidden_states)
    file_path = os.path.join(save_dir, f"hidden_s{session_id}_l{0}.pt")
    torch.save(hidden_states.cpu(), file_path)  
    
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, model.config.head_dim)
    

    query_states = q_proj(hidden_states).view(hidden_shape)
    if model.config.model_type == 'qwen3':
        query_states = q_norm(query_states).transpose(1, 2)
    else:
        query_states = query_states.transpose(1, 2)
    
    key_states = k_proj(hidden_states).view(hidden_shape)
    if model.config.model_type == 'qwen3':
        key_states = k_norm(key_states).transpose(1, 2)
    else:
        key_states = key_states.transpose(1, 2)
    
    position_embeddings = rotary(key_states, position_ids) 
    del key_states, query_states
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            if model.config.model_type == 'qwen3':
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask[layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    )
            file_path = os.path.join(save_dir, f"hidden_s{session_id}_l{i+1}.pt")
            torch.save(hidden_states.cpu(), file_path)    
    '''
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,  
            return_dict=True,
            use_cache=use_cache,
            past_key_values=past_key_values
        )
    '''    
    #for i, hidden_state in enumerate(outputs.hidden_states):
    #    file_path = os.path.join(save_dir, f"hidden_s{session_id}_l{i}.pt")
    #    torch.save(hidden_state.cpu(), file_path)
        

def to_blob(kv_tuples):
    """ Transform a list of tuples of key and value tensors to a single tensor
    """
    return torch.stack([torch.stack(inner_tuple, dim=0).to("cuda:0") for inner_tuple in kv_tuples], dim=0)

def layer_atten_extract_(model, input_ids, attention_mask, layer_id, args, session_id=0):
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
        
        use_cache = None
        past_key_values = None
        cache_position = None
        
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=model.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + model.model.embed_tokens(input_ids).shape[1], device=model.model.embed_tokens(input_ids).device
            )

        mask_kwargs = {
                "config": model.config,
                "input_embeds": model.model.embed_tokens(input_ids),
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
        
        causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
        }
        
        # load model layer
        #print(model.config)
        #print(model.config.model_type)
        layer = model.model.layers[layer_id]
        rotary = model.model.rotary_emb
        layer_norm = layer.input_layernorm
        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        if model.config.model_type == 'qwen3':
            q_norm = layer.self_attn.q_norm
            k_norm = layer.self_attn.k_norm
        

        hidden_states = layer_norm(hidden_states)
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, model.config.head_dim)
        

        query_states = q_proj(hidden_states).view(hidden_shape)
        if model.config.model_type == 'qwen3':
            query_states = q_norm(query_states).transpose(1, 2)
        else:
            query_states = query_states.transpose(1, 2)
        
        key_states = k_proj(hidden_states).view(hidden_shape)
        if model.config.model_type == 'qwen3':
            key_states = k_norm(key_states).transpose(1, 2)
        else:
            key_states = key_states.transpose(1, 2)
        

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
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        # control the cuda memory
        del query_states, key_states, cos, sin, position_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        
        

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float16)
        attn_weights = F.dropout(attn_weights, p=0, training=False)
        
        attn_weights = attn_weights.squeeze(0)
        attn_weights = attn_weights.view(model.config.num_attention_heads // model.config.num_key_value_heads, model.config.num_key_value_heads, *attn_weights.shape[1:])
        attn_weights = attn_weights.sum(dim=0)
        # print(attn_weights.shape)
        attn_path = os.path.join(args.save_att_dir, f"attn_s{session_id}_l{layer_id}.pt")
        torch.save(attn_weights, attn_path)
        

        del attn_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def atten_extract_(model, input_ids, attention_mask, args, session_id=0):
    """
    process all layers in the model
    """
    os.makedirs(args.save_att_dir, exist_ok=True)
    os.makedirs(args.save_hid_dir, exist_ok=True)
    
    for layer_id in range(model.config.num_hidden_layers):
        print(f"Compute the attention weights of {layer_id}-layer...\n")
        layer_atten_extract_(model, input_ids, attention_mask, layer_id, args, session_id)
        
       
'''
def probe_task(kv_pace, kv_tuple, input_idx, attention_maskx, controller, model, tokenizer, decode_token):
    idx = 0
    for k in range(100):
        
        if (k == 0 ):
                # warm-up dummy cycle
            for t in range(1):
                with torch.no_grad():
                    generated = model.generate(
                        input_idx, 
                        attention_mask = attention_maskx,
                        past_key_values=kv_tuple, 
                        max_new_tokens = 1, 
                        return_dict_in_generate=True, 
                        output_scores=True
                    )
            
            kv_tuplex = generated['past_key_values']
            kv_tuple = DynamicCache()

            kv_list = list(kv_tuplex)
            for j in range(len(kv_list)):
                kv_list[j] = list(kv_list[j])
                kv_list[j][0] =  kv_list[j][0][:,:,:-1,:]
                kv_list[j][1] =  kv_list[j][1][:,:,:-1,:]
                #print(kv_list[j][0].shape)
                kv_tuple.update(kv_list[j][0],kv_list[j][1],j)

            del kv_list, kv_tuplex
            controller.warm_up.set()
        
            st = time.perf_counter()


        # if KV cache is not fully streamed
        if not controller.full_event.is_set():

            while (True):

                start = time.perf_counter()
                kv_tuple = controller.probe_tuple(kv_tuple, semantic_seq, target_device='cuda:0')
                end = time.perf_counter()
                elapsed_time = end - start
                print(f"Wait streaming and copy data time: {elapsed_time:.4f}s")

                start = time.perf_counter()
                with torch.no_grad():
                    generated = model.generate(
                        input_idx, 
                        attention_mask = attention_maskx,
                        past_key_values=kv_tuple, 
                        max_new_tokens = 1, 
                        return_dict_in_generate=True, 
                        output_scores=True
                    )

                end = time.perf_counter()
                elapsed_time = end - start
                print(f"Decode token time: {elapsed_time:.4f}s")
                #end = time.perf_counter()
                #elapsed_time = end - st
                #print(f"MOdel time: {elapsed_time:.2f}s")
                
                #next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
                #input_ids = torch.cat([input_ids, next_token_id], dim=1)
                #kv_pace = outputs.past_key_values

                # check the confidence 
                start = time.perf_counter()
                m1 = K_coverage(generated.scores[0]).item()
                m2 = entropy(generated.scores[0]).item()
                data = np.column_stack((m1, m2))
                decide = controller.model.decision_function(data)[0]
                end = time.perf_counter()
                elapsed_time = end - start
                print(f"Metric decide time: {elapsed_time:.4f}s with score {decide}")
                
                
            
                #print(f"Pred score: {decide}")

                if (decide < -0.1):
                    controller.step = 0.2 / ( 1 + 10 * math.e ** (-decide / 20))
                    continue
                else:
                    
                    end = time.perf_counter()
                    if k == 0 : 
                        ttft = end - st
                    
                    token = tokenizer.decode(generated.sequences[0][-1], skip_special_tokens=True)
                    print(token, end="", flush=True)
                    input_idx = (generated.sequences[0]).unsqueeze(0)
                    new_token = torch.tensor([[1]], device=attention_maskx.device)
                    attention_maskx = torch.cat([attention_maskx, new_token], dim=1)
                    kv_tuple = generated['past_key_values']
                    decode_token.append(generated.sequences[0][-1])
                    controller.step = 0.1
                    
                    #print(f"Decoded token_num: {len(decode_token)}")
                    break
        
        # all KV cache is streamed
        else:
            with torch.no_grad():
                
                generated = model.generate(
                    input_idx, 
                    attention_mask = attention_maskx,
                    past_key_values=kv_tuple, 
                    max_new_tokens = 1, 
                    return_dict_in_generate=True, 
                    eos_token_id=tokenizer.eos_token_id, 
                    pad_token_id=tokenizer.eos_token_id, 
                    output_scores=True
                )
            #end = time.time()
            #elapsed_time = end - start
            #print(f"Model time: {elapsed_time:.2f}s")
            input_idx = (generated.sequences[0]).unsqueeze(0)
            new_token = torch.tensor([[1]], device=attention_maskx.device)
            attention_maskx = torch.cat([attention_maskx, new_token], dim=1)
            decode_token.append(generated.sequences[0][-1])
            kv_tuple = generated['past_key_values']
            # print("KV length from model():", kv_tuple.get_seq_length())
            token = tokenizer.decode(generated.sequences[0][-1], skip_special_tokens=True)

            print(token, end="", flush=True)
            #print(f"Decoded token_num: {len(decode_token)}")
'''
