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

def to_blob(kv_tuples):
    """ Transform a list of tuples of key and value tensors to a single tensor
    """
    return torch.stack([torch.stack(inner_tuple, dim=0).to("cuda:0") for inner_tuple in kv_tuples], dim=0)

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
        """根据符号序列构建 Huffman 码本"""
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

        for i in range(n):
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
        
        attention_mask=causal_mask_mapping['full_attention']

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
        print(attn_weights.shape)
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
        
       
