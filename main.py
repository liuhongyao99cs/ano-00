import os
import sys
import time
import torch
import numpy as np
import threading
import argparse
import pickle
import concurrent.futures
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from src import *
from WiKV_interface.WiKV_Controller import WiKV_Controller
from WiKV_interface.WiKV_Encoder import WiKV_Encode
from huggingface_hub import login

# =============================================
# Main controller of WiKV
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

def process_batch(encoder,args,session_id, batch_id):
    # 注意：如果 encoder 不是线程安全 or 有内部状态冲突，应创建副本或重设计
    encoder.calculate_dist_matrix(batch_id=batch_id)
    solu = encoder.constrained_two_opt(batch_id=batch_id)
    
    save_path = f"{args.save_encode_dir}/seq_inflation_{session_id}_batch{batch_id}_.pt"
    torch.save(solu, save_path)
    
    return batch_id, solu  # 可选：返回结果用于后续处理



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

    # load dataset from jsonl
    dataset = args.path_to_context  #f"/home/hoongyao/data/test_data/{data_name}.jsonl"
    data = load_testcases(dataset)

    # Initialize the WiKV controller
    controller = WiKV_Controller(args=args,model=model, tokenizer = tokenizer, shape=(1000, 128), dtype=torch.float32, threshold=0.25)
    
    # learn the predictor
    # controller.Metric()
    controller.boundary_predictor()

    if not os.path.exists(args.save_encode_dir):
        os.makedirs(args.save_encode_dir, exist_ok=True)
    

    # loop all samples in the dataset
    for session_id in range(args.end-args.start):
        
        if data_name in ['longchat', 'tqa', 'nqa']:
            input_text = data[session_id]['prompt'] 
        else:
            input_text = data[session_id]['context']
            
        inputs_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

        input_ids = inputs_ids['input_ids']
        attention_mask = inputs_ids['attention_mask']

        seq_len = input_ids.shape[1]

        encoder = WiKV_Encode(args=args, seq_len=seq_len, config=model.config, session=session_id, window_size=model.config.num_hidden_layers, device=next(model.parameters()).device)
        encoder.Att_Loading()
        kv_quant, kv_dequant = encoder.Semantic_Encode()

        torch.save(kv_quant, f"{args.save_encode_dir}/kv_quant_{session_id}.pt")
        torch.save(encoder.sorted_sequence, f"{args.save_encode_dir}/seq_semantic_{session_id}.pt")

        generated = model.generate(
            input_ids, 
            attention_mask = attention_mask,
            past_key_values=kv_dequant, 
            max_new_tokens = 40, 
            return_dict_in_generate=True, 
            eos_token_id=tokenizer.eos_token_id, 
            pad_token_id=tokenizer.eos_token_id, 
            output_scores=True
        )
        
        prediction = tokenizer.decode(generated.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(prediction)

        # we conduct inflation control on the semantic sequances in each batch
        # load semantic_seq and inflation_control_seq for modification
        # delta coding on modified semantic_seq
        
        # Integrate with decoding paradigm 
        st = time.perf_counter()

        generated = model.generate(
            input_ids, 
            attention_mask = attention_mask,
            past_key_values=kv_dequant, 
            max_new_tokens = 1, 
            return_dict_in_generate=True, 
            eos_token_id=tokenizer.eos_token_id, 
            pad_token_id=tokenizer.eos_token_id, 
            output_scores=True
        )
        kv = generated['past_key_values']
        kv = list(kv)
        key_value = []
        for i in range(len(kv)):
            kv[i] = list(kv[i])
            kv[i][0] = kv[i][0][:, :, :-1][0]
            kv[i][1] = kv[i][1][:, :, :-1][0]
            kv[i] = tuple(kv[i])
        kv = tuple(kv)
        kv = to_blob(kv)
        m1 = K_coverage(generated.scores[0]).item()
        m2 = entropy(generated.scores[0]).item()
        data = np.column_stack((m1,m2))
        decide = controller.model.decision_function(data)
        print(kv.shape, len(generated.scores), m1, m2, decide, input_ids.shape,generated.sequences[0].shape)

        ed = time.perf_counter()
        elapsed_time = ed - st
        print(f"Model .generate time: {elapsed_time:.3f} s")


        '''
        print(encoder.kv_seq_len)
        encoder.Inflation_Seq(session_id)
        semantic_seq, code_size = encoder.Inflation_Control(session_id)


        kv_dequant = to_blob(kv_dequant)
        kv_dequant = kv_dequant.squeeze(2)
        print(kv_dequant.shape)
        
        controller.kv_pool_initialize(kv_dequant)
        controller.start_kv_fill(semantic_seq=semantic_seq, bw_trace=[850,670,960,950,1020,780,640,890.660,780,890,1000,850,670,960,950,1020,780,640,890.660,780,890,1000], kv_gpu=kv_dequant, code_size=code_size)
        def probe_task():
            while (True):
                tensor = controller.probe(target_device='cuda:0')
                if controller.full_event.is_set():
                    break

        probe_thread = threading.Thread(target=probe_task)
        probe_thread.start()
        probe_thread.join()
        '''

        #print(solu)
        #print(max(max(dist_matrix)),min(min(dist_matrix)))
        # print(model.config)
        # WiKV_Encoder.Semantic_Encode()


    # A controller that overlaps KV cache streaming and decoding

'''
if __name__ == "__main__":
    # 创建 Controller，管理一个 (1000, 128) 的 tensor
    initial_th = 0.25
    controller = WiKV_Controller(shape=(1000, 128), dtype=torch.float32, threshold=initial_th)

    
    def probe_task():
        while (True):
            tensor = controller.probe(target_device='cuda:0')
            print(f"🚀 获取到 tensor，shape={tensor.shape}, device={tensor.device}")
            if controller.full_event.is_set():
                print("probe task is ended...\n")
                break

    probe_thread = threading.Thread(target=probe_task)
    probe_thread.start()

    for _ in range(100):
        time.sleep(0.1)
        progress = controller.get_progress()
        print(f"📊 当前填充进度: {progress:.2%}")
        if progress >= 1.0:
            break

    probe_thread.join()
    print("🔚 主程序结束")
'''
