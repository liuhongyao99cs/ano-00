import torch
import torch.nn.functional as F
import time
import threading
import argparse
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from src.utils import *
from WiKV_Interface.WiKV_Controller import WiKV_Controller
from WiKV_Interface.WiKV_Encoder import WiKV_Encode
from huggingface_hub import login

# =============================================
# Compute the metrics of decoding tokens with full att
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

if not os.path.exists(args.save_metric_dir):
    os.makedirs(args.save_metric_dir, exist_ok=True)


for session_id in range(args.start, args.end):
    
    if data_name in ['longchat', 'tqa', 'nqa']:
        input_text = data[session_id]['prompt'] 
    else:
        input_text = data[session_id]['context']
        
    inputs_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

    input_ids = inputs_ids['input_ids']
    attention_mask = inputs_ids['attention_mask']

    seq_len = input_ids.shape[1]

    # WiKV_Encoder = WiKV_Encode(args=args, seq_len=seq_len, config=model.config, session=session_id,device=next(model.parameters()).device)
    # WiKV_Encoder.Att_Loading()
    # WiKV_Encoder.Semantic_Encode()

    # full KV cache contains full attention
    raw_kv = torch.load(f"{args.save_kv_dir}/raw_kv_{session_id}.pt")
    kv = tensor_to_tuple(raw_kv)
    # generate logit scores through model.generate
    generated = model.generate(input_ids, past_key_values=kv, max_new_tokens = 100, return_dict_in_generate=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, output_scores=True)
    prediction = tokenizer.decode(generated.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(prediction)
    print("Dumping the metrics: K-top and entropy for decoded tokens...")
    k_top = []
    entro = []
    for k in range(len(generated.scores)):
        k_top.append(K_coverage(generated.scores[k]).item())
        entro.append(entropy(generated.scores[k]).item())
    torch.save(k_top, f"{args.save_metric_dir}/k_top_{session_id}.pt")
    torch.save(entro, f"{args.save_metric_dir}/entro_{session_id}.pt")
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
