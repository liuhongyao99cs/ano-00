import torch
from pathlib import Path
import torch.nn.functional as F
import time
import argparse
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

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
p.add_argument("--save_dir", type=str)
args = p.parse_args()

model_name = args.model_id #"Qwen/Qwen3-4B"  # 
model_N = args.model #"Qwen3-4B"
data_name = args.dataset_name

# your hf account
login(token = "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")

if __name__ == "__main__":
    # Check if save_dir exists
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # load model, remember use 4bit, half() and flash_attention_2 to reduce memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,              # 开启 4-bit 加载
        bnb_4bit_use_double_quant=True, # 开启双重量化 (进一步节省显存)
        bnb_4bit_quant_type="nf4",      # 使用 NF4 格式 (精度损失最小)
        bnb_4bit_compute_dtype=torch.bfloat16 # 计算时使用的精度 (建议保持 bf16)
    )

    if data_name in ['videomme']:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,  
            device_map="auto",               
            attn_implementation="flash_attention_2"
        )

        processor = AutoProcessor.from_pretrained(model_name)

    else:

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config, 
            dtype=torch.float16, 
            attn_implementation="flash_attention_2",
            device_map="auto",
            output_attentions=False
        )

    # process dataset, assume we are testing 40K tokens
    dataset = args.path_to_context  #f"/home/hoongyao/data/test_data/{data_name}.jsonl"
    data = load_testcases(dataset)

for session_id in range(args.start, args.end):
    
    if data_name in ['longchat', 'tqa', 'nqa']:
        input_text = data[session_id]['prompt'] 
    elif data_name in ['hotpotqa']:
        input_text = data[session_id]['context'] + "Based on given passages, answer the question: " + data[session_id]['input'] 
    elif data_name in ['gov_report']:
        input_text = data[session_id]['context'] + "Summarize the given context in 250 tokens." 
    elif data_name in ['videomme']:
        input_text = "Please answer the following multiple-choice question. Select the correct option (A, B, C, or D) and provide a brief explanation for your choice. Format your response as: Answer: [Option] Explanation: [Your reasoning]" + data[session_id]['question']

    
    # seperate VLM and LLM tasks
    if data_name in ['videomme']:
        url = data[session_id]["url"]
        video_path = Path(dataset).parent
        download_youtube_video(url=url, session_id=session_id, output_folder=video_path)
        video =video_path/f"{session_id}.mp4"
        frames = extract_frames(
            video_path = video,
            time_interval=0.5,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video,
                        "max_pixels": 360 * 420,
                        "fps": 2.0, # 降低FPS以减少token数量方便演示
                    },
                    {"type": "text", "text": input_text},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            videos=[frames],  
            padding=True,
            return_tensors="pt",
        )


        # Move inputs to the same device as the model (GPU)
        inputs = inputs.to(model.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        print("Generating response...")
        # Video-MME often requires longer generation windows (max_new_tokens)
        generated_ids = model.generate(**inputs, max_new_tokens=100, return_dict_in_generate=True)
        print(input_ids.shape[1])
        print(generated_ids.sequences.shape)

        # 5. Decode Output
        # Trim the input tokens from the output to get only the generated text
        answer = processor.batch_decode(
            [generated_ids.sequences[0,input_ids.shape[1]:]],  # 直接传一维列表
            skip_special_tokens=True
        )

        print("-" * 20)
        print("Model Output:")
        print(answer[0])
        print("-" * 20)
        print("Model answer:")
        print(data[session_id]['answer'])
        
    else:
        inputs_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_ids = inputs_ids['input_ids']
        attention_mask = inputs_ids['attention_mask']
        
    print(f"Context length{input_ids.shape} tokens...")
    print(f"Saving the KV cache of dataset: {data_name}, doc {session_id}...")

    
    # since 1 new token, we delete the last token's KV cache and then store
    kv = generated_ids['past_key_values']

    kv = list(kv)
    key_value = []
    for i in range(len(kv)):
        kv[i] = list(kv[i])
        kv[i][0] = kv[i][0][:, :, :input_ids.shape[1]-1][0]
        kv[i][1] = kv[i][1][:, :, :input_ids.shape[1]-1][0]
        kv[i] = tuple(kv[i])
    kv = tuple(kv)
    kv_tensor = to_blob(kv)
    

    
    torch.save(kv_tensor, f"{args.save_dir}/raw_kv_{session_id}.pt")
    if session_id == 0:
        pickle.dump(kv, open(f"{args.save_dir}/raw_kv_{session_id}.pkl", "wb"))

    if data_name in ['videomme']:
        kvx = torch.load(f"{args.save_dir}/raw_kv_{session_id}.pt")
        bin_list = [42,42,42,32,32]
        layer_group = 9
        kv_quant, max_q = layer_quantization(kvx, bin_list, layer_group)
        kv_dequant = layer_dequantize(kv_quant, max_q, bin_list, layer_group)
        kvx = tensor_to_tuple(kv) #kv_dequant

        generated = model.generate(
                        **inputs,       
                        past_key_values=kv_dequant,         
                        max_new_tokens=100,
                        return_dict_in_generate=True,
                        use_cache = True   
                    )

        answer = processor.batch_decode(
                [generated.sequences[0,input_ids.shape[1]:]],  # 直接传一维列表
                skip_special_tokens=True
            )

        print("-" * 20)
        print("Model Output with KV cache:")
        print(answer)
        print("-" * 20)
