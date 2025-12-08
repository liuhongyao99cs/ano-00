import torch
import argparse
from src import *
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info


# 1. Load the Model and Processor
# We use Qwen2_5_VLForConditionalGeneration specifically for this version.
model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

print(f"Loading {model_path}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 开启 4-bit 加载
    bnb_4bit_use_double_quant=True, # 开启双重量化 (进一步节省显存)
    bnb_4bit_quant_type="nf4",      # 使用 NF4 格式 (精度损失最小)
    bnb_4bit_compute_dtype=torch.bfloat16 # 计算时使用的精度 (建议保持 bf16)
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config=bnb_config,  # <--- 传入量化配置
    device_map="auto",               # 自动分配到 GPU
    attn_implementation="flash_attention_2" # 依然推荐开启 Flash Attention
)

processor = AutoProcessor.from_pretrained(model_path)

datafile = "/home/hongyao/WiKV/Datasets/video_mme_test.jsonl"
data = load_testcases(datafile)
url = data[0]["url"]
video_path = "/home/hongyao/WiKV/Datasets/VideoMME_video"
download_youtube_video(url=url, output_folder=video_path)

video_path = "/home/hongyao/WiKV/Datasets/VideoMME_video/fFjv93ACGo8.mp4" 
question = "Answer the question with one answer of A, B, C, D based on the provided video frames." + data[0]['question'] 

frames = extract_frames(
    video_path=video_path,
    time_interval=3.5,
)

# Construct the message format required by Qwen2.5-VL
messages = [
    {
        "role": "user", 
        "content": [
            {"type": "video", "video": None}, # 这里留空或者不写都可以，关键在下面
            {"type": "text", "text": question}
        ]
    }
]

# 3. 生成 input_text
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 4. 直接传给 processor (关键点！)
# videos 参数接受的是：List[List[PIL.Image]] (即一批视频，每个视频是一组图)
inputs = processor(
    text=[text],
    videos=[frames],  # <--- 直接把 PIL 列表传进去
    padding=True,
    return_tensors="pt",
)


# Move inputs to the same device as the model (GPU)
inputs = inputs.to(model.device)

# 4. Generate Response
print("Generating response...")
# Video-MME often requires longer generation windows (max_new_tokens)
generated_ids = model.generate(**inputs, max_new_tokens=10)

# 5. Decode Output
# Trim the input tokens from the output to get only the generated text
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)

print("-" * 20)
print("Model Output:")
print(output_text[0])
print("-" * 20)
print("Model answer:")
print(data[0]['answer'])

p = argparse.ArgumentParser()
args = p.parse_args()
args.dataset_name = 'videomme'
attention_attract_modality(args, model, inputs, 0)