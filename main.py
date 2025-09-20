import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

model_name = "Qwen/Qwen3-8B"

# ========================
# Step 0: 加载模型和 tokenizer
# ========================

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
    output_attentions=True
)

dataset = "C:/Users/14490/Desktop/WiKV/datasets/Academic.jsonl"

data = load_testcases(dataset)
print(data[10]['question'])

