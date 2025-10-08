cd /home/hoongyao/WiKV
source /home/hoongyao/miniconda3/bin/activate WiKV

export MODEL=Qwen3-4B
export MODEL_ID=Qwen/Qwen3-4B

#export MODEL=Qwen3-1.7B
#export MODEL_ID=Qwen/Qwen3-1.7B

#export MODEL=Llama8B
#export MODEL_ID=meta-llama/Llama-3.1-8B-Instruct

#export MODEL=Phi-4
#export MODEL_ID=microsoft/Phi-4-mini-instruct

#export MODEL=Ministral-8B
#export MODEL_ID=mistralai/Ministral-8B-Instruct-2410 

export dataset=/home/hoongyao/data/test_data
export dataname=gov_report
python3 prefill.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name $dataname \
    --path_to_context ${dataset}/$dataname.jsonl \
    --start 0 \
    --end 1 \