# WiKV
Welcome to WiKV, which overlaps the wireless KV progressive streaming with pacing token decoding to reduce the TTFT and latency in long context serving.

### Demo-gif: Summary of long gov report
![](https://github.com/liuhongyao99cs/WiKV/blob/main/WiKV_report.gif)

### Demo video:
https://github.com/user-attachments/assets/a1a69974-fff9-4551-bce6-077e7b70009e


# Install WiKV (RTX 5080 + Ubuntu 24.04)

### First, create an virtual env with the dependencies in the folder
0. cd ~/WiKV
1. conda env create -f env.yml -n WiKV
2. conda activate WiKV
### Install required pytorch (For RTX 5080, CUDA 12.8 is needed)
3. pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
### Install Flash Attention 2 (download wheel from https://github.com/Dao-AILab/flash-attention/releases)
4. pip install flash_attn-2.8.1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Run experiments

## Preperation
1. Generate the attention score for semantic coding:
   
   cd scripts
   bash Attention.sh (specify your hidden, attention save dirs and the dataset name)
   
2. Obtain the KV cache of datasets

   bash KV_cache.sh
