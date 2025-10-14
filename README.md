# WiKV
Welcome to WiKV, which overlaps the wireless KV progressive streaming with pacing token decoding to reduce the TTFT and latency in long context serving.

Demo-gif: Summary of long gov report
![](https://github.com/liuhongyao99cs/WiKV/blob/main/WiKV_report.gif)

Demo video:
https://github.com/user-attachments/assets/a1a69974-fff9-4551-bce6-077e7b70009e


## Install WiKV in LAPTOP RTX 5080 with Linux x86-64
1. Setup virtual python env in miniconda3 and install dependencies
   ```bash
   ...
   cd DOWNLOAD_PATH/WiKV
   conda env create -f env.yml -n WiKV
   conda activate WiKV
   ...
2. Modify the installed pytorch version
   ```bash
   ...
   pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
   ...
3. Install Flash Attention 2 (download wheel from https://github.com/Dao-AILab/flash-attention/releases)
   ```bash
   ...
   pip install flash_attn-2.8.1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   ...

## Install WiKV in Jetson
Jetson is hard to setup a python virtual env since it is hard to find proper wheels in arm64.
We recommend use Jetson-container to build your own container which includes flash-attention, bitsandbytes, transformers and pytorch.
Jetson container: https://github.com/dusty-nv/jetson-containers.
   ```bash
   ...
   git clone https://github.com/dusty-nv/jetson-containers
   bash jetson-containers/install.sh
   jetson-containers build --name=wikv_container pytorch transformers flash-attention bitsandbytes
   ...

## Experiments

1. Generate the attention score for semantic coding, run (specify your hidden, attention save dirs and the dataset name; and if in Jetson docker run your container):
   ```bash
   ...
   cd scripts
   bash Attention.sh
   ...
   
2. Obtain the KV cache of datasets, run:
   ```bash
   bash KV_cache.sh

3. Run WiKV:
   ```bash
   bash main.sh

4. Run baseline: compute / prefill directly
   ```bash
   bash prefill.sh
