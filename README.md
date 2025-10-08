# WiKV
### Topic of Longchat
![](https://github.com/liuhongyao99cs/WiKV/blob/main/WiKV_longchat.gif)
### Answer the question in NQA
<img src="https://github.com/liuhongyao99cs/WiKV/blob/main/WiKV_nqa.gif)" width="210px">
### Summary of long gov report
![](https://github.com/liuhongyao99cs/WiKV/blob/main/WiKV_report.gif)
Demo video:
https://github.com/user-attachments/assets/a1a69974-fff9-4551-bce6-077e7b70009e


# Install in RTX 5080 LAPTOP

### First, create an virtual env with the dependencies in the folder
0. cd ~/WiKV
1. conda env create -f env.yml -n WiKV
2. conda activate WiKV
### Install required pytorch (For RTX 5080, CUDA 12.8)
3. pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
### Install Flash Attention 2 (download wheel from https://github.com/Dao-AILab/flash-attention/releases)
4. pip install flash_attn-2.8.1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
