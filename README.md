# WiKV: Wireless KV Streaming for Efficient LLM Inference

Welcome to **WiKV**.

WiKV overlaps wireless KV progressive streaming with pacing token decoding to significantly reduce the Time-to-First-Token (TTFT) and overall latency of Large Language Model (LLM) inference on mobile and IoT devices.

---

## 📊 Inference Comparison

We benchmark WiKV against three standard inference baselines, evaluating both efficiency and response quality.

### Demos

**1. Summary of Long Government Report**

<p align="center">
  <img src="https://github.com/liuhongyao99cs/WiKV/blob/main/images/WiKV_report.gif" alt="WiKV Government Report Summary" width="80%">
</p>

**2. Video Demonstration**

[Watch the full demo video here](https://github.com/user-attachments/assets/a1a69974-fff9-4551-bce6-077e7b70009e)

---

## ⚙️ Installation

Please follow the instructions below based on your hardware platform.

### Option 1: Linux Laptop (x86-64)
**Target Hardware:** RTX 5080 Mobile / Linux x86-64

1.  **Setup Python Environment**
    Create a virtual environment using Miniconda and install dependencies.
    ```bash
    cd DOWNLOAD_PATH/WiKV
    conda env create -f env.yml -n WiKV
    conda activate WiKV
    ```

2.  **Install PyTorch**
    Install the specific version required for this project (CUDA 12.8).
    ```bash
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
    ```

3.  **Install Flash Attention 2**
    Download the appropriate wheel from the [Flash Attention Releases](https://github.com/Dao-AILab/flash-attention/releases) and install it.
    ```bash
    # Example command (ensure the filename matches your downloaded wheel)
    pip install flash_attn-2.8.1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    ```

### Option 2: NVIDIA Jetson Orin NX / AGX Orin
**Target Hardware:** ARM64 / Tegra

Due to the difficulty of finding proper PyTorch/Flash-Attention wheels for ARM64, we recommend using [jetson-containers](https://github.com/dusty-nv/jetson-containers).

1.  **Build Base Container**
    Clone the repository and build a container with the necessary base libraries (PyTorch, Transformers, Flash-Attention, BitsAndBytes).
    ```bash
    git clone [https://github.com/dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers)
    bash jetson-containers/install.sh
    jetson-containers build --name=wikv_container pytorch transformers flash-attention bitsandbytes
    ```

2.  **Extend Container**
    Create a custom Dockerfile to install `scikit-learn` and other necessary packages using the base container built above.
    
    **Create a file named `Dockerfile`:**
    ```dockerfile
    FROM wikv_container:r36.4.tegra-aarch64-cu126-22.04
    RUN pip install --no-cache-dir scikit-learn
    ```
    
    **Build the final image:**
    ```bash
    sudo docker build -t wikv .
    ```

---

## 🧪 Experiments

### 1. Generate Attention Scores
Generate attention scores for semantic coding.
*Note: Please specify your directories properly in the scripts before running.*

* **For Laptop:**
    ```bash
    cd scripts
    bash Attention.sh
    ```
* **For Jetson:**
    ```bash
    cd scripts/jetson_scripts
    bash Attention_jetson.sh
    ```

### 2. Obtain KV Cache
Generate the KV cache for the datasets.
```bash
bash KV_cache.sh
