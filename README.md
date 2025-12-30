# WiKV: Wireless KV Streaming for Efficient LLM Inference

**WiKV** optimizes Large Language Model (LLM) inference on mobile and IoT devices. By overlapping wireless KV progressive streaming with pacing token decoding, WiKV significantly reduces Time-to-First-Token (TTFT) and overall latency.

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

### Option 1: Linux Laptop (x86-64)
*Target Hardware: RTX 5080 Mobile / Linux x86-64*

1. **Setup Python Environment**
   Create a virtual environment using Miniconda and install dependencies.
   ```bash
   cd DOWNLOAD_PATH/WiKV
   conda env create -f env.yml -n WiKV
   conda activate WiKV
