# sudo update-alternatives --set iptables /usr/sbin/iptables-legacy
# sudo apt reinstall docker-ce
sudo docker run -it --rm \
  --network=host \
  --runtime=nvidia \
  --volume /home/hongyao/Desktop/WiKV:/workspace \
  --volume /home/hongyao/Desktop/jetson-containers/data:/data \
  wikv_container:r36.4.tegra-aarch64-cu126-22.04 \
  /bin/bash -c "
    cd /workspace &&
    export MODEL=Qwen3-8B &&
    export MODEL_ID=Qwen/Qwen3-8B &&
    export dataset=/workspace/Test_data &&
    export dataname=nqa &&
    python3 ./prefill.py \
      --model_id \$MODEL_ID \
      --model \$MODEL \
      --dataset_name \$dataname \
      --path_to_context \${dataset}/\${dataname}.jsonl \
      --start 0 \
      --end 1
  "
