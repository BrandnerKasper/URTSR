#!/bin/bash

# List of file paths
file_paths=(
    "configs/urepss_05_01.yaml"
    "configs/URTSR/Ablation/urtsr_01_bare.yaml"
    "configs/URTSR/Ablation/urtsr_01_buff.yaml"
    "configs/URTSR/Ablation/urtsr_01_warp.yaml"
    "configs/URTSR/Ablation/urtsr_01_warp_buff.yaml"
)

# Loop through each file path and call the Python script
for file_path in "${file_paths[@]}"
do
    echo "Calling train.py with $file_path"
    python train.py "$file_path"
done
