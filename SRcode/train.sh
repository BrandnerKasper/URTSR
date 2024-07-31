#!/bin/bash

# List of file paths
file_paths=(
    "configs/STSS/stss_original_bi_warp_01.yaml"
    "configs/STSS/stss_original_bi_warp_02.yaml"
    "configs/STSS/stss_original_bi_warp_03.yaml"
    "configs/STSS/stss_original_bi_warp_04.yaml"
    "configs/STSS/stss_original_bi_warp_05.yaml"
    "configs/STSS/stss_original_bi_warp_06.yaml"
)

# Loop through each file path and call the Python script
for file_path in "${file_paths[@]}"
do
    echo "Calling train.py with $file_path"
    python train.py "$file_path"
done
