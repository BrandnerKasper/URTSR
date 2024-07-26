#!/bin/bash

# List of file paths
file_paths=(
    "configs/STSS/stss_original_bi_01.yaml"
    "configs/STSS/stss_original_bi_02.yaml"
    "configs/STSS/stss_original_bi_03.yaml"
    "configs/STSS/stss_original_bi_04.yaml"
    "configs/STSS/stss_original_bi_05.yaml"
    "configs/STSS/stss_original_bi_06.yaml"
    "configs/URTSR/urtsr_01.yaml"
    "configs/URTSR/urtsr_06.yaml"
    "configs/EVRNET/evrnet_01.yaml"
    "configs/EVRNET/evrnet_06.yaml"
)

# Loop through each file path and call the Python script
for file_path in "${file_paths[@]}"
do
    echo "Calling train.py with $file_path"
    python train.py "$file_path"
done
