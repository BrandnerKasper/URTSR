#!/bin/bash

# List of file paths
file_paths=(
    "configs/RepNet/repnet_01.yaml"
    "configs/Urteil/urteil_01.yaml"
    "configs/Urteil_2/urteil_2_01.yaml"
    "configs/RepNetRRSR/repnet_rrsr_01.yaml"
)

# Loop through each file path and call the Python script
for file_path in "${file_paths[@]}"
do
    echo "Calling train.py with $file_path"
    python train.py "$file_path"
done
