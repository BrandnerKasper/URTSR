#!/bin/bash

# List of file paths
file_paths=(
    "configs/URepSS_2/urepss_2_01.yaml"
    "configs/URepSS_3/urepss_3_01.yaml"
    "configs/URepSS_4/urepss_4_01.yaml"
)

# Loop through each file path and call the Python script
for file_path in "${file_paths[@]}"
do
    echo "Calling train.py with $file_path"
    python train.py "$file_path"
done
