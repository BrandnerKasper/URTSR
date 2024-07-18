#!/bin/bash

# List of file paths
file_paths=(
    "configs/STSS/stss_original_01.yaml"
    "configs/STSS/stss_original_02.yaml"
    "configs/STSS/stss_original_03.yaml"
    "configs/STSS/stss_original_04.yaml"
    "configs/STSS/stss_original_05.yaml"
)

# Loop through each file path and call the Python script
for file_path in "${file_paths[@]}"
do
    echo "Calling process_config.py with $file_path"
    python train.py "$file_path"
done
