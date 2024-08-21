#!/bin/bash

# List of file paths
file_paths=(
    "configs/URTSR/urtsr_02.yaml"
    "configs/URTSR/urtsr_03.yaml"
    "configs/URTSR/urtsr_04.yaml"
    "configs/URTSR/urtsr_05.yaml"
    "configs/URTSR/urtsr_06.yaml"
    "configs/URTSR/urtsr_all.yaml"
)

# Loop through each file path and call the Python script
for file_path in "${file_paths[@]}"
do
    echo "Calling train.py with $file_path"
    python train.py "$file_path"
done
