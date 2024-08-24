#!/bin/bash

# List of file paths
file_paths=(
    "configs/URTSR/urtsr_all.yaml"
)

# Loop through each file path and call the Python script
for file_path in "${file_paths[@]}"
do
    echo "Calling train.py with $file_path"
    python train.py "$file_path"
done
