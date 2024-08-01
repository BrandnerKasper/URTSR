#!/bin/bash

# List of file paths
file_paths=(
    "pretrained_models/evrnet_01.pth"
    "pretrained_models/evrnet_06.pth"
)

# Loop through each file path and call the Python script
for file_path in "${file_paths[@]}"
do
    echo "Calling train.py with $file_path"
    python evaluate.py "$file_path"
done
