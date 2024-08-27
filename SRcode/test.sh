#!/bin/bash

# List of file paths
file_paths=(
    "pretrained_models/URTSR/urtsr_all.pth"
    "pretrained_models/RTSRN/rtsrn_all.pth"
    "pretrained_models/URTSR/urtsr_all.pth"
    "pretrained_models/NSRRD/nsrrd_01.pth"
    "pretrained_models/NSRRD/nsrrd_02.pth"
    "pretrained_models/NSRRD/nsrrd_03.pth"
    "pretrained_models/NSRRD/nsrrd_04.pth"
    "pretrained_models/NSRRD/nsrrd_05.pth"
    "pretrained_models/NSRRD/nsrrd_06.pth"
)

# Loop through each file path and call the Python script
for file_path in "${file_paths[@]}"
do
    echo "Calling test.py with $file_path"
    python test.py "$file_path"
done
