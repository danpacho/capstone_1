#!/bin/bash

# Find all model.pth files recursively and compress them into .tar.gz
find . -type f -name "model.pth" | while read -r pthfile; do
    # Get the directory containing the model.pth file
    dirname=$(dirname "$pthfile")
    
    # Create a tar.gz archive with the model.pth file
    tarfile="$dirname/model.tar.gz"
    echo "Compressing $pthfile into $tarfile..."
    
    # Compress the model.pth file into model.tar.gz
    tar -czvf "$tarfile" -C "$dirname" "model.pth"
    
    # Check if compression was successful
    if [ $? -eq 0 ]; then
        echo "Successfully compressed $pthfile into $tarfile."
    else
        echo "Failed to compress $pthfile."
    fi
done
