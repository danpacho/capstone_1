#!/bin/bash

# Find all .tar.gz files recursively and extract the model.pth file
find . -type f -name "model.tar.gz" | while read -r tarfile; do
    # Extract the .tar.gz file
    echo "Extracting $tarfile..."
    
    # Create a directory to extract into (optional, keeps things organized)
    dirname=$(dirname "$tarfile")
    
    # Check if model.pth exists inside the archive
    if tar -tzf "$tarfile" | grep -q "model.pth"; then
        # Extract only model.pth from the archive
        tar -xzf "$tarfile" -C "$dirname" "model.pth"
        
        # Check if extraction was successful
        if [ $? -eq 0 ]; then
            echo "Extracted model.pth from $tarfile successfully."
        else
            echo "Failed to extract model.pth from $tarfile."
        fi
    else
        echo "No model.pth found in $tarfile."
    fi
done
