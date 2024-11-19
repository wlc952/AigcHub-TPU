#!/bin/bash

# Create models directory if it doesn't exist
mkdir -p models

# Download the tar.gz file
wget https://modelscope.cn/models/wlc952/aigchub_models/resolve/master/sherpa-onnx/sherpa.tar.gz -O models/sherpa.tar.gz

# Extract the tar.gz file into the models directory
tar -zxvf models/sherpa.tar.gz -C models

# Remove the tar.gz file after extraction
rm models/sherpa.tar.gz

# Print a message indicating the download is complete
echo "Model download complete!"
