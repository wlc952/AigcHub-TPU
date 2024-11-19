#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update package list and install required packages
sudo apt-get update -y
sudo apt-get install -y alsa-utils libasound2-dev

# Create and navigate to the build directory
mkdir -p build
cd build

# Configure and build the project
cmake -DCMAKE_BUILD_TYPE=Release -DTPU_SDK_PATH="$TPU_SDK_PATH" ..
make clean
make -j$(nproc)