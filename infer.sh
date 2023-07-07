#!/bin/bash

# Change to the working directory
cd ./RWKV/rwkv-v4neo

# Activate the virtual environment
source ../venv/bin/activate

# Set environment variables
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

# Run inference src
python infer.py