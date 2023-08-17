#!/bin/bash

# Change to the working directory
cd RWKV/pretrain

# Activate the virtual environment
source venv/bin/activate

# Set environment variables
export OMP_NUM_THREADS=5
export CUDA_VISIBLE_DEVICES=1

# Run train.py
python train.py --encoded_data "RWKV/data/encoded_data" --raw_data "RWKV/data/processed_data" --output_dir "RWKV/pretrain/checkpoint" --logging_dir "RWKV/pretrain/logs" --num_train_epochs 25