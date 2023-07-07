#!/bin/bash

# Change to the working directory
cd ./RWKV/rwkv-v4neo

# Activate the virtual environment
source ../venv/bin/activate

# Install required libraries
pip install -r requirements.txt

# Set environment variables
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

# Run train.py
python train.py --load_model "" --wandb "pho-vlc" --proj_dir "out" --data_file "./data/vlc/vlc.txt" --data_type "utf-8" --ctx_len 192 --epoch_steps 10 --epoch_count 40 --epoch_begin 0 --epoch_save 5 --micro_bsz 32 --n_layer 4 --n_embd 320 --pre_ffn 0 --head_qk 0 --lr_init 8e-4 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --accelerator gpu --devices 1 --precision tf32 --strategy ddp_find_unused_parameters_false --grad_cp 0