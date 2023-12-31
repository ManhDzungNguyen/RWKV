# RWKV Language Model
This is the code for training the RWKV model from scratch to create a pretrained model for Vietnamese. 

However, I do not recommend using this method if you do not have significant resources. According to the original author's calculations, theoretically, it would require at least 22,436 hours of A100 time to train. In practice, RWKV 14B was trained on 64 A100s in parallel, sacrificing a bit of performance for various reasons. RWKV 14B took about 3 months of A100 hours to train, thus achieving approximately 20% theoretical efficiency.

This repository has 2 branches:
- The `main` branch implements training and inference code using the transformers library from HuggingFace.
- The `symato` branch deploys the training and inference code based on the git repository: https://github.com/telexyz/symato with some bug fixes.

RWKV preprint: https://arxiv.org/abs/2305.13048

# Setup
- `python3.8 -m venv venv`
- `pip install --upgrade pip`
- `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`
- `pip install -r requirements.txt`
- `python setup.py`

## Training and Inference
To train, please run ./pretrain/train.sh (requires a GPU).
To infer, please run ./pretrain/infer.py