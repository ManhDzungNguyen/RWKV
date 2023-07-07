# RWKV Language Model
This is the code for training the RWKV model from scratch to create a pretrained model for Vietnamese. 

However, I do not recommend using this method if you do not have significant resources. According to the original author's calculations, theoretically, it would require at least 22,436 hours of A100 time to train. In practice, RWKV 14B was trained on 64 A100s in parallel, sacrificing a bit of performance for various reasons. RWKV 14B took about 3 months of A100 hours to train, thus achieving approximately 20% theoretical efficiency.

This repository has 2 branches:
- The `main` branch implements training and inference code using the transformers library from HuggingFace.
- The `symato` branch deploys the training and inference code based on the git repository: https://github.com/telexyz/symato with some bug fixes.

RWKV preprint: https://arxiv.org/abs/2305.13048

## Training
To train, please run ./train.sh (requires a GPU).

Here's an overview of the different files in the `rwkv-v4neo`:

`dataset.py`: This file is responsible for loading and preparing the training data in the (x, y) format:
    x represents the input token sequence.
    y represents the next token after x.

`model.py`: This file contains the implementation of the RWKV-V4NEO model.

`trainer.py`: This file includes the steps for data preparation, model adjustment, and saving/loading model parameters.

`train.py`: This file contains various training scenarios.

`wkv_cuda.*`: These files implement CUDA operations to accelerate computations for time-mixing (also known as linear attention).

## Inference
To infer, please run ./infer.sh (requires a GPU).