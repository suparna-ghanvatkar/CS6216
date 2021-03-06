#!/bin/bash

python train_dp.py link_prediction with \
dataset='FB15k-237' \
inductive=True \
model='bert-dkrl' \
rel_model='transe' \
loss_fn='margin' \
regularizer=1e-2 \
max_len=32 \
num_negatives=64 \
lr=1e-4 \
use_scheduler=False \
batch_size=64 \
emb_batch_size=512 \
eval_batch_size=128 \
max_epochs=50 \
checkpoint=None \
use_cached_text=False \
NOISE_MULTIPLIER = 1 \
MAX_GRAD_NORM = 1 \
