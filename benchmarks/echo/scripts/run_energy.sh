#!/bin/bash

set -e

echo "====================================================="
echo "Starting ECHO ENERGY Experiments for GHR"
echo "====================================================="

SEEDS=(42 43 44)

for SEED in "${SEEDS[@]}"; do
    echo "-----------------------------------------------------"
    echo "Initializing Run for Seed: $SEED"
    echo "-----------------------------------------------------"

    python ECHO_train.py \
        --task energy  \
        --gnn_type GHR \
        --hidden_dim 32 \
        --lr 0.0003000427934048188 \
        --batch_size 128 \
        --num_layers 1 \
        --l_steps 3 \
        --h_steps 1 \
        --device auto \
        --seed "$SEED"
        
    echo "Seed $SEED completed."
    echo ""
done