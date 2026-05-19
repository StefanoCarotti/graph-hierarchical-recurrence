#!/bin/bash

set -e

echo "====================================================="
echo "Starting ECHO CHARGE Experiments for GHR"
echo "====================================================="

SEEDS=(43 42 44)

for SEED in "${SEEDS[@]}"; do
    echo "-----------------------------------------------------"
    echo "Initializing Run for Seed: $SEED"
    echo "-----------------------------------------------------"

    python ECHO_train.py \
        --task charge \
        --gnn_type GHR \
        --hidden_dim 32 \
        --lr 0.000340759553996452 \
        --batch_size 128 \
        --num_layers 1 \
        --l_steps 3 \
        --h_steps 1 \
        --device auto \
        --use_swiglu \
        --seed "$SEED"
        
    echo "Seed $SEED completed."
    echo ""
done