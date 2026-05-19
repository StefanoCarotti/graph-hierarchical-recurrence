#!/bin/bash

set -e

echo "====================================================="
echo "Starting ECHO DIAM Experiments for GHR"
echo "====================================================="

# Run 43 first to verify reproducibility, then the others
SEEDS=(41 42 44)

for SEED in "${SEEDS[@]}"; do
    echo "-----------------------------------------------------"
    echo "Initializing Run for Seed: $SEED"
    echo "-----------------------------------------------------"

    python ECHO_train.py \
        --task diam \
        --gnn_type GHR \
        --hidden_dim 32 \
        --lr 0.0018327255780015133 \
        --batch_size 128 \
        --num_layers 1 \
        --l_steps 2 \
        --h_steps 2 \
        --device auto \
        --seed "$SEED"
        
    echo "Seed $SEED completed."
    echo ""
done