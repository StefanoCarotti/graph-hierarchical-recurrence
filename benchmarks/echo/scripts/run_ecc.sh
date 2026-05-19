#!/bin/bash

# Exit immediately if any command fails (prevents cascading errors)
set -e

echo "====================================================="
echo "Starting ECHO ECC Experiments for GHR (Seeds 1, 2, 3)"
echo "====================================================="

SEEDS=(1 2 3)

# Loop through each seed
for SEED in "${SEEDS[@]}"; do
    echo "-----------------------------------------------------"
    echo "Initializing Run for Seed: $SEED"
    echo "-----------------------------------------------------"

    # Run the training script with parameters from hparams.yaml
    python ECHO_train.py \
        --task ecc \
        --gnn_type GHR \
        --hidden_dim 32 \
        --lr 0.001 \
        --weight_decay 0.0 \
        --batch_size 256 \
        --num_layers 1 \
        --l_steps 2 \
        --h_steps 3 \
        --device auto \
        --seed "$SEED" \
        
        
    echo "Seed $SEED completed. Logs and checkpoints saved to lightning_logs/"
    echo ""
done

echo "====================================================="
echo "All 3 seeds completed for ECC"
echo "====================================================="