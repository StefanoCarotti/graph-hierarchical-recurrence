#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="2,3,4,5"
OUTDIR="./results"

MODELS="--models gcn_gp"
SCHEDULER="--scheduler none"
N_SAMPLES="--n_samples 32"
NUM_GPUS="--num_gpus=4"
OUTPUT_DIR="--output_dir ${OUTDIR}"
EPOCHS="--max_epochs 50"

COMMON_ARGS=" ${MODELS} ${SCHEDULER} ${N_SAMPLES} ${NUM_GPUS} ${OUTPUT_DIR} ${EPOCHS}"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"

TASK=diam

# this will run a bayesian hyperparameter search for the diameter task 
# downlad data first. note: in case of DRew, specify k parametrers. check download-all.py for more details
python download-all.py --root ./ --task ${TASK} 

python search.py ${COMMON_ARGS} --tasks ${TASK}
