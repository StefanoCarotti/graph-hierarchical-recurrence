# LRIM Model Training

Minimal training pipeline for Long-Range Ising Model (LRIM) node-level regression using PyTorch Geometric.

> **⚠️ Disclaimer**: This is a barebone skeleton to illustrate usage and containerization. The models provided (SimpleGCN and SimpleMLP) are **not reasonable baselines** for competitive performance. They serve as minimal examples to demonstrate the training pipeline, dataset loading, and evaluation setup.

## Dataset

Datasets are hosted on HuggingFace: [jmathys/lrim_graph_benchmark](https://huggingface.co/datasets/jmathys/lrim_graph_benchmark)

## Quick Start

### 1. Get the Dataset Loader

If you don't have `lrim_loader.py`, download it:

```bash
wget https://raw.githubusercontent.com/iJorl/lrim_graph_benchmark/main/lrim_loader.py
```

Or copy it from the repository.

### 2. Setup and Train

Run the setup script to build the container and start training:

```bash
./setup.sh
```

This will:
- Build the Apptainer container (if not exists)
- Automatically download the dataset from HuggingFace
- Train both SimpleGCN and SimpleMLP models
- Save best models and print results


