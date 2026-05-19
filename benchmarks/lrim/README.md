# LRIM Graph Benchmark
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![🌍 Website](https://img.shields.io/badge/🌍-Website-blue)](https://lrim-graphbenchmark.com)
[![🤗 Datasets](https://img.shields.io/badge/🤗-Datasets-yellow)](https://huggingface.co/datasets/jmathys/lrim_graph_benchmark)

## 📄 Paper

**LRIM: A Physics-Based Benchmark for Provably Evaluating Long-Range Capabilities in Graph Learning**

Accurately modeling long-range dependencies in graph-structured data is critical for many real-world applications. However, incorporating long-range interactions beyond the nodes' immediate neighborhood in a _scalable_ manner remains an open challenge for graph machine learning models. Existing benchmarks for evaluating long-range capabilities either cannot _guarantee_ that their tasks actually depend on long-range information or are rather limited. Therefore, claims of long-range modeling improvements based on said performance remain questionable. We introduce the Long-Range Ising Model Graph Benchmark, a physics-based benchmark utilizing the well-studied Ising model whose ground truth _provably_ depends on long-range dependencies. Our benchmark consists of ten datasets that scale from 256 to 65k nodes per graph, and provide controllable long-range dependencies through tunable parameters, allowing precise control over the hardness and ``long-rangedness". We provide model-agnostic evidence that local information is insufficient, further validating the design choices of our benchmark. Via experiments on classical message-passing architectures and graph transformers, we show that both perform far from the optimum, especially those with scalable complexity. Our goal is that our benchmark will foster the development of scalable methodologies that effectively model long-range interactions in graphs.



## 🌐 Website

Visit our project website: [https://lrim-graphbenchmark.com](https://lrim-graphbenchmark.com) for informatin on how to get started, or explore the datastes and oracle baselines in an interactive manner.


## 🗂️ Repository Structure

```
lrim_graph_benchmark/
├── data-generation/          # Generate LRIM datasets from scratch, but we recommend downloading the official datasets
│
├── example-setup/           # Minimal training example
│   ├── setup.sh             # Quick start training script
│   ├── train.py             # Basic GCN, MLP baseline
│   └── README.md            # Example documentation
│
└── model-training/          # Full training pipeline
    ├── setup.sh             # Build container and run sample
    ├── run_training.sh      # Train with custom configs
    ├── run_inference.sh     # Run inference on checkpoints
    ├── verify_checkpoints.sh # Verify against reference models
    ├── lrim_configs/        # Model configurations
    ├── grit/                # model implementations
    └── README.md            # Documentation
```

## 🚀 Quick Start

### Option 1: Minimal Example (Recommended for First-Time Users)

Train a simple baseline model in minutes:

```bash
cd example-setup
./setup.sh
```

This runs a minimal GCN/MLP baseline on one of the LRIM datasets. **Note:** These are demonstration models, not competitive baselines.

### Option 2: Full Training Pipeline

Train entire models:

```bash
cd model-training
./setup.sh
```

This builds the container, trains an example configuration, and runs inference.

### Option 3: Generate Your Own Data

Create custom LRIM datasets, we **strongly emphasize that to use the provided generated datasets**, rather than regenerating them for comparisons:

```bash
cd data-generation
./setup.sh
./lrim_gen.sh 16 0.6 100000  # 16×16 matrices, σ=0.6, 100k samples
```

## 📦 Provided Datasets

Download datasets directly from HuggingFace:
- **Dataset Repository**: [jmathys/lrim_graph_benchmark](https://huggingface.co/datasets/jmathys/lrim_graph_benchmark)
- **Available sizes**: 16, 32, 64, 128, 256
- **Difficulty levels**: σ=0.6 (hard), σ=1.5 (easy)

## 🔧 Requirements

- **Apptainer/Singularity**: For containerized execution (strongly recommended) [download here](https://apptainer.org/docs/admin/main/installation.html)
- **CUDA-capable GPU**: For training (CPU mode available but slower)

All Python dependencies are included in the provided Apptainer containers, therefore, one can recreate the environment in either conda or venv, however, we strongly suggest using apptainer instead.

> **💡 Tip for Compute Clusters**: If you encounter issues building Apptainer containers on your compute cluster, we recommend building the `.sif` file locally on your machine and then transferring it to the cluster. You do not need a GPU locally for container building—only for training.

## 📚 Citation

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{
mathys2026lrim,
title={{LRIM}: a Physics-Based Benchmark for Provably Evaluating Long-Range Capabilities in Graph Learning},
author={Jo{\"e}l Mathys and Henrik Christiansen and Federico Errica and Takashi Maruyama and Francesco Alesiani},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=IAZXEX1dVV}
}
```

## 🔗 Links

- **Website**: [Official Project Website](https://lrim-graphbenchmark.com)
- **Datasets**: [HuggingFace](https://huggingface.co/datasets/jmathys/lrim_graph_benchmark)
- **Paper**: [OpenReview](https://openreview.net/forum?id=IAZXEX1dVV)

---

For detailed documentation, see the README files in each subdirectory:
- [Data Generation](data-generation/README.md)
- [Example Setup](example-setup/README.md)
- [Model Training](model-training/README.md)
