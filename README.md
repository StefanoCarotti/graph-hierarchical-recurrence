# Graph Hierarchical Recurrence (GHR)

*A hierarchical recurrent GNN that couples fine- and coarse-level message passing for long-range reasoning.*

[![arXiv](https://img.shields.io/badge/arXiv-2605.18387-b31b1b.svg)](https://arxiv.org/abs/2605.18387)

GHR is a graph neural network framework designed for tasks that require propagating information across many hops. It maintains two coupled hidden states—one at the resolution of the original graph and one at the resolution of a learned coarse abstraction—and evolves them together over multiple recurrent steps. The coarse level provides long-range context to the fine level at every step; the fine level continuously summarises back upward. At inference time, the joint state is rolled forward across several reasoning steps, so the effective receptive field grows with computation rather than with depth. Unlike transformer-based long-range methods, GHR scales linearly with graph size and uses no positional encoding (on molecular benchmarks Laplacian PE can be used as an optional add-on).

The model is trained end-to-end with backpropagation through time (BPTT) on a temporally-discounted loss that assigns higher weight to later steps, encouraging the recurrent dynamics to converge rather than merely fit the first pass.

---

## The framework

**Dual-level architecture.** Given a graph G = (V, E), a one-level coarsening is computed once (using the Graclus algorithm) to produce a smaller graph `G_H` and a cluster assignment c : V → V_H. GHR maintains two hidden states: `h_L` (fine, one vector per original node) and `h_H` (coarse, one vector per cluster).

**Inner step.** Each recurrent step runs `H_steps` outer iterations. In each outer iteration:

1. **Bottom-up aggregation.** Fine states are pooled into cluster summaries (scatter-max for synthetic tasks; scatter-add for molecular tasks) and added to the coarse state before message passing.
2. **Coarse message passing.** `h_H` is updated by a message-passing layer operating on `G_H` (Often Gated GINEConv is our default configuration).
3. **Top-down guidance.** Each fine node receives a projected copy of its cluster's updated `h_H` as an additive context signal.
4. **Fine message passing.** `h_L` is updated by a second message-passing layer on `G`, repeated `L_steps` times per outer iteration, with the input node feature embedding and top-down context injected at each sub-step.

Both levels use RMSNorm pre-normalisation before each message-passing call, and residual connections accumulate messages additively.

**Global recurrence.** The pair (h_L, h_H) is carried across `T` sequential calls to the model (reasoning steps). At training time `T` is sampled from `{T-1, T, T+1}` for a small amount of regularisation; at evaluation time it is fixed. The training loss is a discounted sum over reasoning steps: L = Σ_r γ^(T-1-r) · MSE(ŷ_r, y), where γ < 1 down-weights earlier (less converged) predictions. 

**Graph preparation.** A `GHRTransform` is applied once per dataset: it runs Graclus clustering, pools edge attributes to the coarse graph via max-pooling, and stores `cluster`, `coarse_edge_index`, and `coarse_edge_attr` alongside the original graph data.

---

## Results

| Benchmark | Task | Metric | GHR | Best baseline |
|---|---|---|---|---|
| ECHO-Synth | Diameter | MAE ↓ | 0.746 | 1.014 (GRIT) |
| ECHO-Synth | Eccentricity | MAE ↓ | 3.456 | 4.651 (DRew) |
| ECHO-Synth | SSSP | MAE ↓ | 0.035 | 0.121 (GRIT) |
| ECHO-Energy | Energy | MAE ↓ | 6.040 | 5.257 (GPS) |
| LRGB | Peptides-struct | MAE ↓ | 0.2821 | 0.2358 (Cache-GNN) |

GHR achieves state-of-the-art on ECHO-Synth and ECHO-Energy while operating at hidden dimension 32 in most runs—one to two orders of magnitude fewer parameters than the strongest baselines on those tasks. On the LRIM benchmark (Ising model graphs up to 65k nodes), GHR is competitive on in-distribution evaluation and shows strong out-of-distribution extrapolation to larger lattice sizes not seen during training. Full numerical results and ablation tables are reported in the accompanying paper.

---

## Quickstart

See [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) for a 5-minute hands-on tutorial: build a small random geometric graph, instantiate GHR, run a forward pass, and visualise the two-level hierarchy.

---

## Installation

**Requirements:** Python 3.11, PyTorch ≥ 2.0, PyTorch Geometric ≥ 2.4, and matching `torch_scatter` / `torch_cluster` wheels.

```bash
# Install torch_scatter and torch_cluster from the PyG wheel index.
# Replace <TORCH> and <CUDA> with your versions, e.g. torch-2.7.0+cu124
pip install torch_scatter torch_cluster \
    -f https://data.pyg.org/whl/torch-<TORCH>+<CUDA>.html

# Install remaining dependencies
pip install -r requirements.txt
```

> **Apple Silicon (MPS):** `torch_scatter`'s `scatter_max` currently errors on MPS. Training on CPU still works but is slower than expected on MPS. Use `--device cpu` to avoid the errors, or run on CUDA.

### Verify your install

```bash
bash smoke_test.sh
```

Expected: `PASS : 6   FAIL : 0   TIMEOUT: 0`

---

## Reproducing paper results

### ECHO benchmark (Synth + Chem)

Download datasets first:

```bash
cd benchmarks/echo
python scripts/download-all.py          # all tasks
python scripts/download-all.py --task sssp   # single task
```

Run GHR using the provided shell scripts, which encode the exact hyperparameters from the paper:

```bash
cd benchmarks/echo/scripts

bash run_sssp.sh      # ECHO-Synth: single-source shortest path  (L=6, H=3, lr=1e-3)
bash run_diam.sh      # ECHO-Synth: graph diameter               (L=2, H=2, lr=1.83e-3)
bash run_ecc.sh       # ECHO-Synth: node eccentricity            (L=2, H=3, lr=1e-3)
bash run_energy.sh    # ECHO-Chem:  molecular energy             (L=3, H=1, lr=3e-4)
bash run_charge.sh    # ECHO-Chem:  atomic partial charge        (L=3, H=1, lr=3.4e-4)
```

Each script runs three seeds and saves metrics under `lightning_logs/`. Training can also be launched directly:

```bash
python scripts/ECHO_train.py \
    --task sssp \
    --gnn_type GHR \
    --hidden_dim 32 \
    --l_steps 6 \
    --h_steps 3 \
    --lr 1e-3 \
    --batch_size 128 \
    --device auto \
    --seed 42
```

### LRIM benchmark

See `benchmarks/lrim/README.md` for full instructions. The quick path:

```bash
cd benchmarks/lrim/example-setup
python LRIM_train.py \
    --dataset_name lrim_32_0.6_10k \
    --hidden_dim 256 \
    --L_steps 3 \
    --H_steps 3 \
    --batch_size 8
```

OOD evaluation (extrapolation to larger lattice sizes) is run by loading a checkpoint trained at a smaller size and evaluating on larger grids.

### LRGB Peptides-struct

```bash
cd benchmarks/lrgb
python LRGB_train.py --config LRGB.yaml
```

The YAML file contains all hyperparameters (`hidden_dim=32, L_steps=2, H_steps=2, reasoning_steps=5, use_swiglu=True`).

### RGG-SSSP ablation

```bash
cd experiments/rgg
python OOR_train.py          # GHR vs ablation baselines on out-of-range SSSP
python train_weighted.py     # GHR on weighted-distance SSSP
```

The ablation compares GHR against (i) flat baselines (deep and recurrent GINE / GatedGINE, with and without the global-recurrent-step logic) and (ii) GHR variants with different message-passing backbones (GINE, GatedGINE, GCN, GAT, A-DGN), on random geometric graphs with weighted SSSP targets. See Section 4.3 of the paper for details.

### City-Network

Not in the current version of the paper, an additional exploratory benchmark.

---

## Repository structure

```
GHR/
├── notebooks/
│   └── quickstart.ipynb       # 5-minute hands-on tutorial 
├── benchmarks/
│   ├── echo/                  # ECHO benchmark (Synth + Chem)
│   │   ├── models/            # GHRModel and baselines (ADGN, DRew, GraphCON, …)
│   │   ├── scripts/           # Training scripts and per-task run_*.sh wrappers
│   │   └── utils/             # Dataset loading, GHRTransform, Lightning modules
│   ├── lrim/                  # LRIM benchmark (Ising model graphs)
│   │   └── example-setup/     # Minimal single-file training example
│   ├── lrgb/                  # LRGB Peptides-struct benchmark
│   └── city_network/          # Urban road-network benchmark
├── experiments/
│   └── rgg/                   # RGG-SSSP ablation study (GHR vs baselines)
├── smoke_test.sh              # Runs all 9 training entry-points for 1 batch each
├── requirements.txt           # Python dependencies
└── LICENSE                    # MIT
```

---

## Citation

If you use this code or the GHR model in your research, please cite:

```bibtex
@misc{carotti2026graphhierarchicalrecurrencelongrange,
      title={Graph Hierarchical Recurrence for Long-Range Generalization}, 
      author={Stefano Carotti and Marco Pacini and Alessio Gravina and Davide Bacciu and Bruno Lepri and Sebastiano Bontorin},
      year={2026},
      eprint={2605.18387},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.18387}, 
}
```

---

## License

This project is released under the [MIT License](LICENSE).
