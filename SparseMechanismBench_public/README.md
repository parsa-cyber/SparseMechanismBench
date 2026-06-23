# SparseMechanismBench

**SparseMechanismBench: Why Sparse Local Plasticity Is Not Enough for Task-Discriminative Learning**

This repository contains code, raw CSVs, summary CSVs, plots, and the manuscript for a mechanism-isolation study of local plasticity, sparsity, homeostasis, replay, frozen readouts, and representation geometry.

## Core scientific question

Are sparse local-plasticity representations weak because the representations themselves are poorly task-aligned, or because the downstream readout/classifier is too weak?

## Main finding

Sparse local plasticity reliably creates sparse representations, but sparsity alone does not guarantee task-discriminative accuracy or stable continual learning. Oja features can preserve task-accessible information, while Sparse Oja and Homeostatic Sparse Oja often retain high sparsity at the cost of weaker class geometry.

## Repository structure

```text
README.md
requirements.txt
data_loading.py
models.py
train_classification.py
train_hebbian_oja.py
train_sparse_oja.py
train_homeostatic_oja.py
run_sample_efficiency.py
run_continual_learning.py
run_full_benchmark.py
run_frozen_readout.py
run_representation_geometry.py
run_frozen_readout_geometry_uploaded.py
run_sparsity_sweep.py
run_sparsity_sweep_subset.py
run_replay_sweep_subset.py
plot_results.py
make_enhanced_figures.py
make_all_figures.py
make_sparsemechanismbench_final.py
results/
figures/
SparseMechanismBench_final_manuscript.pdf
SparseMechanismBench_final_manuscript.docx
```

## Data setup

The package does **not** include raw MNIST/Fashion-MNIST IDX files. Place them under:

```text
data/mnist/train-images.idx3-ubyte
data/mnist/train-labels.idx1-ubyte
data/mnist/t10k-images.idx3-ubyte
data/mnist/t10k-labels.idx1-ubyte

data/fashion_mnist/train-images-idx3-ubyte
data/fashion_mnist/train-labels-idx1-ubyte
data/fashion_mnist/t10k-images-idx3-ubyte
data/fashion_mnist/t10k-labels-idx1-ubyte
```

The scikit-learn digits dataset is loaded automatically.

## Reproduction commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Validate datasets:

```bash
python validate_datasets.py
```

Run full benchmark / prior benchmark scripts:

```bash
python run_full_benchmark.py
python run_sample_efficiency.py
python run_continual_learning.py
```

Run mechanism-isolation analyses:

```bash
python run_sparsity_sweep_subset.py
python run_replay_sweep_subset.py
python run_frozen_readout_geometry_uploaded.py --datasets mnist fashion_mnist --seeds 0 1 2 3 4 5 6 7 8 9 --train-subset 2000 --test-subset 1000 --local-epochs 5
```

Generate figures:

```bash
python plot_results.py
python make_enhanced_figures.py
python make_all_figures.py
python make_interpretability_uploaded.py
```

Generate manuscript:

```bash
python make_sparsemechanismbench_final.py
```

## Integrity notes

- No test-set tuning was performed.
- All numerical results in the manuscript are traceable to CSV files in `results/` and figures in `figures/`.
- Frozen readout and representation-geometry analyses use fixed mechanism-isolation subsets and are not state-of-the-art benchmark claims.
- The MLP/CNN hidden-feature frozen-readout controls were run over 5 seeds; local/raw-pixel frozen-readout experiments were run over 10 seeds.

## Key CSVs

- `results/updated_classification_summary.csv`
- `results/sparsity_sweep_subset_summary.csv`
- `results/replay_sweep_subset_summary.csv`
- `results/frozen_readout_mnist_fashion_by_seed.csv`
- `results/frozen_readout_mnist_fashion_summary.csv`
- `results/representation_geometry_mnist_fashion_by_seed.csv`
- `results/representation_geometry_mnist_fashion_summary.csv`
- `results/frozen_readout_statistical_tests.csv`
- `results/selectivity_summary.csv`
