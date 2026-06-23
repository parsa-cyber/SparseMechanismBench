# SparseMechanismBench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2008178.svg)](https://doi.org/10.5281/zenodo.2008178)

**SparseMechanismBench** is a reproducible mechanism-isolation study testing whether sparse local Hebbian/Oja-style plasticity is sufficient for task-discriminative learning and continual-learning stability.

## Paper

The public manuscript is available here:

**SparseMechanismBench_final_public.pdf**

Archived DOI: https://doi.org/10.5281/zenodo.2008178

## Research Question

Are sparse local-plasticity representations weak because the representations themselves are poorly task-aligned, or because the downstream readout/classifier is too weak?

## Main Result

Sparse Oja-style learning reliably creates sparse representations, but sparsity alone does not guarantee class separability, strong readout performance, or stable continual learning. Oja features can preserve task-accessible information with stronger readouts, but extreme Sparse Oja and Homeostatic Sparse Oja produce highly sparse representations with weaker class-aligned geometry.

## Methods

SparseMechanismBench compares:

* Logistic regression, MLP, and LeNet-style CNN baselines
* Oja, Sparse Oja, and Homeostatic Sparse Oja local learning rules
* Sparsity sweeps
* Replay sweeps
* Frozen-feature readout experiments
* Representation-geometry and feature-health diagnostics

Experiments use digits, MNIST, and Fashion-MNIST.

## Repository Contents

* `SparseMechanismBench_final_public.pdf` — public manuscript
* `SparseMechanismBench_public/` — reproducibility package
* `SparseMechanismBench_public/results/` — raw and summary CSV result files
* `SparseMechanismBench_public/figures/` — generated figures
* `SparseMechanismBench_public/requirements.txt` — Python dependencies
* `SparseMechanismBench_public/run_all_experiments.py` — main experiment runner

## Reproducing Results

Clone this repository and install dependencies:

```bash
pip install -r SparseMechanismBench_public/requirements.txt
```

Then run the relevant experiment scripts from the `SparseMechanismBench_public/` directory.

## Citation

If you use this project, cite the archived release:

```text
Fatehi, P. (2026). SparseMechanismBench: Why Sparse Local Plasticity Is Not Enough for Task-Discriminative Learning (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.2008178
```

## Author

Parsa Fatehi
Langley High School
