# SparseMechanismBench Experiment Plan

## Goal
Test whether sparse local plasticity is sufficient for task-discriminative learning, or whether additional mechanisms such as homeostasis, replay, feedback, and task-relevant credit assignment are required.

## Experiments completed
1. Dataset validation for digits, MNIST, and Fashion-MNIST.
2. Full benchmark classification for supervised baselines and local-learning models.
3. LeNet-style Standard CNN baseline.
4. Homeostatic Sparse Oja.
5. Sparsity sweep over active fractions.
6. Replay sweep over replay levels.
7. Frozen-feature readout experiment on MNIST and Fashion-MNIST using uploaded IDX files.
8. Representation-geometry and feature-health analysis.
9. Statistical tests and effect-size estimates.
10. Interpretability figures for learned local features, activations, PCA, and confusion matrices.

## Planned extensions
1. Reward-modulated Hebbian/Oja learning.
2. Feedback-alignment hybrid.
3. EWC baseline.
4. Larger-scale full-data frozen-readout experiments.
5. Energy measurements and neuromorphic hardware comparisons.
