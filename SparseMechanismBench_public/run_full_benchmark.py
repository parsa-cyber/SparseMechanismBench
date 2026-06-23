"""Convenience wrapper for the full completed benchmark.

This wrapper preserves the benchmark design in the manuscript. It is intentionally
conservative: run individual scripts for long experiments if you need resumability.
"""
import subprocess, sys
cmds = [
    [sys.executable, 'validate_datasets.py'],
    [sys.executable, 'train_classification.py', '--model', 'standard_cnn', '--datasets', 'digits', 'mnist', 'fashion_mnist', '--seeds', '0','1','2','3','4', '--epochs', '2'],
    [sys.executable, 'train_classification.py', '--model', 'homeostatic_sparse_oja', '--datasets', 'digits', 'mnist', 'fashion_mnist', '--seeds', '0','1','2','3','4', '--epochs', '2'],
    [sys.executable, 'run_sparsity_sweep_subset.py'],
    [sys.executable, 'run_replay_sweep_subset.py'],
    [sys.executable, 'plot_results.py'],
]
for cmd in cmds:
    print('RUN:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
