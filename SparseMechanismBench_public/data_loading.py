from pathlib import Path
import struct
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

DATA_ROOT = Path(__file__).resolve().parent / 'data'
DEFAULT_EXTERNAL_DATA = Path(__file__).resolve().parent / 'data'

def _read_idx_images(path):
    path = Path(path)
    with open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    return data.astype(np.float32) / 255.0

def _read_idx_labels(path):
    path = Path(path)
    with open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)

def _idx_paths(dataset, external_root=DEFAULT_EXTERNAL_DATA):
    if dataset == 'mnist':
        d = Path(external_root) / 'mnist'
        return d/'train-images.idx3-ubyte', d/'train-labels.idx1-ubyte', d/'t10k-images.idx3-ubyte', d/'t10k-labels.idx1-ubyte'
    if dataset == 'fashion_mnist':
        d = Path(external_root) / 'fashion_mnist'
        return d/'train-images-idx3-ubyte', d/'train-labels-idx1-ubyte', d/'t10k-images-idx3-ubyte', d/'t10k-labels-idx1-ubyte'
    raise ValueError(dataset)

def load_dataset(dataset, external_root=DEFAULT_EXTERNAL_DATA, digits_seed=42):
    """Return (X_train_flat, y_train, X_test_flat, y_test, image_shape)."""
    dataset = dataset.lower()
    if dataset == 'digits':
        X, y = load_digits(return_X_y=True)
        X = (X.astype(np.float32) / 16.0)
        X_train, X_test, y_train, y_test = train_test_split(X, y.astype(np.int64), test_size=0.2, random_state=digits_seed, stratify=y)
        return X_train, y_train, X_test, y_test, (8, 8)
    if dataset in ['mnist', 'fashion_mnist']:
        tr_i, tr_l, te_i, te_l = _idx_paths(dataset, external_root)
        X_train = _read_idx_images(tr_i)
        y_train = _read_idx_labels(tr_l)
        X_test = _read_idx_images(te_i)
        y_test = _read_idx_labels(te_l)
        return X_train.reshape(len(X_train), -1), y_train, X_test.reshape(len(X_test), -1), y_test, X_train.shape[1:]
    raise ValueError(f'Unknown dataset {dataset}')

def subset_by_fraction(X, y, fraction, seed):
    rng = np.random.default_rng(seed)
    n = max(1, int(round(len(X) * fraction)))
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]

def split_tasks(X, y):
    m1 = y < 5
    m2 = y >= 5
    return X[m1], y[m1], X[m2], y[m2]
