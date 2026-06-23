import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.0, n_classes=10):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, n_classes))
        self.net = nn.Sequential(*layers)
        self.hidden = self.net[0]
        self.dropout = dropout
    def forward(self, x):
        return self.net(x)
    def encode(self, x):
        h = torch.relu(self.hidden(x))
        return h

class LeNetStyleCNN(nn.Module):
    def __init__(self, image_shape=(28,28), n_classes=10):
        super().__init__()
        h, w = image_shape
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2)
        )
        with torch.no_grad():
            dummy = torch.zeros(1,1,h,w)
            flat = self.conv(dummy).view(1,-1).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat, 128), nn.ReLU(), nn.Linear(128, n_classes))
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))
    def encode(self, x):
        z = self.conv(x).view(x.size(0), -1)
        return torch.relu(self.fc[0](z))

class LocalOjaFeatures:
    def __init__(self, input_dim, hidden_dim=128, lr=0.02, active_frac=None, homeostasis=False, target_rate=0.10, theta_lr=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1/np.sqrt(input_dim), size=(hidden_dim, input_dim)).astype(np.float32)
        self.theta = np.zeros(hidden_dim, dtype=np.float32)
        self.lr = lr
        self.active_frac = active_frac
        self.homeostasis = homeostasis
        self.target_rate = target_rate
        self.theta_lr = theta_lr
        self.hidden_dim = hidden_dim
        self.running_rate = np.zeros(hidden_dim, dtype=np.float32)
    def _activate_batch(self, X):
        Z = X @ self.W.T - self.theta[None, :]
        Z = np.maximum(Z, 0)
        if self.active_frac is not None and self.active_frac < 1.0:
            k = max(1, int(round(self.hidden_dim * self.active_frac)))
            idx = np.argpartition(Z, -k, axis=1)[:, -k:]
            mask = np.zeros_like(Z, dtype=bool)
            rows = np.arange(Z.shape[0])[:, None]
            mask[rows, idx] = True
            Z = np.where(mask, Z, 0)
        return Z
    def fit(self, X, epochs=5, batch_size=1024, seed=0):
        rng = np.random.default_rng(seed)
        n = len(X)
        X = X.astype(np.float32)
        for ep in range(epochs):
            for start in range(0, n, batch_size):
                # shuffled mini-batches each epoch
                pass
            idxs = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = X[idxs[start:start+batch_size]]
                Z = self._activate_batch(batch)
                # batch Oja update: yx - y^2*w
                mean_yx = (Z.T @ batch) / max(1, len(batch))
                mean_y2 = (Z**2).mean(axis=0)[:, None]
                self.W += self.lr * (mean_yx - mean_y2 * self.W)
                # normalize rows for numerical stability
                norms = np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-8
                self.W /= norms
                if self.homeostasis:
                    active = (Z > 0).mean(axis=0).astype(np.float32)
                    self.running_rate = 0.95*self.running_rate + 0.05*active
                    # If fires too often, increase threshold; if too rarely, lower it.
                    self.theta += self.theta_lr * (active - self.target_rate)
            print(f'local epoch {ep+1}/{epochs} done', flush=True)
        return self
    def transform(self, X):
        return self._activate_batch(X.astype(np.float32)).astype(np.float32)
    def sparsity(self, X):
        Z = self.transform(X)
        return float(np.mean(Z == 0))
    def feature_images(self, image_shape):
        return self.W.reshape((self.hidden_dim,) + tuple(image_shape))
