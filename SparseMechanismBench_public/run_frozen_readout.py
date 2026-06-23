"""Frozen Feature Readout Experiment.

Trains unsupervised/local feature extractors on the training split only,
freezes features, and trains multiple supervised readouts on those frozen
representations. Results are written after every seed/readout so interrupted
runs remain auditable.

If MNIST/Fashion-MNIST IDX data are not available, the script records a
failed_missing_data status rather than fabricating results.
"""
import argparse, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import torch

from data_loading import load_dataset
from models import LocalOjaFeatures, MLP, LeNetStyleCNN
from train_classification import train_torch_model

RESULTS = Path(__file__).resolve().parent / 'results'
RESULTS.mkdir(exist_ok=True)

READOUTS = ['logreg', 'ridge', 'linear_svm', 'knn5', 'small_mlp']
LOCAL_SOURCES = ['oja', 'sparse_oja', 'homeostatic_sparse_oja']
COMPARE_SOURCES = ['raw_pixels', 'mlp_hidden', 'standard_cnn_penultimate']


def fit_feature_source(source, Xtr, ytr, Xte, yte, image_shape, seed, local_epochs, hidden_dim, max_train_for_features=None):
    t0 = time.time()
    if max_train_for_features and len(Xtr) > max_train_for_features:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Xtr), size=max_train_for_features, replace=False)
        Xfit, yfit = Xtr[idx], ytr[idx]
    else:
        Xfit, yfit = Xtr, ytr

    if source == 'raw_pixels':
        Ztr, Zte = Xtr.astype(np.float32), Xte.astype(np.float32)
        return Ztr, Zte, 0.0, time.time()-t0, 'ok'

    if source in LOCAL_SOURCES:
        active = None if source == 'oja' else 0.10
        homeo = (source == 'homeostatic_sparse_oja')
        learner = LocalOjaFeatures(Xtr.shape[1], hidden_dim=hidden_dim, lr=0.02, active_frac=active,
                                    homeostasis=homeo, target_rate=active or 0.10, theta_lr=0.01, seed=seed)
        learner.fit(Xfit, epochs=local_epochs, batch_size=1024, seed=seed)
        Ztr, Zte = learner.transform(Xtr), learner.transform(Xte)
        sparsity = float(np.mean(Zte == 0))
        return Ztr, Zte, sparsity, time.time()-t0, 'ok'

    if source == 'mlp_hidden':
        # supervised comparison feature: trained only on training split.
        model = MLP(Xtr.shape[1], hidden_dim=128, dropout=0.0)
        acc, secs, params, sp, pred, curve, trained = train_torch_model(
            model, Xfit, yfit, Xte, yte, image_shape=None, epochs=5, lr=1e-3, batch_size=512, seed=seed)
        with torch.no_grad():
            Ztr = trained.encode(torch.tensor(Xtr, dtype=torch.float32)).numpy()
            Zte = trained.encode(torch.tensor(Xte, dtype=torch.float32)).numpy()
        sparsity = float(np.mean(Zte == 0))
        return Ztr, Zte, sparsity, time.time()-t0, 'ok'

    if source == 'standard_cnn_penultimate':
        model = LeNetStyleCNN(image_shape=image_shape)
        acc, secs, params, sp, pred, curve, trained = train_torch_model(
            model, Xfit, yfit, Xte, yte, image_shape=image_shape, epochs=5 if image_shape==(8,8) else 2, lr=1e-3, batch_size=512, seed=seed)
        with torch.no_grad():
            Xtr_t = torch.tensor(Xtr.reshape(-1,1,*image_shape), dtype=torch.float32)
            Xte_t = torch.tensor(Xte.reshape(-1,1,*image_shape), dtype=torch.float32)
            Ztr = trained.encode(Xtr_t).numpy()
            Zte = trained.encode(Xte_t).numpy()
        sparsity = float(np.mean(Zte == 0))
        return Ztr, Zte, sparsity, time.time()-t0, 'ok'

    raise ValueError(source)


def fit_readout(readout, Ztr, ytr, Zte, yte, seed):
    t0 = time.time()
    if readout == 'logreg':
        clf = LogisticRegression(max_iter=500, solver='lbfgs', random_state=seed, n_jobs=1)
    elif readout == 'ridge':
        clf = RidgeClassifier(alpha=1.0)
    elif readout == 'linear_svm':
        clf = LinearSVC(C=1.0, max_iter=3000, dual='auto', random_state=seed)
    elif readout == 'knn5':
        clf = KNeighborsClassifier(n_neighbors=5)
    elif readout == 'small_mlp':
        clf = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', max_iter=300, alpha=1e-4, random_state=seed, early_stopping=True)
    else:
        raise ValueError(readout)
    clf.fit(Ztr, ytr)
    pred = clf.predict(Zte)
    return float(accuracy_score(yte, pred)), time.time() - t0


def append_rows(path, rows):
    df_new = pd.DataFrame(rows)
    if path.exists():
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)


def summarize(by_seed_path, summary_path):
    df = pd.read_csv(by_seed_path)
    ok = df[df['status'] == 'ok'].copy()
    if ok.empty:
        pd.DataFrame().to_csv(summary_path, index=False); return
    rows=[]
    for keys, g in ok.groupby(['dataset','feature_source','readout']):
        ds, src, ro = keys
        row={'dataset':ds,'feature_source':src,'readout':ro,'n':len(g)}
        for c in ['accuracy','feature_sparsity','feature_train_time_s','readout_train_time_s']:
            row[c+'_mean']=float(g[c].mean())
            row[c+'_std']=float(g[c].std(ddof=0))
        rows.append(row)
    pd.DataFrame(rows).to_csv(summary_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['digits','mnist','fashion_mnist'])
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    ap.add_argument('--feature-sources', nargs='+', default=LOCAL_SOURCES + COMPARE_SOURCES)
    ap.add_argument('--readouts', nargs='+', default=READOUTS)
    ap.add_argument('--hidden-dim', type=int, default=128)
    ap.add_argument('--local-epochs', type=int, default=5)
    ap.add_argument('--max-train-for-features', type=int, default=10000)
    ap.add_argument('--max-train-for-readout', type=int, default=15000)
    args = ap.parse_args()

    by_seed = RESULTS / 'frozen_readout_by_seed.csv'
    summary = RESULTS / 'frozen_readout_summary.csv'
    existing = pd.read_csv(by_seed) if by_seed.exists() else pd.DataFrame()

    for dataset in args.datasets:
        try:
            Xtr, ytr, Xte, yte, shape = load_dataset(dataset)
        except Exception as e:
            rows=[{'dataset':dataset,'seed':s,'feature_source':'ALL','readout':'ALL','status':'failed_missing_data','error':repr(e)} for s in args.seeds]
            append_rows(by_seed, rows)
            print(f'{dataset} failed: {e}', flush=True)
            continue

        # Limit readout train size deterministically for runtime, but keep test full.
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            if args.max_train_for_readout and len(Xtr) > args.max_train_for_readout:
                idx_ro = rng.choice(len(Xtr), size=args.max_train_for_readout, replace=False)
                Xro, yro = Xtr[idx_ro], ytr[idx_ro]
            else:
                Xro, yro = Xtr, ytr

            for source in args.feature_sources:
                # Skip if all readouts already finished.
                try:
                    Ztr, Zte, sp, ftime, stat = fit_feature_source(source, Xro, yro, Xte, yte, shape, seed,
                                                                  args.local_epochs, args.hidden_dim,
                                                                  args.max_train_for_features)
                    rows=[]
                    for readout in args.readouts:
                        if not existing.empty and ((existing.get('dataset')==dataset)&(existing.get('seed')==seed)&(existing.get('feature_source')==source)&(existing.get('readout')==readout)&(existing.get('status')=='ok')).any():
                            continue
                        try:
                            acc, rtime = fit_readout(readout, Ztr, yro, Zte, yte, seed)
                            rows.append({'dataset':dataset,'seed':seed,'feature_source':source,'readout':readout,
                                         'accuracy':acc,'feature_sparsity':sp,'feature_train_time_s':ftime,
                                         'readout_train_time_s':rtime,'n_train_readout':len(yro),'n_test':len(yte),
                                         'hidden_dim':args.hidden_dim,'local_epochs':args.local_epochs,'status':'ok','error':''})
                        except Exception as e:
                            rows.append({'dataset':dataset,'seed':seed,'feature_source':source,'readout':readout,
                                         'accuracy':np.nan,'feature_sparsity':sp,'feature_train_time_s':ftime,
                                         'readout_train_time_s':np.nan,'n_train_readout':len(yro),'n_test':len(yte),
                                         'hidden_dim':args.hidden_dim,'local_epochs':args.local_epochs,'status':'failed_readout','error':repr(e)})
                    if rows:
                        append_rows(by_seed, rows)
                        existing = pd.read_csv(by_seed)
                        print(f'{dataset} seed={seed} source={source} done', flush=True)
                except Exception as e:
                    rows=[{'dataset':dataset,'seed':seed,'feature_source':source,'readout':r,'status':'failed_feature','error':repr(e)} for r in args.readouts]
                    append_rows(by_seed, rows)
                    print('failed', dataset, seed, source, e, flush=True)
                    traceback.print_exc()
    summarize(by_seed, summary)

if __name__ == '__main__':
    main()
