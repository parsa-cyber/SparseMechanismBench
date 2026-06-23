"""Representation Geometry Analysis.

Computes class-separability and feature-health metrics for raw pixels,
local Oja features, Sparse Oja features, Homeostatic Sparse Oja features,
MLP hidden layers, and Standard CNN penultimate features. Missing IDX data
are logged rather than silently ignored.
"""
import argparse, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import torch

from data_loading import load_dataset
from models import LocalOjaFeatures, MLP, LeNetStyleCNN
from train_classification import train_torch_model

RESULTS = Path(__file__).resolve().parent / 'results'; RESULTS.mkdir(exist_ok=True)
FIGURES = Path(__file__).resolve().parent / 'figures'; FIGURES.mkdir(exist_ok=True)


def get_features(source, Xtr, ytr, Xte, yte, shape, seed, hidden_dim=128, local_epochs=5, max_train=10000):
    if max_train and len(Xtr) > max_train:
        rng = np.random.default_rng(seed); idx = rng.choice(len(Xtr), size=max_train, replace=False)
        Xfit, yfit = Xtr[idx], ytr[idx]
    else:
        Xfit, yfit = Xtr, ytr
    if source == 'raw_pixels':
        return Xtr.astype(np.float32), Xte.astype(np.float32), 0.0
    if source in ['oja','sparse_oja','homeostatic_sparse_oja']:
        active = None if source=='oja' else 0.10
        homeo = source=='homeostatic_sparse_oja'
        learner = LocalOjaFeatures(Xtr.shape[1], hidden_dim=hidden_dim, lr=0.02, active_frac=active,
                                    homeostasis=homeo, target_rate=active or 0.10, theta_lr=0.01, seed=seed)
        learner.fit(Xfit, epochs=local_epochs, batch_size=1024, seed=seed)
        return learner.transform(Xtr), learner.transform(Xte), float(np.mean(learner.transform(Xte)==0))
    if source == 'mlp_hidden':
        model = MLP(Xtr.shape[1], hidden_dim=128, dropout=0.0)
        acc, secs, params, sp, pred, curve, trained = train_torch_model(model, Xfit, yfit, Xte, yte, image_shape=None, epochs=5, lr=1e-3, batch_size=512, seed=seed)
        with torch.no_grad():
            Ztr = trained.encode(torch.tensor(Xtr,dtype=torch.float32)).numpy()
            Zte = trained.encode(torch.tensor(Xte,dtype=torch.float32)).numpy()
        return Ztr, Zte, float(np.mean(Zte==0))
    if source == 'standard_cnn_penultimate':
        model = LeNetStyleCNN(image_shape=shape)
        acc, secs, params, sp, pred, curve, trained = train_torch_model(model, Xfit, yfit, Xte, yte, image_shape=shape, epochs=5 if shape==(8,8) else 2, lr=1e-3, batch_size=512, seed=seed)
        with torch.no_grad():
            Ztr = trained.encode(torch.tensor(Xtr.reshape(-1,1,*shape),dtype=torch.float32)).numpy()
            Zte = trained.encode(torch.tensor(Xte.reshape(-1,1,*shape),dtype=torch.float32)).numpy()
        return Ztr, Zte, float(np.mean(Zte==0))
    raise ValueError(source)


def scatter_metrics(Z, y, max_samples=2000, seed=0):
    rng = np.random.default_rng(seed)
    if len(Z) > max_samples:
        idx = rng.choice(len(Z), size=max_samples, replace=False)
        Zs, ys = Z[idx], y[idx]
    else:
        Zs, ys = Z, y
    # Standardize for distance-based metrics.
    Zs = Zs.astype(np.float32)
    Zs = (Zs - Zs.mean(axis=0, keepdims=True)) / (Zs.std(axis=0, keepdims=True) + 1e-6)
    sil = silhouette_score(Zs, ys) if len(np.unique(ys)) > 1 else np.nan
    db = davies_bouldin_score(Zs, ys) if len(np.unique(ys)) > 1 else np.nan
    labels = np.unique(ys)
    means = np.stack([Zs[ys==c].mean(axis=0) for c in labels])
    global_mean = Zs.mean(axis=0)
    between = sum(((ys==c).sum()) * np.sum((means[i]-global_mean)**2) for i,c in enumerate(labels)) / len(Zs)
    within = sum(np.sum((Zs[ys==c]-means[i])**2) for i,c in enumerate(labels)) / len(Zs)
    fisher = float(between/(within+1e-8))
    # distance ratio via centroids: mean distance to own centroid vs other centroids
    own=[]; other=[]
    for i,z in enumerate(Zs):
        ci=np.where(labels==ys[i])[0][0]
        own.append(np.linalg.norm(z-means[ci]))
        others=np.delete(means, ci, axis=0)
        other.append(np.min(np.linalg.norm(others-z, axis=1)))
    intra=float(np.mean(own)); inter=float(np.mean(other)); ratio=inter/(intra+1e-8)
    return sil, db, fisher, intra, inter, ratio


def feature_health(Z, y):
    sp=float(np.mean(Z==0))
    mean_unit=Z.mean(axis=0)
    dead=float(np.mean(np.abs(mean_unit) < 1e-7))
    active_counts=(Z>0).mean(axis=0)
    util_entropy=float(-np.sum((active_counts/(active_counts.sum()+1e-8))*np.log((active_counts/(active_counts.sum()+1e-8))+1e-8)))
    # class selectivity index by unit using class means.
    cs=[]
    for j in range(Z.shape[1]):
        m=np.array([Z[y==c,j].mean() for c in np.unique(y)])
        mx=m.max(); rest=(m.sum()-mx)/max(1,len(m)-1)
        cs.append((mx-rest)/(mx+rest+1e-8))
    return sp, dead, float(np.mean(mean_unit)), util_entropy, float(np.mean(cs))


def append_csv(path, rows):
    df_new=pd.DataFrame(rows)
    if path.exists():
        df=pd.concat([pd.read_csv(path), df_new], ignore_index=True)
    else:
        df=df_new
    df.to_csv(path,index=False)


def plot_pca(dataset, feature_packs, yte, outpath, seed=0, max_samples=1000):
    n=len(feature_packs)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), squeeze=False)
    rng=np.random.default_rng(seed)
    idx = rng.choice(len(yte), size=min(max_samples,len(yte)), replace=False)
    for ax,(source,Z) in zip(axes[0], feature_packs):
        Zs=Z[idx]
        if Zs.shape[1] > 2:
            P=PCA(n_components=2, random_state=seed).fit_transform(Zs)
        else:
            P=Zs
        sc=ax.scatter(P[:,0], P[:,1], c=yte[idx], s=6, cmap='tab10', alpha=0.75)
        ax.set_title(source)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'Representation PCA: {dataset}')
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['digits','mnist','fashion_mnist'])
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    ap.add_argument('--sources', nargs='+', default=['raw_pixels','oja','sparse_oja','homeostatic_sparse_oja','mlp_hidden','standard_cnn_penultimate'])
    ap.add_argument('--local-epochs', type=int, default=5)
    args=ap.parse_args()
    by_seed=RESULTS/'representation_geometry_by_seed.csv'
    summary=RESULTS/'representation_geometry_summary.csv'
    pca_summary=[]
    for dataset in args.datasets:
        try:
            Xtr,ytr,Xte,yte,shape=load_dataset(dataset)
        except Exception as e:
            rows=[{'dataset':dataset,'seed':s,'source':'ALL','status':'failed_missing_data','error':repr(e)} for s in args.seeds]
            append_csv(by_seed, rows); print(dataset,'failed missing data',e); continue
        for seed in args.seeds:
            pca_packs=[]
            for source in args.sources:
                try:
                    t0=time.time(); Ztr,Zte,sp_decl=get_features(source,Xtr,ytr,Xte,yte,shape,seed,local_epochs=args.local_epochs)
                    lp=LogisticRegression(max_iter=500, solver='lbfgs', random_state=seed, n_jobs=1).fit(Ztr, ytr)
                    lp_acc=float(accuracy_score(yte, lp.predict(Zte)))
                    sil,db,fisher,intra,inter,ratio=scatter_metrics(Zte,yte,seed=seed)
                    sp,dead,mean_act,util,select=feature_health(Zte,yte)
                    rows=[{'dataset':dataset,'seed':seed,'source':source,'linear_probe_accuracy':lp_acc,
                           'silhouette':sil,'davies_bouldin':db,'fisher_ratio':fisher,
                           'mean_intra_class_distance':intra,'mean_inter_class_distance':inter,
                           'inter_intra_ratio':ratio,'activation_sparsity':sp,'dead_unit_rate':dead,
                           'mean_activation_per_unit':mean_act,'utilization_entropy':util,
                           'class_selectivity_index':select,'time_s':time.time()-t0,'status':'ok','error':''}]
                    append_csv(by_seed, rows)
                    if seed == args.seeds[0]: pca_packs.append((source,Zte))
                    print('geometry',dataset,seed,source,lp_acc, flush=True)
                except Exception as e:
                    append_csv(by_seed, [{'dataset':dataset,'seed':seed,'source':source,'status':'failed','error':repr(e)}])
                    traceback.print_exc()
            if seed == args.seeds[0] and pca_packs:
                plot_pca(dataset, pca_packs, yte, FIGURES/f'representation_pca_{dataset}.png', seed=seed)
    df=pd.read_csv(by_seed)
    ok=df[df.status=='ok'].copy()
    if not ok.empty:
        rows=[]
        metrics=[c for c in ok.columns if c not in ['dataset','seed','source','status','error'] and pd.api.types.is_numeric_dtype(ok[c])]
        for keys,g in ok.groupby(['dataset','source']):
            row=dict(zip(['dataset','source'],keys)); row['n']=len(g)
            for c in metrics:
                row[c+'_mean']=float(g[c].mean()); row[c+'_std']=float(g[c].std(ddof=0))
            rows.append(row)
        pd.DataFrame(rows).to_csv(summary,index=False)

if __name__=='__main__': main()
