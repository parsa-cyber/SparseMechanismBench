"""Run Frozen Feature Readout and Representation Geometry experiments on uploaded IDX datasets.

This script is intentionally resumable: every completed feature-source/seed/dataset writes
CSV rows immediately. It uses fixed mechanism-isolation subsets for runtime and explicitly
records subset sizes. It does not tune on the test set.
"""
from pathlib import Path
import time, traceback, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats

import torch

from data_loading import load_dataset
from models import LocalOjaFeatures, MLP, LeNetStyleCNN
from train_classification import train_torch_model

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'; RESULTS.mkdir(exist_ok=True)
FIGURES = ROOT/'figures'; FIGURES.mkdir(exist_ok=True)

LOCAL_SOURCES = ['oja','sparse_oja','homeostatic_sparse_oja']
SUP_SOURCES = ['raw_pixels','mlp_hidden','standard_cnn_penultimate']
READOUTS = ['logreg','ridge','linear_svm','knn5','small_mlp']


def deterministic_subset(X, y, n, seed, stratified=True):
    rng = np.random.default_rng(seed)
    if n is None or len(X) <= n:
        return X, y
    if not stratified:
        idx = rng.choice(len(X), n, replace=False); return X[idx], y[idx]
    classes = np.unique(y)
    per = n // len(classes)
    extra = n - per*len(classes)
    idxs=[]
    for i,c in enumerate(classes):
        inds=np.where(y==c)[0]
        k = per + (1 if i < extra else 0)
        k=min(k, len(inds))
        idxs.extend(rng.choice(inds,k,replace=False).tolist())
    idxs=np.array(idxs)
    rng.shuffle(idxs)
    return X[idxs], y[idxs]


def fit_feature_source(source, Xtr, ytr, Xte, yte, shape, seed, hidden_dim, local_epochs):
    t0=time.time()
    status='ok'; err=''
    if source=='raw_pixels':
        return Xtr.astype(np.float32), Xte.astype(np.float32), 0.0, time.time()-t0, status, err
    if source in LOCAL_SOURCES:
        active = None if source=='oja' else 0.10
        homeo = source=='homeostatic_sparse_oja'
        learner=LocalOjaFeatures(Xtr.shape[1], hidden_dim=hidden_dim, lr=0.02, active_frac=active,
                                  homeostasis=homeo, target_rate=active or 0.10, theta_lr=0.01, seed=seed)
        learner.fit(Xtr, epochs=local_epochs, batch_size=1024, seed=seed)
        Ztr=learner.transform(Xtr); Zte=learner.transform(Xte)
        return Ztr, Zte, float(np.mean(Zte==0)), time.time()-t0, status, err
    if source=='mlp_hidden':
        model=MLP(Xtr.shape[1], hidden_dim=128, dropout=0.0)
        acc,secs,params,sp,pred,curve,trained=train_torch_model(model,Xtr,ytr,Xte,yte,image_shape=None,
                                                                  epochs=5,lr=1e-3,batch_size=512,seed=seed)
        with torch.no_grad():
            Ztr=trained.encode(torch.tensor(Xtr,dtype=torch.float32)).numpy()
            Zte=trained.encode(torch.tensor(Xte,dtype=torch.float32)).numpy()
        return Ztr,Zte,float(np.mean(Zte==0)),time.time()-t0,status,err
    if source=='standard_cnn_penultimate':
        model=LeNetStyleCNN(image_shape=shape)
        acc,secs,params,sp,pred,curve,trained=train_torch_model(model,Xtr,ytr,Xte,yte,image_shape=shape,
                                                                  epochs=3,lr=1e-3,batch_size=512,seed=seed)
        with torch.no_grad():
            Ztr=trained.encode(torch.tensor(Xtr.reshape(-1,1,*shape),dtype=torch.float32)).numpy()
            Zte=trained.encode(torch.tensor(Xte.reshape(-1,1,*shape),dtype=torch.float32)).numpy()
        return Ztr,Zte,float(np.mean(Zte==0)),time.time()-t0,status,err
    raise ValueError(source)


def fit_readout(readout, Ztr, ytr, Zte, yte, seed):
    t0=time.time()
    if readout=='logreg':
        # lbfgs works well on small fixed subsets.
        clf=SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=1000, tol=1e-3, random_state=seed)
    elif readout=='ridge':
        clf=RidgeClassifier(alpha=1.0)
    elif readout=='linear_svm':
        clf=SGDClassifier(loss='hinge', alpha=1e-4, max_iter=1000, tol=1e-3, random_state=seed)
    elif readout=='knn5':
        clf=KNeighborsClassifier(n_neighbors=5)
    elif readout=='small_mlp':
        clf=MLPClassifier(hidden_layer_sizes=(64,), max_iter=60, alpha=1e-4, random_state=seed, early_stopping=True, n_iter_no_change=5)
    else:
        raise ValueError(readout)
    clf.fit(Ztr,ytr)
    pred=clf.predict(Zte)
    return float(accuracy_score(yte,pred)), time.time()-t0


def feature_health(Z, y):
    Z=np.asarray(Z)
    sparsity=float(np.mean(Z==0))
    unit_mean=Z.mean(axis=0)
    dead=float(np.mean(np.abs(unit_mean)<1e-8))
    active=(Z>0).mean(axis=0)
    p=active/(active.sum()+1e-12)
    util=float(-(p*np.log(p+1e-12)).sum())
    cs=[]
    labels=np.unique(y)
    for j in range(Z.shape[1]):
        means=np.array([Z[y==c,j].mean() for c in labels])
        mx=means.max(); rest=(means.sum()-mx)/max(1,len(means)-1)
        cs.append(float((mx-rest)/(mx+rest+1e-12)))
    return sparsity, dead, float(unit_mean.mean()), util, float(np.mean(cs))


def geometry_metrics(Z, y, seed, max_geo=1000):
    rng=np.random.default_rng(seed)
    if len(Z)>max_geo:
        idx=rng.choice(len(Z), size=max_geo, replace=False)
        Z=Z[idx]; y=y[idx]
    Z=Z.astype(np.float32)
    # Drop zero-variance columns before scaling.
    std=Z.std(axis=0)
    keep=std>1e-8
    if keep.sum()<2:
        return {k:np.nan for k in ['silhouette','davies_bouldin','fisher_ratio','mean_intra_class_distance','mean_inter_class_distance','inter_intra_ratio']}
    Z=Z[:,keep]
    Z=(Z-Z.mean(axis=0))/(Z.std(axis=0)+1e-6)
    labels=np.unique(y)
    try: sil=float(silhouette_score(Z,y))
    except Exception: sil=np.nan
    try: db=float(davies_bouldin_score(Z,y))
    except Exception: db=np.nan
    means=np.stack([Z[y==c].mean(axis=0) for c in labels])
    global_mean=Z.mean(axis=0)
    between=sum(((y==c).sum())*np.sum((means[i]-global_mean)**2) for i,c in enumerate(labels))/len(Z)
    within=sum(np.sum((Z[y==c]-means[i])**2) for i,c in enumerate(labels))/len(Z)
    fisher=float(between/(within+1e-12))
    own=[]; other=[]
    for i,z in enumerate(Z):
        ci=int(np.where(labels==y[i])[0][0])
        own.append(np.linalg.norm(z-means[ci]))
        others=np.delete(means,ci,axis=0)
        other.append(np.min(np.linalg.norm(others-z,axis=1)))
    intra=float(np.mean(own)); inter=float(np.mean(other)); ratio=float(inter/(intra+1e-12))
    return {'silhouette':sil,'davies_bouldin':db,'fisher_ratio':fisher,'mean_intra_class_distance':intra,'mean_inter_class_distance':inter,'inter_intra_ratio':ratio}


def append(path, rows):
    new=pd.DataFrame(rows)
    if path.exists():
        old=pd.read_csv(path)
        df=pd.concat([old,new],ignore_index=True)
    else:
        df=new
    df.to_csv(path,index=False)


def summarize():
    fr=RESULTS/'frozen_readout_mnist_fashion_by_seed.csv'
    gs=RESULTS/'representation_geometry_mnist_fashion_by_seed.csv'
    if fr.exists():
        df=pd.read_csv(fr); ok=df[df.status=='ok']
        rows=[]
        for keys,g in ok.groupby(['dataset','feature_source','readout']):
            row=dict(zip(['dataset','feature_source','readout'],keys)); row['n']=len(g)
            for c in ['accuracy','feature_sparsity','feature_train_time_s','readout_train_time_s']:
                row[c+'_mean']=float(g[c].mean()); row[c+'_std']=float(g[c].std(ddof=1)) if len(g)>1 else 0.0
                se=(g[c].std(ddof=1)/np.sqrt(len(g))) if len(g)>1 else 0.0
                row[c+'_ci95_low']=float(g[c].mean()-1.96*se); row[c+'_ci95_high']=float(g[c].mean()+1.96*se)
            rows.append(row)
        pd.DataFrame(rows).to_csv(RESULTS/'frozen_readout_mnist_fashion_summary.csv',index=False)
    if gs.exists():
        df=pd.read_csv(gs); ok=df[df.status=='ok']
        rows=[]
        group=['dataset','feature_source']
        num=[c for c in ok.columns if c not in group+['seed','status','error'] and pd.api.types.is_numeric_dtype(ok[c])]
        for keys,g in ok.groupby(group):
            row=dict(zip(group,keys)); row['n']=len(g)
            for c in num:
                row[c+'_mean']=float(g[c].mean()); row[c+'_std']=float(g[c].std(ddof=1)) if len(g)>1 else 0.0
                se=(g[c].std(ddof=1)/np.sqrt(len(g))) if len(g)>1 else 0.0
                row[c+'_ci95_low']=float(g[c].mean()-1.96*se); row[c+'_ci95_high']=float(g[c].mean()+1.96*se)
            rows.append(row)
        pd.DataFrame(rows).to_csv(RESULTS/'representation_geometry_mnist_fashion_summary.csv',index=False)


def make_figures():
    # Frozen readout heatmap per dataset.
    path=RESULTS/'frozen_readout_mnist_fashion_summary.csv'
    if path.exists():
        df=pd.read_csv(path)
        for ds in df.dataset.unique():
            sub=df[df.dataset==ds]
            piv=sub.pivot(index='feature_source', columns='readout', values='accuracy_mean')
            fig,ax=plt.subplots(figsize=(8,4.5))
            im=ax.imshow(piv.values, cmap='viridis', vmin=0, vmax=1)
            ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=30, ha='right')
            ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    ax.text(j,i,f'{piv.values[i,j]:.2f}',ha='center',va='center',color='white' if piv.values[i,j]<0.55 else 'black',fontsize=8)
            fig.colorbar(im, ax=ax, label='Accuracy')
            ax.set_title(f'Frozen readout accuracy ({ds})')
            fig.tight_layout(); fig.savefig(FIGURES/f'frozen_readout_{ds}_heatmap.png',dpi=200,bbox_inches='tight'); plt.close(fig)
    path=RESULTS/'representation_geometry_mnist_fashion_summary.csv'
    if path.exists():
        df=pd.read_csv(path)
        for ds in df.dataset.unique():
            sub=df[df.dataset==ds].copy()
            fig,axes=plt.subplots(1,3,figsize=(13,4))
            metrics=['linear_probe_accuracy_mean','silhouette_mean','inter_intra_ratio_mean']
            titles=['Linear probe accuracy','Silhouette','Inter/intra distance ratio']
            for ax,m,t in zip(axes,metrics,titles):
                ax.bar(sub.feature_source, sub[m])
                ax.set_title(t); ax.tick_params(axis='x', labelrotation=30)
            fig.suptitle(f'Representation geometry ({ds})')
            fig.tight_layout(); fig.savefig(FIGURES/f'representation_geometry_{ds}_summary.png',dpi=200,bbox_inches='tight'); plt.close(fig)


def pca_plot_one_seed(dataset, packs, y, seed):
    rng=np.random.default_rng(seed); n=min(1200,len(y)); idx=rng.choice(len(y),n,replace=False)
    fig,axes=plt.subplots(1,len(packs),figsize=(4*len(packs),4),squeeze=False)
    for ax,(source,Z) in zip(axes[0],packs):
        Zs=Z[idx]
        P=PCA(n_components=2,random_state=seed).fit_transform(Zs) if Zs.shape[1]>2 else Zs
        ax.scatter(P[:,0],P[:,1],c=y[idx],cmap='tab10',s=6,alpha=0.75)
        ax.set_title(source); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'Representation PCA ({dataset}, seed {seed})')
    fig.tight_layout(); fig.savefig(FIGURES/f'representation_pca_{dataset}_uploaded.png',dpi=200,bbox_inches='tight'); plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--datasets',nargs='+',default=['mnist','fashion_mnist'])
    ap.add_argument('--seeds',nargs='+',type=int,default=list(range(10)))
    ap.add_argument('--feature-sources',nargs='+',default=LOCAL_SOURCES+SUP_SOURCES)
    ap.add_argument('--readouts',nargs='+',default=READOUTS)
    ap.add_argument('--train-subset',type=int,default=10000)
    ap.add_argument('--test-subset',type=int,default=2000)
    ap.add_argument('--hidden-dim',type=int,default=128)
    ap.add_argument('--local-epochs',type=int,default=5)
    args=ap.parse_args()
    fr_path=RESULTS/'frozen_readout_mnist_fashion_by_seed.csv'
    geo_path=RESULTS/'representation_geometry_mnist_fashion_by_seed.csv'
    existing_fr=pd.read_csv(fr_path) if fr_path.exists() else pd.DataFrame()
    existing_geo=pd.read_csv(geo_path) if geo_path.exists() else pd.DataFrame()
    for ds in args.datasets:
        Xtr_all,ytr_all,Xte_all,yte_all,shape=load_dataset(ds)
        for seed in args.seeds:
            Xtr,ytr=deterministic_subset(Xtr_all,ytr_all,args.train_subset,seed)
            Xte,yte=deterministic_subset(Xte_all,yte_all,args.test_subset,1000+seed)
            pca_packs=[]
            for source in args.feature_sources:
                # Skip if feature + all readouts already done and geometry done.
                if not existing_fr.empty:
                    have=existing_fr[(existing_fr.dataset==ds)&(existing_fr.seed==seed)&(existing_fr.feature_source==source)&(existing_fr.status=='ok')]['readout'].unique().tolist()
                    if set(args.readouts).issubset(set(have)):
                        print('skip readout',ds,seed,source,flush=True)
                        continue
                try:
                    Ztr,Zte,sp,ftime,status,err=fit_feature_source(source,Xtr,ytr,Xte,yte,shape,seed,args.hidden_dim,args.local_epochs)
                    # Representation geometry once per feature source.
                    if existing_geo.empty or not (((existing_geo.dataset==ds)&(existing_geo.seed==seed)&(existing_geo.feature_source==source)&(existing_geo.status=='ok')).any()):
                        try:
                            logreg=RidgeClassifier(alpha=1.0).fit(Ztr,ytr)
                            lp_acc=float(accuracy_score(yte,logreg.predict(Zte)))
                            gm=geometry_metrics(Zte,yte,seed,max_geo=min(800,len(yte)))
                            fh=feature_health(Zte,yte)
                            grow={'dataset':ds,'seed':seed,'feature_source':source,'linear_probe_accuracy':lp_acc,
                                  **gm,'activation_sparsity':fh[0],'dead_unit_rate':fh[1],'mean_activation_per_unit':fh[2],
                                  'utilization_entropy':fh[3],'class_selectivity_index':fh[4],
                                  'train_subset_n':len(Xtr),'test_subset_n':len(Xte),'status':'ok','error':''}
                            append(geo_path,[grow]); existing_geo=pd.read_csv(geo_path)
                            print('geometry',ds,seed,source,lp_acc,flush=True)
                            if seed==args.seeds[0]: pca_packs.append((source,Zte))
                        except Exception as eg:
                            append(geo_path,[{'dataset':ds,'seed':seed,'feature_source':source,'status':'failed','error':repr(eg)}])
                            traceback.print_exc()
                    # Readouts.
                    rows=[]
                    for ro in args.readouts:
                        if not existing_fr.empty and (((existing_fr.dataset==ds)&(existing_fr.seed==seed)&(existing_fr.feature_source==source)&(existing_fr.readout==ro)&(existing_fr.status=='ok')).any()):
                            continue
                        try:
                            acc,rtime=fit_readout(ro,Ztr,ytr,Zte,yte,seed)
                            rows.append({'dataset':ds,'seed':seed,'feature_source':source,'readout':ro,'accuracy':acc,
                                         'feature_sparsity':sp,'feature_train_time_s':ftime,'readout_train_time_s':rtime,
                                         'train_subset_n':len(Xtr),'test_subset_n':len(Xte),'status':'ok','error':''})
                            print('readout',ds,seed,source,ro,acc,flush=True)
                        except Exception as er:
                            rows.append({'dataset':ds,'seed':seed,'feature_source':source,'readout':ro,'status':'failed','error':repr(er)})
                            traceback.print_exc()
                    if rows:
                        append(fr_path,rows); existing_fr=pd.read_csv(fr_path)
                except Exception as e:
                    traceback.print_exc()
                    rows=[{'dataset':ds,'seed':seed,'feature_source':source,'readout':ro,'status':'failed_feature','error':repr(e)} for ro in args.readouts]
                    append(fr_path,rows); existing_fr=pd.read_csv(fr_path)
            if seed==args.seeds[0] and pca_packs:
                pca_plot_one_seed(ds,pca_packs,yte,seed)
    summarize(); make_figures()

if __name__=='__main__':
    main()
