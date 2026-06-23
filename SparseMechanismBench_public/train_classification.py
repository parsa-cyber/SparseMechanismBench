import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from data_loading import load_dataset, subset_by_fraction
from models import MLP, LeNetStyleCNN, LocalOjaFeatures

RESULTS = Path(__file__).resolve().parent / 'results'
RESULTS.mkdir(exist_ok=True)

def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)

def summarize(path, out_path, group_cols):
    df = pd.read_csv(path)
    num_cols = [c for c in df.columns if c not in group_cols and pd.api.types.is_numeric_dtype(df[c])]
    rows=[]
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple): keys=(keys,)
        row=dict(zip(group_cols,keys))
        for c in num_cols:
            row[c+'_mean']=g[c].mean(); row[c+'_std']=g[c].std(ddof=0); row[c+'_n']=g[c].count()
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path,index=False)

# ----- Supervised baselines -----
def train_logreg(Xtr,ytr,Xte,yte,seed=0,max_iter=200):
    t0=time.time(); clf=LogisticRegression(max_iter=max_iter, solver='lbfgs', n_jobs=1, random_state=seed)
    clf.fit(Xtr,ytr); secs=time.time()-t0
    pred=clf.predict(Xte); acc=accuracy_score(yte,pred)
    params = Xtr.shape[1]*10 + 10
    return acc, secs, params, clf

def train_random_features(Xtr,ytr,Xte,yte,seed=0,n_features=256):
    rng=np.random.default_rng(seed); W=rng.normal(0,1/np.sqrt(Xtr.shape[1]),size=(Xtr.shape[1],n_features)).astype(np.float32); b=rng.normal(0,0.1,size=n_features).astype(np.float32)
    Ztr=np.maximum(Xtr@W+b,0); Zte=np.maximum(Xte@W+b,0)
    acc,secs,params,clf=train_logreg(Ztr,ytr,Zte,yte,seed=seed,max_iter=200)
    return acc,secs,params+W.size+b.size,clf,W,b

def train_pca_linear(Xtr,ytr,Xte,yte,seed=0,n_components=64):
    n_components=min(n_components, Xtr.shape[1], len(Xtr)-1)
    t0=time.time(); pca=PCA(n_components=n_components, random_state=seed); Ztr=pca.fit_transform(Xtr); Zte=pca.transform(Xte)
    clf=LogisticRegression(max_iter=200, solver='lbfgs', n_jobs=1, random_state=seed); clf.fit(Ztr,ytr)
    secs=time.time()-t0; acc=accuracy_score(yte,clf.predict(Zte)); params=n_components*10+10 + pca.components_.size
    return acc,secs,params,pca,clf

def train_torch_model(model, Xtr, ytr, Xte, yte, image_shape=None, epochs=5, lr=1e-3, batch_size=512, seed=0):
    set_seed(seed); torch.set_num_threads(4)
    if image_shape is None:
        Xtr_t=torch.tensor(Xtr,dtype=torch.float32); Xte_t=torch.tensor(Xte,dtype=torch.float32)
    else:
        Xtr_t=torch.tensor(Xtr.reshape(-1,1,*image_shape),dtype=torch.float32); Xte_t=torch.tensor(Xte.reshape(-1,1,*image_shape),dtype=torch.float32)
    ytr_t=torch.tensor(ytr,dtype=torch.long); yte_t=torch.tensor(yte,dtype=torch.long)
    dl=DataLoader(TensorDataset(Xtr_t,ytr_t),batch_size=batch_size,shuffle=True,num_workers=0)
    opt=torch.optim.Adam(model.parameters(),lr=lr); loss_fn=nn.CrossEntropyLoss()
    t0=time.time(); curve=[]
    for ep in range(epochs):
        model.train()
        for xb,yb in dl:
            opt.zero_grad(); out=model(xb); loss=loss_fn(out,yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred=model(Xte_t).argmax(1); acc=float((pred==yte_t).float().mean().item())
        curve.append({'epoch':ep+1,'accuracy':acc}); print(f'epoch {ep+1}/{epochs} acc={acc:.4f}', flush=True)
    secs=time.time()-t0
    with torch.no_grad():
        logits=model(Xte_t); pred=logits.argmax(1).numpy(); acc=accuracy_score(yte,pred)
        try:
            H=model.encode(Xte_t[:2000]).detach().numpy(); sparsity=float(np.mean(H==0))
        except Exception:
            sparsity=np.nan
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return acc,secs,params,sparsity,pred,curve,model

def train_local_model(kind, Xtr,ytr,Xte,yte,seed=0,hidden=128,epochs=5,active_frac=None):
    if kind=='oja':
        learner=LocalOjaFeatures(Xtr.shape[1],hidden_dim=hidden,lr=0.02,active_frac=None,homeostasis=False,seed=seed)
    elif kind=='sparse_oja':
        learner=LocalOjaFeatures(Xtr.shape[1],hidden_dim=hidden,lr=0.02,active_frac=active_frac or 0.10,homeostasis=False,seed=seed)
    elif kind=='homeostatic_sparse_oja':
        learner=LocalOjaFeatures(Xtr.shape[1],hidden_dim=hidden,lr=0.02,active_frac=active_frac or 0.10,homeostasis=True,target_rate=active_frac or 0.10,theta_lr=0.01,seed=seed)
    else:
        raise ValueError(kind)
    t0=time.time(); learner.fit(Xtr,epochs=epochs,batch_size=1024,seed=seed)
    Ztr=learner.transform(Xtr); Zte=learner.transform(Xte)
    clf=SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=20, tol=1e-3, random_state=seed)
    clf.fit(Ztr,ytr); secs=time.time()-t0
    pred=clf.predict(Zte); acc=accuracy_score(yte,pred); sparsity=float(np.mean(Zte==0)); params=learner.W.size + hidden*10 + 10
    return acc,secs,params,sparsity,pred,learner,clf

if __name__=='__main__':
    import argparse, os
    ap=argparse.ArgumentParser(); ap.add_argument('--datasets',nargs='+',default=['digits','mnist','fashion_mnist']); ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4]); ap.add_argument('--model',default='standard_cnn'); ap.add_argument('--epochs',type=int,default=3)
    args=ap.parse_args()
    out=RESULTS/f'{args.model}_by_seed.csv'
    existing=pd.read_csv(out) if out.exists() else pd.DataFrame()
    rows=[]
    for dataset in args.datasets:
        Xtr,ytr,Xte,yte,shape=load_dataset(dataset)
        for seed in args.seeds:
            if not existing.empty and ((existing['dataset']==dataset)&(existing['seed']==seed)&(existing['model']==args.model)).any():
                continue
            if args.model=='standard_cnn':
                model=LeNetStyleCNN(image_shape=shape); acc,secs,params,sparsity,pred,curve,trained=train_torch_model(model,Xtr,ytr,Xte,yte,image_shape=shape,epochs=args.epochs,lr=1e-3,batch_size=512,seed=seed)
            elif args.model=='homeostatic_sparse_oja':
                acc,secs,params,sparsity,pred,learner,clf=train_local_model('homeostatic_sparse_oja',Xtr,ytr,Xte,yte,seed=seed,hidden=128,epochs=args.epochs,active_frac=0.10)
            else:
                raise ValueError(args.model)
            row={'dataset':dataset,'model':args.model,'seed':seed,'accuracy':acc,'train_time_s':secs,'param_count':params,'activation_sparsity':sparsity,'epochs':args.epochs}
            rows.append(row); pd.DataFrame((existing.to_dict('records') if not existing.empty else [])+rows).to_csv(out,index=False)
            print(row, flush=True)
    summarize(out, RESULTS/f'{args.model}_summary.csv', ['dataset','model'])
