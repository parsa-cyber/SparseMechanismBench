import time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from data_loading import load_dataset, split_tasks
from models import LocalOjaFeatures
RESULTS=Path(__file__).resolve().parent/'results'; OUT=RESULTS/'replay_sweep_subset_by_seed.csv'; SUM=RESULTS/'replay_sweep_subset_summary.csv'

def summarize():
    if not OUT.exists(): return
    df=pd.read_csv(OUT)
    rows=[]
    for keys,g in df.groupby(['dataset','split','model','replay_frac']):
        d,sp,m,r=keys; row={'dataset':d,'split':sp,'model':m,'replay_frac':r,'n':len(g)}
        for c in ['A_before','A_after','A_T2','avg_final','raw_forgetting','normalized_forgetting']:
            row[c+'_mean']=g[c].mean(); row[c+'_std']=g[c].std(ddof=0)
        rows.append(row)
    pd.DataFrame(rows).to_csv(SUM,index=False)

existing=pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
for dataset in ['mnist','fashion_mnist']:
    Xtr,ytr,Xte,yte,shape=load_dataset(dataset)
    rng0=np.random.default_rng(321); idx=rng0.choice(len(Xtr), size=10000, replace=False); Xtr,ytr=Xtr[idx],ytr[idx]
    X1,y1,X2,y2=split_tasks(Xtr,ytr); Xt1,yt1,Xt2,yt2=split_tasks(Xte,yte)
    for model_name, af, homeo in [('oja',None,False),('sparse_oja',0.10,False),('homeostatic_sparse_oja',0.10,True)]:
        for seed in [0,1,2,3,4]:
            # skip if all replay levels present
            if not existing.empty:
                done=existing[(existing.dataset==dataset)&(existing.model==model_name)&(existing.seed==seed)]
                if set(done.replay_frac.round(2).tolist()) >= {0.0,0.01,0.05,0.10,0.20}:
                    continue
            learner=LocalOjaFeatures(Xtr.shape[1],hidden_dim=128,lr=0.02,active_frac=af,homeostasis=homeo,target_rate=0.10,seed=seed)
            learner.fit(X1,epochs=1,batch_size=1024,seed=seed)
            Z1=learner.transform(X1); Z2=learner.transform(X2); Zt1=learner.transform(Xt1); Zt2=learner.transform(Xt2)
            clf1=SGDClassifier(loss='log_loss',max_iter=20,tol=1e-3,random_state=seed)
            clf1.fit(Z1,y1); A_before=accuracy_score(yt1,clf1.predict(Zt1)); rng=np.random.default_rng(seed)
            for replay in [0.0,0.01,0.05,0.10,0.20]:
                if not existing.empty and ((existing.dataset==dataset)&(existing.model==model_name)&(existing.seed==seed)&(abs(existing.replay_frac-replay)<1e-9)).any():
                    continue
                n_rep=int(round(len(Z1)*replay))
                if n_rep>0:
                    rep_idx=rng.choice(len(Z1),size=n_rep,replace=False); Ztrain=np.vstack([Z2,Z1[rep_idx]]); ytrain=np.concatenate([y2,y1[rep_idx]])
                else:
                    Ztrain=Z2; ytrain=y2
                clf2=SGDClassifier(loss='log_loss',max_iter=20,tol=1e-3,random_state=seed)
                clf2.fit(Ztrain,ytrain)
                A_after=accuracy_score(yt1,clf2.predict(Zt1)); A_T2=accuracy_score(yt2,clf2.predict(Zt2)); raw=A_before-A_after; norm=raw/A_before if A_before>0 else np.nan
                row={'dataset':dataset,'split':'fixed_10k_train_full_test','model':model_name,'seed':seed,'replay_frac':replay,'A_before':A_before,'A_after':A_after,'A_T2':A_T2,'avg_final':(A_after+A_T2)/2,'raw_forgetting':raw,'normalized_forgetting':norm}
                existing=pd.concat([existing,pd.DataFrame([row])],ignore_index=True); existing.to_csv(OUT,index=False); summarize(); print(row, flush=True)
summarize()
