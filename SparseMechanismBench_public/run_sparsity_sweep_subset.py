import time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from data_loading import load_dataset
from models import LocalOjaFeatures
RESULTS=Path(__file__).resolve().parent/'results'; OUT=RESULTS/'sparsity_sweep_subset_by_seed.csv'; SUM=RESULTS/'sparsity_sweep_subset_summary.csv'

def write_summary():
    if not OUT.exists(): return
    df=pd.read_csv(OUT)
    summary=[]
    for keys,g in df.groupby(['dataset','split','model','active_frac']):
        d,sp,m,a=keys
        summary.append({'dataset':d,'split':sp,'model':m,'active_frac':a,'accuracy_mean':g.accuracy.mean(),'accuracy_std':g.accuracy.std(ddof=0),'sparsity_mean':g.activation_sparsity.mean(),'sparsity_std':g.activation_sparsity.std(ddof=0),'n':len(g)})
    pd.DataFrame(summary).to_csv(SUM,index=False)
existing=pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
for dataset in ['mnist','fashion_mnist']:
    Xtr,ytr,Xte,yte,shape=load_dataset(dataset)
    rng0=np.random.default_rng(123)
    train_idx=rng0.choice(len(Xtr), size=min(10000,len(Xtr)), replace=False)
    test_idx=rng0.choice(len(Xte), size=min(2000,len(Xte)), replace=False)
    Xtr_s,ytr_s=Xtr[train_idx],ytr[train_idx]; Xte_s,yte_s=Xte[test_idx],yte[test_idx]
    for active_frac in [0.05,0.10,0.20,0.40,0.60,1.0]:
        for seed in [0,1,2,3,4]:
            if not existing.empty and ((existing.dataset==dataset)&(existing.seed==seed)&(existing.active_frac==active_frac)).any():
                continue
            t0=time.time(); af=None if active_frac>=1.0 else active_frac
            learner=LocalOjaFeatures(Xtr_s.shape[1],hidden_dim=128,lr=0.02,active_frac=af,homeostasis=False,target_rate=active_frac,seed=seed)
            learner.fit(Xtr_s,epochs=1,batch_size=2048,seed=seed)
            Ztr=learner.transform(Xtr_s); Zte=learner.transform(Xte_s)
            clf=SGDClassifier(loss='log_loss',max_iter=20,tol=1e-3,random_state=seed)
            clf.fit(Ztr,ytr_s); pred=clf.predict(Zte)
            row={'dataset':dataset,'split':'fixed_10k_train_2k_test','model':'Oja' if active_frac>=1.0 else 'Sparse Oja','active_frac':active_frac,'seed':seed,'accuracy':accuracy_score(yte_s,pred),'activation_sparsity':float(np.mean(Zte==0)),'train_time_s':time.time()-t0,'epochs':1,'hidden':128}
            existing=pd.concat([existing,pd.DataFrame([row])],ignore_index=True)
            existing.to_csv(OUT,index=False); write_summary(); print(row, flush=True)
write_summary()
