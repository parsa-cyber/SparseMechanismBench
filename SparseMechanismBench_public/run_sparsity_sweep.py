import time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from data_loading import load_dataset
from models import LocalOjaFeatures

RESULTS=Path(__file__).resolve().parent/'results'; RESULTS.mkdir(exist_ok=True)
OUT=RESULTS/'sparsity_sweep_by_seed.csv'

def summarize():
    df=pd.read_csv(OUT)
    rows=[]
    for keys,g in df.groupby(['dataset','model','active_frac']):
        d,m,a=keys
        rows.append({'dataset':d,'model':m,'active_frac':a,'accuracy_mean':g.accuracy.mean(),'accuracy_std':g.accuracy.std(ddof=0),'sparsity_mean':g.activation_sparsity.mean(),'sparsity_std':g.activation_sparsity.std(ddof=0),'train_time_s_mean':g.train_time_s.mean(),'n':len(g)})
    pd.DataFrame(rows).to_csv(RESULTS/'sparsity_sweep_summary.csv',index=False)

existing=pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
rows=[]
for dataset in ['mnist','fashion_mnist']:
    Xtr,ytr,Xte,yte,shape=load_dataset(dataset)
    for active_frac in [0.05,0.10,0.20,0.40,0.60,1.0]:
        for seed in [0,1,2,3,4]:
            if not existing.empty and ((existing.dataset==dataset)&(existing.seed==seed)&(existing.active_frac==active_frac)).any():
                continue
            t0=time.time()
            af=None if active_frac>=1.0 else active_frac
            learner=LocalOjaFeatures(Xtr.shape[1],hidden_dim=128,lr=0.02,active_frac=af,homeostasis=False,target_rate=active_frac,seed=seed)
            learner.fit(Xtr,epochs=1,batch_size=2048,seed=seed)
            Ztr=learner.transform(Xtr); Zte=learner.transform(Xte)
            clf=SGDClassifier(loss='log_loss',max_iter=20,tol=1e-3,random_state=seed)
            clf.fit(Ztr,ytr)
            pred=clf.predict(Zte)
            acc=accuracy_score(yte,pred); sp=float(np.mean(Zte==0)); secs=time.time()-t0
            model='Oja' if active_frac>=1.0 else 'Sparse Oja'
            row={'dataset':dataset,'model':model,'active_frac':active_frac,'seed':seed,'accuracy':acc,'activation_sparsity':sp,'train_time_s':secs,'epochs':1,'hidden':128}
            rows.append(row)
            pd.DataFrame((existing.to_dict('records') if not existing.empty else [])+rows).to_csv(OUT,index=False)
            summarize()
            print(row, flush=True)
