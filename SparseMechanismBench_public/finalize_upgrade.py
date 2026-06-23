import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
root=Path(__file__).resolve().parent; results=root/'results'; figs=root/'figures'; figs.mkdir(exist_ok=True)
old=Path('/mnt/data/final_polished_results')
base=pd.read_csv(old/'classification_by_seed.csv')
std=pd.read_csv(results/'standard_cnn_by_seed.csv')[['dataset','model','seed','accuracy']]
hso=pd.read_csv(results/'homeostatic_sparse_oja_by_seed.csv')[['dataset','model','seed','accuracy']]
# replace old compact cnn remains as 'compact_cnn'; new as standard_cnn
base=base.rename(columns={'model':'model'})
base['model']=base['model'].replace({'cnn':'compact_cnn'})
all_cls=pd.concat([base,std,hso],ignore_index=True)
all_cls.to_csv(results/'updated_classification_by_seed.csv',index=False)
summary=all_cls.groupby(['dataset','model']).accuracy.agg(['mean','std','count']).reset_index().rename(columns={'mean':'accuracy_mean','std':'accuracy_std','count':'n'})
summary['accuracy_std']=summary['accuracy_std'].fillna(0)
summary.to_csv(results/'updated_classification_summary.csv',index=False)
# confidence intervals
ci=[]
for (d,m),g in all_cls.groupby(['dataset','model']):
    vals=g.accuracy.values; n=len(vals); mean=vals.mean(); sd=vals.std(ddof=1) if n>1 else 0
    tcrit=stats.t.ppf(0.975,n-1) if n>1 else np.nan
    half=tcrit*sd/np.sqrt(n) if n>1 else np.nan
    ci.append({'dataset':d,'model':m,'mean':mean,'sd':sd,'n':n,'ci95_low':mean-half if n>1 else np.nan,'ci95_high':mean+half if n>1 else np.nan})
pd.DataFrame(ci).to_csv(results/'confidence_interval_table.csv',index=False)
# statistical tests for core comparisons
comparisons=[]
def test_pair(dataset, model_a, model_b):
    a=all_cls[(all_cls.dataset==dataset)&(all_cls.model==model_a)].sort_values('seed')
    b=all_cls[(all_cls.dataset==dataset)&(all_cls.model==model_b)].sort_values('seed')
    seeds=sorted(set(a.seed)&set(b.seed))
    av=np.array([a[a.seed==s].accuracy.iloc[0] for s in seeds])
    bv=np.array([b[b.seed==s].accuracy.iloc[0] for s in seeds])
    if len(seeds)<2: return None
    diff=av-bv
    try: t,p=stats.ttest_rel(av,bv)
    except Exception: t,p=np.nan,np.nan
    try: w,pw=stats.wilcoxon(av,bv,zero_method='wilcox',correction=False)
    except Exception: w,pw=np.nan,np.nan
    cohen=diff.mean()/(diff.std(ddof=1)+1e-12)
    return {'dataset':dataset,'model_a':model_a,'model_b':model_b,'n':len(seeds),'mean_a':av.mean(),'mean_b':bv.mean(),'mean_diff_a_minus_b':diff.mean(),'paired_t_p':p,'wilcoxon_p':pw,'cohens_dz':cohen}
for d in ['mnist','fashion_mnist']:
    for ma,mb in [('mlp','hebbian_oja'),('mlp','sparse_hebbian_oja'),('mlp','homeostatic_sparse_oja'),('standard_cnn','compact_cnn'),('standard_cnn','mlp'),('sparse_hebbian_oja','homeostatic_sparse_oja')]:
        r=test_pair(d,ma,mb)
        if r: comparisons.append(r)
pd.DataFrame(comparisons).to_csv(results/'statistical_tests.csv',index=False)
# use same as summary for now
pd.DataFrame(comparisons).to_csv(results/'statistical_tests_summary.csv',index=False)
print(summary)
print(pd.DataFrame(comparisons))
