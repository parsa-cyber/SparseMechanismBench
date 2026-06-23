from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent; results=root/'results'; figs=root/'figures'; figs.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':160, 'font.size':9})
name_map={'logistic_regression':'LogReg','random_features_linear':'Rand+Lin','pca_features_linear':'PCA+Lin','mlp':'MLP','mlp_dropout':'MLP+DO','compact_cnn':'cCNN','standard_cnn':'Std CNN','hebbian_oja':'Oja','sparse_hebbian_oja':'Sparse Oja','homeostatic_sparse_oja':'Homeo Sparse Oja'}
cls=pd.read_csv(results/'updated_classification_summary.csv')
# classification by dataset
for dataset in ['digits','mnist','fashion_mnist']:
    df=cls[cls.dataset==dataset].copy(); df['label']=df.model.map(name_map).fillna(df.model)
    df=df.sort_values('accuracy_mean',ascending=False)
    fig,ax=plt.subplots(figsize=(9,4.8)); ax.bar(df.label, df.accuracy_mean, yerr=df.accuracy_std, capsize=3)
    ax.set_ylim(0,1.05); ax.set_ylabel('Test accuracy'); ax.set_title(f'Classification accuracy ({dataset.replace("_","-")})'); ax.tick_params(axis='x',rotation=25); ax.grid(axis='y',alpha=.25)
    fig.tight_layout(); fig.savefig(figs/f'classification_{dataset}.png'); plt.close(fig)
# standard vs compact cnn
cnn=cls[cls.model.isin(['compact_cnn','standard_cnn'])].copy(); cnn['label']=cnn.model.map(name_map)
fig,ax=plt.subplots(figsize=(7,4));
for label,g in cnn.groupby('label'):
    g=g.set_index('dataset').loc[['digits','mnist','fashion_mnist']]
    ax.plot(['digits','MNIST','Fashion'], g.accuracy_mean, marker='o', label=label)
ax.set_ylim(0,1.05); ax.set_ylabel('Accuracy'); ax.set_title('Compact CNN vs LeNet-style Standard CNN'); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figs/'standard_vs_compact_cnn.png'); plt.close(fig)
# mechanism ladder
mech=cls[cls.model.isin(['hebbian_oja','sparse_hebbian_oja','homeostatic_sparse_oja','mlp','standard_cnn'])].copy(); mech['label']=mech.model.map(name_map)
fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,d in zip(axs,['mnist','fashion_mnist']):
    df=mech[mech.dataset==d].copy(); order=['Oja','Sparse Oja','Homeo Sparse Oja','MLP','Std CNN']; df['label']=pd.Categorical(df.label, order); df=df.sort_values('label')
    ax.bar(df.label.astype(str),df.accuracy_mean,yerr=df.accuracy_std,capsize=3); ax.set_title(d.replace('_','-')); ax.tick_params(axis='x',rotation=25); ax.set_ylim(0,1.05); ax.grid(axis='y',alpha=.25)
axs[0].set_ylabel('Accuracy'); fig.suptitle('Mechanism ladder: local plasticity, sparsity, homeostasis, and task-directed baselines'); fig.tight_layout(); fig.savefig(figs/'mechanism_ladder_accuracy.png'); plt.close(fig)
# sparsity sweep subset
sw=pd.read_csv(results/'sparsity_sweep_subset_summary.csv')
fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,d in zip(axs,['mnist','fashion_mnist']):
    df=sw[sw.dataset==d].sort_values('active_frac')
    # x = sparsity; y=accuracy
    ax.errorbar(df.sparsity_mean,df.accuracy_mean,yerr=df.accuracy_std,fmt='o-',capsize=3)
    for _,r in df.iterrows(): ax.annotate(f"{int(r.active_frac*100)}% active",(r.sparsity_mean,r.accuracy_mean),fontsize=7,xytext=(4,4),textcoords='offset points')
    ax.set_xlabel('Activation sparsity'); ax.set_title(d.replace('_','-')); ax.grid(alpha=.25); ax.set_xlim(0,1)
axs[0].set_ylabel('Accuracy'); fig.suptitle('Sparse Oja sweep on fixed 10k/2k subset'); fig.tight_layout(); fig.savefig(figs/'ablation_accuracy_vs_sparsity.png'); plt.close(fig)
# replay sweep partial (if exists)
try:
    rp=pd.read_csv(results/'replay_sweep_subset_summary.csv')
    fig,ax=plt.subplots(figsize=(7,4));
    for m,g in rp[rp.dataset=='mnist'].groupby('model'):
        ax.errorbar(g.replay_frac*100,g.normalized_forgetting_mean,yerr=g.normalized_forgetting_std,marker='o',capsize=3,label=m)
    ax.set_xlabel('Replay from Task 1 during Task 2 (%)'); ax.set_ylabel('Normalized forgetting'); ax.set_title('Replay sweep (partial MNIST subset)'); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figs/'replay_sweep_partial_mnist.png'); plt.close(fig)
except Exception as e: pass
# mechanism map
fig,ax=plt.subplots(figsize=(10,5)); ax.axis('off')
boxes=[('Local plasticity\n(Oja)',.08,.65),('Sparse coding\n(k-WTA)',.28,.65),('Homeostasis\n(target firing)',.48,.65),('Replay\n(memory stabilization)',.68,.65),('Task-directed\nbackprop/feedback',.48,.25)]
for text,x,y in boxes:
    ax.text(x,y,text,ha='center',va='center',bbox=dict(boxstyle='round,pad=.5',fc='#e8f2ff',ec='#336699'))
for x1,y1,x2,y2 in [(.15,.65,.21,.65),(.35,.65,.41,.65),(.55,.65,.61,.65),(.48,.58,.48,.34)]:
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.5))
ax.text(.48,.06,'Core interpretation: sparsity alone is not enough; biological efficiency likely requires interacting mechanisms.',ha='center',fontsize=11)
fig.tight_layout(); fig.savefig(figs/'mechanism_map.png'); plt.close(fig)
print('plots written to',figs)
