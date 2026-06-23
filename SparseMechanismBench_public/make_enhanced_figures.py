from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'
FIGURES = ROOT/'figures'
FIGURES.mkdir(exist_ok=True)

def save_frozen_readout():
    p=RESULTS/'frozen_readout_summary.csv'
    if not p.exists(): return
    df=pd.read_csv(p)
    ok=df[df['dataset']=='digits'].copy()
    if ok.empty: return
    # local and raw feature sources only for clean plot
    order=['raw_pixels','oja','sparse_oja','homeostatic_sparse_oja','mlp_hidden','standard_cnn_penultimate']
    ok['feature_source']=pd.Categorical(ok['feature_source'], categories=order, ordered=True)
    pivot=ok.pivot_table(index='feature_source', columns='readout', values='accuracy_mean', observed=False)
    fig, ax = plt.subplots(figsize=(9,4.8))
    im=ax.imshow(pivot.values, vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=35, ha='right')
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([str(x).replace('_',' ') for x in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v=pivot.values[i,j]
            if np.isfinite(v): ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im, ax=ax, label='Test accuracy')
    ax.set_title('Frozen readout accuracy on digits')
    fig.tight_layout(); fig.savefig(FIGURES/'frozen_readout_digits_heatmap.png', dpi=220); plt.close(fig)


def save_geometry():
    p=RESULTS/'representation_geometry_summary.csv'
    if not p.exists(): return
    df=pd.read_csv(p)
    df=df[df['dataset']=='digits'].copy()
    if df.empty: return
    df['label']=df['source'].str.replace('_',' ', regex=False)
    metrics=[('linear_probe_accuracy_mean','Linear probe accuracy'),('silhouette_mean','Silhouette score'),('fisher_ratio_mean','Fisher ratio'),('activation_sparsity_mean','Activation sparsity')]
    fig, axes = plt.subplots(2,2,figsize=(10,7))
    for ax,(col,title) in zip(axes.ravel(),metrics):
        ax.bar(df['label'], df[col], yerr=df[col.replace('_mean','_std')] if col.replace('_mean','_std') in df else None)
        ax.set_title(title); ax.tick_params(axis='x', rotation=30); ax.grid(axis='y', alpha=.25)
    fig.suptitle('Representation geometry and feature health (digits)')
    fig.tight_layout(); fig.savefig(FIGURES/'representation_geometry_digits_summary.png', dpi=220); plt.close(fig)


def save_status():
    # Simple visual mechanism map, improved from earlier.
    fig, ax=plt.subplots(figsize=(10,4))
    ax.axis('off')
    boxes=[('Local plasticity\n(Oja)',0.08,0.65),('Sparse coding\n(k-WTA)',0.28,0.65),('Homeostasis\n(target firing)',0.48,0.65),('Replay\n(memory stabilization)',0.68,0.65),('Task signal\n(backprop/feedback/reward)',0.48,0.25)]
    for text,x,y in boxes:
        ax.text(x,y,text,ha='center',va='center',bbox=dict(boxstyle='round,pad=.45',fc='#eaf3ff',ec='#336699'),fontsize=10)
    arrows=[((0.16,0.65),(0.22,0.65)),((0.36,0.65),(0.42,0.65)),((0.56,0.65),(0.62,0.65)),((0.48,0.54),(0.48,0.36))]
    for (x1,y1),(x2,y2) in arrows:
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.5))
    ax.text(0.5,0.05,'Mechanism-isolation claim: sparsity alone is insufficient; useful learning likely requires interacting mechanisms.',ha='center',fontsize=11)
    fig.tight_layout(); fig.savefig(FIGURES/'mechanism_map_v2.png',dpi=220); plt.close(fig)

if __name__=='__main__':
    save_frozen_readout(); save_geometry(); save_status()
    print('enhanced figures saved')
