from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import torch
from data_loading import load_dataset
from models import LocalOjaFeatures, LeNetStyleCNN, MLP
from train_classification import train_torch_model

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)

def subset(X,y,n,seed):
    rng=np.random.default_rng(seed); idx=[]; labels=np.unique(y); per=n//len(labels)
    for i,c in enumerate(labels):
        inds=np.where(y==c)[0]; idx.extend(rng.choice(inds, min(per,len(inds)), replace=False).tolist())
    idx=np.array(idx); rng.shuffle(idx); return X[idx], y[idx]

def train_source(src,Xtr,ytr,Xte,yte,shape,seed=0):
    if src=='raw_pixels': return Xtr,Xte
    if src in ['oja','sparse_oja','homeostatic_sparse_oja']:
        active=None if src=='oja' else 0.10; homeo=(src=='homeostatic_sparse_oja')
        l=LocalOjaFeatures(Xtr.shape[1],hidden_dim=128,lr=0.02,active_frac=active,homeostasis=homeo,target_rate=active or 0.10,theta_lr=0.01,seed=seed)
        l.fit(Xtr,epochs=5,batch_size=1024,seed=seed)
        return l.transform(Xtr), l.transform(Xte), l
    if src=='mlp_hidden':
        m=MLP(Xtr.shape[1],hidden_dim=128,dropout=0)
        acc,secs,params,sp,pred,curve,trained=train_torch_model(m,Xtr,ytr,Xte,yte,image_shape=None,epochs=5,lr=1e-3,batch_size=512,seed=seed)
        with torch.no_grad():
            return trained.encode(torch.tensor(Xtr,dtype=torch.float32)).numpy(), trained.encode(torch.tensor(Xte,dtype=torch.float32)).numpy(), trained
    if src=='standard_cnn_penultimate':
        m=LeNetStyleCNN(image_shape=shape)
        acc,secs,params,sp,pred,curve,trained=train_torch_model(m,Xtr,ytr,Xte,yte,image_shape=shape,epochs=3,lr=1e-3,batch_size=512,seed=seed)
        with torch.no_grad():
            Ztr=trained.encode(torch.tensor(Xtr.reshape(-1,1,*shape),dtype=torch.float32)).numpy()
            Zte=trained.encode(torch.tensor(Xte.reshape(-1,1,*shape),dtype=torch.float32)).numpy()
        return Ztr,Zte,trained

def plot_weight_grids(dataset='mnist', seed=0):
    Xtr,ytr,Xte,yte,shape=load_dataset(dataset)
    Xs,ys=subset(Xtr,ytr,2000,seed)
    fig,axes=plt.subplots(3,16,figsize=(16,3.5))
    titles=['Oja','Sparse Oja','Homeostatic Sparse Oja']
    for r,src in enumerate(['oja','sparse_oja','homeostatic_sparse_oja']):
        active=None if src=='oja' else 0.10; homeo=(src=='homeostatic_sparse_oja')
        l=LocalOjaFeatures(Xs.shape[1],hidden_dim=128,lr=0.02,active_frac=active,homeostasis=homeo,target_rate=active or 0.10,theta_lr=0.01,seed=seed)
        l.fit(Xs,epochs=5,batch_size=1024,seed=seed)
        W=l.W[:16].reshape((16,)+shape)
        for i in range(16):
            ax=axes[r,i]; ax.imshow(W[i],cmap='gray'); ax.axis('off')
        axes[r,0].set_ylabel(titles[r],fontsize=9)
    fig.suptitle(f'Learned local features ({dataset}, seed {seed})')
    fig.tight_layout(); fig.savefig(FIG/f'learned_features_{dataset}.png',dpi=200,bbox_inches='tight'); plt.close(fig)

def plot_confusions(dataset='mnist', seed=0):
    Xtr,ytr,Xte,yte,shape=load_dataset(dataset)
    Xs,ys=subset(Xtr,ytr,2000,seed); Xt,yt=subset(Xte,yte,1000,1000+seed)
    sources=['raw_pixels','oja','sparse_oja','homeostatic_sparse_oja','mlp_hidden','standard_cnn_penultimate']
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for ax,src in zip(axes.ravel(),sources):
        res=train_source(src,Xs,ys,Xt,yt,shape,seed)
        Ztr,Zte=res[0],res[1]
        clf=RidgeClassifier(alpha=1.0).fit(Ztr,ys)
        pred=clf.predict(Zte)
        cm=confusion_matrix(yt,pred,labels=np.arange(10),normalize='true')
        im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=1)
        ax.set_title(src)
        ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.tick_params(labelsize=6)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.75,label='Normalized count')
    fig.suptitle(f'Confusion matrices using ridge readout ({dataset}, seed {seed})')
    fig.savefig(FIG/f'confusion_matrices_{dataset}.png',dpi=200,bbox_inches='tight'); plt.close(fig)

def plot_histograms(dataset='mnist', seed=0):
    Xtr,ytr,Xte,yte,shape=load_dataset(dataset)
    Xs,ys=subset(Xtr,ytr,2000,seed); Xt,yt=subset(Xte,yte,1000,1000+seed)
    sources=['oja','sparse_oja','homeostatic_sparse_oja','mlp_hidden','standard_cnn_penultimate']
    fig,axes=plt.subplots(1,len(sources),figsize=(15,3))
    for ax,src in zip(axes,sources):
        res=train_source(src,Xs,ys,Xt,yt,shape,seed)
        Zte=res[1]
        vals=Zte.ravel()
        ax.hist(vals, bins=40, log=True)
        ax.set_title(src); ax.set_xlabel('activation'); ax.set_ylabel('count (log)')
    fig.suptitle(f'Activation histograms ({dataset}, seed {seed})')
    fig.tight_layout(); fig.savefig(FIG/f'activation_histograms_{dataset}.png',dpi=200,bbox_inches='tight'); plt.close(fig)

if __name__=='__main__':
    for ds in ['mnist','fashion_mnist']:
        plot_weight_grids(ds,0)
        plot_confusions(ds,0)
        plot_histograms(ds,0)
