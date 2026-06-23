from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from data_loading import load_dataset
results=Path(__file__).resolve().parent/'results'; figs=Path(__file__).resolve().parent/'figures'; results.mkdir(exist_ok=True); figs.mkdir(exist_ok=True)
rows=[]
fig,axs=plt.subplots(3,10,figsize=(10,3.3))
for r,dataset in enumerate(['digits','mnist','fashion_mnist']):
    Xtr,ytr,Xte,yte,shape=load_dataset(dataset)
    rows.append({'dataset':dataset,'train_size':len(Xtr),'test_size':len(Xte),'image_shape':str(shape),'train_min':float(Xtr.min()),'train_max':float(Xtr.max()),'test_min':float(Xte.min()),'test_max':float(Xte.max()),'train_class_counts':dict(zip(*np.unique(ytr,return_counts=True))),'test_class_counts':dict(zip(*np.unique(yte,return_counts=True)))})
    for c in range(10):
        idx=np.where(ytr==c)[0][0]
        axs[r,c].imshow(Xtr[idx].reshape(shape),cmap='gray'); axs[r,c].axis('off')
        if r==0: axs[r,c].set_title(str(c),fontsize=8)
    axs[r,0].set_ylabel(dataset,fontsize=8)
pd.DataFrame(rows).to_csv(results/'dataset_validation.csv',index=False)
fig.tight_layout(); fig.savefig(figs/'example_images_grid.png')
