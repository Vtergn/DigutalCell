"""Blind protocol-family test for the conditioned neural virtual cell."""
from pathlib import Path
import json, numpy as np, h5py, torch
from torch import nn
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'human_bidirectional_output'; OUT=ROOT/'virtual_cell_unseen_protocol'; OUT.mkdir(exist_ok=True)
torch.manual_seed(23); np.random.seed(23)
with h5py.File(DATA/'dynamic_discrepancy_development.mat') as f:
    wave=f['waveforms'][:,1,:,:,:].astype('float32'); lam=f['lambda_tracks'][:].astype('float32'); grid=f['grid'][:].ravel().astype('float32')
valid=np.isfinite(wave).all(axis=(1,2,3)); wave=wave[valid]; ncell,nprot,_,_=wave.shape
step=10; y=wave[:,:,:,::step]; lam=lam[:,::step]; t=grid[::step]/1000; nt=len(t)
train_cells=np.arange(int(.75*ncell)); test_cells=np.arange(int(.75*ncell),ncell); held_protocol=3; seen_protocols=[0,1,2,4,5]
mu=y[train_cells].reshape(-1,3).mean(0); sd=y[train_cells].reshape(-1,3).std(0)+1e-6; yn=(y-mu[None,None,:,None])/sd[None,None,:,None]
features=[]; targets=[]
for c in train_cells:
    for p in seen_protocols:
        for k in range(nt):
            features.append([t[k],lam[p,k],*yn[c,0,:,k],p/(nprot-1)]); targets.append(yn[c,p,:,k]-yn[c,0,:,k])
X=torch.tensor(np.asarray(features),dtype=torch.float32); Y=torch.tensor(np.asarray(targets),dtype=torch.float32)
model=nn.Sequential(nn.Linear(6,96),nn.Tanh(),nn.Linear(96,96),nn.Tanh(),nn.Linear(96,3)); opt=torch.optim.Adam(model.parameters(),lr=2e-3)
for ep in range(500):
    p=model(X); loss=((p-Y)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
def predict(c,p):
    xx=np.asarray([[t[k],lam[p,k],*yn[c,0,:,k],p/(nprot-1)] for k in range(nt)],dtype='float32')
    with torch.no_grad(): d=model(torch.tensor(xx)).numpy()
    return y[c,0]+(d*sd[None,:]).T
pred=np.asarray([predict(c,held_protocol) for c in test_cells]); truth=y[test_cells,held_protocol]
err=np.mean(np.abs(pred-truth),axis=(0,2)); metrics={'held_out_protocol':held_protocol,'seen_protocols':seen_protocols,'train_cells':train_cells.tolist(),'test_cells':test_cells.tolist(),'V_MAE_mV':float(err[0]),'Ca_MAE_uM':float(err[1]),'T_MAE_kPa':float(err[2]),'overall_MAE_mixed_units':float(np.mean(np.abs(pred-truth))),'note':'protocol-family blind test; no long-hold outcomes used during training'}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); np.savez_compressed(OUT/'predictions.npz',pred=pred,truth=truth,time=t,lambda_tracks=lam[held_protocol]); torch.save(model.state_dict(),OUT/'decoder.pt'); print(json.dumps(metrics,indent=2))
fig,axs=plt.subplots(3,1,figsize=(9,7),sharex=True,constrained_layout=True)
labels=['V (mV)','Ca (µM)','T (kPa)']; colors=['#2166ac','#d6604d']
for q,ax in enumerate(axs):
    ax.plot(t*1000,truth[:,q].mean(0),color=colors[0],lw=2,label='native truth')
    ax.plot(t*1000,pred[:,q].mean(0),color=colors[1],lw=2,ls='--',label='neural blind prediction')
    ax.fill_between(t*1000,np.quantile(truth[:,q],.1,axis=0),np.quantile(truth[:,q],.9,axis=0),color=colors[0],alpha=.12)
    ax.set_ylabel(labels[q]); ax.grid(alpha=.2)
axs[-1].set_xlabel('Time (ms)'); axs[0].set_title('Held-out long-hold protocol: cell-held-out neural prediction',fontweight='bold'); axs[0].legend(frameon=False)
fig.savefig(OUT/'blind_protocol_prediction.png',dpi=220,bbox_inches='tight'); fig.savefig(OUT/'blind_protocol_prediction.pdf',bbox_inches='tight'); plt.close(fig)
