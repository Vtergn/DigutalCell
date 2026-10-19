"""Pre-UDE gate: learn a constrained output discrepancy between two SAC forms.

This is intentionally an output residual benchmark, not a full UDE. The prior
trace is supplied and the network predicts the saturating-SAC minus prior
correction. Long-hold protocol and test cells are sealed.
"""
from pathlib import Path
import json, numpy as np, h5py, torch
from torch import nn
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'human_bidirectional_output'; OUT=ROOT/'model_form_residual_decoder'; OUT.mkdir(exist_ok=True)
torch.manual_seed(31); np.random.seed(31)
with h5py.File(DATA/'dynamic_discrepancy_development.mat') as f:
    allw=f['waveforms'][:].astype('float32'); lam=f['lambda_tracks'][:].astype('float32'); grid=f['grid'][:].ravel().astype('float32')
# forms: prior=0, saturating=1; MATLAB dimensions cell, form, protocol, channel, time
valid=np.isfinite(allw[:,0]).all(axis=(1,2,3)) & np.isfinite(allw[:,1]).all(axis=(1,2,3)); allw=allw[valid]
step=10; prior=allw[:,0,:,:,::step]; truth=allw[:,1,:,:,::step]; lam=lam[:,::step]; t=grid[::step]/1000; ncell,nprot,nchan,nt=prior.shape
train=np.arange(int(.75*ncell)); test=np.arange(int(.75*ncell),ncell); hold=3; seen=[0,1,2,4,5]
mu=prior[train].reshape(-1,3).mean(0); sd=prior[train].reshape(-1,3).std(0)+1e-6
def norm(a): return (a-mu[None,None,:,None])/sd[None,None,:,None]
pn=norm(prior); tn=norm(truth)
X=[]; Y=[]
for c in train:
    for p in seen:
        for k in range(nt):
            X.append([t[k],lam[p,k],p/(nprot-1),*pn[c,p,:,k],*pn[c,0,:,k]])
            Y.append(tn[c,p,:,k]-pn[c,p,:,k])
X=torch.tensor(np.asarray(X),dtype=torch.float32); Y=torch.tensor(np.asarray(Y),dtype=torch.float32)
model=nn.Sequential(nn.Linear(9,96),nn.Tanh(),nn.Linear(96,96),nn.Tanh(),nn.Linear(96,3)); opt=torch.optim.Adam(model.parameters(),lr=2e-3)
for ep in range(600):
    pred=model(X); loss=((pred-Y)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
def predict(c,p):
    xx=np.asarray([[t[k],lam[p,k],p/(nprot-1),*pn[c,p,:,k],*pn[c,0,:,k]] for k in range(nt)],dtype='float32')
    with torch.no_grad(): d=model(torch.tensor(xx)).numpy()
    corrected=prior[c,p]+(d*sd[None,:]).T
    return corrected
pred=np.asarray([predict(c,hold) for c in test]); true=truth[test,hold]; base=prior[test,hold]
err_base=np.mean(np.abs(base-true),axis=(0,2)); err_corr=np.mean(np.abs(pred-true),axis=(0,2)); metrics={'train_cells':train.tolist(),'test_cells':test.tolist(),'seen_protocols':seen,'held_protocol':hold,'baseline_prior_MAE':[float(x) for x in err_base],'residual_corrected_MAE':[float(x) for x in err_corr],'baseline_overall_MAE':float(np.mean(np.abs(base-true))),'residual_overall_MAE':float(np.mean(np.abs(pred-true))),'note':'output residual pre-UDE gate; no ODE flux was changed'}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); torch.save(model.state_dict(),OUT/'residual_decoder.pt'); np.savez_compressed(OUT/'predictions.npz',prior=base,corrected=pred,true=true,time=t)
fig,axs=plt.subplots(3,1,figsize=(9,7),sharex=True,constrained_layout=True); labels=['V (mV)','Ca (µM)','T (kPa)']
for q,ax in enumerate(axs):
    ax.plot(t*1000,true[:,q].mean(0),color='#2166ac',lw=2,label='saturating truth')
    ax.plot(t*1000,base[:,q].mean(0),color='#777777',lw=1.6,label='prior')
    ax.plot(t*1000,pred[:,q].mean(0),color='#d6604d',ls='--',lw=2,label='residual corrected')
    ax.set_ylabel(labels[q]); ax.grid(alpha=.2)
axs[-1].set_xlabel('Time (ms)'); axs[0].set_title('Pre-UDE gate: held-out long-hold feedback-form shift',fontweight='bold'); axs[0].legend(frameon=False)
fig.savefig(OUT/'heldout_residual_comparison.png',dpi=220,bbox_inches='tight'); fig.savefig(OUT/'heldout_residual_comparison.pdf',bbox_inches='tight'); plt.close(fig)
print(json.dumps(metrics,indent=2))
