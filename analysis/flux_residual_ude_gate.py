"""Flux-level residual diagnostic before a full ODE-UDE implementation.

The network predicts d(truth-prior)/dt from the prior state and mechanical
input. It is then integrated from the prior initial state. This is a narrow
diagnostic: the prior trajectory is used to form the known vector field, so it
is not yet a native ToR-ORd flux replacement.
"""
from pathlib import Path
import json, numpy as np, h5py, torch
from torch import nn
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'human_bidirectional_output'; OUT=ROOT/'flux_residual_ude_gate'; OUT.mkdir(exist_ok=True)
torch.manual_seed(41); np.random.seed(41)
with h5py.File(DATA/'dynamic_discrepancy_development.mat') as f:
    allw=f['waveforms'][:].astype('float32'); lam=f['lambda_tracks'][:].astype('float32'); grid=f['grid'][:].ravel().astype('float32')
valid=np.isfinite(allw[:,0]).all(axis=(1,2,3)) & np.isfinite(allw[:,1]).all(axis=(1,2,3)); allw=allw[valid]
step=10; prior=allw[:,0,:,:,::step]; truth=allw[:,1,:,:,::step]; lam=lam[:,::step]; t=grid[::step]; dt=float(t[1]-t[0]); ncell,nprot,nchan,nt=prior.shape
train=np.arange(int(.75*ncell)); test=np.arange(int(.75*ncell),ncell); hold=3; seen=[0,1,2,4,5]
scale=np.array([20.,.1,20.],dtype='float32')
pn=prior/scale[None,None,:,None]; tn=truth/scale[None,None,:,None]
dp=np.gradient(pn,dt,axis=-1); dtarg=np.gradient(tn,dt,axis=-1)-dp
X=[]; Y=[]
for c in train:
    for p in seen:
        for k in range(nt):
            # λ=1 gates the correction to zero; include a causal one-step state.
            gate=max(float(lam[p,k]-1),0.)
            X.append([pn[c,p,0,k],pn[c,p,1,k],pn[c,p,2,k],dp[c,p,0,k],dp[c,p,1,k],dp[c,p,2,k],lam[p,k],gate,p/(nprot-1)])
            Y.append(dtarg[c,p,:,k]*gate)
X=torch.tensor(np.asarray(X),dtype=torch.float32); Y=torch.tensor(np.asarray(Y),dtype=torch.float32)
model=nn.Sequential(nn.Linear(9,96),nn.Tanh(),nn.Linear(96,96),nn.Tanh(),nn.Linear(96,3)); opt=torch.optim.Adam(model.parameters(),lr=2e-3)
for ep in range(500):
    q=model(X); loss=((q-Y)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()

def corrected(c,p):
    # Integrate prior derivative plus predicted discrepancy along the sealed prior trajectory.
    out=np.zeros((3,nt),dtype='float32'); out[:,0]=prior[c,p,:,0]
    for k in range(nt-1):
        gate=max(float(lam[p,k]-1),0.)
        xx=np.array([*pn[c,p,:,k],*dp[c,p,:,k],lam[p,k],gate,p/(nprot-1)],dtype='float32')
        with torch.no_grad(): r=model(torch.tensor(xx[None,:])).numpy()[0]
        out[:,k+1]=out[:,k]+scale*(dp[c,p,:,k]+r)*dt
    return out
pred=np.asarray([corrected(c,hold) for c in test]); base=prior[test,hold]; true=truth[test,hold]
errb=np.mean(np.abs(base-true),axis=(0,2)); errr=np.mean(np.abs(pred-true),axis=(0,2))
metrics={'train_cells':train.tolist(),'test_cells':test.tolist(),'seen_protocols':seen,'held_protocol':hold,'prior_MAE':[float(x) for x in errb],'flux_residual_MAE':[float(x) for x in errr],'prior_overall_MAE':float(np.mean(np.abs(base-true))),'flux_residual_overall_MAE':float(np.mean(np.abs(pred-true))),'note':'diagnostic flux residual; prior trajectory supplies the known derivative and this is not a native ODE-UDE'}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); torch.save(model.state_dict(),OUT/'flux_residual.pt'); np.savez_compressed(OUT/'predictions.npz',prior=base,pred=pred,true=true,time=t)
fig,axs=plt.subplots(3,1,figsize=(9,7),sharex=True,constrained_layout=True); labels=['V (mV)','Ca (µM)','T (kPa)']
for q,ax in enumerate(axs):
    ax.plot(t,true[:,q].mean(0),color='#2166ac',lw=2,label='truth')
    ax.plot(t,base[:,q].mean(0),color='#777777',lw=1.5,label='prior')
    ax.plot(t,pred[:,q].mean(0),color='#d6604d',ls='--',lw=2,label='flux residual')
    ax.set_ylabel(labels[q]); ax.grid(alpha=.2)
axs[-1].set_xlabel('Time (ms)'); axs[0].set_title('Flux residual UDE diagnostic: held-out long-hold',fontweight='bold'); axs[0].legend(frameon=False)
fig.savefig(OUT/'comparison.png',dpi=220,bbox_inches='tight'); fig.savefig(OUT/'comparison.pdf',bbox_inches='tight'); plt.close(fig)
print(json.dumps(metrics,indent=2))
