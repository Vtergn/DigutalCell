"""Conditioned neural behavior decoder for the virtual-cell renderer.

Unlike the recursive state model, this first practical decoder conditions each
time point on the same cell's baseline trajectory. It is a transparent neural
surrogate for reconstruction and visualization; it is not a replacement for
the native mechanistic simulator.
"""
from pathlib import Path
import json
import numpy as np
import h5py
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'human_bidirectional_output'; OUT=ROOT/'virtual_cell_decoder'; OUT.mkdir(exist_ok=True)
torch.manual_seed(17); np.random.seed(17)
with h5py.File(DATA/'dynamic_discrepancy_development.mat') as f:
    wave=f['waveforms'][:,1,:,:,:].astype('float32'); lam=f['lambda_tracks'][:].astype('float32'); grid=f['grid'][:].ravel().astype('float32')
valid=np.isfinite(wave).all(axis=(1,2,3)); wave=wave[valid]; ncell,nprot,nchan,ntime=wave.shape
step=10; y=wave[:,:,:,::step]; t=grid[::step]/1000; lam=lam[:,::step]; ntime=len(t)
train=np.arange(int(.75*ncell)); test=np.arange(int(.75*ncell),ncell)
mu=y[train].reshape(-1,3).mean(0); sd=y[train].reshape(-1,3).std(0)+1e-6
yn=(y-mu[None,None,:,None])/sd[None,None,:,None]

# Inputs: time, imposed length, baseline V/Ca/T at the same time, protocol one-hot.
features=[]; targets=[]
for cell in train:
    for p in range(nprot):
        for k in range(ntime):
            features.append([t[k],lam[p,k],*yn[cell,0,:,k],p/(nprot-1)])
            targets.append(yn[cell,p,:,k]-yn[cell,0,:,k])
X=torch.tensor(np.asarray(features),dtype=torch.float32); Y=torch.tensor(np.asarray(targets),dtype=torch.float32)
class Decoder(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(6,96),nn.Tanh(),nn.Linear(96,96),nn.Tanh(),nn.Linear(96,3))
    def forward(self,x): return self.net(x)
model=Decoder(); opt=torch.optim.Adam(model.parameters(),lr=2e-3)
for ep in range(500):
    pred=model(X); loss=((pred-Y)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    if ep in (0,99,299,499): print('epoch',ep+1,'loss',float(loss.detach()))

def predict(cell,p):
    xx=np.asarray([[t[k],lam[p,k],*yn[cell,0,:,k],p/(nprot-1)] for k in range(ntime)],dtype='float32')
    with torch.no_grad(): z=model(torch.tensor(xx)).numpy()
    delta=(z*sd[None,:]).T
    return y[cell,0]+delta
pred=np.zeros_like(y[test])
for a,c in enumerate(test):
    for p in range(nprot): pred[a,p]=predict(c,p)
err=np.mean(np.abs(pred-y[test]),axis=(0,1,3)); metrics={'valid_cells':int(ncell),'train_cells':train.tolist(),'test_cells':test.tolist(),'time_step_ms':10,'V_MAE_mV':float(err[0]),'Ca_MAE_uM':float(err[1]),'T_MAE_kPa':float(err[2]),'note':'cell-held-out conditioned decoder; native baseline trajectory is provided as an input'}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); torch.save(model.state_dict(),OUT/'decoder.pt')
np.savez_compressed(OUT/'predictions.npz',pred=pred,true=y[test],time=t,lambda_tracks=lam)

# Render first held-out cell under the long-hold protocol (p=3).
cell=int(test[0]); p=3; j=0; truth=y[cell,p]; forecast=pred[j,p]
fig=plt.figure(figsize=(11,6)); gs=fig.add_gridspec(2,2,width_ratios=[1,1.7]); axc=fig.add_subplot(gs[:,0]); axv=fig.add_subplot(gs[0,1]); axt=fig.add_subplot(gs[1,1]); axc.set_aspect('equal'); axc.set_xlim(-1.5,1.5); axc.set_ylim(-1.25,1.25); axc.axis('off'); axc.set_title(f'Neural virtual cell (held-out cell {cell})')
cellpatch=Ellipse((0,0),1.4,1,facecolor='#ed8da9',edgecolor='#333',lw=2); mem=Ellipse((0,0),1.5,1.08,fill=False,edgecolor='#2166ac',lw=8); axc.add_patch(cellpatch); axc.add_patch(mem); txt=axc.text(0,0,'',ha='center',va='center',color='white',fontsize=18,fontweight='bold')
axv.set_title('Voltage: native truth vs neural reconstruction'); axv.set_ylabel('mV'); axv.set_xlim(0,1000); axv.set_ylim(-100,50); axv.grid(alpha=.2); lv,=axv.plot([],[],color='#2166ac',label='truth'); lp,=axv.plot([],[],color='#d6604d',ls='--',label='neural'); axv.legend(frameon=False)
axt.set_title('Tension: native truth vs neural reconstruction'); axt.set_ylabel('kPa'); axt.set_xlabel('Time (ms)'); axt.set_xlim(0,1000); axt.grid(alpha=.2); lt,=axt.plot([],[],color='#2166ac',label='truth'); ltp,=axt.plot([],[],color='#d6604d',ls='--',label='neural'); axt.legend(frameon=False)
frames=np.arange(ntime)[::2]
def update(k):
    q=frames[k]; tt=t[:q+1]*1000; lv.set_data(tt,truth[0,:q+1]); lp.set_data(tt,forecast[0,:q+1]); lt.set_data(tt,truth[2,:q+1]); ltp.set_data(tt,forecast[2,:q+1]); ca=np.clip((forecast[1,q]-.05)/.45,0,1); cellpatch.set_facecolor((1-.5*ca,.35+.45*ca,.65+.25*ca)); mem.set_edgecolor('#762a83' if forecast[0,q]>-20 else '#2166ac'); mem.width=1.5*(1+3*(lam[p,q]-1)); cellpatch.width=1.4*(1+3*(lam[p,q]-1)); txt.set_text(f'Ca\n{forecast[1,q]:.3f} µM\nλ={lam[p,q]:.3f}'); return lv,lp,lt,ltp,cellpatch,mem,txt
ani=FuncAnimation(fig,update,frames=len(frames),interval=35,blit=False); ani.save(OUT/'conditioned_virtual_cell.gif',writer=PillowWriter(fps=25)); plt.close(fig)
print(json.dumps(metrics,indent=2))
