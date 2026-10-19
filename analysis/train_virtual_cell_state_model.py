"""Train and evaluate a small recurrent virtual-cell behavior model.

The model is deliberately modest: it learns a shared state transition from
many simulated cells, while the native ToR-ORd/SAC–Land model remains the
ground truth. Splits are by cell, so this is a real held-out-cell test.
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

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'human_bidirectional_output'; OUT=ROOT/'virtual_cell_state_model'; OUT.mkdir(exist_ok=True)
torch.manual_seed(11); np.random.seed(11)
with h5py.File(DATA/'dynamic_discrepancy_development.mat') as f:
    # MATLAB/HDF5 order: cell, feedback-form, protocol, channel, time.
    wave=f['waveforms'][:,1,:,:,:].astype('float32')
    lam=f['lambda_tracks'][:].astype('float32')
    grid=f['grid'][:].ravel().astype('float32')
assert wave.shape==(64,6,3,1001), wave.shape

# Downsample to 10 ms for a compact but dynamic state-space benchmark.
valid_cells=np.isfinite(wave).all(axis=(1,2,3)); wave=wave[valid_cells]
step=10; y=wave[:,:,:,::step]; lam_ds=lam[:,::step]; u=np.broadcast_to(lam_ds[None,:,:],(wave.shape[0],lam_ds.shape[0],lam_ds.shape[1])); t=grid[::step]; ncell,nprot,nchan,nt=y.shape
ntrain=int(np.floor(.75*ncell)); train_cells=np.arange(ntrain); test_cells=np.arange(ntrain,ncell)
mu=y[train_cells].reshape(-1,3).mean(0); sd=y[train_cells].reshape(-1,3).std(0)+1e-6
yn=(y-mu[None,None,:,None])/sd[None,None,:,None]

class StateModel(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(4,96),nn.Tanh(),nn.Linear(96,96),nn.Tanh(),nn.Linear(96,3))
    def step(self,h,inp): return h, self.net(inp)

model=StateModel(); opt=torch.optim.Adam(model.parameters(),lr=2e-3)
train_y=torch.tensor(yn[train_cells]); train_u=torch.tensor(u[train_cells])
inp=torch.cat([train_u[:,:,:,None],train_y.permute(0,1,3,2)],dim=3)[:,:,:-1,:].reshape(-1,4)
target=train_y[:,:,:,1:].permute(0,1,3,2).reshape(-1,3)
for epoch in range(300):
    p=model.net(inp); loss=((p-target)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    if epoch in (0,49,149,299): print('epoch',epoch+1,'loss',float(loss.detach()))

def rollout(cell,protocol):
    h=torch.zeros(64); pred=torch.zeros(nt,3); pred[0]=torch.tensor(yn[cell,protocol,:,0])
    for k in range(nt-1):
        inp=torch.cat([torch.tensor([u[cell,protocol,k]]),pred[k]],0).float(); h,p=model.step(h,inp); pred[k+1]=p.detach()
    return (pred.numpy()*sd+mu)

pred=np.zeros_like(y[test_cells]);
for a,cell in enumerate(test_cells):
    for protocol in range(nprot): pred[a,protocol]=rollout(cell,protocol).T
err=np.mean(np.abs(pred-y[test_cells]),axis=(0,1,3)); # channel errors
all_mae=float(np.mean(np.abs(pred-y[test_cells])))
metrics={'train_cells':train_cells.tolist(),'test_cells':test_cells.tolist(),'protocols':nprot,'time_step_ms':10,'V_MAE_mV':float(err[0]),'Ca_MAE_uM':float(err[1]),'T_MAE_kPa':float(err[2]),'overall_MAE_mixed_units':all_mae,'note':'held-out-cell rollout; all six protocols are seen during training'}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); torch.save(model.state_dict(),OUT/'state_model.pt')
np.savez_compressed(OUT/'heldout_predictions.npz',pred=pred,true=y[test_cells],time=t,lambda_tracks=u[test_cells])

# Render one held-out cell under the unseen-to-this-cell long-hold trajectory.
cell=int(test_cells[0]); protocol=3; j=np.where(test_cells==cell)[0][0]; truth=y[cell,protocol].T; forecast=pred[j,protocol].T
fig=plt.figure(figsize=(11,6)); gs=fig.add_gridspec(2,2,width_ratios=[1,1.7]); axc=fig.add_subplot(gs[:,0]); axv=fig.add_subplot(gs[0,1]); axt=fig.add_subplot(gs[1,1]); axc.set_aspect('equal'); axc.set_xlim(-1.5,1.5); axc.set_ylim(-1.25,1.25); axc.axis('off')
axc.set_title(f'Held-out virtual cell {cell}'); cellpatch=Ellipse((0,0),1.4,1,facecolor='#ed8da9',edgecolor='#333',lw=2); mem=Ellipse((0,0),1.5,1.08,fill=False,edgecolor='#2166ac',lw=8); axc.add_patch(cellpatch); axc.add_patch(mem); txt=axc.text(0,0,'',ha='center',va='center',color='white',fontsize=18,fontweight='bold')
axv.set_title('Voltage: native truth vs neural rollout'); axv.set_ylabel('mV'); axv.set_xlim(0,1000); axv.set_ylim(-100,50); axv.grid(alpha=.2); lv,=axv.plot([],[],color='#2166ac',label='truth'); lp,=axv.plot([],[],color='#d6604d',ls='--',label='neural'); axv.legend(frameon=False)
axt.set_title('Tension: native truth vs neural rollout'); axt.set_ylabel('kPa'); axt.set_xlabel('Time (ms)'); axt.set_xlim(0,1000); axt.grid(alpha=.2); lt,=axt.plot([],[],color='#2166ac',label='truth'); ltp,=axt.plot([],[],color='#d6604d',ls='--',label='neural'); axt.legend(frameon=False)
frames=np.arange(nt)[::2]
def update(k):
    q=frames[k]; tt=t[:q+1]; lv.set_data(tt,truth[:q+1,0]); lp.set_data(tt,forecast[:q+1,0]); lt.set_data(tt,truth[:q+1,2]); ltp.set_data(tt,forecast[:q+1,2]);
    ca=np.clip((forecast[q,1]-.05)/.45,0,1); cellpatch.set_facecolor((1-.5*ca,.35+.45*ca,.65+.25*ca)); mem.set_edgecolor('#762a83' if forecast[q,0]>-20 else '#2166ac'); mem.width=1.5*(1+3*(u[cell,protocol,q]-1)); cellpatch.width=1.4*(1+3*(u[cell,protocol,q]-1)); txt.set_text(f'Ca\n{forecast[q,1]:.3f} µM\nλ={u[cell,protocol,q]:.3f}'); return lv,lp,lt,ltp,cellpatch,mem,txt
ani=FuncAnimation(fig,update,frames=len(frames),interval=35,blit=False); ani.save(OUT/'heldout_virtual_cell.gif',writer=PillowWriter(fps=25)); plt.close(fig)
print(json.dumps(metrics,indent=2))
