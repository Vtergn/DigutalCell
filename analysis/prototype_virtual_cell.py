"""Small neural virtual-cell prototype and animated state renderer.

This is a representation demo, not an independent digital-twin validation.
The network learns the existing eight waveform traces and the renderer shows
the learned V/Ca/TnC state together with the imposed length pulse.
"""
from pathlib import Path
import csv, json
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "human_bidirectional_output"
OUT = ROOT / "virtual_cell_prototype"
OUT.mkdir(exist_ok=True)
torch.manual_seed(7); np.random.seed(7)

configs=[("trpn0_sac0",0,0,"feedback off"),("trpn1_sac0",1,0,"TnC–Ca only"),
         ("trpn0_sac1",0,1,"SAC only"),("trpn1_sac1",1,1,"both feedbacks")]
rows=[]; traces={}
for onset in (150,600):
    for suffix,trpn,sac,label in configs:
        path=DATA/f"pulse_t{onset}_lam110_{suffix}.csv"
        with path.open(newline='',encoding='utf-8') as h:
            d=list(csv.DictReader(h))
        t=np.array([float(r['time_ms']) for r in d],dtype=np.float32)
        y=np.array([[float(r['Vm_mV']),float(r['Cai_uM']),float(r['TRPN_fraction'])] for r in d],dtype=np.float32)
        lam=np.ones_like(t); m=(t>=onset)&(t<onset+1); lam[m]=1+(t[m]-onset)*.1
        m=(t>=onset+1)&(t<onset+11); lam[m]=1.1
        m=(t>=onset+11)&(t<onset+12); lam[m]=1.1-(t[m]-(onset+11))*.1
        traces[(onset,suffix)]=(t,lam,y)
        for ti,li,yi in zip(t,lam,y): rows.append([ti/1000,li,onset/1000,trpn,sac,*yi])
data=np.asarray(rows,np.float32)
X=torch.tensor(data[:,:5]); Y=torch.tensor(data[:,5:])
mu=Y.mean(0); sd=Y.std(0)
class Decoder(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(5,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,3))
    def forward(self,x): return self.net(x)
model=Decoder(); opt=torch.optim.Adam(model.parameters(),lr=2e-3)
Yn=(Y-mu)/sd
for epoch in range(3000):
    pred=model(X); loss=((pred-Yn)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad(): pred=model(X)*sd+mu
rmse=torch.sqrt(((pred-Y)**2).mean(0)).numpy()
metrics={"training_traces":8,"training_points":int(len(X)),"rmse_V_mV":float(rmse[0]),"rmse_Ca_uM":float(rmse[1]),"rmse_TRPN":float(rmse[2]),"status":"representation/reconstruction prototype; no independent generalization claim"}
(OUT/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding='utf-8')
torch.save(model.state_dict(),OUT/"decoder.pt")

# Render the learned both-feedback cell under the early pulse.
onset=150; suffix="trpn1_sac1"; t,lam,y=traces[(onset,suffix)]
with torch.no_grad():
    xx=torch.tensor(np.c_[t/1000,lam,np.full_like(t,onset/1000),np.ones_like(t),np.ones_like(t)],dtype=torch.float32)
    yh=(model(xx)*sd+mu).numpy()
frames=np.arange(0,len(t),10)
fig=plt.figure(figsize=(11,6)); gs=fig.add_gridspec(2,2,width_ratios=[1.0,1.65],height_ratios=[1,1])
axcell=fig.add_subplot(gs[:,0]); axv=fig.add_subplot(gs[0,1]); axca=fig.add_subplot(gs[1,1])
axcell.set_aspect('equal'); axcell.set_xlim(-1.5,1.5); axcell.set_ylim(-1.25,1.25); axcell.axis('off')
axcell.set_title('Virtual cardiomyocyte state',fontweight='bold')
cell=Ellipse((0,0),1.4,1.0,facecolor='#f4a3b8',edgecolor='#333',lw=2); axcell.add_patch(cell)
mem=Ellipse((0,0),1.5,1.08,fill=False,edgecolor='#2166ac',lw=8); axcell.add_patch(mem)
ca_txt=axcell.text(0,0.1,'Ca',ha='center',va='center',fontsize=24,color='white',fontweight='bold')
state_txt=axcell.text(0,-.28,'',ha='center',va='center',fontsize=11)
axv.set_title('Membrane voltage'); axv.set_ylabel('mV'); axv.set_xlim(0,1000); axv.set_ylim(-100,50); axv.grid(alpha=.2)
axca.set_title('Intracellular Ca and imposed stretch'); axca.set_ylabel('Ca (µM)'); axca.set_xlabel('Time (ms)'); axca.set_xlim(0,1000); axca.set_ylim(0,.55); axca.grid(alpha=.2)
lv,=axv.plot([],[],color='#762a83',lw=1.8,label='V(t)'); lc,=axca.plot([],[],color='#d6604d',lw=1.8,label='Caᵢ(t)'); ls=axca.twinx(); ls.set_ylabel('λ'); ls.set_ylim(.99,1.11); ll,=ls.plot([],[],color='#1b7837',lw=1.8,label='λ(t)')
axv.legend(frameon=False); axca.legend([lc,ll],['Caᵢ','λ'],frameon=False,loc='upper right')
def update(k):
    j=frames[k]; lv.set_data(t[:j+1],yh[:j+1,0]); lc.set_data(t[:j+1],yh[:j+1,1]); ll.set_data(t[:j+1],lam[:j+1])
    width=1.4*(1+3*(lam[j]-1)); cell.width=width; mem.width=width*1.07
    ca_norm=np.clip((yh[j,1]-.05)/.45,0,1); cell.set_facecolor((1-.5*ca_norm,.35+.45*ca_norm,.65+.25*ca_norm))
    v_norm=np.clip((yh[j,0]+90)/125,0,1); mem.set_edgecolor((.15+.7*v_norm,.2,.8-.6*v_norm))
    ca_txt.set_text(f'Ca\n{yh[j,1]:.3f} µM'); state_txt.set_text(f'V = {yh[j,0]:.1f} mV\nλ = {lam[j]:.3f}')
    return lv,lc,ll,cell,mem,ca_txt,state_txt
ani=FuncAnimation(fig,update,frames=len(frames),interval=35,blit=False)
ani.save(OUT/"virtual_cell_demo.gif",writer=PillowWriter(fps=25))
plt.close(fig)
print(json.dumps(metrics,indent=2))
