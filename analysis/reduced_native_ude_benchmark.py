"""Reduced native ODE/UDE gate.

A small, real RK4 rollout benchmark. It is intentionally not the full
ToR-ORd/SAC-Land UDE: the prior vector field is a fitted 3-output field.
"""
from pathlib import Path
import json, time
import numpy as np, torch
from scipy.io import loadmat
from torch import nn
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"human_bidirectional_output"/"dynamic_discrepancy_development.mat"
OUT=ROOT/"reduced_native_ude_benchmark"; OUT.mkdir(exist_ok=True)
torch.manual_seed(17); np.random.seed(17)
f=loadmat(str(DATA.with_name("dynamic_discrepancy_development_v5.mat")))
waveforms=np.transpose(f["waveforms"].astype("float32"),(4,1,2,3,0))
lam=f["lambda_tracks"].astype("float32"); lam=np.squeeze(lam)
grid=f["grid"].ravel().astype("float32")
valid=np.isfinite(waveforms[:,0]).all(axis=(1,2,3)) & np.isfinite(waveforms[:,1]).all(axis=(1,2,3))
waveforms=waveforms[valid]
step=10
prior=waveforms[:,0,:,:,::step].transpose(0,1,3,2)
truth=waveforms[:,1,:,:,::step].transpose(0,1,3,2)
u=lam[::step,:].T; t=grid[::step]; dt=float(t[1]-t[0])
ncell,nprot,nt,nch=prior.shape
train=np.arange(min(20,int(.75*ncell))); test=np.arange(int(.75*ncell),ncell)
hold=3; seen=[0,1,2]
scale=np.array([20.,.1,20.],dtype="float32")
P=prior/scale; Y=truth/scale
X=[]; Z=[]
for c in train:
    for p in seen:
        x=P[c,p,:-1]; dx=np.diff(P[c,p],axis=0)/dt
        uu=np.column_stack([u[p,:-1],np.ones(nt-1,dtype="float32")])
        X.append(np.column_stack([x,uu])); Z.append(dx)
X=np.concatenate(X); Z=np.concatenate(Z)
coef=np.linalg.lstsq(X,Z,rcond=None)[0].astype("float32")
C=torch.tensor(coef)

def prior_rhs(x,uu):
    return torch.cat([x,uu,torch.ones_like(uu)],-1) @ C

class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4,48),nn.Tanh(),nn.Linear(48,48),nn.Tanh(),nn.Linear(48,3))
    def forward(self,x,uu):
        return 0.20*torch.tanh(self.net(torch.cat([x,uu],-1)))

model=Residual(); opt=torch.optim.Adam(model.parameters(),lr=2e-3)

def rk4_step(x,uu,h):
    k1=prior_rhs(x,uu)+model(x,uu)
    k2=prior_rhs(x+0.5*h*k1,uu)+model(x+0.5*h*k1,uu)
    k3=prior_rhs(x+0.5*h*k2,uu)+model(x+0.5*h*k2,uu)
    k4=prior_rhs(x+h*k3,uu)+model(x+h*k3,uu)
    return x+(h/6.)*(k1+2*k2+2*k3+k4)

def rollout(x0,uv):
    xs=[x0]; x=x0
    for k in range(uv.shape[1]-1):
        x=rk4_step(x,uv[:,k:k+1],dt); xs.append(x)
    return torch.stack(xs,1)

starts=torch.tensor(P[train][:,seen,0,:],dtype=torch.float32)
targets=torch.tensor(Y[train][:,seen,:,:],dtype=torch.float32)
uv=torch.tensor(u[seen,:],dtype=torch.float32).unsqueeze(0).expand(len(train),-1,-1)
t0=time.perf_counter()
for ep in range(20):
    opt.zero_grad(); total=0.
    for j in range(len(seen)):
        total=total+torch.mean(torch.abs(rollout(starts[:,j],uv[:,j])-targets[:,j]))
    total=total/len(seen)+1e-3*sum((q*q).mean() for q in model.parameters())
    total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
elapsed=time.perf_counter()-t0

@torch.no_grad()
def predict(cells,p):
    x0=torch.tensor(P[cells,p,0],dtype=torch.float32)
    uu=torch.tensor(np.tile(u[p][None,:],(len(cells),1)),dtype=torch.float32)
    return rollout(x0,uu).numpy()*scale

pred=predict(test,hold); base=prior[test,hold]*scale; true=truth[test,hold]*scale
mae=np.mean(np.abs(pred-true),axis=(0,1)); bmae=np.mean(np.abs(base-true),axis=(0,1))
metrics={"train_cells":train.tolist(),"test_cells":test.tolist(),"held_protocol":int(hold),"epochs":20,"elapsed_seconds":float(elapsed),"n_parameters":int(sum(q.numel() for q in model.parameters())),"prior_MAE_V_Ca_T":[float(x) for x in bmae],"native_reduced_ude_MAE_V_Ca_T":[float(x) for x in mae],"prior_overall_MAE":float(np.mean(np.abs(base-true))),"native_reduced_ude_overall_MAE":float(np.mean(np.abs(pred-true))),"note":"True RK4 rollout of a reduced vector field; not the full native ToR-ORd/SAC-Land RHS."}
(OUT/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
torch.save({"model":model.state_dict(),"coef":coef},OUT/"reduced_native_ude.pt")
np.savez_compressed(OUT/"predictions.npz",pred=pred,prior=base,true=true,time=t)
fig,axs=plt.subplots(3,1,figsize=(9,7),sharex=True,constrained_layout=True)
for q,(ax,label) in enumerate(zip(axs,["V (mV)","Ca (µM)","T (kPa)"])):
    ax.plot(t,true[:,:,q].mean(0),color="#2166ac",lw=2,label="truth")
    ax.plot(t,base[:,:,q].mean(0),color="#777777",lw=1.3,label="reduced prior")
    ax.plot(t,pred[:,:,q].mean(0),color="#d6604d",ls="--",lw=1.8,label="native reduced UDE")
    ax.set_ylabel(label); ax.grid(alpha=.2)
axs[-1].set_xlabel("Time (ms)"); axs[0].set_title("Reduced native UDE: held-out protocol",fontweight="bold"); axs[0].legend(frameon=False)
fig.savefig(OUT/"comparison.png",dpi=220,bbox_inches="tight"); fig.savefig(OUT/"comparison.pdf",bbox_inches="tight"); plt.close(fig)
(OUT/"README.md").write_text("This is a real RK4 rollout test with a bounded derivative residual. The prior is a fitted three-output reduced vector field, not the full native ToR-ORd/SAC-Land RHS. Results are therefore a feasibility gate, not a full cardiac UDE claim.",encoding="utf-8")
print(json.dumps(metrics,indent=2))










