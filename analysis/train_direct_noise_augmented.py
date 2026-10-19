"""Same direct network, trained with noisy calibration inputs only.

The target loads remain 1.025, 1.05, 1.075. Lambda 1.125 and all test
cells remain excluded from training/validation and model selection.
"""
from pathlib import Path
import copy
import json
import numpy as np
import torch
from torch.nn import functional as F
from train_direct_load_predictor import DirectLoadNetwork,get_data

root=Path(__file__).resolve().parents[1]
base=root/'analysis/virtual_cardiomyocyte_5000'
torch.set_num_threads(4)
torch.manual_seed(20260930)
rng=np.random.default_rng(20260930)
ap,ca,t,split,lam=get_data(base/'merged_cohort.h5')
ix=lambda x:int(np.argmin(abs(lam-x)))
cal_ids=[ix(1.),ix(1.1)]
target_ids=[ix(x) for x in [1.025,1.05,1.075]]
assert np.allclose(lam[cal_ids],[1.,1.1])
assert np.allclose(lam[target_ids],[1.025,1.05,1.075])
train_ids=np.flatnonzero(split==0)
valid_ids=np.flatnonzero(split==1)
model=DirectLoadNetwork(2,ap.shape[1])
optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-3)

def perturb(a,c,tt,kind,generator):
    if kind=='clean':return a,c,tt
    if kind=='low_noise':sa,sc,st,step,dc,dt=1.,.005,.5,1,0.,0.
    elif kind=='high_noise':sa,sc,st,step,dc,dt=3.,.015,1.5,1,0.,0.
    elif kind=='drift_sparse':sa,sc,st,step,dc,dt=1.,.005,.5,10,.004,1.
    else:raise ValueError(kind)
    def noisy(q,sigma):
        if step==1:return q+torch.randn(q.shape,generator=generator)*sigma
        coarse=q[...,::step]+torch.randn(q[...,::step].shape,generator=generator)*sigma
        return F.interpolate(coarse.reshape(-1,1,coarse.shape[-1]),size=q.shape[-1],
                             mode='linear',align_corners=True).reshape(q.shape)
    a=noisy(a,sa);c=noisy(c,sc);tt=noisy(tt,st)
    if dc:
        frac=torch.linspace(0,1,c.shape[-1])[None,:]
        c=c+(torch.rand((len(c),1),generator=generator)*2-1)*dc
        c=c+(torch.rand((len(c),1),generator=generator)*2-1)*dc*frac
    if dt:
        frac=torch.linspace(0,1,tt.shape[-1])[None,None,:]
        tt=tt+(torch.rand((len(tt),2,1),generator=generator)*2-1)*dt
        tt=tt+(torch.rand((len(tt),2,1),generator=generator)*2-1)*dt*frac
    return a,torch.clamp(c,min=.001),tt

# Fixed, noisy validation traces from validation cells only.
validation=[]
for k,kind in enumerate(['clean','low_noise','high_noise','drift_sparse']):
    ids=valid_ids[:min(len(valid_ids),256)]
    g=torch.Generator().manual_seed(20260930+k)
    validation.append((kind,ids,perturb(ap[ids],ca[ids],t[ids][:,cal_ids,:],kind,g)))

def validate():
    model.eval();errors=[]
    with torch.no_grad():
        for _,ids,(va,vc,vt) in validation:
            cal_lam=torch.tensor([1.,1],dtype=torch.float32).repeat(len(ids),1)
            cal_lam[:,1]=1.1
            for j in target_ids:
                pred=model(va,vc,vt,cal_lam,torch.full((len(ids),),float(lam[j])))
                errors.append(float(torch.mean(abs(pred-t[ids,j,:]))))
    return float(np.mean(errors))

best=float('inf');best_state=None;stale=0;history=[]
rows=np.array([(i,j) for i in train_ids for j in target_ids])
for epoch in range(100):
    model.train();rng.shuffle(rows);losses=[]
    for start in range(0,len(rows),128):
        batch=rows[start:start+128];ids=batch[:,0];targets=batch[:,1]
        g=torch.Generator().manual_seed(20260930+epoch*1000+start)
        kind=['clean','low_noise','high_noise','drift_sparse'][rng.integers(0,4)]
        va,vc,vt=perturb(ap[ids],ca[ids],t[ids][:,cal_ids,:],kind,g)
        pred=model(va,vc,vt,torch.tensor([1.,1.1]).repeat(len(ids),1),
                   torch.tensor(lam[targets],dtype=torch.float32))
        loss=F.huber_loss(pred/100,t[ids,targets,:]/100,delta=.1)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        losses.append(float(loss.detach()))
    v=validate()
    history.append({'epoch':epoch+1,'training_loss':float(np.mean(losses)),'validation_MAE_kPa':v})
    if v<best-1e-4:
        best=v;best_state=copy.deepcopy(model.state_dict());stale=0
    else:stale+=1
    if (epoch+1)%10==0:print(f'epoch {epoch+1} validation {v:.3f} best {best:.3f}',flush=True)
    if stale>=20:break
out=base/'direct_2load_noise_augmented'
out.mkdir(exist_ok=True)
torch.save(best_state,out/'model.pt')
(out/'manifest.json').write_text(json.dumps({'epochs_run':len(history),
    'best_validation_MAE_kPa':best,'training_target_lambdas':[1.025,1.05,1.075],
    'training_cells':len(train_ids),'validation_cells':len(valid_ids),
    'augmentation':['clean','low_noise','high_noise','drift_sparse'],
    'architecture':'Same DirectLoadNetwork as clean baseline',
    'selection':'Validation cells only, across four observation conditions'},indent=2))
(out/'history.json').write_text(json.dumps(history,indent=2))
print(json.dumps({'epochs':len(history),'best_validation_MAE_kPa':best},indent=2))
