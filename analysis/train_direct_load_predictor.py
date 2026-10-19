"""Direct, load-conditioned waveform baseline with a sealed outside-range load.

Target lambda=1.125 is removed from all training/validation targets. Cell
splits stay frozen; lambda=1.05 is an in-range test condition.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from load_conditioned_posterior import TraceEncoder


class DirectLoadNetwork(nn.Module):
    def __init__(self, n_calibration_loads: int, n_times: int):
        super().__init__()
        self.ap = TraceEncoder(1)
        self.ca = TraceEncoder(1)
        self.tension = TraceEncoder(2)
        self.query = nn.Sequential(
            nn.Linear(48 * 3 + 1, 192), nn.GELU(),
            nn.Linear(192, 192), nn.GELU(),
            nn.Linear(192, n_times),
        )
        self.n_calibration_loads = n_calibration_loads

    def forward(self, ap, ca, t_cal, cal_lambda, target_lambda):
        b, k, length = t_cal.shape
        ap_z = self.ap((ap / 100).unsqueeze(1))
        ca_z = self.ca(ca.unsqueeze(1))
        lam_series = ((cal_lambda - 1) / .125)[:, :, None].expand(-1, -1, length)
        load_input = torch.stack((t_cal / 100, lam_series), dim=2)
        load_z = self.tension(load_input.reshape(b*k, 2, length)).reshape(b,k,48).mean(1)
        target = ((target_lambda - 1) / .125)[:, None]
        return self.query(torch.cat((ap_z, ca_z, load_z, target), dim=1)) * 100


def get_data(path: Path):
    with h5py.File(path) as f:
        if path.name == 'cohort.mat':
            v = f["V"][()].T
            ca = f["Cai_uM"][()].T
            t = f["Ta_kPa"][()].transpose(2,1,0)
            split = f["split_code"][()].ravel().astype(int)
            lam = f["lambda"][()].ravel()
        else:
            v = f["V"][()]
            ca = f["Cai_uM"][()]
            t = f["Ta_kPa"][()]
            split = f["split_code"][()].astype(int)
            lam = f["lambda"][()]
    time = np.arange(0, 601, 2)
    return (
        torch.from_numpy(v[:, time].copy()).float(),
        torch.from_numpy(ca[:, time].copy()).float(),
        torch.from_numpy(t[:, :, time].copy()).float(),
        split, lam,
    )


@torch.no_grad()
def score(model, data, cell_ids, cal_ids, target_ids):
    model.eval()
    ap, ca, t, _, lam = data
    rows = []
    for j in target_ids:
        actual = t[cell_ids, j, :]
        cal = t[cell_ids][:, cal_ids, :]
        cal_lam = torch.tensor(lam[cal_ids], dtype=torch.float32).repeat(len(cell_ids),1)
        query = torch.full((len(cell_ids),),float(lam[j]))
        predicted = model(ap[cell_ids],ca[cell_ids],cal,cal_lam,query)
        rows.append({
            "lambda": float(lam[j]),
            "waveform_MAE_kPa": float((actual-predicted).abs().mean()),
            "peak_MAE_kPa": float((actual.max(1).values-predicted.max(1).values).abs().mean()),
        })
    return rows


def static_load_baseline(data, cell_ids, cal_ids, target_ids):
    _, _, t, _, lam = data
    rows=[]
    for j in target_ids:
        observed=t[cell_ids,cal_ids[0],:]
        if len(cal_ids)==1:
            predicted=observed
        else:
            second=t[cell_ids,cal_ids[1],:]
            fraction=(float(lam[j])-float(lam[cal_ids[0]])) / (float(lam[cal_ids[1]])-float(lam[cal_ids[0]]))
            predicted=observed+fraction*(second-observed)
        actual=t[cell_ids,j,:]
        rows.append({
            'lambda':float(lam[j]),
            'waveform_MAE_kPa':float((actual-predicted).abs().mean()),
            'peak_MAE_kPa':float((actual.max(1).values-predicted.max(1).values).abs().mean()),
        })
    return rows


def train(path: Path, epochs: int, n_cal: int):
    torch.set_num_threads(4)
    torch.manual_seed(20260928)
    np.random.seed(20260928)
    random.seed(20260928)
    data = get_data(path)
    ap, ca, t, split, lam = data
    def index(value):
        j = int(np.argmin(abs(lam - value)))
        assert abs(lam[j] - value) < 1e-6
        return j
    cal_ids = [index(1.0)] if n_cal == 1 else [index(1.0), index(1.1)]
    # The 1.125 output is sealed from training. Calibration observations
    # are also excluded as targets; for two-load calibration, use 1.025-.075.
    train_target_ids = [index(x) for x in [.025 + 1, 1.05, 1.075, 1.1] if index(x) not in cal_ids]
    test_ids = np.flatnonzero(split == 2)
    train_ids = np.flatnonzero(split == 0)
    valid_ids = np.flatnonzero(split == 1)
    if min(len(train_ids),len(valid_ids),len(test_ids))<10:
        raise ValueError("Too few cells in frozen split")
    model = DirectLoadNetwork(n_cal,ap.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-3)
    best = float('inf'); best_weights = None; stale = 0
    rng = np.random.default_rng(20260928)
    history = []
    for epoch in range(epochs):
        model.train()
        rows = np.array([(i,j) for i in train_ids for j in train_target_ids])
        rng.shuffle(rows)
        losses=[]
        for start in range(0,len(rows),128):
            batch=rows[start:start+128]
            cells=batch[:,0]; targets=batch[:,1]
            cal=t[cells][:,cal_ids,:]
            cal_lam=torch.tensor(lam[cal_ids],dtype=torch.float32).repeat(len(cells),1)
            target_lam=torch.tensor(lam[targets],dtype=torch.float32)
            actual=t[cells,targets,:]
            prediction=model(ap[cells],ca[cells],cal,cal_lam,target_lam)
            loss=torch.nn.functional.huber_loss(prediction/100,actual/100,delta=.1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        val=score(model,data,valid_ids,cal_ids,train_target_ids)
        val_loss=float(np.mean([x['waveform_MAE_kPa'] for x in val]))
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_MAE_kPa':val_loss})
        if val_loss<best-1e-4:
            best=val_loss; best_weights=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
        if (epoch+1)%10==0:
            print(f'epoch {epoch+1} train={history[-1]["train_loss"]:.4f} val={val_loss:.3f} best={best:.3f}',flush=True)
        if stale>=20: break
    model.load_state_dict(best_weights)
    result={
        'cohort':str(path.resolve()),'calibration_loads':lam[cal_ids].tolist(),
        'training_target_loads':lam[train_target_ids].tolist(),
        'train_n_cells':len(train_ids),'validation_n_cells':len(valid_ids),
        'test_n_cells':len(test_ids),'epochs_run':len(history),
        'in_range_test':score(model,data,test_ids,cal_ids,[index(1.05)]),
        'outside_range_test':score(model,data,test_ids,cal_ids,[index(1.125)]),
        'static_in_range_test':static_load_baseline(data,test_ids,cal_ids,[index(1.05)]),
        'static_outside_range_test':static_load_baseline(data,test_ids,cal_ids,[index(1.125)]),
        'note':'Direct neural baseline; target lambda=1.125 excluded from all training targets.',
    }
    folder=path.parent / f'direct_{n_cal}load'
    folder.mkdir(exist_ok=True)
    torch.save(model.state_dict(),folder/'model.pt')
    (folder/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (folder/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('cohort',type=Path)
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--calibration-loads',type=int,choices=[1,2],default=1)
    options=parser.parse_args()
    train(options.cohort,options.epochs,options.calibration_loads)
