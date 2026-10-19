"""Export explicit parameter draws for true-equation held-out-load propagation.

Usage: python analysis/export_posterior_draws.py COHORT.h5 MODEL.pt OUTPUT.csv
Only frozen test cells are exported. Mode 2 uses T1.0; mode 3 uses T1.0+T1.1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from load_conditioned_posterior import LoadConditionedPosterior
from train_load_posterior import mode_masks, read_data


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('cohort',type=Path)
    parser.add_argument('model',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--cells',type=int,default=24)
    parser.add_argument('--draws',type=int,default=16)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260928)
    data=read_data(args.cohort)
    test=np.flatnonzero(data[5].numpy()==2)
    rng=np.random.default_rng(20260928)
    selected=np.sort(rng.choice(test,size=min(args.cells,len(test)),replace=False))
    saved=torch.load(args.model,map_location='cpu',weights_only=False)
    model=LoadConditionedPosterior(n_times=data[0].shape[1])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    lo,hi=data[6],data[7]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['cell_index_zero_based','mode','draw_index']+[f'theta_{i+1}' for i in range(12)])
        with torch.no_grad():
            for mode in [2,3]:
                index=torch.tensor(selected)
                ap,ca,tension,lambdas=(x[index] for x in data[:4])
                cm,lm=mode_masks(mode,len(index),ap.device)
                posterior=model(ap,ca,tension,lambdas,cm,lm)
                unit=model.sample(posterior,args.draws).numpy()
                physical=lo+unit*(hi-lo)
                for row,cell in enumerate(selected):
                    for draw in range(args.draws):
                        writer.writerow([int(cell),mode,draw]+physical[row,draw].tolist())
    manifest={
        'cohort':str(args.cohort.resolve()),
        'model':str(args.model.resolve()),
        'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'target_lambda':1.125,'test_cell_indices_zero_based':selected.tolist(),
        'draws_per_cell_per_mode':args.draws,'modes':[2,3],
        'note':'Parameter draws only; run torord_land_given_theta for true forward prediction.',
    }
    args.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'exported {len(selected)} test cells x 2 modes x {args.draws} draws to {args.output}')


if __name__=='__main__':
    main()
