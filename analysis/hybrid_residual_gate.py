"""Pre-neural gate for few-shot cross-model discrepancy correction.

Source A is an intentionally wrong mechanistic prior for independent B cells.
Matching and per-cell correction use only baseline and lambda=1.05 B data.
The held-out 1.075/1.10 B outputs are used only to score predictions.
This is an output-discrepancy screen, not a hybrid ODE or a PINN.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

import score_cross_model_static_geometry as geom

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'mechchem_cross_model/output/hybrid_residual_gate.json'
TARGET=geom.EC
LOADS=np.array([1.075,1.10])


def main():
    a,qa=geom.clean_features(geom.A_PATH)
    b,qb=geom.clean_features(geom.B_PATH)
    # The match sees only broad, physically comparable baseline/calibration
    # observations. Scales are fixed physical resolutions, not future B labels.
    match_cols=np.array([0,1,2])
    scale=np.array([30,.08,12.])
    a_match=np.concatenate([a[:,0,match_cols]/scale,
                            (a[:,1,match_cols]-a[:,0,match_cols])/scale],axis=1)
    b_match=np.concatenate([b[:,0,match_cols]/scale,
                            (b[:,1,match_cols]-b[:,0,match_cols])/scale],axis=1)
    match=cdist(b_match,a_match).argmin(axis=1)
    ap=a[match]

    # Predict future responses relative to each B cell's observed baseline.
    truth=np.stack([b[:,j,TARGET]-b[:,0,TARGET] for j in (2,3)],axis=1)
    physics=np.stack([ap[:,j,TARGET]-ap[:,0,TARGET] for j in (2,3)],axis=1)
    r=(LOADS-1)/.05
    linear=r[None,:,None]*(b[:,1,TARGET]-b[:,0,TARGET])[:,None,:]
    physics_linear=r[None,:,None]*(ap[:,1,TARGET]-ap[:,0,TARGET])[:,None,:]
    curvature=physics-physics_linear
    hybrid=linear+curvature
    zero=np.zeros_like(truth)
    # Same future-scale normalization as the geometry analysis, with
    # measurement-resolution floors. It is fixed for evaluation only.
    floor=geom.FLOOR[TARGET]
    ys=np.maximum(truth.std(axis=0),floor)
    methods={'zero_change':zero,'wrong_physics':physics,
             'direct_two_point_linear':linear,
             'wrong_physics_plus_affine_residual':hybrid}
    scores={}
    for name,pred in methods.items():
        err=np.abs((pred-truth)/ys)
        scores[name]={
            'mean_standardized_MAE':float(err.mean()),
            'by_load_standardized_MAE':err.mean(axis=(0,2)).tolist(),
            'by_endpoint_standardized_MAE':err.mean(axis=(0,1)).tolist(),
        }
    out={
        'n_B_cells':len(b),'n_A_library_cells':len(a),
        'A_candidate_indices_zero_based':match.tolist(),
        'match_distance_median':float(np.median(np.min(cdist(b_match,a_match),axis=1))),
        'future_loads':LOADS.tolist(),
        'endpoint_order':['APD','Ca_peak','V_min','V_max','Ca_200','Ca_400','Ca_600','Ca_AUC'],
        'scores':scores,
        'mean_abs_curvature_A':np.mean(np.abs(curvature),axis=(0,1)).tolist(),
        'mean_abs_curvature_B':np.mean(np.abs(truth-linear),axis=(0,1)).tolist(),
        'interpretation':'Output-level affine discrepancy gate; no neural or ODE residual fitted.'
    }
    OUT.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
