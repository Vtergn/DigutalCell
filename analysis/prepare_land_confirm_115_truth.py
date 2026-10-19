"""Prepare hidden true-parameter input for an independent λ=1.15 check.

This file must only be used by the separate truth generator and scorer.
"""
from pathlib import Path
import argparse
import json
import h5py
import numpy as np
from scipy.io import savemat

base=Path(__file__).resolve().parent/'virtual_cardiomyocyte_5000'
parser=argparse.ArgumentParser()
parser.add_argument('--manifest',default='land_stress_confirmation_manifest.json')
parser.add_argument('--output-file',default='land_confirm_115_truth_input.mat')
args=parser.parse_args()
metadata=json.loads((base/args.manifest).read_text())
cells=np.array(metadata.get('cells',metadata.get('selected_test_cells')))
with h5py.File(base/'merged_cohort.h5') as f:
    theta=f['theta'][cells,4:12]
    cai=f['Cai_uM'][cells]
assert len(cells)==64 and theta.shape==(64,8)
savemat(base/args.output_file,
        {'cell_index_zero_based':cells.astype(float),'Cai_uM':cai,
         'theta_true':theta,'target_lambdas':np.array([1.15])},do_compression=True)
print({'cells':len(cells),'theta_shape':theta.shape})
