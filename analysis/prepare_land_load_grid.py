"""Freeze load-grid inputs; true parameters stay in a separate MATLAB file."""
from pathlib import Path
import json
import hashlib
import h5py
import numpy as np
from scipy.io import savemat

root=Path(__file__).resolve().parents[1]
base=root/'analysis/virtual_cardiomyocyte_5000'
selection=json.loads((base/'land_blind_128_selection.json').read_text())
cells=np.array(selection['selected_test_cells'],dtype=int)
lam=np.array([.95,.975,1.105,1.115,1.125,1.15])
with h5py.File(base/'merged_cohort.h5') as f:
    ca=f['Cai_uM'][cells]
    true_theta=f['theta'][cells,4:12]
    assert np.all(f['split_code'][cells]==2)
estimated=[]
for i,cell in enumerate(cells,1):
    with h5py.File(base/'land_blind_128_results'/f'land_refine_test_{i:02d}.mat') as f:
        r=f['result']
        assert int(r['cell_index_zero_based'][()].item())==cell
        estimated.append(r['theta_after'][()].ravel())
estimated=np.stack(estimated)
pred=base/'land_load_grid_predict_input.mat'
truth=base/'land_load_grid_truth_input.mat'
savemat(pred,{'cell_index_zero_based':cells,'Cai_uM':ca,
              'theta_after':estimated,'target_lambdas':lam},do_compression=True)
savemat(truth,{'cell_index_zero_based':cells,'Cai_uM':ca,
               'theta_true':true_theta,'target_lambdas':lam},do_compression=True)
manifest={'cells':len(cells),'target_lambdas':lam.tolist(),
          'calibration_lambdas':[1.,1.1],
          'predict_input_has_no_true_parameters_or_target_traces':True,
          'predict_input_sha256':hashlib.sha256(pred.read_bytes()).hexdigest(),
          'truth_input_sha256':hashlib.sha256(truth.read_bytes()).hexdigest(),
          'no_refitting_or_hyperparameter_selection':True}
(base/'land_load_grid_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
