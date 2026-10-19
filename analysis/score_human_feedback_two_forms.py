"""Score original-law twins on altered SAC and TnC feedback equations."""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent / "human_bidirectional_output"
inp = np.loadtxt(ROOT / "feedback_13param_two_forms_native_input.csv", delimiter=",")
native = np.loadtxt(ROOT / "feedback_13param_two_forms_native_predictions.csv", delimiter=",")
source = np.loadtxt(ROOT / "feedback_13param_native_input.csv", delimiter=",")
assert inp.shape == (24, 64) and native.shape == (24, 22)
assert np.allclose(inp[:, :2], native[:, :2]) and np.all(native[:, 3] == 1)
floor = np.array([.1,.1,.001,.1,.1,.0005])
source_truth = source[source[:, 1] == 0, 28:46].reshape(12,3,6)
scale = source_truth.std(axis=0)
active = scale >= floor


def summary(pred, truth):
    e = np.abs(pred-truth)
    es = e/np.maximum(scale, floor)
    per = (es*active).sum(axis=(0,2))/(len(pred)*active.sum(axis=1))
    return {"mean": float(per.mean()), "per_target": per.tolist(),
            "physical_MAE": e.mean(axis=0).tolist()}


out = {"n_test_cells":12,
       "truth_models": {"saturating_SAC": "Effective ionic length saturates with scale 0.05, normalized to match linear law at lambda 1.10.",
                        "reduced_TnC_buffer": "Dynamic troponin calcium-buffering capacity is 75% of source; alternate cells were repaced."},
       "warning": "These are two controlled alternate laws in the same model framework, not independent published electromechanical truth models.",
       "results": {}}
for label, scenario in [("saturating_SAC",2),("reduced_TnC_buffer",3)]:
    mask=inp[:,1]==scenario
    truth=inp[mask,28:46].reshape(12,3,6)
    out["results"][label]={
        "native_ODE":summary(native[mask,4:22].reshape(12,3,6),truth),
        "direct_RBF":summary(inp[mask,46:64].reshape(12,3,6),truth),
        "parameter_normalized_MAE":float(np.mean(np.abs(inp[mask,15:28]-inp[mask,2:15])/
            np.array([.2,.3,.3,.3,.2,.4,1.2,.11,.4,1,1,3.2,40]))),
    }
path=ROOT/"feedback_13param_two_forms_score.json"
path.write_text(json.dumps(out,indent=2),encoding="utf-8")
for key,value in out["results"].items():
    print(key,"native",value["native_ODE"]["mean"],
          "direct RBF",value["direct_RBF"]["mean"])
