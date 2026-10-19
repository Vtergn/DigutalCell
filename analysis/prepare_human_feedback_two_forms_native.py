"""Set up original-equation twin propagation against two alternate laws."""

from pathlib import Path
import numpy as np

from score_human_feedback_observability import fit_predict

ROOT = Path(__file__).resolve().parent / "human_bidirectional_output"
source = np.loadtxt(ROOT / "feedback_13param_cohort.csv", delimiter=",")
forms = np.loadtxt(ROOT / "feedback_13param_two_forms.csv", delimiter=",")
assert forms.shape == (24, 66) and np.all(forms[:, -1] == 1)
assert np.all(forms[:12, 0] == forms[12:, 0])
source_f = source[:, 16:-1].reshape(64, 12, 8).transpose(0, 2, 1)
source_x = np.c_[source[:, 13:16], source_f[:, 2]]
source_y = source_f[:, 5:8, :][:, :, [0, 2, 3, 6, 8, 10]].reshape(64, -1)
test = forms[:12, 0].astype(int) - 1
train = np.setdiff1d(np.arange(64), test)
assert len(train) == 52
lo = np.array([.9,.85,.85,.85,.9,.8,.4,.75,1.8,4.5,1.8,-4.1,100])
hi = np.array([1.1,1.15,1.15,1.15,1.1,1.2,1.6,.86,2.2,5.5,2.8,-.9,140])
rows = []
for scenario, block in [(2, forms[:12]), (3, forms[12:])]:
    feat = block[:, 17:-1].reshape(12, 4, 12)
    x = np.c_[block[:, 14:17], feat[:, 0]]
    truth = feat[:, 1:, :][:, :, [0, 2, 3, 6, 8, 10]].reshape(12, -1)
    inferred = np.clip(fit_predict(source_x[train], source[train, :13], x), lo, hi)
    direct = fit_predict(source_x[train], source_y[train], x)
    rows.append(np.c_[block[:, 0], np.full(12, scenario), block[:, 1:14],
                      inferred, truth, direct])
out = np.vstack(rows)
assert out.shape == (24, 64)
path = ROOT / "feedback_13param_two_forms_native_input.csv"
np.savetxt(path, out, delimiter=",", fmt="%.12g")
print(path)
