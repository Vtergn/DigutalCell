"""Create a numerical-sensitivity runner with tighter ODE settings only."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "analysis/mct_feedback_audit"
src = ROOT / "mct2024_reverse_3param.m"
dst = ROOT / "mct2024_reverse_3param_tight.m"
s = src.read_text(encoding="utf-8")
old = "function output = mct2024_reverse_3param(CL_requested,nbeat,init_file,mct_gain,pCa0_scale,serca_scale)"
new = "function output = mct2024_reverse_3param_tight(CL_requested,nbeat,init_file,mct_gain,pCa0_scale,serca_scale)"
if s.count(old) != 1:
    raise ValueError("Function signature mismatch")
s = s.replace(old, new)
old = "odeset('RelTol',1e-5,'MaxStep',1,'Stats','on')"
new = "odeset('RelTol',1e-6,'MaxStep',0.5,'Stats','on')"
if s.count(old) != 1:
    raise ValueError("ODE option mismatch")
dst.write_text(s.replace(old, new), encoding="utf-8")
print(dst)
