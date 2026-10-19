"""Expose two controlled parameters in a separate MCT2024 pilot runner.

The validated single-condition reference remains untouched. This derivative
preserves its equations except for multiplying basal L-type Ca permeability
by a caller-provided scale and retaining the already exposed MCT gain.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "analysis/mct_feedback_audit/mct2024_single_condition.m"
DST = ROOT / "analysis/mct_feedback_audit/mct2024_reverse_variant.m"


def replace_once(s: str, a: str, b: str) -> str:
    if s.count(a) != 1:
        raise ValueError(f"Expected exactly one occurrence of {a!r}; got {s.count(a)}")
    return s.replace(a, b)


def main() -> None:
    s = SRC.read_text(encoding="utf-8")
    s = replace_once(s,
        "function output = mct2024_single_condition(CL_requested,nbeat,init_file,mct_gain)",
        "function output = mct2024_reverse_variant(CL_requested,nbeat,init_file,mct_gain,pCa0_scale)")
    s = replace_once(s, "p = mct_gain;  % pass feedback gain into the local ODE function",
                     "p = [mct_gain,pCa0_scale];  % two controlled virtual-cell parameters")
    s = replace_once(s, "pCa0= 5.4e-4;", "pCa0= 5.4e-4*p(2);")
    s = replace_once(s, "pCa = 1 + p*0.2*", "pCa = 1 + p(1)*0.2*")
    s = replace_once(s, "'nbeat',nbeat,'mct_gain',mct_gain);",
                     "'nbeat',nbeat,'mct_gain',mct_gain,'pCa0_scale',pCa0_scale);")
    DST.write_text(s, encoding="utf-8")
    print(DST)


if __name__ == "__main__":
    main()
