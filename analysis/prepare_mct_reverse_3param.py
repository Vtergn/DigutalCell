"""Make a separate 3-parameter MCT runner for robustness development.

Adds a multiplier to native maximal SR Ca uptake (SERCA) without changing the
previously checked two-parameter runner or the original author code.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "analysis/mct_feedback_audit/mct2024_reverse_variant.m"
DST = ROOT / "analysis/mct_feedback_audit/mct2024_reverse_3param.m"


def once(s: str, old: str, new: str) -> str:
    n = s.count(old)
    if n != 1:
        raise ValueError(f"Expected one occurrence, got {n}: {old}")
    return s.replace(old, new)


def main() -> None:
    s = SRC.read_text(encoding="utf-8")
    s = once(s,
        "function output = mct2024_reverse_variant(CL_requested,nbeat,init_file,mct_gain,pCa0_scale)",
        "function output = mct2024_reverse_3param(CL_requested,nbeat,init_file,mct_gain,pCa0_scale,serca_scale)")
    s = once(s, "p = [mct_gain,pCa0_scale];  % two controlled virtual-cell parameters",
             "p = [mct_gain,pCa0_scale,serca_scale];  % three controlled parameters")
    s = once(s, "Vmax_SRCaP = 5.3114e-3;", "Vmax_SRCaP = 5.3114e-3*p(3);")
    s = once(s, "'pCa0_scale',pCa0_scale);",
             "'pCa0_scale',pCa0_scale,'serca_scale',serca_scale);")
    DST.write_text(s, encoding="utf-8")
    print(DST)


if __name__ == "__main__":
    main()
