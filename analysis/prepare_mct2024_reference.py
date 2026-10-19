"""Make a short, non-invasive two-condition runner from the downloaded MCT code.

The native package sweeps dozens of PCLs for 300 beats, assumes a missing
yfinal_3000.mat, writes to ./result, and opens figures. This generated copy
keeps its ODE/equations intact while exposing a single PCL/load condition.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "Data/MCT2024-MCT_v1.0.0/UTokyo-FML-MCT2024-147ccd2/"
          "rabbit_negroni2015_PCLsweep_passive.m")
OUTPUT = ROOT / "analysis/mct_feedback_audit/mct2024_single_condition.m"


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected one occurrence, got {count}: {old[:65]}")
    return source.replace(old, new)


def main() -> None:
    body = SOURCE.read_text(encoding="utf-8-sig")
    body = replace_once(body,
        "function output = rabbit_negroni2015_PCLsweep_passive()",
        "function output = mct2024_single_condition(CL_requested,nbeat,init_file,mct_gain)")
    body = replace_once(body, "p = 0;  % Parameter array for passing nondefault conditions",
                        "p = mct_gain;  % pass feedback gain into the local ODE function")
    body = replace_once(body, "tarray=0.0:1.e-3:1e4;", "tarray=zeros(1,1000000);")
    body = replace_once(body, "PCLspan=[span1 span2];", "PCLspan=1000;")
    body = replace_once(body, "load('yfinal_3000');", "load(init_file,'yfinal','Lneg0','Qkarray0');")
    body = replace_once(body, "nbeat = 300;", "% nbeat is supplied by the caller")
    body = replace_once(body, "CL = 0.10; % closs linker is constant this time.",
                        "CL = CL_requested; % imposed gel crosslinker concentration")
    body = replace_once(body,
        "pCa = 1.0*(0.2*(-d0*gelF*(gelF-d1)+smin)/(d2+d3*(gelF-d1)^2)+1);",
        "pCa = 1 + p*0.2*(-d0*gelF*(gelF-d1)+smin)/(d2+d3*(gelF-d1)^2);")
    start = body.index("%Find peak value of each beat")
    next_fun = body.index("function output = f(", start)
    body = body[:start] + (
        "% Output raw closed-loop traces; skip the original PCL-sweep plots/files.\n"
        "output=struct('t_ms',t,'Vm_mV',y(:,39),'Cai_uM',1000*y(:,38), ...\n"
        "    't_mech_ms',tarray(:),'Lneg',Lneg(:),'force',force(:), ...\n"
        "    'gelForce',gelForce(:),'pCaCoeff',pCaCoeffarray(:), ...\n"
        "    'CL',CL,'PCL_ms',PCL,'nbeat',nbeat,'mct_gain',mct_gain);\n"
        "end\n\n") + body[next_fun:]
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(body, encoding="utf-8")
    print(f"Prepared {OUTPUT}; source ODE remains intact")


if __name__ == "__main__":
    main()
