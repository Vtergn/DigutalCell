"""Package the coupled sarcomere/electrical results for an offline viewer."""
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/"sarcomere_closed_loop_viewer"


def pack(array, digits):
    return np.round(np.asarray(array,dtype=float),digits).tolist()


def main():
    OUT.mkdir(exist_ok=True)
    native=np.load(HERE/"sarcomere_closed_loop.npz")
    stochastic=np.load(HERE/"sarcomere_stochastic_cosim.npz")
    names={
        "closed_pulse":"闭环 + 机械脉冲",
        "no_external_pulse":"闭环 + 无机械脉冲",
        "pulse_no_sac":"机械脉冲 + 关闭 SAC",
        "pulse_no_motor":"机械脉冲 + 关闭 Ca→肌节驱动",
        "pulse_no_mechanical_feedback":"机械脉冲 + 关闭机械→电/Ca",
        "stochastic":"闭环 + 原论文随机肌节动力学",
    }
    cases={}
    for key,name in names.items():
        if key=="stochastic":
            state=stochastic["state"]
            lam=stochastic["lambda_mean"]
            local=stochastic["local_lambda"]
            tension=None
            time=stochastic["time_ms"]
        else:
            state=native[f"{key}_state"][:461]
            lam=native[f"{key}_lambda"][:461]
            local=native[f"{key}_local_lambda"][:461]
            tension=native[f"{key}_tension_kpa"][:461]
            time=np.arange(461)
        if state.shape!=(461,90) or local.shape!=(461,20):
            raise RuntimeError(f"unexpected state shape for {key}")
        if not (np.isfinite(state).all() and np.isfinite(lam).all()
                and np.isfinite(local).all()):
            raise RuntimeError(f"non-finite output in {key}")
        cases[key]={"name":name,"time_ms":pack(time,0),
                    "V_mV":pack(state[:,0],3),
                    "Ca_uM":pack(1000*state[:,5],5),
                    "lambda":pack(lam,5),
                    "local_lambda":pack(local,5),
                    "tension_kPa":pack(tension,3) if tension is not None else None,
                    "pulse":key!="no_external_pulse",
                    "sac":key!="pulse_no_sac" and key!="pulse_no_mechanical_feedback",
                    "motor":key!="pulse_no_motor",
                    "feedback":key!="pulse_no_mechanical_feedback",
                    "stochastic":key=="stochastic"}
    payload={"scenarios":cases,
             "source":"Härtter sarcomere model (MIT) + ToR-ORd/SAC-Land human cell",
             "limits":"Motor activation and length conversion are exploratory; no measured force-unit bridge or paired AP/Ca/sarcomere validation."}
    (OUT/"viewer_data.js").write_text("window.SPATIAL_LOOP_DATA="+
        json.dumps(payload,ensure_ascii=False,separators=(",",":"))+";\n",encoding="utf-8")
    report={"scenario_count":len(cases),"frames":461,
            "all_scenarios_20_sarcomeres":True,
            "stochastic_data_present":True,
            "viewer_is_physics_output_not_experimental_reconstruction":True}
    (OUT/"validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
