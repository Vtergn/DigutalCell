"""Piezo1 eLife T50/k interquartile sensitivity under whole-date holdout."""

import csv
import json
from collections import defaultdict

import numpy as np

from causal_cnn_benchmark import metrics
from dynamic_input_screen import HERE
from mechanistic_tension_personalization import LAG_MS, lagged
from mechanistic_tension_simulator import assemble_records, physical_parameters, predict, select_shared


def main():
    source = json.loads((HERE / "external_physics_bridge_results.json").read_text(encoding="utf-8"))["elife"]
    t = source["t50_mN_per_m"]
    k = source["slope_mN_per_m"]
    settings = [("median",t["median"],k["median"])] + [
        (f"T50_{a}_k_{b}",t[a],k[b]) for a in ("q25","q75") for b in ("q25","q75")
    ]
    base = physical_parameters()
    rows = [r for r in assemble_records() if r["channel"] == "mp1"]
    scores=[]
    for label,t50,slope in settings:
        physics={**base,"elife_piezo1_T50_mN_per_m":t50,"elife_piezo1_k_mN_per_m":slope}
        for date in sorted({r["date"] for r in rows}):
            train=[r for r in rows if r["date"]!=date]
            test=[r for r in rows if r["date"]==date]
            selected=select_shared(train,physics)
            for row in test:
                shifted={**row,"force":lagged(row["force"],LAG_MS["mp1"])}
                wave=predict(shifted,selected["selected_shared_T0_mN_per_m"],
                             selected["selected_ETH_transfer_multiplier"],physics)
                scores.append({"prior_setting":label,"T50_mN_per_m":t50,"k_mN_per_m":slope,
                               "heldout_date":date,"cell_id":row["cell_id"],"sweep":row["sweep"],
                               "shared_T0_mN_per_m":selected["selected_shared_T0_mN_per_m"],
                               "ETH_multiplier":selected["selected_ETH_transfer_multiplier"],
                               **metrics(row,wave)})
    with (HERE / "mechanistic_tension_prior_sensitivity_scores.csv").open(
            "w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(scores[0]))
        writer.writeheader();writer.writerows(scores)
    for label,_,_ in settings:
        group=defaultdict(list)
        for row in scores:
            if row["prior_setting"]==label:group[row["cell_id"]].append(float(row["mae_pA"]))
        print(label,"median cell MAE",round(float(np.median([np.mean(v) for v in group.values()])),2),
              "T0 choices",sorted({row["shared_T0_mN_per_m"] for row in scores
                                   if row["prior_setting"]==label}))


if __name__=="__main__":
    main()
