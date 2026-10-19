"""Check native MCT2024 load × feedback contrasts on its own rabbit model."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "mct_feedback_audit"


def load_case(name):
    with h5py.File(OUT / f"mct_{name}_20.mat") as f:
        group = f["a"]
        return {key: group[key][()].ravel() for key in group}


def beats(case, final_beat=20, window=4):
    t, vm, ca = (case[k] for k in ["t_ms", "Vm_mV", "Cai_uM"])
    mt, length, force, stress, pca = (case[k] for k in
        ["t_mech_ms", "Lneg", "force", "gelForce", "pCaCoeff"])
    result = []
    for beat in range(final_beat-window, final_beat):
        selection = (t >= beat*1000) & (t < (beat+1)*1000)
        mechanical = (mt >= beat*1000) & (mt < (beat+1)*1000)
        grid = np.arange(beat*1000, (beat+1)*1000)
        voltage = np.interp(grid, t[selection], vm[selection])
        calcium = ca[selection]
        result.append({
            "beat": beat+1,
            "Ca_peak_uM": float(calcium.max()),
            "Ca_amplitude_uM": float(np.ptp(calcium)),
            "APD_above_minus67_mV_ms": int(np.sum(voltage > -67)),
            "sarcomere_state_excursion": float(np.ptp(length[mechanical])),
            "force_peak_kPa": float(force[mechanical].max()),
            "gel_stress_peak_kPa": float(stress[mechanical].max()),
            "pCa_multiplier_peak": float(pca[mechanical].max()),
        })
    return result


def main() -> None:
    cases = {name: load_case(name) for name in ["lo_on", "lo_off", "hi_on", "hi_off"]}
    per_case = {name: beats(case) for name, case in cases.items()}
    last = {name: rows[-1] for name, rows in per_case.items()}
    fields = ["Ca_peak_uM", "Ca_amplitude_uM", "APD_above_minus67_mV_ms",
              "sarcomere_state_excursion", "force_peak_kPa", "gel_stress_peak_kPa",
              "pCa_multiplier_peak"]
    contrast = {}
    for field in fields:
        contrast[field] = {
            "load_effect_with_MCT_on": last["hi_on"][field] - last["lo_on"][field],
            "load_effect_with_MCT_off": last["hi_off"][field] - last["lo_off"][field],
            "feedback_specific_load_interaction": (
                last["hi_on"][field] - last["lo_on"][field]
                - last["hi_off"][field] + last["lo_off"][field]),
        }
    challenge_amplification = {}
    for field in ["Ca_peak_uM", "APD_above_minus67_mV_ms"]:
        low_feedback_gap = last["lo_on"][field] - last["lo_off"][field]
        high_feedback_gap = last["hi_on"][field] - last["hi_off"][field]
        challenge_amplification[field] = {
            "MCT_on_minus_off_at_low_load": low_feedback_gap,
            "MCT_on_minus_off_at_high_load": high_feedback_gap,
            "high_to_low_gap_ratio": high_feedback_gap / low_feedback_gap,
        }
    result = {
        "species_model": "rabbit Shannon-Bers/Negroni MCT2024; not ToR-ORd/Land",
        "CL_low_high": [0.02, 0.10], "beats": 20, "PCL_ms": 1000,
        "definition": "MCT-off fixes the model's pCa multiplier to one; all other native ODEs remain unchanged",
        "per_case_last_four_beats": per_case,
        "last_beat_load_x_feedback_contrast": contrast,
        "mechanical_challenge_amplification": challenge_amplification,
        "scope": "Model ablation using common initial state. Check last-beat drift before claiming steady-state physiology.",
    }
    (OUT / "native_mct_ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    labels = {"lo_on": "Low load, MCT on", "lo_off": "Low load, MCT off",
              "hi_on": "High load, MCT on", "hi_off": "High load, MCT off"}
    for name, case in cases.items():
        t = case["t_ms"]; mt = case["t_mech_ms"]
        e = (t >= 19000) & (t <= 20000)
        m = (mt >= 19000) & (mt <= 20000)
        style = "-" if name.endswith("on") else "--"
        color = "#e15759" if name.startswith("hi") else "#4e79a7"
        for ax, x, y in [
            (axes[0], t[e], case["Vm_mV"][e]),
            (axes[1], t[e], case["Cai_uM"][e]),
            (axes[2], mt[m], case["Lneg"][m]),
            (axes[3], mt[m], case["gelForce"][m]),
        ]:
            ax.plot((x-19000)/1000, y, style, color=color, lw=1.3, label=labels[name])
    for ax, label in zip(axes, ["Voltage (mV)", "Ca (µM)", "Sarcomere state", "Gel stress (kPa)"]):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend(ncol=2, fontsize=8, frameon=False)
    axes[-1].set_xlabel("Time within beat 20 (s)")
    fig.savefig(OUT / "native_mct_ablation.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"last_beat": last, "contrasts": contrast}, indent=2))


if __name__ == "__main__":
    main()
