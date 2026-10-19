"""Publication-style comparison of native MATLAB and optimized PyTorch beats."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/"human_bidirectional_output"

fig, axes=plt.subplots(3,2,figsize=(10.5,7.8),sharex=True,constrained_layout=True)
colors=("#234f77","#b6512e","#28806c")
fields=("Vm_mV","Cai_uM","Ta_kPa")
labels=("Membrane voltage (mV)","Free Ca (µM)","Active tension (kPa)")
pred_fields=("predicted_voltage","predicted_cai","predicted_active_tension")
for col,(code,length) in enumerate((("100",1.00),("110",1.10))):
    ref=pd.read_csv(OUT/f"trpn1_sac1_lambda{code}.csv")
    pred=np.load(HERE/f"full_beat_validation_lambda{code}.npz")
    t=ref.iloc[:,0].to_numpy()
    for row,(field,label,pfield,color) in enumerate(zip(fields,labels,pred_fields,colors)):
        ax=axes[row,col]
        ax.plot(t,ref[field],color=color,lw=2.0,label="MATLAB ode15s")
        ax.plot(pred["time"],pred[pfield],color="#20252b",lw=1.15,ls="--",label="PyTorch RHS + BDF")
        ax.set_ylabel(label if col==0 else "")
        ax.grid(alpha=.16)
        if row==0:
            ax.set_title(f"Sustained length λ = {length:.2f}",fontsize=11,weight="bold")
        if row==2:
            ax.set_xlabel("Time (ms)")
axes[0,0].legend(frameon=False,loc="upper right",fontsize=8)
fig.suptitle("50-state electromechanical model: full-beat numerical agreement",fontsize=13,weight="bold")
fig.savefig(HERE/"full_beat_validation.png",dpi=220,bbox_inches="tight")
fig.savefig(HERE/"full_beat_validation.pdf",bbox_inches="tight")
plt.close(fig)
