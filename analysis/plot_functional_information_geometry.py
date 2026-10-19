"""Compact figure for the exploratory functional-information feasibility test."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent/"human_bidirectional_output"
R=json.loads((ROOT/"feedback_functional_information_geometry.json").read_text())
source=R["source_neighbors"]
pred=R["predictive_latent_5fold"]
cross=R["source_trained_latent_cross_form"]

plt.rcParams.update({"font.size":9,"axes.spines.top":False,
                     "axes.spines.right":False,"figure.dpi":160})
fig,axs=plt.subplots(1,3,figsize=(11.8,3.4),layout="constrained")

labels=["Rest","150 ms","300 ms","600 ms","300+600"]
keys=[None,"early150_110","mid300_110","late600_110","300+600"]
vals=[source["baseline"]]+[source["conditions"][k]["future_neighbor_distance"] for k in keys[1:]]
color=["#8c9ba6"]+["#4386a4"]*3+["#c27b39"]
axs[0].bar(np.arange(len(vals)),vals,color=color,width=.7)
axs[0].set_xticks(np.arange(len(vals)),labels,rotation=30,ha="right")
axs[0].set_ylabel("Future distance of 5 nearest neighbours")
axs[0].set_title("A  Observational neighbourhoods")
axs[0].set_ylim(0,1.3)

for name,col,marker in [("baseline","#8c9ba6","o"),
                        ("mid300","#4386a4","s"),
                        ("mid+late","#c27b39","^")]:
    dims=[d for d in [1,2,3,5] if f"PLS{d}" in pred[name]]
    axs[1].plot(dims,[pred[name][f"PLS{d}"] for d in dims],
                color=col,marker=marker,label=name)
axs[1].set_xticks([1,2,3,5])
axs[1].set_xlabel("Predictive latent coordinates (PLS)")
axs[1].set_ylabel("Held-out future-response L2 error")
axs[1].set_title("B  Low-dimensional functional state")
axs[1].legend(frameon=False)

conditions=["Source","Gain exponent","SAC saturation","TnC capacity"]
base=[cross["baseline"]["source_test_L2"],
      cross["baseline"]["gain_exponent_test_L2"],
      cross["baseline"]["saturating_SAC_test_L2"],
      cross["baseline"]["reduced_TnC_buffer_test_L2"]]
challenge=[cross["mid300"]["source_test_L2"],
           cross["mid300"]["gain_exponent_test_L2"],
           cross["mid300"]["saturating_SAC_test_L2"],
           cross["mid300"]["reduced_TnC_buffer_test_L2"]]
x=np.arange(4); width=.36
axs[2].bar(x-width/2,base,width,label="Rest only",color="#8c9ba6")
axs[2].bar(x+width/2,challenge,width,label="Rest + challenge",color="#4386a4")
axs[2].set_xticks(x,conditions,rotation=30,ha="right")
axs[2].set_ylabel("Source-trained latent prediction error")
axs[2].set_title("C  Transfer across controlled laws")
axs[2].legend(frameon=False,loc="upper left")
fig.suptitle("Exploratory functional information geometry (64 synthetic cells)",y=1.05,fontsize=11)
for ext in ["png","svg"]:
    fig.savefig(ROOT/f"feedback_functional_information_geometry.{ext}",bbox_inches="tight")
print(ROOT/"feedback_functional_information_geometry.png")
