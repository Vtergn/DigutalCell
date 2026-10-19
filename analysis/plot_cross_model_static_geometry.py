"""Small transparent plot of the prespecified cross-model geometry result."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

import score_cross_model_static_geometry as score

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'mechchem_cross_model/output/cross_model_static_score.json'
OUT=ROOT/'mechchem_cross_model/output/cross_model_static_summary.png'

result=json.loads(DATA.read_text(encoding='utf-8'))
b,_=score.clean_features(score.B_PATH)
xb,cb,yb=score.vectors(b,score.source_scaling(b))
baseline,challenge=score.neighbors(xb,cb,yb)

fig,ax=plt.subplots(1,2,figsize=(8.2,3.8),layout='constrained')
labels=['ToR-ORd / SAC–Land','ORd–MechChem']
v0=[result['A']['baseline_future_neighbor_distance'],result['B']['baseline_future_neighbor_distance']]
v1=[result['A']['challenge_future_neighbor_distance'],result['B']['challenge_future_neighbor_distance']]
x=np.arange(2)
ax[0].bar(x-.18,v0,.34,color='#8da9c4',label='Baseline')
ax[0].bar(x+.18,v1,.34,color='#de8758',label='Baseline + length challenge')
ax[0].set_xticks(x,labels,rotation=12,ha='right')
ax[0].set_ylabel('Future AP/Ca distance of 5 neighbors')
ax[0].legend(frameon=False,fontsize=8)
ax[0].set_title('Static preload, model-specific scales')
ax[0].spines[['top','right']].set_visible(False)

ax[1].scatter(baseline,challenge,color='#b55345',alpha=.8)
lim=max(float(np.max(baseline)),float(np.max(challenge)))*1.06
ax[1].plot([0,lim],[0,lim],color='0.5',linestyle='--',linewidth=1)
ax[1].set(xlim=(0,lim),ylim=(0,lim),xlabel='Baseline neighbor future distance',
          ylabel='After-challenge neighbor future distance',
          title=f"Independent MechChem cells (n={len(b)})")
ax[1].spines[['top','right']].set_visible(False)
fig.suptitle('Does a mechanical challenge reorganize functional neighborhoods?',fontsize=12)
fig.savefig(OUT,dpi=180)
print(OUT)
