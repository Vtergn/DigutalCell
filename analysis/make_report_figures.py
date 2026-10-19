"""Create publication-style summary figures for the project briefing."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report_figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                     "axes.titlesize":11, "axes.labelsize":10,
                     "figure.dpi":150, "savefig.dpi":300,
                     "axes.spines.top":False, "axes.spines.right":False})
BLUE, ORANGE, GREEN, RED, GREY = "#2166ac", "#d6604d", "#1b7837", "#762a83", "#666666"

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

def box(ax, x, y, w, h, text, color, fs=10):
    p = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.03,rounding_size=0.08",
                       linewidth=1.5,edgecolor=color,facecolor="white")
    ax.add_patch(p)
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color="#222")

def arrow(ax,x1,y1,x2,y2,color="#555",ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,
                                 linewidth=1.6,color=color,linestyle=ls))

# Figure 1: platform architecture
fig, ax = plt.subplots(figsize=(12,5.2)); ax.set_xlim(0,12); ax.set_ylim(0,6); ax.axis("off")
box(ax,.4,4.35,2.2,.8,"Electrical state\nV(t), ion currents",BLUE)
box(ax,4.9,4.35,2.2,.8,"Ca handling\nCaᵢ(t), TnC",ORANGE)
box(ax,9.4,4.35,2.2,.8,"Mechanical output\nT(t), shortening",GREEN)
arrow(ax,2.6,4.75,4.9,4.75,BLUE); arrow(ax,7.1,4.75,9.4,4.75,ORANGE)
box(ax,4.9,1.85,2.2,.8,"Mechanical input\nλ(t), load protocol",RED)
box(ax,1.0,1.85,2.6,.8,"SAC feedback\nλ → Iₘₑcₕ → V/Ca",RED)
box(ax,8.3,1.85,2.8,.8,"TnC–Ca feedback\nλ → binding → Caᵢ",RED)
arrow(ax,6.0,2.65,2.3,4.35,RED); arrow(ax,6.8,2.65,10.0,4.35,RED)
arrow(ax,3.6,2.25,4.9,2.25,RED); arrow(ax,9.4,2.25,7.1,2.25,RED)
ax.text(6,5.55,"Human bidirectional electromechanical virtual-cell platform",ha="center",fontsize=14,fontweight="bold")
ax.text(6,.85,"Individualization: baseline + mechanical calibration → cell-specific state → blind counterfactual prediction",ha="center",fontsize=11,color="#333")
ax.text(.4,.2,"Solid arrows: implemented coupling. Red boxes: perturbable mechanical feedback paths.",fontsize=8.5,color=GREY)
save(fig,"fig1_platform_architecture")

# Figure 2: core numerical results
fig, axs = plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
labels=["none","TnC only","SAC only","both"]; apd=[0,4,518,519]; ca=[0,-.104,.00028,-.104]
x=np.arange(4); w=.36; ax=axs[0,0]; ax2=ax.twinx()
b1=ax.bar(x-w/2,apd,w,color=BLUE,label="ΔAPD (ms)"); b2=ax2.bar(x+w/2,ca,w,color=ORANGE,label="ΔCa peak (µM)")
ax.set_xticks(x,labels,rotation=22); ax.set_ylabel("ΔAPD (ms)"); ax2.set_ylabel("ΔCa peak (µM)")
ax.set_title("A  Feedback-path ablation at λ=1.10"); ax.set_ylim(-20,570); ax2.set_ylim(-.13,.03)
ax.legend([b1,b2],["ΔAPD","ΔCa peak"],loc="upper left",frameon=False)
names=["Baseline","Selected\nchallenge","Fixed mid\n300 ms","Balanced\nrandom"]; vals=[.2412,.2067,.2074,.2029]
ax=axs[0,1]; bars=ax.bar(names,vals,color=[GREY,BLUE,GREEN,ORANGE]); ax.set_ylabel("13-parameter normalized MAE")
ax.set_title("B  Mechanical challenge adds selective observability"); ax.set_ylim(0,.28)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+.006,f"{v:.3f}",ha="center",fontsize=9)
ax.text(.02,.04,"Feedback-gain MAE: 0.347 → 0.057",transform=ax.transAxes,color=RED,fontsize=9)
names=["Native ODE\n(same model)","Direct RBF\n(target held out)","Native ODE\n10% mismatch"]; vals=[.157,1.860,.265]
ax=axs[1,0]; bars=ax.bar(names,vals,color=[GREEN,ORANGE,RED]); ax.set_ylabel("Standardized future-output MAE")
ax.set_title("C  Blind prediction of unseen mechanical protocols"); ax.set_ylim(0,2.1)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+.05,f"{v:.3f}",ha="center",fontsize=9)
ax.text(.02,.04,"Same-model synthetic benchmark; not real-cell validation",transform=ax.transAxes,color=GREY,fontsize=8.5)
names=["No probe","Fixed C3","Random","Goal / risk"]; vals=[12.04,9.16,10.29,11.09]
ax=axs[1,1]; bars=ax.bar(names,vals,color=[GREY,BLUE,ORANGE,RED]); ax.set_ylabel("Long-hold V MAE (mV)")
ax.set_title("D  Current adaptive-probe pilot"); ax.set_ylim(0,14)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+.25,f"{v:.2f}",ha="center",fontsize=9)
ax.text(.02,.04,"Risk-targeted design did not beat fixed C3",transform=ax.transAxes,color=RED,fontsize=8.5)
save(fig,"fig2_core_results")

# Figure 3: status ladder
fig, ax = plt.subplots(figsize=(11,4.8)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,5)
items=[("Implemented","Bidirectional ToR-ORd/SAC–Land\nplatform + pathway ablation",GREEN),
       ("Supported","Mechanical challenge improves\nfeedback-state observability",BLUE),
       ("Conditional","Mechanistic twin predicts unseen\nloads in same-model synthetic data",ORANGE),
       ("Open","Cross-model and real human-cell\nblind validation",RED)]
for i,(head,body,c) in enumerate(items):
    x=.35+i*2.45; box(ax,x,2.0,2.05,1.25,head,c,11); ax.text(x+1.025,1.62,body,ha="center",va="top",fontsize=8.8,color="#333")
    if i<3: arrow(ax,x+2.1,2.62,x+2.35,2.62,c)
ax.text(5,4.35,"Current project status",ha="center",fontsize=15,fontweight="bold")
ax.text(5,.55,"Engineering platform: complete  |  Scientific digital-twin validation: incomplete",ha="center",fontsize=11,color="#333")
save(fig,"fig3_status_ladder")

# Figure 4: mechanism-resolved time traces.
import csv
fig, axs = plt.subplots(3,2,figsize=(12,8),sharex='col',constrained_layout=True)
configs=[('trpn0_sac0','Both feedbacks off','#777777'),
         ('trpn1_sac0','TnC–Ca feedback only',ORANGE),
         ('trpn0_sac1','SAC feedback only',BLUE),
         ('trpn1_sac1','Both feedbacks on',RED)]
for col,(onset,title) in enumerate([(150,'Early pulse: onset 150 ms'),(600,'Late pulse: onset 600 ms')]):
    t=None
    for suffix,label,color in configs:
        f=ROOT/'human_bidirectional_output'/f'pulse_t{onset}_lam110_{suffix}.csv'
        with f.open(newline='', encoding='utf-8') as h:
            rows=list(csv.DictReader(h))
        t=np.array([float(r['time_ms']) for r in rows])
        vm=np.array([float(r['Vm_mV']) for r in rows])
        cai=np.array([float(r['Cai_uM']) for r in rows])
        if suffix=='trpn0_sac0':
            # prescribed mechanical input: 1 ms ramp, 10 ms hold, 1 ms release
            lam=np.ones_like(t); m=(t>=onset)&(t<onset+1); lam[m]=1+(t[m]-onset)*.1
            m=(t>=onset+1)&(t<onset+11); lam[m]=1.1
            m=(t>=onset+11)&(t<onset+12); lam[m]=1.1-(t[m]-(onset+11))*.1
            axs[0,col].plot(t,lam,color=GREEN,lw=2.2,label='λ(t)')
        axs[1,col].plot(t,vm,color=color,lw=1.15,label=label)
        axs[2,col].plot(t,cai,color=color,lw=1.15,label=label)
    axs[0,col].set_title(title,fontsize=11,fontweight='bold'); axs[0,col].set_ylabel('Length ratio λ')
    axs[1,col].set_ylabel('V (mV)'); axs[2,col].set_ylabel('Caᵢ (µM)'); axs[2,col].set_xlabel('Time (ms)')
    axs[0,col].set_ylim(.98,1.12); axs[0,col].legend(frameon=False,loc='upper right')
    axs[1,col].set_ylim(-100,50); axs[2,col].set_ylim(0,.55)
    for row in range(3): axs[row,col].axvspan(onset,onset+12,color=GREEN,alpha=.08,lw=0)
axs[1,0].legend(frameon=False,fontsize=8,loc='lower right')
fig.suptitle('Mechanistic separation of the two mechanical feedback routes',fontsize=14,fontweight='bold')
fig.text(.5,.005,'The same λ(t) perturbation is applied; changing feedback paths changes which electrical/Ca features move.',ha='center',fontsize=9,color=GREY)
save(fig,'fig4_mechanistic_waveforms')
print(OUT)
