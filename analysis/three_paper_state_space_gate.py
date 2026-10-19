"""Frozen exploratory audit: continuum, stable components, and physical gradients.

The 64-cell source cohort defines all scaling and PCA axes. The independent
64-cell confirmation cohort is scored without refitting those axes. These are
synthetic cells from the same equations, so this is a feasibility audit only.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
ROOT = HERE / "human_bidirectional_output"
PARAM = ["CaL", "SERCA", "NCX", "IKr", "IK1", "INaL", "g_ionic",
         "ca50", "TRPN_n", "nperm", "beta0", "beta1", "Tref"]
BF = np.array([.1, .001, .1])
CF = np.array([.1,.1,.1,.001,.001,.1,.1,.1,.1,.1,.0005,.0005])


def unpack(path: Path):
    a = np.loadtxt(path, delimiter=",")
    assert a.shape == (64, 113) and np.all(a[:, -1] == 1)
    f = a[:, 16:-1].reshape(64, 12, 8).transpose(0, 2, 1)
    return a[:, :13], a[:, 13:16], f[:, :5]


ta, ba, ca = unpack(ROOT / "feedback_13param_cohort.csv")
tb, bb, cb = unpack(ROOT / "feedback_13param_confirm_cohort.csv")


def observation(base, challenge):
    # The same 300-ms mechanical calibration and frozen block weighting as the
    # existing functional-neighborhood analysis.
    bsd = ba.std(0); bkeep = bsd >= BF
    b = (base[:, bkeep] - ba[:, bkeep].mean(0)) / np.maximum(bsd[bkeep], BF[bkeep])
    csd = ca[:, 2].std(0); ckeep = csd >= CF
    c = (challenge[:, 2, ckeep] - ca[:, 2, ckeep].mean(0)) / np.maximum(csd[ckeep], CF[ckeep])
    return np.concatenate([b / np.sqrt(b.shape[1]), c / np.sqrt(c.shape[1])], axis=1)


xa, xb = observation(ba, ca), observation(bb, cb)
pca = PCA(n_components=5).fit(xa)
za, zb = pca.transform(xa), pca.transform(xb)


def balanced_mst_gap(x, min_group=8):
    """Largest MST edge that divides at least min_group cells on each side."""
    d = cdist(x, x)
    tree = minimum_spanning_tree(d).tocoo()
    edges = list(zip(tree.row.tolist(), tree.col.tolist(), tree.data.tolist()))
    median = float(np.median([e[2] for e in edges]))
    graph = nx.Graph(); graph.add_nodes_from(range(len(x)))
    graph.add_weighted_edges_from(edges)
    candidates = []
    for i, j, w in edges:
        graph.remove_edge(i, j)
        sizes = sorted(len(c) for c in nx.connected_components(graph))
        if len(sizes) == 2 and sizes[0] >= min_group:
            candidates.append((w / median, sizes[0], sizes[1]))
        graph.add_edge(i, j, weight=w)
    return max(candidates, default=(0., 0, 0))


rng = np.random.default_rng(20261018)
gaps = {}
for name, z in [("source", za), ("confirmation", zb)]:
    actual = balanced_mst_gap(z)
    gaussian = rng.multivariate_normal(z.mean(0), np.cov(z, rowvar=False),
                                        size=(500, len(z)))
    null = np.array([balanced_mst_gap(v)[0] for v in gaussian])
    gaps[name] = {"balanced_mst_edge_over_median": float(actual[0]),
                  "cut_group_sizes": [int(actual[1]), int(actual[2])],
                  "covariance_matched_gaussian_95pct": float(np.quantile(null, .95)),
                  "null_tail_fraction": float(np.mean(null >= actual[0]))}

mixtures = []
for k in range(1, 6):
    g = GaussianMixture(n_components=k, covariance_type="full", reg_covar=.01,
                        n_init=10, random_state=42).fit(za)
    mixtures.append({"k": k, "source_bic": float(g.bic(za)),
                     "confirmation_log_density_per_cell": float(g.score(zb)),
                     "confirmation_component_counts": np.bincount(
                         g.predict(zb), minlength=k).tolist()})

# A gradient must be interpretable on an untouched parameter-draw cohort.
rho_a = np.array([[spearmanr(za[:, i], ta[:, j]).statistic
                   for j in range(13)] for i in range(5)])
rho_b = np.array([[spearmanr(zb[:, i], tb[:, j]).statistic
                   for j in range(13)] for i in range(5)])
best = np.unravel_index(np.argmax(abs(rho_a)), rho_a.shape)
permutation_max = []
for _ in range(1000):
    tp = ta[rng.permutation(len(ta))]
    vals = [abs(spearmanr(za[:, i], tp[:, j]).statistic)
            for i in range(5) for j in range(13)]
    permutation_max.append(max(vals))
gradient = {"source_selected_pc": int(best[0] + 1),
            "source_selected_parameter": PARAM[best[1]],
            "source_spearman": float(rho_a[best]),
            "confirmation_spearman_frozen_axes": float(rho_b[best]),
            "source_familywise_permutation_p": float((1 + np.sum(
                np.array(permutation_max) >= abs(rho_a[best]))) / 1001),
            "all_source_and_confirmation_correlations": [
                {"pc": i+1, "parameter": PARAM[j],
                 "source": float(rho_a[i,j]), "confirmation": float(rho_b[i,j])}
                for i in range(5) for j in range(13)]}

# Negative control: at baseline lambda=1, Land beta1 is absent from the
# length-dependent pathway. Compare with a source-trained baseline space.
bkeep = ba.std(0) >= BF
base_a = (ba[:,bkeep] - ba[:,bkeep].mean(0)) / np.maximum(ba[:,bkeep].std(0), BF[bkeep])
base_b = (bb[:,bkeep] - ba[:,bkeep].mean(0)) / np.maximum(ba[:,bkeep].std(0), BF[bkeep])
base_pca = PCA(n_components=base_a.shape[1]).fit(base_a)
base_za, base_zb = base_pca.transform(base_a), base_pca.transform(base_b)
gradient["beta1_baseline_pc_correlations_source"] = [float(spearmanr(base_za[:,i], ta[:,11]).statistic)
                                                    for i in range(base_za.shape[1])]
gradient["beta1_baseline_pc_correlations_confirmation"] = [float(spearmanr(base_zb[:,i], tb[:,11]).statistic)
                                                          for i in range(base_zb.shape[1])]

out = {"note": "Exploratory same-equation synthetic geometry; not a measured biological topology.",
       "source_cells": 64, "confirmation_cells": 64,
       "observation_features": int(xa.shape[1]),
       "pca_explained_variance_ratio_source": pca.explained_variance_ratio_.tolist(),
       "balanced_h0_proxy": gaps, "gaussian_mixtures": mixtures,
       "physical_gradient": gradient}
dest = HERE / "three_paper_state_space_gate.json"
dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
for panel, z, theta, title in [(ax[0], za, ta, "Source: 64 virtual cells"),
                               (ax[1], zb, tb, "Confirmation: 64 virtual cells")]:
    scatter = panel.scatter(z[:,0], z[:,3], c=theta[:,11], cmap="viridis",
                            vmin=-4.1, vmax=-.9, s=28, alpha=.85)
    panel.set(xlabel="PC1 (source trained)", ylabel="PC4", title=title)
fig.colorbar(scatter, ax=[ax[0],ax[1]], label="Land beta1", shrink=.85)
ks = [r["k"] for r in mixtures]
ax[2].plot(ks, [r["confirmation_log_density_per_cell"] for r in mixtures], "o-")
ax[2].set(xlabel="Gaussian mixture components", ylabel="Held-out log density / cell",
          title="Density fit (not state count)")
fig.savefig(HERE / "three_paper_state_space_gate.png", dpi=190)
print(json.dumps({"gaps": gaps, "mixtures": mixtures, "gradient": {
    k:v for k,v in gradient.items() if k!="all_source_and_confirmation_correlations"},
    "pca_var": out["pca_explained_variance_ratio_source"]}, indent=2))
