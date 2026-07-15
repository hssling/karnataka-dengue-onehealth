"""
Revision R1 — Supplementary Figure S-spatial.
Panel A: Moran scatterplot of pooled NB Pearson residuals (standardised residual
vs spatial lag), annotated with global Moran's I and permutation p.
Panel B: LISA cluster map of pooled residuals (HH/LL/HL/LH, significant at p<0.05).
"""
import warnings
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local
warnings.filterwarnings("ignore"); np.random.seed(42)

OUT = r"D:\Dengue project\analysis\submission\ohb_special_issue\revision_R1\analysis"
FIGDIR = r"D:\Dengue project\analysis\submission\ohb_special_issue\revision_R1"
gd = gpd.read_file(OUT + r"\spatial_lisa.gpkg")
w = Queen.from_dataframe(gd, use_index=False); w.transform = "r"

y = gd["resid"].values.astype(float)
z = (y - y.mean()) / y.std()
lag = (w.full()[0] @ z)
mi = Moran(y, w, permutations=9999)
lm = Moran_Local(y, w, permutations=9999, seed=42)
sig = lm.p_sim < 0.05
qcol = {1: "#c0392b", 2: "#5dade2", 3: "#2471a3", 4: "#e59866"}  # HH LH LL HL
qname = {1: "High–High", 2: "Low–High", 3: "Low–Low", 4: "High–Low"}
gd["cat"] = np.where(sig, lm.q, 0)

fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))

# Panel A: Moran scatterplot
axA = ax[0]
axA.axhline(0, color="0.7", lw=0.8); axA.axvline(0, color="0.7", lw=0.8)
axA.scatter(z, lag, s=42, c="#34495e", edgecolor="white", zorder=3)
b = np.polyfit(z, lag, 1)[0]
xs = np.linspace(z.min(), z.max(), 50)
axA.plot(xs, b * xs, color="#c0392b", lw=2)
axA.set_xlabel("Standardised NB Pearson residual (z)")
axA.set_ylabel("Spatial lag of residual (W·z)")
axA.set_title("A. Moran scatterplot, model residuals", loc="left", fontweight="bold")
axA.text(0.03, 0.94, f"Global Moran's I = {mi.I:+.3f}\np(9999 perm) = {mi.p_sim:.3f}",
         transform=axA.transAxes, va="top", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9f9", ec="0.7"))

# Panel B: LISA cluster map
axB = ax[1]
gd.plot(ax=axB, color="#ecf0f1", edgecolor="0.6", lw=0.5)
for q in (1, 2, 3, 4):
    sub = gd[gd["cat"] == q]
    if len(sub):
        sub.plot(ax=axB, color=qcol[q], edgecolor="0.3", lw=0.5,
                 label=f"{qname[q]} (n={len(sub)})")
axB.set_title("B. Local Moran (LISA) clusters, p<0.05", loc="left", fontweight="bold")
axB.axis("off")
axB.legend(loc="lower left", fontsize=9, frameon=True, title="Significant clusters")

plt.tight_layout()
for ext, dpi in [("png", 300), ("tiff", 300)]:
    fig.savefig(FIGDIR + rf"\FigureS1_spatial_autocorrelation.{ext}", dpi=dpi, bbox_inches="tight")
print("Saved FigureS1_spatial_autocorrelation.png/.tiff")
print(f"Moran's I={mi.I:+.3f}, p={mi.p_sim:.3f}; significant LISA units={int(sig.sum())}")
