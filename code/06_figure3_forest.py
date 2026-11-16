"""Figure 3 (final): forest plot of the multi-layer NB GLM, generated from
05_consistency_reanalysis_results.json so it cannot drift from Table 2.
All nine Table 2 layers (including the vegetation condition index), plus same-year
rainfall from the ten-layer model shown separately with an open marker. Log x-axis
because the vegetation estimate (0.25, 0.06-1.02) is far from the others."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
R = json.load(open(OUT / "results" / "05_consistency_reanalysis_results.json"))
plt.rcParams.update({"font.family": "serif", "font.size": 10})

ROWS = [("Chikungunya co-occurring incidence (shared vector)", "Chik_Incidence"),
        ("Female literacy (socioeconomic)", "Female_Literacy"),
        ("Livestock per capita (animal)", "Livestock_pc"),
        ("Poultry per capita (animal)", "Poultry_pc"),
        ("Improved water (socioeconomic)", "Water_Improved"),
        ("Large-ruminant density (animal)", "LargeRuminant_pc"),
        ("Clean-fuel use (socioeconomic)", "Clean_Fuel"),
        ("Vegetation condition index (environment)", "Vegetation_Condition_Index"),
        ("Improved sanitation (socioeconomic)", "Sanitation_Improved")]
data = [(lab, R["multi"][k]) for lab, k in ROWS]
rain = ("Same-year annual rainfall (climate)†", R["ten_layer"]["Rain_mm"])

n = len(data)
ys = list(np.arange(n + 1, 1, -1)) + [0]          # gap between the nine and rainfall
fig, ax = plt.subplots(figsize=(7.2, 4.9))
for (lab, e), y in zip(data + [rain], ys):
    excl = e["lo"] > 1 or e["hi"] < 1
    col = "#de2d26" if excl else "#08519c"
    is_rain = lab.startswith("Same-year")
    ax.errorbar(e["irr"], y, xerr=[[e["irr"] - e["lo"]], [e["hi"] - e["irr"]]],
                fmt="D" if is_rain else "o", mfc="white" if is_rain else col,
                color=col, ecolor=col, capsize=3, lw=1.5, ms=6.5)
    ax.text(3.35, y, f"{e['irr']:.2f} ({e['lo']:.2f}–{e['hi']:.2f})", va="center", fontsize=8.5)
ax.axvline(1.0, color="grey", ls="--", lw=1)
ax.axhline(1, color="lightgrey", lw=0.8)
ax.set_xscale("log")
ax.set_xlim(0.05, 3.2)
ticks = [0.05, 0.1, 0.25, 0.5, 1, 2]
ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()
ax.set_yticks(ys); ax.set_yticklabels([lab for lab, _ in data] + [rain[0]])
ax.set_ylim(-0.8, n + 1.8)
ax.text(3.35, n + 1.5, "IRR (95% CI)", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_xlabel("Incidence rate ratio per 1-SD increase (95% CI, log scale)")
ax.set_title("Multi-layer Negative Binomial model of district dengue incidence, 2021–2025", fontsize=10, pad=14)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.subplots_adjust(right=0.80)
for ext in ("tiff", "png"):
    kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
    fig.savefig(OUT / "figures" / f"Figure3_forest_IRR.{ext}", dpi=720, bbox_inches="tight", **kw)
print("rows:", [(lab[:22], round(e["irr"], 2), round(e["lo"], 2), round(e["hi"], 2)) for lab, e in data + [rain]])
