"""Generate OHB figures (TIFF) from the verified analysis panel."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 10})

OUT = r"D:\Dengue project\analysis\submission\ohb_special_issue"
panel = pd.read_csv(OUT + r"\..\jch_karnataka_onehealth\analysis_panel.csv")

def save(fig, name):
    for ext, dpi in [("tiff", 720), ("png", 720)]:
        try:
            fig.savefig(f"{OUT}\\{name}.{ext}", dpi=dpi, bbox_inches="tight",
                        pil_kwargs={"compression": "tiff_lzw"} if ext == "tiff" else None)
        except Exception as e:
            fig.savefig(f"{OUT}\\{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# Figure 1: state burden (bars) + mean district incidence (line)
years = [2021, 2022, 2023, 2024, 2025]
state = [7393, 9889, 19300, 32886, 7325]
inc = [11.5, 14.6, 16.3, 33.7, 6.7]
fig, ax1 = plt.subplots(figsize=(6.5, 4))
ax1.bar(years, state, color="#9ecae1", edgecolor="#3182bd", label="State dengue cases")
ax1.set_xlabel("Year"); ax1.set_ylabel("State dengue cases")
ax1.set_xticks(years)
for x, v in zip(years, state):
    ax1.text(x, v + 600, f"{v:,}", ha="center", fontsize=8)
ax2 = ax1.twinx()
ax2.plot(years, inc, "o-", color="#de2d26", lw=2, label="Mean district incidence")
ax2.set_ylabel("Mean district incidence (/100,000)", color="#de2d26")
ax2.tick_params(axis="y", labelcolor="#de2d26")
ax1.set_title("Karnataka dengue burden and mean district incidence, 2021-2025")
save(fig, "Figure1_burden_trend")

# Figure 2: district dengue incidence 2024 (ranked horizontal bars)
d24 = panel[panel.Year == 2024][["d", "Incidence"]].sort_values("Incidence")
fig, ax = plt.subplots(figsize=(6.5, 7))
ax.barh(d24["d"].str.title(), d24["Incidence"], color="#74c476", edgecolor="#31a354")
ax.set_xlabel("Dengue incidence, 2024 (/100,000)")
ax.set_title("District dengue incidence in the 2024 surge (n=29 districts)")
ax.margins(y=0.01)
save(fig, "Figure2_district_incidence_2024")

# Figure 3: forest plot of multi-layer One Health NB GLM incidence rate ratios.
# Chikungunya co-incidence (shared-vector layer) is the one signal above 1.
labels = ["Chikungunya co-incidence", "Female literacy", "Livestock per capita",
          "Poultry per capita", "Improved water", "Large-ruminant density",
          "Clean-fuel use", "Improved sanitation"]
irr = [1.121, 1.260, 1.094, 1.029, 0.936, 0.928, 0.904, 0.994]
lo  = [1.030, 0.919, 0.961, 0.880, 0.736, 0.791, 0.733, 0.667]
hi  = [1.220, 1.728, 1.247, 1.203, 1.190, 1.088, 1.116, 1.482]
colors = ["#de2d26"] + ["#08519c"] * (len(labels) - 1)  # highlight chikungunya
y = np.arange(len(labels))[::-1]
fig, ax = plt.subplots(figsize=(6.8, 4.2))
for yi, m, l, h, c in zip(y, irr, lo, hi, colors):
    ax.errorbar(m, yi, xerr=[[m - l], [h - m]], fmt="o", color=c, ecolor=c,
                capsize=3, lw=1.6, ms=7)
ax.axvline(1.0, color="grey", ls="--", lw=1)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel("Incidence rate ratio (95% CI), multi-layer Negative Binomial GLM")
ax.set_title("Shared-vector ecology, not animal or climate proxies,\npredicts district dengue incidence")
ax.set_xlim(0.6, 1.9)
save(fig, "Figure3_forest_IRR")

print("Figures written:", d24.shape, "districts in Fig2")
