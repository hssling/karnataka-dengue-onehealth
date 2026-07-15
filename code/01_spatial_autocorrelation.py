"""
Revision R1 — Reviewer 2 (spatial dependence).
Assess spatial autocorrelation of the multi-layer NB GLM residuals (global Moran's I,
per-year Moran's I, local Moran / LISA) using queen-contiguity weights from the
Karnataka district shapefile, and run spatial-regression robustness checks (spreg
spatial error and spatial lag on the 2024 cross-section, plus a spatially-lagged
neighbour-incidence covariate added to the panel NB model).

Reuses the exact panel-building code from ../../onehealth_layers_v2.py so that all
descriptive numbers match the original submission.
"""
import os, sys, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local
import spreg
warnings.filterwarnings("ignore"); np.random.seed(42)

OHB = r"D:\Dengue project\analysis\submission\ohb_special_issue"
ROOT = r"D:\Dengue project\Dengue Karnataka 2021-2025"
SHP = ROOT + r"\karnataka_project\data\geospatial\District.shp"
OUT = OHB + r"\revision_R1\analysis"

# --- 1. Rebuild the exact analysis panel (reuse original panel code) ---
src = open(OHB + r"\onehealth_layers_v2.py", encoding="utf-8").read().split("# Full multi-layer model")[0]
ns = {}; exec(src, ns)
p = ns["p"]; layers = ns["layers"]; norm = ns["norm"]; SYN = ns["SYN"]

def nb(formula, data):
    po = smf.glm(formula, data=data, family=sm.families.Poisson(), offset=data["logpop"]).fit()
    bb = ((data["Dengue_Cases"] - po.mu) ** 2 - data["Dengue_Cases"]) / po.mu
    a = max(0.05, smf.ols("bb~mu-1", pd.DataFrame({"bb": bb, "mu": po.mu})).fit().params.iloc[0])
    m = smf.glm(formula, data=data, family=sm.families.NegativeBinomial(alpha=a),
                offset=data["logpop"]).fit(cov_type="cluster", cov_kwds={"groups": data["d"]})
    return m, a

log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

P("="*70); P("REVISION R1 - SPATIAL AUTOCORRELATION (Reviewer 2)"); P("="*70)
P(f"Panel: n={len(p)} district-years, {p.d.nunique()} districts")

# --- 2. Fit the full multi-layer NB model and extract Pearson residuals ---
f = "Dengue_Cases ~ " + " + ".join("z_" + c for c in layers) + " + C(Year)"
m, alpha = nb(f, p)
p = p.copy()
p["mu"] = m.mu
p["pearson_resid"] = (p["Dengue_Cases"] - p["mu"]) / np.sqrt(p["mu"] + alpha * p["mu"]**2)
P(f"\nFull multi-layer NB GLM refitted: n={int(m.nobs)}, alpha={alpha:.3f}")

# --- 3. Load shapefile, harmonise names, build queen-contiguity weights ---
g = gpd.read_file(SHP)
g["d"] = g["KGISDist_1"].map(norm)
# harmonise a few shapefile spellings to the panel's normalised keys
extra = {"chamarajanagara": "chamarajanagar", "bengaluru (rural)": "bengaluru rural",
         "bengaluru (urban)": "bengaluru urban", "vijayanagara": "vijayanagara",
         "kalburgi": "kalaburagi", "kolara": "kolar"}
g["d"] = g["d"].replace(extra)
g = g.dissolve(by="d").reset_index()   # ensure one polygon per district key

panel_d = set(p["d"].unique())
shp_d = set(g["d"].unique())
P(f"\nDistricts in panel: {len(panel_d)}; in shapefile: {len(shp_d)}")
missing = sorted(panel_d - shp_d)
P(f"Panel districts NOT matched to a polygon: {missing if missing else 'none'}")

# district-level mean Pearson residual (pooled) and mean incidence
dist = (p.groupby("d")
          .agg(resid=("pearson_resid", "mean"), incidence=("Incidence", "mean"),
               chik=("Chik_Incidence", "mean"))
          .reset_index())
gd = g.merge(dist, on="d", how="inner").reset_index(drop=True)
P(f"Districts with both geometry and data (spatial analysis set): {len(gd)}")

w = Queen.from_dataframe(gd, use_index=False)
w.transform = "r"
n_islands = len(w.islands)
P(f"Queen-contiguity weights built: {gd.shape[0]} areas, {n_islands} island(s).")

# --- 4. Global Moran's I on pooled mean residuals and on mean incidence ---
def moran_report(y, label):
    mi = Moran(np.asarray(y, float), w, permutations=9999)
    P(f"  Moran's I [{label}]: I={mi.I:+.3f}, E[I]={mi.EI:+.3f}, "
      f"z={mi.z_sim:+.2f}, p(perm)={mi.p_sim:.4f}")
    return mi

P("\n--- Global Moran's I (queen weights, 9999 permutations) ---")
moran_report(gd["resid"], "pooled mean NB Pearson residual")
moran_report(gd["incidence"], "pooled mean raw incidence (reference)")

# --- 5. Per-year Moran's I on residuals ---
P("\n--- Per-year Moran's I on NB Pearson residuals ---")
for yr in sorted(p["Year"].unique()):
    dy = p[p.Year == yr][["d", "pearson_resid"]]
    gy = g.merge(dy, on="d", how="inner").reset_index(drop=True)
    wy = Queen.from_dataframe(gy, use_index=False); wy.transform = "r"
    mi = Moran(gy["pearson_resid"].values.astype(float), wy, permutations=9999)
    P(f"  {int(yr)}: I={mi.I:+.3f}, z={mi.z_sim:+.2f}, p(perm)={mi.p_sim:.4f} (n={len(gy)})")

# --- 6. Local Moran (LISA) on pooled residuals ---
P("\n--- Local Moran's I (LISA) on pooled residuals, hotspot count ---")
lm = Moran_Local(gd["resid"].values.astype(float), w, permutations=9999, seed=42)
sig = lm.p_sim < 0.05
labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
gd["lisa_q"] = np.where(sig, lm.q, 0)
gd["lisa_p"] = lm.p_sim
counts = {labels[k]: int(((lm.q == k) & sig).sum()) for k in labels}
P(f"  Significant local clusters (p<0.05): {counts}  (total sig={int(sig.sum())})")

# --- 7. Spatial-regression robustness on the 2024 cross-section (log incidence) ---
P("\n--- spreg spatial regressions, 2024 cross-section (log dengue incidence) ---")
p24 = p[p.Year == 2024].copy()
g24 = g.merge(p24, on="d", how="inner").reset_index(drop=True)
w24 = Queen.from_dataframe(g24, use_index=False); w24.transform = "r"
Xcols = ["z_Chik_Incidence", "z_LargeRuminant_pc", "z_Livestock_pc",
         "z_Female_Literacy", "z_Vegetation_Condition_Index"]
Y = np.log(g24["Incidence"].values + 1).reshape(-1, 1)
X = g24[Xcols].values
ols = spreg.OLS(Y, X, w=w24, spat_diag=True, moran=True, name_x=Xcols, name_y="log_inc")
P("  OLS diagnostics (Moran's I on OLS residuals, LM tests):")
P(f"    Moran's I (resid): {float(ols.moran_res[0]):+.3f}, p={float(ols.moran_res[2]):.4f}")
P(f"    LM-lag p={ols.lm_lag[1]:.4f}; LM-error p={ols.lm_error[1]:.4f}; "
  f"robust LM-lag p={ols.rlm_lag[1]:.4f}; robust LM-error p={ols.rlm_error[1]:.4f}")
try:
    err = spreg.ML_Error(Y, X, w=w24, name_x=Xcols, name_y="log_inc")
    chik_b = err.betas[1][0]  # first covariate after constant
    P(f"  ML spatial-error model: chik beta={chik_b:+.3f}, lambda={err.betas[-1][0]:+.3f}")
except Exception as e:
    P(f"  ML_Error failed: {e}")
try:
    lag = spreg.ML_Lag(Y, X, w=w24, name_x=Xcols, name_y="log_inc")
    rho = float(np.ravel(lag.rho)[0])
    P(f"  ML spatial-lag model: chik beta={float(lag.betas[1][0]):+.3f}, rho={rho:+.3f}")
except Exception as e:
    P(f"  ML_Lag failed: {e}")

# --- 8. Spatially-lagged neighbour-incidence covariate in the panel NB model ---
P("\n--- Panel NB GLM + spatially-lagged neighbour dengue incidence covariate ---")
# build a full-district weight (one row per district) to compute neighbour means per year
gfull = g[g["d"].isin(panel_d)].reset_index(drop=True)
wf = Queen.from_dataframe(gfull, use_index=False); wf.transform = "r"
idx = {d: i for i, d in enumerate(gfull["d"])}
Wmat = wf.full()[0]
rows = []
for yr, dy in p.groupby("Year"):
    inc_by_d = dict(zip(dy["d"], dy["Incidence"]))
    vec = np.array([inc_by_d.get(d, np.nan) for d in gfull["d"]])
    lagv = Wmat @ np.nan_to_num(vec)   # neighbour-average incidence
    for d, lv in zip(gfull["d"], lagv):
        rows.append((d, yr, lv))
lagdf = pd.DataFrame(rows, columns=["d", "Year", "SpatialLag_Incidence"])
p2 = p.merge(lagdf, on=["d", "Year"], how="left").dropna(subset=["SpatialLag_Incidence"]).copy()
p2["z_SpatialLag_Incidence"] = (p2["SpatialLag_Incidence"] - p2["SpatialLag_Incidence"].mean()) / p2["SpatialLag_Incidence"].std()
f_sl = "Dengue_Cases ~ z_Chik_Incidence + z_SpatialLag_Incidence + C(Year)"
P(f"  (spatial-lag panel n={len(p2)} district-years across {p2.d.nunique()} districts with polygons)")
msl, _ = nb(f_sl, p2)
for v in ["z_Chik_Incidence", "z_SpatialLag_Incidence"]:
    P(f"  {v:26s} IRR={np.exp(msl.params[v]):.3f} "
      f"95%CI[{np.exp(msl.conf_int().loc[v,0]):.3f},{np.exp(msl.conf_int().loc[v,1]):.3f}] "
      f"p={msl.pvalues[v]:.4f}")

# save residual/LISA table for the figure script
gd[["d", "resid", "incidence", "chik", "lisa_q", "lisa_p"]].to_csv(OUT + r"\spatial_residuals.csv", index=False)
gd.to_file(OUT + r"\spatial_lisa.gpkg", driver="GPKG")
open(OUT + r"\01_spatial_results.txt", "w", encoding="utf-8").write("\n".join(log))
P("\nSaved: spatial_residuals.csv, spatial_lisa.gpkg, 01_spatial_results.txt")
P("DONE")
