"""
Revision R1 — Reviewer 3 (temporal scale / lagged climate).
Routine district data for Karnataka are available only at ANNUAL resolution
(annual district dengue counts; annual district rainfall total and departure from
normal, 2021-2024). No monthly dengue or monthly rainfall series exist in the
routine record, so sub-annual seasonality and short lags cannot be modelled.
Within the annual data we test (a) same-year rainfall, (b) prior-year (t-1) lagged
rainfall, and (c) rainfall departure from normal, to probe whether an annual
climate signal is present at all.
"""
import warnings, glob
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
warnings.filterwarnings("ignore"); np.random.seed(42)

OHB = r"D:\Dengue project\analysis\submission\ohb_special_issue"
ROOT = r"D:\Dengue project\Dengue Karnataka 2021-2025"
OUT = OHB + r"\revision_R1\analysis"

src = open(OHB + r"\onehealth_layers_v2.py", encoding="utf-8").read().split("# Full multi-layer model")[0]
ns = {}; exec(src, ns)
p = ns["p"].copy(); norm = ns["norm"]

log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

def nb(formula, data, var):
    po = smf.glm(formula, data=data, family=sm.families.Poisson(), offset=data["logpop"]).fit()
    bb = ((data["Dengue_Cases"] - po.mu) ** 2 - data["Dengue_Cases"]) / po.mu
    a = max(0.05, smf.ols("bb~mu-1", pd.DataFrame({"bb": bb, "mu": po.mu})).fit().params.iloc[0])
    m = smf.glm(formula, data=data, family=sm.families.NegativeBinomial(alpha=a),
                offset=data["logpop"]).fit(cov_type="cluster", cov_kwds={"groups": data["d"]})
    return (np.exp(m.params[var]), np.exp(m.conf_int().loc[var, 0]),
            np.exp(m.conf_int().loc[var, 1]), m.pvalues[var], int(m.nobs))

P("="*70); P("REVISION R1 - ANNUAL / LAGGED CLIMATE (Reviewer 3)"); P("="*70)

# --- Load annual rainfall (actual + departure) for 2021-2024 ---
def load_rain(path, year):
    r = pd.read_csv(path); r.columns = [str(c).replace("﻿", "").strip() for c in r.columns]
    low = {c: c.lower() for c in r.columns}
    def pick(*k):
        for key in k:
            for c, l in low.items():
                if key in l: return c
    cd = pick("district"); ct = pick("total actual", "annual actual", "actual")
    cdep = pick("total departure", "annual departure", "%dep", "departure")
    r = r[r[cd].astype(str).str.upper().str.strip() != "STATE"]
    out = pd.DataFrame({"d": r[cd].map(norm),
                        "Rain_mm": pd.to_numeric(r[ct], errors="coerce"),
                        "Rain_dep": pd.to_numeric(r[cdep], errors="coerce") if cdep else np.nan,
                        "Year": year}).dropna(subset=["d"])
    return out

rain = pd.concat([load_rain(glob.glob(ROOT + rf"\Karnataka district rainfall\*{y}*.csv")[0], y)
                  for y in (2021, 2022, 2023, 2024)]).drop_duplicates(["d", "Year"])
P(f"Annual rainfall records: {len(rain)} district-years (2021-2024). "
  f"Departure available: {rain['Rain_dep'].notna().sum()}.")
P("NOTE: No monthly dengue or monthly rainfall series exist in the routine record; "
  "sub-annual seasonality and short (weekly/monthly) lags cannot be estimated.")

# lagged rainfall: rain of year (t-1) matched to dengue of year t
rain_lag = rain.copy(); rain_lag["Year"] = rain_lag["Year"] + 1
rain_lag = rain_lag.rename(columns={"Rain_mm": "Rain_mm_lag1", "Rain_dep": "Rain_dep_lag1"})[
    ["d", "Year", "Rain_mm_lag1", "Rain_dep_lag1"]]

p = p.drop(columns=[c for c in ["Rain_mm", "Rain_dep"] if c in p.columns])
pm = (p.merge(rain[["d", "Year", "Rain_mm", "Rain_dep"]], on=["d", "Year"], how="left")
        .merge(rain_lag, on=["d", "Year"], how="left"))

def zc(df, col):
    df = df.dropna(subset=[col]).copy()
    df["z_" + col] = (df[col] - df[col].mean()) / df[col].std()
    return df

P("\n--- Annual climate NB GLM (log-pop offset, year FE, cluster SE) ---")
for col, lab in [("Rain_mm", "Same-year rainfall (mm)"),
                 ("Rain_dep", "Same-year rainfall departure (%)"),
                 ("Rain_mm_lag1", "Prior-year (t-1) rainfall (mm)"),
                 ("Rain_dep_lag1", "Prior-year (t-1) rainfall departure (%)")]:
    d = zc(pm, col)
    irr, lo, hi, pv, n = nb(f"Dengue_Cases ~ z_{col} + C(Year)", d, f"z_{col}")
    P(f"  {lab:42s} IRR={irr:.3f} 95%CI[{lo:.3f},{hi:.3f}] p={pv:.3f} (n={n})")

# --- Adjusted rainfall: within the full multi-layer model (matches manuscript framing) ---
P("\n--- Rainfall adjusted for the other One Health layers (multivariable) ---")
layers = ns["layers"]
d = zc(pm, "Rain_mm")
for c in layers:
    d["z_" + c] = (d[c] - d[c].mean()) / d[c].std()
f_adj = "Dengue_Cases ~ z_Rain_mm + " + " + ".join("z_" + c for c in layers) + " + C(Year)"
irr, lo, hi, pv, n = nb(f_adj, d, "z_Rain_mm")
P(f"  Same-year rainfall, fully adjusted        IRR={irr:.3f} 95%CI[{lo:.3f},{hi:.3f}] p={pv:.3f} (n={n})")
d2 = zc(pm, "Rain_mm_lag1")
for c in layers:
    d2["z_" + c] = (d2[c] - d2[c].mean()) / d2[c].std()
f_adj2 = "Dengue_Cases ~ z_Rain_mm_lag1 + " + " + ".join("z_" + c for c in layers) + " + C(Year)"
irr, lo, hi, pv, n = nb(f_adj2, d2, "z_Rain_mm_lag1")
P(f"  Prior-year rainfall, fully adjusted       IRR={irr:.3f} 95%CI[{lo:.3f},{hi:.3f}] p={pv:.3f} (n={n})")

open(OUT + r"\02_lagged_climate_results.txt", "w", encoding="utf-8").write("\n".join(log))
P("\nSaved: 02_lagged_climate_results.txt"); P("DONE")
