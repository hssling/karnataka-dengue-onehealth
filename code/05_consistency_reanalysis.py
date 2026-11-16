"""
Consistency reanalysis: re-derives every model estimate reported in the article and its
supplementary material from the same
panel-building code as the primary analysis (onehealth_layers_v2.py), and adds:
  * predictor SDs in natural units (to define 'per 1 SD' IRRs);
  * explicit year windows / n for every rainfall model;
  * 2025-exclusion sensitivity for BOTH single-layer and multi-layer models;
  * Ballari+Vijayanagara merged-geography sensitivity (Census-2011 boundaries);
  * raw-count (no offset) livestock models behind the 'denominator artifact' statement;
  * full nine-layer estimates for Figure 3.
"""
import sys, glob, warnings, json
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore"); np.random.seed(42)
sys.stdout.reconfigure(encoding="utf-8")
OHB = r"D:\Dengue project\analysis\submission\ohb_special_issue"
ROOT = r"D:\Dengue project\Dengue Karnataka 2021-2025"
OUT = OHB + r"\revision_R2\final_response\analysis"
src = open(OHB + r"\onehealth_layers_v2.py", encoding="utf-8").read().split("# Full multi-layer model")[0]
ns = {}; exec(src, ns)
p = ns["p"].copy(); norm = ns["norm"]; layers = ns["layers"]
R = {}


def nbfit(formula, data, offset=True):
    off = data["logpop"] if offset else None
    po = smf.glm(formula, data=data, family=sm.families.Poisson(), offset=off).fit()
    bb = ((data["Dengue_Cases"] - po.mu) ** 2 - data["Dengue_Cases"]) / po.mu
    a = max(0.05, smf.ols("bb~mu-1", pd.DataFrame({"bb": bb, "mu": po.mu})).fit().params.iloc[0])
    return smf.glm(formula, data=data, family=sm.families.NegativeBinomial(alpha=a), offset=off).fit(
        cov_type="cluster", cov_kwds={"groups": data["d"]})


def est(m, v):
    ci = m.conf_int().loc[v]
    return dict(irr=float(np.exp(m.params[v])), lo=float(np.exp(ci[0])),
                hi=float(np.exp(ci[1])), p=float(m.pvalues[v]), n=int(m.nobs))


def zall(df, cols):
    df = df.copy()
    for c in cols:
        df["z_" + c] = (df[c] - df[c].mean()) / df[c].std()
    return df


FULL = "Dengue_Cases ~ " + " + ".join("z_" + c for c in layers) + " + C(Year)"

print(f"Panel n={len(p)}, districts={p.d.nunique()}, years={sorted(p.Year.unique())}")
print("Districts:", sorted(p.d.unique()))
R["panel"] = dict(n=len(p), districts=p.d.nunique(), years=sorted(map(int, p.Year.unique())))

# --- SDs of predictors (natural units) ---
R["sd"] = {c: dict(mean=float(p[c].mean()), sd=float(p[c].std())) for c in layers}
print("\nPredictor mean / SD (natural units):")
for c in layers:
    print(f"  {c:28s} mean={p[c].mean():10.3f} sd={p[c].std():10.3f}")

# --- Primary multi-layer model ---
m = nbfit(FULL, p)
q = multipletests([m.pvalues["z_" + c] for c in layers], method="fdr_bh")[1]
R["multi"] = {}
print("\nPrimary multi-layer NB GLM (2021-2025):")
for c, qq in zip(layers, q):
    e = est(m, "z_" + c); e["q"] = float(qq); R["multi"][c] = e
    print(f"  {c:28s} IRR {e['irr']:.3f} ({e['lo']:.3f}-{e['hi']:.3f}) p={e['p']:.4f} q={qq:.4f} n={e['n']}")

# --- Single-layer models, all nine layers ---
R["single"] = {}
print("\nSingle-layer NB GLMs (2021-2025):")
for c in layers:
    e = est(nbfit(f"Dengue_Cases ~ z_{c} + C(Year)", p), "z_" + c); R["single"][c] = e
    print(f"  {c:28s} IRR {e['irr']:.3f} ({e['lo']:.3f}-{e['hi']:.3f}) p={e['p']:.4f}")
r, pv = stats.spearmanr(p["Incidence"], p["Chik_Incidence"])
R["spearman_chik"] = dict(rho=float(r), p=float(pv))
print(f"\nSpearman dengue vs chik incidence rho={r:.3f} p={pv:.2e}")

# --- Sensitivity: exclude 2025 (z-scores recomputed on the restricted panel) ---
p24 = zall(p[p.Year <= 2024], layers)
m24 = nbfit(FULL, p24)
R["no2025"] = dict(single={c: est(nbfit(f"Dengue_Cases ~ z_{c} + C(Year)", p24), "z_" + c) for c in layers},
                   multi={c: est(m24, "z_" + c) for c in layers})
print("\nWithout 2025 (n=%d), z restandardised:" % len(p24))
for c in layers:
    s, mm_ = R["no2025"]["single"][c], R["no2025"]["multi"][c]
    print(f"  {c:28s} single {s['irr']:.3f} ({s['lo']:.3f}-{s['hi']:.3f}) p={s['p']:.4f} | "
          f"multi {mm_['irr']:.3f} ({mm_['lo']:.3f}-{mm_['hi']:.3f}) p={mm_['p']:.4f}")
p24b = p[p.Year <= 2024].copy()   # z kept from full panel (as in sensitivity_no2025.py)
R["no2025_fullz"] = dict(
    single_chik=est(nbfit("Dengue_Cases ~ z_Chik_Incidence + C(Year)", p24b), "z_Chik_Incidence"),
    single_live=est(nbfit("Dengue_Cases ~ z_Livestock_pc + C(Year)", p24b), "z_Livestock_pc"),
    multi={c: est(nbfit(FULL, p24b), "z_" + c) for c in layers})
print("  (z from full panel) chik single", R["no2025_fullz"]["single_chik"],
      "\n  live single", R["no2025_fullz"]["single_live"])
for c in layers:
    print(f"   multi(fullz) {c:28s} {R['no2025_fullz']['multi'][c]}")

# --- Sensitivity: Ballari + Vijayanagara merged (undivided district, Census-2011 boundary) ---
dn = pd.read_csv(ROOT + r"\karnataka_project\data\Datasets\karnataka_district_dengue_chik_2021_2025_verified.csv")
dn["d"] = dn["District"].map(norm)
vj = dn[dn.d == "vijayanagara"].set_index("Year")[["Dengue_Cases", "Chik_Cases"]]
print("\nVijayanagara reported:", vj.to_dict())
pm = p.copy()
for y in vj.index:
    i = (pm.d == "ballari") & (pm.Year == y)
    pm.loc[i, "Dengue_Cases"] += vj.loc[y, "Dengue_Cases"]
    pm.loc[i, "Chik_Cases"] += vj.loc[y, "Chik_Cases"]
pm["Incidence"] = pm.Dengue_Cases / pm.Population * 1e5
pm["Chik_Incidence"] = pm.Chik_Cases / pm.Population * 1e5
pm = zall(pm, layers)
mm = nbfit(FULL, pm)
R["merged"] = dict(single_chik=est(nbfit("Dengue_Cases ~ z_Chik_Incidence + C(Year)", pm), "z_Chik_Incidence"),
                   single_live=est(nbfit("Dengue_Cases ~ z_Livestock_pc + C(Year)", pm), "z_Livestock_pc"),
                   multi={c: est(mm, "z_" + c) for c in layers})
qm = multipletests([mm.pvalues["z_" + c] for c in layers], method="fdr_bh")[1]
R["merged"]["q"] = dict(zip(layers, map(float, qm)))
print("Merged Ballari+Vijayanagara: chik single", R["merged"]["single_chik"])
for c in layers:
    print(f"  multi {c:28s} {R['merged']['multi'][c]} q={R['merged']['q'][c]:.3f}")
print("Ballari incidence merged:", pm[pm.d == "ballari"][["Year", "Incidence"]].round(1).values.tolist())

# --- Raw-count (no offset) livestock models ---
p["log_livestock_total"] = np.log(p["Livestock_pc"] * p["Population"] / 1000)
p = zall(p, layers + ["log_livestock_total"])
R["raw"] = {}
for lab, f, off in [("pc_offset", "Dengue_Cases ~ z_Livestock_pc + C(Year)", True),
                    ("pc_nooffset", "Dengue_Cases ~ z_Livestock_pc + C(Year)", False),
                    ("total_nooffset", "Dengue_Cases ~ z_log_livestock_total + C(Year)", False),
                    ("total_offset", "Dengue_Cases ~ z_log_livestock_total + C(Year)", True)]:
    v = f.split("~")[1].split("+")[0].strip()
    R["raw"][lab] = est(nbfit(f, p, off), v)
    print(f"Livestock {lab:15s} {R['raw'][lab]}")
g = p.groupby("d")
rr = stats.spearmanr(g.Dengue_Cases.sum(), g.log_livestock_total.first())
print(f"Spearman district total cases vs total livestock rho={rr[0]:.3f} p={rr[1]:.3f}")
R["raw"]["spearman_total"] = dict(rho=float(rr[0]), p=float(rr[1]))


# --- Rainfall models: explicit windows ---
def load_rain(path, year):
    r = pd.read_csv(path); r.columns = [str(c).replace("\ufeff", "").strip() for c in r.columns]
    low = {c: c.lower() for c in r.columns}

    def pick(*k):
        for key in k:
            for c, l in low.items():
                if key in l:
                    return c
    cd = pick("district"); ct = pick("total actual", "annual actual", "actual")
    cdep = pick("total departure", "annual departure", "%dep", "departure")
    r = r[r[cd].astype(str).str.upper().str.strip() != "STATE"]
    return pd.DataFrame({"d": r[cd].map(norm), "Rain_mm": pd.to_numeric(r[ct], errors="coerce"),
                         "Rain_dep": pd.to_numeric(r[cdep], errors="coerce"), "Year": year}).dropna(subset=["d"])


RAIN_FILES = {y: glob.glob(ROOT + rf"\Karnataka district rainfall\*{y}*.csv")[0] for y in (2021, 2022, 2023, 2024)}
RAIN_FILES[2025] = OUT + r"\data\ka_dist_rainfall_2025.csv"   # KSNDMC Annual Report 2025
RAIN_FILES[2020] = OUT + r"\data\ka_dist_rainfall_2020.csv"   # KSNDMC Annual Report 2020 (prep_rain2020.py)
rain = pd.concat([load_rain(f, y) for y, f in sorted(RAIN_FILES.items())]).drop_duplicates(["d", "Year"])
print("\nRainfall files cover years:", sorted(rain.Year.unique()), "rows", len(rain))
# normals should be stable across years: compare 2025 file to 2024 file
_n = {}
for y in (2024, 2025):
    rr_ = pd.read_csv(RAIN_FILES[y], encoding="utf-8-sig"); rr_.columns = [c.strip() for c in rr_.columns]
    cn = [c for c in rr_.columns if "normal" in c.lower() and ("total" in c.lower() or "annual" in c.lower())][0]
    _n[y] = rr_.set_index(rr_["District"].map(norm))[cn]
cmpn = pd.concat(_n, axis=1).dropna()
print("Annual normal 2024 vs 2025 file: max abs diff (mm) =", float((cmpn[2024] - cmpn[2025]).abs().max()),
      "over", len(cmpn), "districts")
R["rain_meta"] = dict(years=sorted(map(int, rain.Year.unique())),
                      normal_maxdiff=float((cmpn[2024] - cmpn[2025]).abs().max()))
pp_ = p.drop(columns=["Rain_mm"], errors="ignore").merge(rain, on=["d", "Year"], how="left")
print("Panel district-years lacking same-year rainfall:", int(pp_["Rain_mm"].isna().sum()))
lag = rain.copy(); lag["Year"] += 1
lag = lag.rename(columns={"Rain_mm": "Rain_mm_lag1", "Rain_dep": "Rain_dep_lag1"})[
    ["d", "Year", "Rain_mm_lag1", "Rain_dep_lag1"]]
pr = (p.drop(columns=[c for c in ["Rain_mm"] if c in p.columns])
      .merge(rain, on=["d", "Year"], how="left").merge(lag, on=["d", "Year"], how="left"))
R["rain"] = {}
for col in ["Rain_mm", "Rain_dep", "Rain_mm_lag1", "Rain_dep_lag1"]:
    d = pr.dropna(subset=[col]).copy(); d = zall(d, layers + [col])
    e = est(nbfit(f"Dengue_Cases ~ z_{col} + C(Year)", d), "z_" + col)
    e["dengue_years"] = sorted(map(int, d.Year.unique())); e["sd"] = float(d[col].std())
    if col in ("Rain_mm", "Rain_mm_lag1"):
        e["adj"] = est(nbfit(f"Dengue_Cases ~ z_{col} + " + " + ".join("z_" + c for c in layers)
                             + " + C(Year)", d), "z_" + col)
    if col.endswith("lag1"):
        e["rain_years"] = sorted(map(int, (d.Year - 1).unique()))
    R["rain"][col] = e
    print(f"  {col:14s} {e}")

# --- Ten-layer model: nine layers + same-year rainfall (all coefficients) ---
d10 = zall(pr.dropna(subset=["Rain_mm"]).copy(), layers + ["Rain_mm"])
m10 = nbfit("Dengue_Cases ~ z_Rain_mm + " + " + ".join("z_" + c for c in layers) + " + C(Year)", d10)
R["ten_layer"] = {c: est(m10, "z_" + c) for c in ["Rain_mm"] + layers}
print("\nTen-layer model (nine layers + same-year rainfall), n=%d:" % int(m10.nobs))
for c, e in R["ten_layer"].items():
    print(f"  {c:28s} IRR {e['irr']:.3f} ({e['lo']:.3f}-{e['hi']:.3f}) p={e['p']:.4f}")

json.dump(R, open(OUT + r"\05_consistency_reanalysis_results.json", "w"), indent=1, default=float)
print("\nSaved 05_consistency_reanalysis_results.json")
