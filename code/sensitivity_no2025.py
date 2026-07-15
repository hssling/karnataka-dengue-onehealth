"""Sensitivity: do findings change if 2025 (possibly incomplete district data) is excluded?"""
import warnings, numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm, statsmodels.formula.api as smf
warnings.filterwarnings("ignore"); np.random.seed(42)

# Reuse the panel-building code from onehealth_layers_v2 (everything before the model section)
code = open("onehealth_layers_v2.py", encoding="utf-8").read().split("# Full multi-layer model")[0]
ns = {}; exec(code, ns)
p = ns["p"]

def nb(formula, data, var):
    po = smf.glm(formula, data=data, family=sm.families.Poisson(), offset=data["logpop"]).fit()
    bb = ((data["Dengue_Cases"] - po.mu) ** 2 - data["Dengue_Cases"]) / po.mu
    a = max(0.05, smf.ols("bb~mu-1", pd.DataFrame({"bb": bb, "mu": po.mu})).fit().params.iloc[0])
    m = smf.glm(formula, data=data, family=sm.families.NegativeBinomial(alpha=a),
                offset=data["logpop"]).fit(cov_type="cluster", cov_kwds={"groups": data["d"]})
    ci = m.conf_int()
    return np.exp(m.params[var]), np.exp(ci.loc[var, 0]), np.exp(ci.loc[var, 1]), m.pvalues[var]

for yrs, label in [([2021, 2022, 2023, 2024, 2025], "WITH 2025"),
                   ([2021, 2022, 2023, 2024], "WITHOUT 2025")]:
    pp = p[p.Year.isin(yrs)].copy()
    chik = nb("Dengue_Cases ~ z_Chik_Incidence + C(Year)", pp, "z_Chik_Incidence")
    live = nb("Dengue_Cases ~ z_Livestock_pc + C(Year)", pp, "z_Livestock_pc")
    rr = stats.spearmanr(pp["Incidence"], pp["Chik_Incidence"])
    print(f"--- {label} (n={len(pp)} district-years) ---")
    print(f"  Chikungunya co-incidence: IRR={chik[0]:.3f} [{chik[1]:.3f},{chik[2]:.3f}] p={chik[3]:.4f}; Spearman rho={rr[0]:.3f} p={rr[1]:.4f}")
    print(f"  Livestock per-capita:     IRR={live[0]:.3f} [{live[1]:.3f},{live[2]:.3f}] p={live[3]:.4f}")
