"""
Recheck + add missing One Health layers, and test whether One Health
connections genuinely predict district dengue incidence (Karnataka 2021-2025).
New layers: livestock per-capita; large-ruminant (cattle+buffalo) per-capita;
poultry per-capita; vegetation condition index (VCI); chikungunya co-incidence.
"""
import re, glob, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore"); np.random.seed(42)

ROOT = r"D:\Dengue project\Dengue Karnataka 2021-2025"; PROJ = ROOT + r"\karnataka_project"
SYN = {  # district name harmonization
 "bangalore":"bengaluru urban","bengaluru urban":"bengaluru urban","bangaluru urban":"bengaluru urban",
 "bangalore rural":"bengaluru rural","bengaluru rural":"bengaluru rural","bangaluru rural":"bengaluru rural",
 "belgaum":"belagavi","belagavi":"belagavi","bijapur":"vijayapura","vijayapura":"vijayapura",
 "gulbarga":"kalaburagi","kalaburagi":"kalaburagi","bellary":"ballari","ballari":"ballari",
 "shimoga":"shivamogga","shivamogga":"shivamogga","mysore":"mysuru","mysuru":"mysuru",
 "tumkur":"tumakuru","tumakuru":"tumakuru","chikmagalur":"chikkamagaluru","chikkamagaluru":"chikkamagaluru",
 "chikballapur":"chikkaballapura","chikkaballapura":"chikkaballapura","chamarajanagara":"chamarajanagar",
 "chamarajanagar":"chamarajanagar","dakshin kannad":"dakshina kannada","dakshina kannada":"dakshina kannada",
 "uttar kannad":"uttara kannada","uttara":"uttara kannada","uttara kannada":"uttara kannada",
 "koppala":"koppal","koppal":"koppal","ramanagara":"ramanagar","ramanagar":"ramanagar",
 "vijayanagar":"vijayanagara","vijayanagara":"vijayanagara","bagalkot":"bagalkote","bagalkote":"bagalkote",
 "davangere":"davanagere","davanagere":"davanagere","dharwada":"dharwad","yadageri":"yadgir"}
def norm(n): s=re.sub(r"\s+"," ",str(n).strip().lower()); return SYN.get(s,s)
POP={"belagavi":4779661,"mysuru":3001127,"tumakuru":2678980,"kalaburagi":2566326,"ballari":2452595,
 "vijayapura":2177331,"dakshina kannada":2089649,"davanagere":1945497,"raichur":1928812,"bagalkote":1889752,
 "dharwad":1847023,"mandya":1805769,"hassan":1776421,"shivamogga":1752753,"bidar":1703300,"chitradurga":1659456,
 "haveri":1597668,"kolar":1536401,"uttara kannada":1437169,"koppal":1389920,"chikkaballapura":1255104,
 "udupi":1177361,"yadgir":1174271,"chikkamagaluru":1137961,"ramanagar":1082636,"gadag":1064570,
 "chamarajanagar":1020791,"bengaluru rural":990923,"kodagu":554519}

dn=pd.read_csv(PROJ+r"\data\Datasets\karnataka_district_dengue_chik_2021_2025_verified.csv")
dn["d"]=dn["District"].map(norm); dn=dn[~dn["d"].isin(["bbmp","bengaluru urban"])]

# Livestock species -> per-capita layers
lv=pd.read_csv(PROJ+r"\data\processed\karnataka_livestock_district_totals.csv")
lv=lv[lv["District"].str.lower()!="district_name"].copy(); lv["d"]=lv["District"].map(norm)
for c in ["Cattle","Buffalo","Sheep","Goat","Pig","Poultry"]: lv[c]=pd.to_numeric(lv[c],errors="coerce")
lv["pop"]=lv["d"].map(POP)
lv["LargeRuminant_pc"]=(lv["Cattle"]+lv["Buffalo"])/lv["pop"]*1000
lv["Livestock_pc"]=(lv["Cattle"]+lv["Buffalo"]+lv["Sheep"]+lv["Goat"]+lv["Pig"])/lv["pop"]*1000
lv["Poultry_pc"]=lv["Poultry"]/lv["pop"]*1000
lv=lv[["d","LargeRuminant_pc","Livestock_pc","Poultry_pc"]].dropna().drop_duplicates("d")

# NFHS-5
nf=pd.read_csv(PROJ+r"\data\Datasets\karnataka_nfhs5_indicators.csv"); nf["d"]=nf["district"].map(norm)
nf=nf.rename(columns={"14. Women who are literate4 (%)":"Female_Literacy",
 "10. Households using clean fuel for cooking3 (%)":"Clean_Fuel",
 "8. Population living in households with an improved drinking-water source1 (%)":"Water_Improved",
 "9. Population living in households that use an improved sanitation facility2 (%)":"Sanitation_Improved"})
nf=nf[["d","Female_Literacy","Clean_Fuel","Water_Improved","Sanitation_Improved"]].drop_duplicates("d")

# VCI (environment) from master, by district-year
ms=pd.read_csv(PROJ+r"\data\Datasets\karnataka_one_health_master_dataset.csv")
ms["d"]=ms["District"].map(norm)
vci=ms[["d","Year","Vegetation_Condition_Index"]].dropna().drop_duplicates(["d","Year"])

# Rainfall annual
def load_rain(path,year):
    r=pd.read_csv(path); r.columns=[str(c).replace("﻿","").strip() for c in r.columns]
    low={c:c.lower() for c in r.columns}
    def pick(*k):
        for key in k:
            for c,l in low.items():
                if key in l: return c
    cd,ct,cdep=pick("district"),pick("total actual","annual actual","actual"),pick("total departure","annual departure","%dep","departure")
    r=r[r[cd].astype(str).str.upper().str.strip()!="STATE"]
    return pd.DataFrame({"d":r[cd].map(norm),"Rain_mm":pd.to_numeric(r[ct],errors="coerce"),"Year":year}).dropna(subset=["d"])
rain=pd.concat([load_rain(glob.glob(ROOT+rf"\Karnataka district rainfall\*{y}*.csv")[0],y) for y in (2021,2022,2023,2024)]).drop_duplicates(["d","Year"])

# Assemble
p=dn[dn.Year.isin([2021,2022,2023,2024,2025])][["d","Year","Dengue_Cases","Chik_Cases"]].copy()
p["Population"]=p["d"].map(POP)
p=p.merge(lv,on="d",how="left").merge(nf,on="d",how="left").merge(vci,on=["d","Year"],how="left").merge(rain,on=["d","Year"],how="left")
p=p.dropna(subset=["Population","LargeRuminant_pc","Female_Literacy","Vegetation_Condition_Index"]).copy()
p["Incidence"]=p["Dengue_Cases"]/p["Population"]*1e5
p["Chik_Incidence"]=p["Chik_Cases"]/p["Population"]*1e5
p["logpop"]=np.log(p["Population"])
layers=["LargeRuminant_pc","Livestock_pc","Poultry_pc","Vegetation_Condition_Index","Chik_Incidence",
        "Female_Literacy","Clean_Fuel","Water_Improved","Sanitation_Improved"]
for c in layers: p["z_"+c]=(p[c]-p[c].mean())/p[c].std()
print(f"Panel: n={len(p)} district-years, {p.d.nunique()} districts")

def nb(formula,data):
    po=smf.glm(formula,data=data,family=sm.families.Poisson(),offset=data["logpop"]).fit()
    bb=((data["Dengue_Cases"]-po.mu)**2-data["Dengue_Cases"])/po.mu
    a=max(0.05,smf.ols("bb~mu-1",pd.DataFrame({"bb":bb,"mu":po.mu})).fit().params.iloc[0])
    return smf.glm(formula,data=data,family=sm.families.NegativeBinomial(alpha=a),offset=data["logpop"]).fit(
        cov_type="cluster",cov_kwds={"groups":data["d"]}),a

# Full multi-layer model
f="Dengue_Cases ~ "+" + ".join("z_"+c for c in layers)+" + C(Year)"
m,a=nb(f,p)
print(f"\n=== Multi-layer One Health NB GLM (incidence, offset, year FE, cluster SE), n={int(m.nobs)} ===")
rows=[]
for c in layers:
    n="z_"+c; rows.append((c,m.pvalues[n]))
    print(f"  {c:28s} IRR={np.exp(m.params[n]):.3f}  95%CI[{np.exp(m.conf_int().loc[n,0]):.3f},{np.exp(m.conf_int().loc[n,1]):.3f}]  p={m.pvalues[n]:.4f}")
q=multipletests([r[1] for r in rows],method="fdr_bh")[1]
print("  -- FDR-adjusted q-values --")
for (c,pv),qq in zip(rows,q): print(f"  {c:28s} q={qq:.4f}")

# Focused test: shared-vector One Health connection (chikungunya co-incidence)
print("\n=== Shared-vector test: dengue vs chikungunya incidence ===")
r,pv=stats.spearmanr(p["Incidence"],p["Chik_Incidence"]); print(f"  Spearman rho={r:+.3f}, p={pv:.4f} (pooled district-years)")
mc,_=nb("Dengue_Cases ~ z_Chik_Incidence + C(Year)",p)
print(f"  NB GLM chik-only: IRR={np.exp(mc.params['z_Chik_Incidence']):.3f} "
      f"95%CI[{np.exp(mc.conf_int().loc['z_Chik_Incidence',0]):.3f},{np.exp(mc.conf_int().loc['z_Chik_Incidence',1]):.3f}] p={mc.pvalues['z_Chik_Incidence']:.4f}")

# Focused test: large-ruminant density (water-storage animals)
print("\n=== Large-ruminant (cattle+buffalo) per-capita, single-layer ===")
mr,_=nb("Dengue_Cases ~ z_LargeRuminant_pc + C(Year)",p)
print(f"  IRR={np.exp(mr.params['z_LargeRuminant_pc']):.3f} "
      f"95%CI[{np.exp(mr.conf_int().loc['z_LargeRuminant_pc',0]):.3f},{np.exp(mr.conf_int().loc['z_LargeRuminant_pc',1]):.3f}] p={mr.pvalues['z_LargeRuminant_pc']:.4f}")
print("\nDONE")
