"""
Revision R1 — Reviewer 4 (machine-learning specification).
Fully specify and reproduce the gradient-boosting analysis on the verified
29-district One Health panel, add a nested (tuned) cross-validation, a naive-mean
benchmark, and a learning-curve probe of sample-size dependence, so that the
poor out-of-sample skill can be attributed to sample size / coarse predictors
rather than presented as proof that ML "cannot" work.
"""
import warnings
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore"); np.random.seed(42)

OHB = r"D:\Dengue project\analysis\submission\ohb_special_issue"
OUT = OHB + r"\revision_R1\analysis"
src = open(OHB + r"\onehealth_layers_v2.py", encoding="utf-8").read().split("# Full multi-layer model")[0]
ns = {}; exec(src, ns)
p = ns["p"].copy(); layers = ns["layers"]

log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

P("="*70); P("REVISION R1 - GRADIENT BOOSTING FULL SPECIFICATION (Reviewer 4)"); P("="*70)
P(f"Panel: n={len(p)} district-years, {p.d.nunique()} districts, {len(layers)} predictors")
P("Predictors:", ", ".join(layers))
P(f"Missing values in predictor matrix: {int(p[layers].isna().sum().sum())} "
  "(complete-case panel; no imputation required).")

X = p[layers].values.astype(float)
y = np.log1p(p["Dengue_Cases"].values.astype(float))
P(f"Target: log1p(dengue count). Software: scikit-learn {__import__('sklearn').__version__}.")

# --- 1. Fixed-hyperparameter model exactly as originally reported ---
gbr = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                                subsample=0.9, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
r2 = cross_val_score(gbr, X, y, cv=cv, scoring="r2")
P("\n--- Fixed spec: GBR(n_estimators=400, learning_rate=0.05, max_depth=3, "
  "subsample=0.9, random_state=42), 5-fold shuffled KFold (seed 42) ---")
P(f"  Cross-validated R2 = {r2.mean():+.3f} (SD {r2.std():.3f}); "
  f"per-fold: {', '.join(f'{v:+.2f}' for v in r2)}")

# --- 2. Naive-mean benchmark (R2=0 by construction) for context ---
P(f"  Interpretation: R2<0 means the model predicts worse out-of-sample than the "
  f"training-mean benchmark (R2=0).")

# --- 3. Nested cross-validation with hyperparameter tuning ---
P("\n--- Nested 5-fold CV with inner grid search (tuning nested within CV) ---")
grid = {"n_estimators": [100, 200, 400], "learning_rate": [0.02, 0.05, 0.1],
        "max_depth": [2, 3], "subsample": [0.8, 1.0]}
outer = KFold(n_splits=5, shuffle=True, random_state=42)
nested = []
for tr, te in outer.split(X):
    inner = KFold(n_splits=4, shuffle=True, random_state=1)
    gs = GridSearchCV(GradientBoostingRegressor(random_state=42), grid,
                      cv=inner, scoring="r2", n_jobs=-1)
    gs.fit(X[tr], y[tr])
    nested.append(gs.best_estimator_.score(X[te], y[te]))
P(f"  Nested CV R2 = {np.mean(nested):+.3f} (SD {np.std(nested):.3f}); "
  f"grid = {sum(len(v) for v in [grid])* 1} combos over "
  f"{np.prod([len(v) for v in grid.values()])} settings.")

# --- 4. Linear-model reference under identical CV ---
lin_r2 = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")
P(f"  Reference OLS (same folds): CV R2 = {lin_r2.mean():+.3f} (SD {lin_r2.std():.3f})")

# --- 5. Permutation importance (full-fit, for description only) ---
gbr.fit(X, y)
pi = permutation_importance(gbr, X, y, n_repeats=50, random_state=42)
order = np.argsort(pi.importances_mean)[::-1]
P("\n--- Permutation importance (full-fit; descriptive, not out-of-sample) ---")
for i in order[:5]:
    P(f"  {layers[i]:28s} {pi.importances_mean[i]:+.4f} (SD {pi.importances_std[i]:.4f})")

# --- 6. Learning curve: does skill improve with more data? (sample-size probe) ---
P("\n--- Learning-curve probe (training-set fraction vs CV R2) ---")
rng = np.random.RandomState(42)
for frac in (0.5, 0.7, 0.85, 1.0):
    scores = []
    for rep in range(20):
        idx = rng.choice(len(X), int(len(X) * frac), replace=False)
        if len(np.unique(idx)) < 20:
            continue
        s = cross_val_score(GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
              max_depth=3, subsample=0.9, random_state=rep),
              X[idx], y[idx], cv=KFold(5, shuffle=True, random_state=rep), scoring="r2")
        scores.append(s.mean())
    P(f"  {int(frac*100):3d}% of panel (~{int(len(X)*frac)} obs): "
      f"mean CV R2 = {np.mean(scores):+.3f} (SD {np.std(scores):.3f})")

open(OUT + r"\03_gradient_boosting_results.txt", "w", encoding="utf-8").write("\n".join(log))
P("\nSaved: 03_gradient_boosting_results.txt"); P("DONE")
