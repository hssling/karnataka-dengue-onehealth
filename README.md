# Karnataka Dengue & One Health — Reproducibility Repository

Code and public-domain data for the study:

> **From Data to Action: The 2024 Dengue Surge in Karnataka, India, and the Limits of Routine One Health Data for Early Warning, 2021–2025.**
> *Submitted to One Health Bulletin (Special Section on One Health-Based Surveillance and Early Warning). Manuscript ID ohbl_31_26.*

This repository lets anyone reproduce every number, table, and figure in the manuscript and its revision, including the spatial-dependence, lagged-climate, and machine-learning analyses added in response to peer review.

---

## 1. What the study asks

Can routinely available, district-level One Health data (livestock, climate, vegetation, socioeconomic indicators) predict where dengue concentrates in Karnataka? Using a verified five-year district panel (29 districts, 145 district-years) with proper population denominators, the answer is: **among routine layers, only shared *Aedes* vector ecology (chikungunya co-incidence) predicts district dengue; animal-reservoir and climate proxies do not** — a finding we interpret as the limit of coarse, annual, routine data rather than proof of no biological relationship.

## 2. Repository layout

```
code/       Analysis scripts (Python)
  onehealth_layers_v2.py     Layer construction + multi-layer Negative Binomial GLM (primary model)
  sensitivity_no2025.py      Sensitivity: refit excluding 2025
  generate_figures.py        Main manuscript figures (1–3)
  01_spatial_autocorrelation.py   Moran's I (global/local/per-year), LM diagnostics, spatial-error/lag, neighbour-incidence covariate
  02_lagged_climate.py            Annual + prior-year (t−1) rainfall models (unadjusted & adjusted)
  03_gradient_boosting.py         Full ML spec, nested CV, learning curve, benchmarks
  04_spatial_figure.py            Supplementary Figure S1 (Moran scatterplot + LISA map)
data/       Input data (public domain)
  karnataka_district_dengue_chik_2021_2025_verified.csv   Verified surveillance panel (IDSP/NCVBDC)
  karnataka_livestock_district_totals.csv                 20th Livestock Census
  karnataka_nfhs5_indicators.csv                          NFHS-5 socioeconomic indicators
  karnataka_one_health_master_dataset.csv                 Assembled master (incl. vegetation condition index)
  analysis_panel.csv                                      Assembled district-year modelling panel
  rainfall/                                               Annual district rainfall 2021–2024 (KSNDMC)
  geospatial/District.*                                   Karnataka district shapefile (KGIS; queen-contiguity weights)
figures/    Main figures (1–3) and Supplementary Figure S1 (PNG)
results/    Text logs of the spatial, climate, and machine-learning runs
```

## 3. Data sources (all public domain)

| Layer | Source |
| :--- | :--- |
| Dengue & chikungunya cases/deaths | Integrated Disease Surveillance Programme / National Centre for Vector Borne Diseases Control (NCVBDC); cross-checked to the national dashboard |
| Livestock (cattle, buffalo, poultry, etc.) | 20th Livestock Census (2019), Dept. of Animal Husbandry & Dairying |
| Rainfall (annual total, departure) | Karnataka State Natural Disaster Monitoring Centre (KSNDMC) |
| Socioeconomic indicators | National Family Health Survey-5 (2019–21) |
| District population (denominators) | Census of India 2011 |
| District boundaries | Karnataka GIS (KGIS) district shapefile |

## 4. Environment

```
Python 3.14
numpy, pandas, scipy
statsmodels 0.14      # Negative Binomial GLM, cluster-robust SE
scikit-learn 1.7      # gradient boosting, cross-validation
geopandas 1.1         # shapefile handling
libpysal 4.14         # spatial weights
esda 2.10             # Moran's I / LISA
spreg 1.9             # spatial regression (ML error/lag, LM diagnostics)
matplotlib            # figures
```

Install with `pip install -r requirements.txt`.

## 5. How to reproduce

The scripts were written against the original project tree and reference the input files by path at the top of each script (constants `ROOT`, `PROJ`, `OHB`). To run them against this repository, set those constants to point at the repository's `data/` directory (a single edit per script), then:

```bash
python code/onehealth_layers_v2.py        # primary multi-layer NB GLM + FDR (Table 2)
python code/sensitivity_no2025.py         # Supplementary Table S6 (with/without 2025)
python code/01_spatial_autocorrelation.py # Supplementary Table S7 (Reviewer 2)
python code/02_lagged_climate.py          # Supplementary Table S5 (Reviewer 3)
python code/03_gradient_boosting.py       # Supplementary Table S4 (Reviewer 4)
python code/04_spatial_figure.py          # Supplementary Figure S1
python code/generate_figures.py           # Main Figures 1–3
```

All random seeds are fixed (42 unless otherwise stated), so results are deterministic.

## 6. Key results (as reproduced by these scripts)

- **Primary model (Negative Binomial GLM, offset, year FE, cluster-robust SE):** only chikungunya co-incidence is associated with dengue incidence (IRR 1.12, 95% CI 1.03–1.22, p=0.008 adjusted; 1.15, 1.06–1.24, p<0.001 single-layer). No animal, climate, vegetation, or socioeconomic layer survives Benjamini–Hochberg FDR control.
- **Spatial dependence (Reviewer 2):** no residual spatial autocorrelation overall (global Moran's I = −0.07, p=0.43, queen contiguity, 9,999 permutations); significant only transiently in 2021 and 2023; LISA identifies 4 clusters. 2024 LM diagnostics non-significant (LM-lag p=0.92, LM-error p=0.81); ML spatial-error λ=−0.08, spatial-lag ρ=−0.03. The chikungunya signal is unchanged when a neighbour-incidence covariate is added (IRR 1.15, p<0.001). District-clustered SEs are adequate.
- **Climate at annual scale (Reviewer 3):** same-year rainfall is borderline unadjusted (IRR 1.19, p=0.043) but null after adjustment (1.10, p=0.43); prior-year lag null after adjustment (1.02, p=0.87). No monthly/weekly data exist, so seasonality/short lags are not estimable.
- **Machine learning (Reviewer 4):** gradient boosting has negligible out-of-sample skill (5-fold CV R²=−0.31; tuned nested CV R²=+0.07, no better than OLS +0.02). A learning curve shows R² improving with sample size (−0.44 → −0.30), indicating the limit is sample size and coarse predictors, not the method.

## 7. Ethics

The study used aggregated, de-identified, public-domain secondary data. Institutional Ethics Committee and Scientific and Research Committee approval with waiver of consent were obtained (Ref. SRC2026003; approved 19 January 2026).

## 8. Citation

If you use this repository, please cite the manuscript (details to be updated on acceptance) and this repository.

## 9. License

Code is released under the MIT License (see `LICENSE`). Data files are redistributed from public-domain government sources; please also cite the original providers listed in Section 3.
