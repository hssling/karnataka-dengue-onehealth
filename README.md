# Karnataka Dengue & One Health — Reproducibility Repository

Code and public-domain data for the study:

> **From Data to Action: The 2024 Dengue Surge in Karnataka, India, and the Limits of Routine One Health Data for Early Warning, 2021–2025.**
> *One Health Bulletin (Special Section on One Health-Based Surveillance and Early Warning). Manuscript ID ohbl-31-26.*

This repository reproduces every number, table, and figure in the manuscript and its supplementary material.

---

## 1. What the study asks

Can routinely available, district-level One Health data predict where dengue concentrates in Karnataka? The data cover livestock, climate, vegetation, and socioeconomic indicators. The analysis uses a verified five-year district panel (29 districts, 2021–2025, 145 district-years) with Census-2011 population denominators.

The answer: no animal, climate, vegetation, or socioeconomic layer is robustly associated with district dengue incidence. Among routine layers, only shared *Aedes* vector ecology (chikungunya co-occurring incidence) is consistently associated with dengue, across single-layer, multi-layer, sensitivity, and spatially adjusted models. We read this as the limit of coarse, annual, routine data, not as proof that no biological relationship exists.

## 2. Repository layout

```
code/
  onehealth_layers_v2.py          Layer construction + multi-layer Negative Binomial GLM (primary model, Table 2)
  sensitivity_no2025.py           Earlier 2025-exclusion script (superseded by 05_consistency_reanalysis.py)
  generate_figures.py             Figures 1–2 (and the earlier Figure 3)
  01_spatial_autocorrelation.py   Moran's I (global/local/per-year), LM diagnostics, spatial-error/lag,
                                  neighbour-incidence covariate (Supplementary Table S7)
  02_lagged_climate.py            Earlier 2021–2024 rainfall models (superseded by 05_...)
  03_gradient_boosting.py         Full ML specification, nested CV, learning curve (Supplementary Table S4)
  04_spatial_figure.py            Supplementary Figure S1
  00_prep_rainfall_2020.py        Extracts the KSNDMC 2020 district rainfall table and cross-checks it
  05_consistency_reanalysis.py    Final consistency reanalysis: every estimate reported in the paper
                                  (Table 2, Tables S3, S5, S6, SDs for per-SD IRRs, rainfall 2020–2025,
                                  2025 exclusion, combined Ballari–Vijayanagara district, raw-count livestock)
  06_figure3_forest.py            Figure 3, drawn directly from 05's results
data/
  karnataka_district_dengue_chik_2021_2025_verified.csv   Verified surveillance panel (IDSP/NCVBDC)
  karnataka_livestock_district_totals.csv                 20th Livestock Census
  karnataka_nfhs5_indicators.csv                          NFHS-5 socioeconomic indicators
  karnataka_one_health_master_dataset.csv                 Assembled master (incl. vegetation condition index)
  analysis_panel.csv                                      Assembled district-year modelling panel
  rainfall/dist rainfall 2021–2024.csv                    Annual district rainfall (KSNDMC)
  rainfall/ka_dist_rainfall_2020.csv                      KSNDMC Annual Report 2020, district table (annual columns)
  rainfall/ka_dist_rainfall_2025.csv                      KSNDMC Annual Report 2025, district table
  rainfall/raw/ksndmc_2020_district_table.csv             2020 table as extracted from the report (all seasons)
  geospatial/District.*                                   Karnataka district shapefile (KGIS; queen-contiguity weights)
figures/   Figures 1–3 and Supplementary Figure S1 (PNG)
results/   Logs and machine-readable results (05_consistency_reanalysis_results.json is the source of all
           reported estimates)
```

## 3. Data sources (all public domain)

| Layer | Source |
| :--- | :--- |
| Dengue & chikungunya cases/deaths | Integrated Disease Surveillance Programme / National Centre for Vector Borne Diseases Control (NCVBDC); cross-checked to the national dashboard |
| Livestock (cattle, buffalo, sheep, goat, pig, poultry) | 20th Livestock Census (2019), Dept. of Animal Husbandry & Dairying |
| Rainfall (annual total, departure from normal), 2020–2025 | Karnataka State Natural Disaster Monitoring Centre (KSNDMC) annual reports |
| Socioeconomic indicators | National Family Health Survey-5 (2019–21) |
| District population (denominators) | Census of India 2011 |
| District boundaries | Karnataka GIS (KGIS) district shapefile |

**Rainfall checks.** The 2025 annual totals match the KSNDMC *Annual Report 2025* district table exactly for all 31 districts. The 2020 values come from Table 1.1 of the KSNDMC *Annual Report 2020*. The same report's district summary table differs from Table 1.1 by at most 2 mm in 5 of 30 districts. Using either table gives the same results at the published precision. The 2020 Ballari value refers to the undivided district.

**Districts.** 29 of Karnataka's 31 districts are modelled. Bengaluru Urban and the BBMP city reporting unit are excluded because their populations overlap. Vijayanagara is excluded because it was created from Ballari in 2021 and has no Census-2011 population; it reported cases separately only from 2023. A sensitivity analysis adds Vijayanagara's cases back to Ballari on the undivided-district denominator.

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

The scripts were written against the original project tree. Each one names its input paths in constants at the top (`ROOT`, `PROJ`, `OHB`, `OUT`, `SRC`). Point these at this repository's `data/`, `code/` and `results/` directories (one edit per script), then run:

```bash
python code/onehealth_layers_v2.py          # primary multi-layer NB GLM + FDR (Table 2)
python code/00_prep_rainfall_2020.py        # 2020 rainfall extraction + cross-check
python code/05_consistency_reanalysis.py    # all reported estimates -> results/05_...json
python code/06_figure3_forest.py            # Figure 3
python code/01_spatial_autocorrelation.py   # Supplementary Table S7
python code/03_gradient_boosting.py         # Supplementary Table S4
python code/04_spatial_figure.py            # Supplementary Figure S1
python code/generate_figures.py             # Figures 1–2
```

All random seeds are fixed (42 unless otherwise stated), so results are deterministic.

## 6. Key results (as reproduced by these scripts)

IRRs are per one-standard-deviation increase in each layer. One SD of chikungunya co-occurring incidence is 4.1 cases per 100,000. Benjamini–Hochberg FDR is applied across the nine layers, with q<0.05 considered significant.

- **Primary multi-layer model (NB GLM, offset, year FE, district-clustered SE; 145 district-years):** no layer meets q<0.05. Chikungunya co-occurring incidence is the only layer whose 95% CI excludes unity: IRR 1.12 (1.03–1.22), P=0.008, q=0.07. Single-layer: IRR 1.15 (1.06–1.24), P<0.001.
- **Vegetation condition index:** inverse when modelled alone (IRR 0.24, 0.14–0.40) but imprecise after adjustment (0.25, 0.06–1.02; q=0.24).
- **Livestock:** total livestock shows a non-significant positive association with raw counts (IRR 1.12, 0.90–1.38) that reverses with the population offset (0.88, 0.73–1.05), consistent with a denominator effect.
- **Rainfall (2020–2025; all models use the same 145 district-years):**
  - Same-year: unadjusted IRR 1.16 (0.99–1.36), P=0.065; ten-layer adjusted 1.06 (0.86–1.31), P=0.57.
  - Prior-year (2020–2024 rainfall paired with 2021–2025 dengue): unadjusted 1.17 (0.99–1.37), P=0.058; adjusted 1.10 (0.89–1.36), P=0.38.
- **Sensitivity:** the chikungunya association persists when 2025 is excluded (multi-layer IRR 1.13, 1.04–1.23), when Vijayanagara is combined with Ballari (1.11, 1.02–1.21), and when rainfall is added (1.12, 1.03–1.22).
- **Spatial dependence:** there is no residual spatial autocorrelation overall (global Moran's I −0.07, P=0.43). The chikungunya signal is unchanged when a neighbour-incidence covariate is added to the single-layer model (IRR 1.15, P<0.001).
- **Machine learning:** the fixed-specification gradient boosting model has no out-of-sample skill (5-fold CV R² −0.31). Nested tuning gives negligible skill (+0.07, SD 0.09), comparable to OLS (+0.02, SD 0.12). R² rises with sample size (−0.44 → −0.30) but stays below zero.

## 7. Ethics

The study used aggregated, de-identified, public-domain secondary data. Institutional Ethics Committee and Scientific and Research Committee approval with waiver of consent were obtained (Ref. SRC2026003; approved 19 January 2026).

## 8. Citation

If you use this repository, please cite the article (details to be added on publication) and this repository (release v1.1).

## 9. License

Code is released under the MIT License (see `LICENSE`). Data files are redistributed from public-domain government sources; please also cite the original providers listed in Section 3.
