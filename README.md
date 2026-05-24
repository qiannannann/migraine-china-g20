# Migraine burden in China and G20 economies, 1990–2023 — analysis code

Analysis code, derived datasets, intermediate tables, and figures supporting the manuscript:

> **"Migraine burden in China and G20 economies from 1990 to 2023 and exploratory projections to 2053: a descriptive analysis of GBD 2023 modeled estimates."**
> Qian N, Wei T, Song Y, et al. *iScience* (in press).

This repository contains **analysis and figure-generation code only**. All figures are produced in PDF format; all tables are produced as CSV.

## Repository structure

```
.
├── README.md
├── LICENSE                       MIT for code; CC-BY 4.0 for derived data
├── requirements.txt              Python dependencies
├── sessionInfo_R.txt             R packages used
├── code/                         analysis and figure-generation scripts
│   ├── 01_build_dataset.py
│   ├── 02_joinpoint_ui.py
│   ├── 03_decomposition.py
│   ├── 04_sdi_regression.py
│   ├── 05_haqi_endpoint.py
│   ├── 06_tier_sensitivity.py
│   ├── 07_forecast.py
│   ├── 08_eapc_ztest.py
│   ├── 09_joinpoint_sensitivity.py
│   ├── 10_joinpoint_redraw.py
│   ├── 11_arima_diagnostics.py
│   ├── 12_figure2.py            main-text Figure 2
│   ├── 13_prepare_bapc_data.py
│   ├── 14_bapc.R                BAPC primary projection
│   ├── 15_figure5.py            main-text Figure 5
│   ├── 16_table1_assembly.py    Table 1 CSV
│   ├── 17_compute_g20_eapc.py
│   ├── figure1_g20_choropleth.R main-text Figure 1
│   └── figure4_age_sex.R        main-text Figure 4
├── data/                         derived analytic inputs (CSV)
│   ├── merged_analytic.csv
│   ├── migraine_clean.csv
│   ├── sdi_clean.csv
│   ├── haqi_clean.csv
│   ├── tier_clean.csv
│   └── bapc_data/                BAPC input matrices (cases, populations, std_pop)
├── tables/                       analytic outputs (CSV)
│   ├── table1_enhanced.csv
│   ├── ui_overlap_1990_vs_2023.csv
│   ├── joinpoint_segment_APCs.csv
│   ├── joinpoint_sensitivity.csv
│   ├── decomposition_results.csv
│   ├── sdi_regression_results.csv
│   ├── haqi_delta_analysis.csv
│   ├── haqi_correlation_results.csv
│   ├── tier_sensitivity_eapc.csv
│   ├── arima_order_selection.csv
│   ├── arima_diagnostics_summary.csv
│   ├── forecast_validation.csv
│   ├── projections_2024_2053.csv
│   ├── bapc_projections.csv
│   ├── bapc_projections_2024_2053.csv
│   ├── bapc_projections_anchored.csv
│   ├── eapc_ztest_results.csv
│   └── country_eapc.csv
├── figures/                      main-text figures (PDF only)
│   ├── figure2.pdf
│   ├── figure3.pdf
│   └── figure5.pdf
└── docs/
    └── BAPC_README.md
```

Figures 1 and 4 are regenerated from their respective R scripts when the user runs the pipeline; the resulting `figure1_choropleth.pdf` and `figure4_age_sex.pdf` will appear in `figures/`.

## Data sources

This study does **not** redistribute raw GBD 2023 data. Investigators wishing to reproduce the analysis end-to-end should re-extract the data from the IHME Global Health Data Exchange (GHDx) using the parameters below.

| Source | URL | Extract date | Selection |
|---|---|---|---|
| GBD 2023 migraine | https://vizhub.healthdata.org/gbd-results | November 2025 | Cause: Migraine; Measures: Prevalence, Incidence, DALYs; Metrics: Rate per 100,000 and Number; Sex: Male, Female, Both; Age: all ages, age-standardized, 5-year age bands; Year: 1990–2023; Locations: China + 19 G20 sovereign members + IHME G20 aggregate (44586) + EU (4743). |
| GBD 2023 SDI | https://ghdx.healthdata.org | November 2025 | G20 sovereigns + China + G20 aggregate + EU, 1990–2023 annual. |
| GBD 2019 HAQI | https://ghdx.healthdata.org | November 2025 | G20 sovereigns + China + EU, 1990 and 2019 endpoints. |
| GBD world 2014 standard population | bundled with R BAPC package | n/a | Used for BAPC age-standardization. |

Derived datasets produced by `code/01_build_dataset.py` are included under `data/` for full reproducibility of all downstream analyses without re-extracting raw GBD data.

## How to reproduce

### Environment

```bash
# Python 3.10
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# R 4.4.2 with INLA
install.packages(c("BAPC", "INLA", "Epi", "sf", "rnaturalearth",
                   "rnaturalearthdata", "patchwork", "classInt",
                   "ggplot2", "dplyr", "tidyr", "scales"))
```

### Pipeline

Execute from inside `code/`:

```bash
# 1. Build the master analytic dataset
python 01_build_dataset.py

# 2. Descriptive trends, Joinpoint, EAPC Z-tests
python 02_joinpoint_ui.py
python 09_joinpoint_sensitivity.py
python 10_joinpoint_redraw.py
python 08_eapc_ztest.py
python 17_compute_g20_eapc.py

# 3. Decomposition and ecological analyses
python 03_decomposition.py
python 04_sdi_regression.py
python 05_haqi_endpoint.py
python 06_tier_sensitivity.py

# 4. ARIMA + ETS forecasting (Python)
python 07_forecast.py
python 11_arima_diagnostics.py

# 5. BAPC primary projection (R)
python 13_prepare_bapc_data.py
Rscript 14_bapc.R

# 6. Main-text figures
Rscript figure1_g20_choropleth.R
python 12_figure2.py
python 03_decomposition.py            # produces figure3.pdf
Rscript figure4_age_sex.R
python 15_figure5.py

# 7. Main-text Table 1
python 16_table1_assembly.py
```

## Software versions

| Tool | Version |
|---|---|
| Python | 3.10 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| statsmodels | 0.14.6 |
| matplotlib | 3.10.x |
| R | 4.4.2 |
| BAPC | 0.0.37 |
| INLA | 24.x stable |
| sf | 1.0.x |
| rnaturalearth + rnaturalearthdata | 1.0.x |
| patchwork | 1.2.x |
| classInt | 0.4.x |
| Joinpoint Regression Program | 5.4.0 (NCI standalone) |

## Citation

Qian N, Wei T, Song Y, Wang W, Han H, Wang M, Shi Q, Shao N, Qin G, Yang W, Yang Y. Migraine burden in China and G20 economies from 1990 to 2023 and exploratory projections to 2053: a descriptive analysis of GBD 2023 modeled estimates. *iScience* 2026 (in press).

## License

Code: MIT (see `LICENSE`). Derived datasets: CC BY 4.0. Raw GBD data are subject to the IHME Free-of-Charge Non-Commercial User Agreement.

## Contact

Lead contact: Wenming Yang — yangwm8810@126.com
