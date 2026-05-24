# BAPC Projection — Running Instructions

This directory contains everything needed to produce Bayesian Age-Period-Cohort (BAPC) projections for the iScience revision (ISCIENCE-D-26-04532).

## Files

| File | Purpose |
|---|---|
| `13_prepare_bapc_data.py` | Python script that built the input CSVs (already run) |
| `bapc_data/` | Input CSVs ready for R |
| `bapc_data/cases_<country>_<sex>_<measure>.csv` | Age × year case-count matrices |
| `bapc_data/pop_<country>_<sex>.csv` | Age × year population matrices (extrapolated to 2053) |
| `bapc_data/gbd_world_standard_population.csv` | World standard population weights |
| `bapc_data/_manifest.csv` | Index of all CSV files |
| `14_bapc.R` | The R script that fits BAPC and produces projections |

## Steps to run

### 1. Install R packages (one-time setup)

```r
# INLA (Bayesian inference backend)
install.packages("INLA",
  repos = c(getOption("repos"),
            INLA = "https://inla.r-inla-download.org/R/stable"),
  dep = TRUE)

# BAPC
# Try CRAN first
install.packages("BAPC")
# If that fails, R-Forge:
install.packages("BAPC", repos = "http://R-Forge.R-project.org")

# Epi (used internally)
install.packages("Epi")
```

INLA installation can take 5–10 minutes; it pulls in many dependencies. On Windows you may need Rtools.

### 2. Set working directory in R

```r
setwd("/sessions/adoring-charming-euler/mnt/outputs/analysis/bapc_data")
# or whatever the actual path is on your machine
```

### 3. Run the script

```r
source("../14_bapc.R")
```

Expected runtime: **~5–10 minutes** for all 18 BAPC fits (2 countries × 3 sexes × 3 measures).

### 4. Output

Files appear in `./bapc_output/`:

- `bapc_results.rds` — full BAPC model objects (for further analysis in R)
- `bapc_projections_2024_2053.csv` — projected age-standardized rates with 95% credible intervals, one row per year × country × sex × measure
- `bapc_projections_for_merge.csv` — same data but in the same column format as the Python ARIMA/ETS output (`projections_2024_2053.csv`), ready to be combined for the three-model comparison figure
- `bapc_projections.pdf` — visual diagnostic plots of all projections

### 5. Send `bapc_projections_for_merge.csv` back for figure integration

Once BAPC has run, send that CSV back into this session and we will:
- Merge with existing ARIMA/ETS projections
- Generate the final main-text Figure 5 (three-model fan chart)
- Update Methods §3.4 and Results §Projection with BAPC point estimates and 95% CrIs
- Update Response Letter §5.6 with BAPC results

## Notes on BAPC modeling choices

- **Priors**: RW2 (second-order random walk) on age, period, cohort effects — the standard for projecting smooth trends. RW2 priors imply that the projected trajectory continues with a continuous derivative (no abrupt change at the projection boundary).
- **Hyperprior**: loggamma(1, 0.00005) on precision — relatively informative, encourages smooth fits as recommended by Knoll et al. (2020) for sparse data.
- **Overdispersion**: IID overdispersion term included to absorb additional variation.
- **Standardization**: GBD World 2014 standard population (provided as CSV).
- **Future population**: Log-linear extrapolation from the last 15 years' age-specific populations. For publication-quality work, you may want to replace with UN World Population Prospects 2024 projections. The R script reads `pop_*.csv` files, so swapping in alternative future population estimates is straightforward.

## Reference

Riebler A, Held L. Projecting the future burden of cancer: Bayesian age-period-cohort analysis with integrated nested Laplace approximations. *Biom J.* 2017;59(3):531-549.
