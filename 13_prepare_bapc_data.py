"""
Prepare BAPC input data:
For each (country, sex, measure), build:
  - cases matrix: rows = age groups, cols = years (1990-2023 actual + 2024-2053 NA)
  - population matrix: rows = age groups, cols = years (1990-2023 actual + 2024-2053 extrapolated)

Output: one CSV per (country, sex, measure) plus a combined README.
"""
import pandas as pd
import numpy as np
from pathlib import Path

UPLOADS = Path("../data/raw")
OUT = Path("../bapc_data")
OUT.mkdir(parents=True, exist_ok=True)

# Load GBD migraine raw data
print("Loading GBD migraine data...")
mig = pd.read_csv(UPLOADS / "IHME-GBD_2023_DATA-7500ef0b-1.csv")

# Standardize country names
NAME_MAP = {
    "United States of America": "United States",
    "Republic of Korea": "South Korea",
    "Russian Federation": "Russia",
    "Türkiye": "Turkey",
    "European Union": "EU",
}
mig['country'] = mig['location_name'].replace(NAME_MAP)

# Age groups in 5-year order
AGE_ORDER = ['<5 years','5-9 years','10-14 years','15-19 years','20-24 years','25-29 years',
             '30-34 years','35-39 years','40-44 years','45-49 years','50-54 years','55-59 years',
             '60-64 years','65-69 years','70-74 years','75-79 years','80-84 years','85-89 years',
             '90-94 years','95+ years']

MEASURE_SHORT = {
    'Prevalence': 'prev',
    'Incidence': 'inc',
    'DALYs (Disability-Adjusted Life Years)': 'daly',
}

# Years
HISTORICAL_YEARS = list(range(1990, 2024))
FUTURE_YEARS = list(range(2024, 2054))
ALL_YEARS = HISTORICAL_YEARS + FUTURE_YEARS

def get_age_year_matrix(country, sex, measure_long, metric):
    """Return matrix (age × year) for given country/sex/measure/metric."""
    d = mig[(mig['country']==country) & (mig['sex_name']==sex) &
            (mig['measure_name']==measure_long) & (mig['metric_name']==metric) &
            (mig['age_name'].isin(AGE_ORDER))]
    pivot = d.pivot_table(index='age_name', columns='year', values='val')
    # Reorder rows by AGE_ORDER
    pivot = pivot.reindex(AGE_ORDER)
    # Ensure all historical years present
    for y in HISTORICAL_YEARS:
        if y not in pivot.columns:
            pivot[y] = np.nan
    pivot = pivot[HISTORICAL_YEARS]
    return pivot

def extrapolate_population(pop_hist):
    """
    Project population for 2024-2053 by:
    - Fit linear trend on log scale to each age group's historical population
    - Project forward
    - This is a SIMPLIFIED projection; for publication, user should swap in
      UN WPP 2024 projections if available.
    """
    projected = pd.DataFrame(index=pop_hist.index, columns=FUTURE_YEARS, dtype=float)
    for age in pop_hist.index:
        y_hist = pop_hist.loc[age].values.astype(float)
        # Only use last 15 years for projection (more reflective of recent trend)
        recent_y = HISTORICAL_YEARS[-15:]
        recent_v = y_hist[-15:]
        # Filter positive values for log
        mask = recent_v > 0
        if mask.sum() < 5:
            # Fallback: hold last value constant
            projected.loc[age] = y_hist[-1]
            continue
        xs = np.array(recent_y)[mask].astype(float)
        ys = np.log(recent_v[mask])
        # Fit line
        slope, intercept = np.polyfit(xs, ys, 1)
        # Project
        future_logs = intercept + slope * np.array(FUTURE_YEARS, dtype=float)
        projected.loc[age] = np.exp(future_logs)
    return projected

# Build matrices for each (country, sex, measure)
focus_countries = ['China', 'G20']
focus_sexes = ['Both', 'Male', 'Female']
focus_measures = ['Prevalence', 'Incidence', 'DALYs (Disability-Adjusted Life Years)']

manifest_records = []

for country in focus_countries:
    for sex in focus_sexes:
        # Get population (derive from Prevalence Number ÷ Prevalence Rate × 100,000)
        cases_num_prev = get_age_year_matrix(country, sex, 'Prevalence', 'Number')
        cases_rate_prev = get_age_year_matrix(country, sex, 'Prevalence', 'Rate')
        pop_hist = (cases_num_prev / cases_rate_prev) * 100000.0
        pop_hist = pop_hist.replace([np.inf, -np.inf], np.nan)

        # Extrapolate population
        pop_future = extrapolate_population(pop_hist)
        pop_all = pd.concat([pop_hist, pop_future], axis=1)
        pop_all.columns = ALL_YEARS
        pop_all.index.name = 'age_group'

        # Save population
        pop_filename = f"pop_{country.replace(' ','')}_{sex}.csv"
        pop_all.to_csv(OUT / pop_filename)
        manifest_records.append({
            'country': country, 'sex': sex, 'measure': 'population',
            'filename': pop_filename, 'rows_age_groups': len(AGE_ORDER),
            'cols_years': len(ALL_YEARS),
            'period': f'1990-2023 actual + 2024-2053 extrapolated (log-linear, last 15 years)',
        })

        # Now build cases matrix for each measure
        for measure in focus_measures:
            cases_num = get_age_year_matrix(country, sex, measure, 'Number')
            # Future columns: NA (BAPC will project)
            cases_future = pd.DataFrame(index=cases_num.index, columns=FUTURE_YEARS, dtype=float)
            cases_all = pd.concat([cases_num, cases_future], axis=1)
            cases_all.columns = ALL_YEARS
            cases_all.index.name = 'age_group'

            short = MEASURE_SHORT[measure]
            cases_filename = f"cases_{country.replace(' ','')}_{sex}_{short}.csv"
            cases_all.to_csv(OUT / cases_filename)
            manifest_records.append({
                'country': country, 'sex': sex, 'measure': measure,
                'filename': cases_filename, 'rows_age_groups': len(AGE_ORDER),
                'cols_years': len(ALL_YEARS),
                'period': f'1990-2023 actual + 2024-2053 NA (for BAPC projection)',
            })

manifest = pd.DataFrame(manifest_records)
manifest.to_csv(OUT / "_manifest.csv", index=False)

# Standard population for age-standardization — WHO World 2014
# (per IHME GBD documentation; values sum to ~1.0)
# v2.5 fix: previous version had decimal-shift bug (0.86 instead of 0.0860 etc.)
gbd_world_std = pd.Series({
    '<5 years':       0.0886,
    '5-9 years':      0.0869,
    '10-14 years':    0.0860,
    '15-19 years':    0.0847,
    '20-24 years':    0.0822,
    '25-29 years':    0.0793,
    '30-34 years':    0.0761,
    '35-39 years':    0.0715,
    '40-44 years':    0.0659,
    '45-49 years':    0.0604,
    '50-54 years':    0.0537,
    '55-59 years':    0.0455,
    '60-64 years':    0.0372,
    '65-69 years':    0.0296,
    '70-74 years':    0.0221,
    '75-79 years':    0.0152,
    '80-84 years':    0.0091,
    '85-89 years':    0.0044,
    '90-94 years':    0.0015,
    '95+ years':      0.0004,
}, name='gbd_world_std_pop')
# Normalize (sum is ~1.0003; final values essentially unchanged)
gbd_world_std = gbd_world_std / gbd_world_std.sum()
gbd_world_std.to_csv(OUT / "gbd_world_standard_population.csv")

print(f"\n=== BAPC input data preparation complete ===")
print(f"Output directory: {OUT}")
print(f"Files created: {len(list(OUT.glob('*.csv')))}")
print(f"Manifest:\n{manifest.to_string(index=False)}")
print(f"\nNext step: run 14_bapc.R in R/RStudio using these CSVs as input")
