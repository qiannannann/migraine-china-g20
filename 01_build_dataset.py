"""
Build merged analytic dataset:
  migraine (ASPR/ASIR/ASDR + counts + UIs) × SDI × HAQI × Tier × population
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

UPLOADS = Path("../data/raw")
OUT = Path("../data")

# ---------- 1. Migraine ----------
print("Loading migraine GBD 2023 data...")
mig = pd.read_csv(UPLOADS / "IHME-GBD_2023_DATA-7500ef0b-1.csv")
print(f"  rows: {len(mig):,}  countries: {mig['location_name'].nunique()}  years: {mig['year'].nunique()}")

# Clean: drop African Union (not relevant), keep G20 + EU + 19 sovereigns + China
keep_locs = ['China','Argentina','Australia','Brazil','Canada','France','Germany',
             'India','Indonesia','Italy','Japan','Mexico','Republic of Korea',
             'Russian Federation','Saudi Arabia','South Africa','Türkiye',
             'United Kingdom','United States of America','European Union','G20']
mig = mig[mig['location_name'].isin(keep_locs)].copy()

# Standardize country names
NAME_MAP = {
    "United States of America": "United States",
    "Republic of Korea": "South Korea",
    "Russian Federation": "Russia",
    "Türkiye": "Turkey",
    "European Union": "EU",
}
mig['country'] = mig['location_name'].replace(NAME_MAP)
print(f"  filtered to {mig['country'].nunique()} entities")

# ---------- 2. SDI ----------
print("Loading SDI...")
sdi = pd.read_csv(UPLOADS / "IHME_GBD_2023_SDI_1950_2023_Y2025M10D12.csv")
sdi = sdi[(sdi['year_id']>=1990)].copy()
# Same country renaming
sdi['country'] = sdi['location_name'].replace({**NAME_MAP, "United States of America": "United States"})
sdi_keep_locs = list(set(mig['country']))
sdi = sdi[sdi['country'].isin(sdi_keep_locs)].copy()
sdi = sdi.rename(columns={'year_id':'year','mean_value':'sdi'})[['country','year','sdi']]
print(f"  rows: {len(sdi):,}")

# ---------- 3. HAQI ----------
print("Loading HAQI...")
haqi = pd.read_csv("/tmp/haq/IHME_GBD_2019_HAQ_1990_2019_DATA_Y2022M012D21.CSV")
haqi = haqi[(haqi['indicator_id']==100) & (haqi['haq_index_age_type']=='Overall')].copy()
haqi['country'] = haqi['location_name'].replace({**NAME_MAP, "United States of America": "United States"})
haqi = haqi[haqi['country'].isin(sdi_keep_locs)].copy()
haqi = haqi.rename(columns={'year_id':'year','val':'haqi','lower':'haqi_lo','upper':'haqi_hi'})[['country','year','haqi','haqi_lo','haqi_hi']]
print(f"  rows: {len(haqi):,}  unique years: {sorted(haqi['year'].unique())}")

# ---------- 4. Tier classification ----------
print("Loading Tier classification...")
with open('/tmp/g20_summary.json') as f:
    tier_data = json.load(f)
tier_df = pd.DataFrame(tier_data)
tier_df['country'] = tier_df['country'].replace({**NAME_MAP, "United States of America": "United States"})
tier_df = tier_df[['country','n_total','n_lit','n_survey','tier']].rename(columns={'n_total':'n_input_studies'})
print(f"  rows: {len(tier_df)}")
print(tier_df.sort_values(['tier','n_input_studies'], ascending=[True, False]).to_string(index=False))

# ---------- 5. Save individual cleaned files ----------
mig.to_csv(OUT / "migraine_clean.csv", index=False)
sdi.to_csv(OUT / "sdi_clean.csv", index=False)
haqi.to_csv(OUT / "haqi_clean.csv", index=False)
tier_df.to_csv(OUT / "tier_clean.csv", index=False)

# ---------- 6. Build wide-format core dataset ----------
# For each country × year, get ASPR/ASIR/ASDR (Rate, Both sexes, Age-standardized)
asr = mig[(mig['age_name']=='Age-standardized') & (mig['sex_name']=='Both') & (mig['metric_name']=='Rate')].copy()

# Manually pivot to avoid pivot_table column-mapping ambiguity
MEASURE_SHORT = {
    'Prevalence': 'ASPR',
    'Incidence': 'ASIR',
    'DALYs (Disability-Adjusted Life Years)': 'ASDR',
}
asr['short'] = asr['measure_name'].map(MEASURE_SHORT)

frames = []
for short_name, group in asr.groupby('short'):
    g = group[['country','year','val','lower','upper']].rename(columns={
        'val':   f'{short_name}_val',
        'lower': f'{short_name}_lower',
        'upper': f'{short_name}_upper',
    })
    frames.append(g)
asr_wide = frames[0]
for f in frames[1:]:
    asr_wide = asr_wide.merge(f, on=['country','year'], how='outer')

# Counts (Number) too
counts = mig[(mig['age_name']=='All ages') & (mig['sex_name']=='Both') & (mig['metric_name']=='Number')]
counts_wide = counts.pivot_table(index=['country','year'], columns='measure_name', values='val').reset_index()
counts_wide.columns = ['country','year','DALY_count','Incident_cases','Prevalent_cases']

# Merge
merged = asr_wide.merge(counts_wide, on=['country','year'], how='left')
merged = merged.merge(sdi, on=['country','year'], how='left')
# HAQI only 1990 and 2019
merged = merged.merge(haqi[['country','year','haqi']], on=['country','year'], how='left')
# Tier
merged = merged.merge(tier_df[['country','tier','n_input_studies']], on='country', how='left')
# EU and G20 don't have Tier — assign separately
merged.loc[merged['country']=='EU', 'tier'] = 0  # 0 = aggregate
merged.loc[merged['country']=='G20', 'tier'] = 0
merged.loc[merged['country']=='EU', 'n_input_studies'] = np.nan
merged.loc[merged['country']=='G20', 'n_input_studies'] = np.nan

# Column order
core_cols = ['country','year','tier','n_input_studies','sdi','haqi',
             'ASPR_val','ASPR_lower','ASPR_upper',
             'ASIR_val','ASIR_lower','ASIR_upper',
             'ASDR_val','ASDR_lower','ASDR_upper',
             'Prevalent_cases','Incident_cases','DALY_count']
merged = merged[core_cols].sort_values(['country','year']).reset_index(drop=True)
merged.to_csv(OUT / "merged_analytic.csv", index=False)

print(f"\n=== Merged dataset built ===")
print(f"  shape: {merged.shape}")
print(f"  countries: {merged['country'].nunique()}")
print(f"  years: {merged['year'].min()}–{merged['year'].max()}")
print(f"  saved: {OUT / 'merged_analytic.csv'}")
print(f"\nSample (China, every 5 years):")
print(merged[(merged['country']=='China') & (merged['year'].isin([1990,1995,2000,2005,2010,2015,2020,2023]))].to_string(index=False))
