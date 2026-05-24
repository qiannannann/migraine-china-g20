"""
Analysis 5: Tier-stratified sensitivity analysis
- Compute alternative G20 aggregates: population-weighted vs unweighted vs median
- Compute G20-ex-China and G20 restricted to data-rich tiers
- Compare China EAPC against these alternative aggregates
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

DATA = Path("../data")
FIG = Path("../figures")
TABLES = Path("../tables")
UPLOADS = Path("../data/raw")

df = pd.read_csv(DATA / "merged_analytic.csv")
df_sov = df[df['tier'].between(1,4)].copy()

# We need population counts to do population-weighted aggregation
# Derive from the migraine data (number/rate * 100k = population)
mig_raw = pd.read_csv(UPLOADS / "IHME-GBD_2023_DATA-7500ef0b-1.csv")
NAME_MAP = {"United States of America":"United States","Republic of Korea":"South Korea",
            "Russian Federation":"Russia","Türkiye":"Turkey","European Union":"EU"}
mig_raw['country'] = mig_raw['location_name'].replace(NAME_MAP)

# All-ages, Both sexes — for total population
allages_b = mig_raw[(mig_raw['age_name']=='All ages') & (mig_raw['sex_name']=='Both') &
                    (mig_raw['measure_name']=='Prevalence')]
nums = allages_b[allages_b['metric_name']=='Number'][['country','year','val']].rename(columns={'val':'n_cases'})
rates = allages_b[allages_b['metric_name']=='Rate'][['country','year','val']].rename(columns={'val':'crude_rate'})
pop = nums.merge(rates, on=['country','year'])
pop['population'] = np.where(pop['crude_rate']>0, pop['n_cases']/pop['crude_rate']*100000, 0)

df_sov = df_sov.merge(pop[['country','year','population']], on=['country','year'], how='left')

# ---------- Compute EAPC for each country, each measure ----------
def compute_eapc(y, x):
    """EAPC = (exp(beta) - 1) * 100 from log-linear regression of rate on year."""
    valid = (y>0)
    if valid.sum() < 5: return None, None, None
    logy = np.log(y[valid])
    xs = x[valid]
    X = sm.add_constant(xs)
    fit = sm.OLS(logy, X).fit()
    beta = fit.params[1]
    se = fit.bse[1]
    eapc = (np.exp(beta) - 1) * 100
    eapc_lo = (np.exp(beta - 1.96*se) - 1) * 100
    eapc_hi = (np.exp(beta + 1.96*se) - 1) * 100
    return eapc, eapc_lo, eapc_hi

# Per-country EAPC
country_eapc = []
for c in df_sov['country'].unique():
    d = df_sov[df_sov['country']==c].sort_values('year')
    for m in ['ASPR_val','ASIR_val','ASDR_val']:
        eapc, lo, hi = compute_eapc(d[m].values, d['year'].values)
        country_eapc.append({'country':c, 'tier':int(d['tier'].iloc[0]), 'measure':m.replace('_val',''),
                            'eapc':eapc, 'eapc_lo':lo, 'eapc_hi':hi})
country_eapc_df = pd.DataFrame(country_eapc)

# ---------- Compute alternative aggregates ----------
def alt_aggregate(df, countries, measure, weights=None):
    """Compute aggregate ASR over time for given country set."""
    d = df[df['country'].isin(countries)]
    agg = []
    for year, g in d.groupby('year'):
        if weights is None:
            v = g[measure].mean()
        elif weights == 'pop':
            w = g['population'].values
            v = np.average(g[measure].values, weights=w)
        elif weights == 'median':
            v = g[measure].median()
        agg.append({'year': year, 'value': v})
    return pd.DataFrame(agg).sort_values('year')

all_sovs = list(df_sov['country'].unique())
all_sovs_ex_china = [c for c in all_sovs if c != 'China']
tier12 = list(df_sov[df_sov['tier'].between(1,2)]['country'].unique())
tier12_ex_china = [c for c in tier12 if c != 'China']
tier123 = list(df_sov[df_sov['tier'].between(1,3)]['country'].unique())

aggregates = {
    'IHME official G20': None,  # use the G20 row directly
    'Unweighted mean, all 19 G20': (all_sovs, None),
    'Pop-weighted mean, all 19 G20': (all_sovs, 'pop'),
    'Median, all 19 G20': (all_sovs, 'median'),
    'Pop-weighted mean, G20 ex-China': (all_sovs_ex_china, 'pop'),
    'Pop-weighted mean, Tier 1+2 (data-rich)': (tier12, 'pop'),
    'Pop-weighted mean, Tier 1+2 ex-China': (tier12_ex_china, 'pop'),
    'Pop-weighted mean, Tier 1+2+3 (exclude zero-data)': (tier123, 'pop'),
}

# Compute EAPC for each aggregate and each measure
agg_eapc = []
for label, spec in aggregates.items():
    for measure in ['ASPR_val','ASIR_val','ASDR_val']:
        if spec is None:
            # Use IHME's G20 row
            d = df[df['country']=='G20'].sort_values('year')
            series = d[[measure,'year']].rename(columns={measure:'value'})
        else:
            countries, weight = spec
            series = alt_aggregate(df_sov, countries, measure, weight)
        eapc, lo, hi = compute_eapc(series['value'].values, series['year'].values)
        agg_eapc.append({
            'aggregate': label, 'measure': measure.replace('_val',''),
            'eapc': eapc, 'eapc_95CI_lo': lo, 'eapc_95CI_hi': hi,
            'series_1990': series['value'].iloc[0],
            'series_2023': series['value'].iloc[-1],
        })

# China EAPC for comparison
china_eapc = country_eapc_df[country_eapc_df['country']=='China'].set_index('measure')[['eapc','eapc_lo','eapc_hi']]

# Pull together comparison table
comparison = []
for row in agg_eapc:
    measure = row['measure']
    china_e = china_eapc.loc[measure]
    comparison.append({
        'aggregate': row['aggregate'],
        'measure': measure,
        'aggregate_EAPC': round(row['eapc'],3),
        'aggregate_95CI': f"({row['eapc_95CI_lo']:.2f}, {row['eapc_95CI_hi']:.2f})",
        'china_EAPC': round(china_e['eapc'],3),
        'china_95CI': f"({china_e['eapc_lo']:.2f}, {china_e['eapc_hi']:.2f})",
        'china_minus_agg': round(china_e['eapc'] - row['eapc'], 3),
    })

comp_df = pd.DataFrame(comparison)
comp_df.to_csv(TABLES / "tier_sensitivity_eapc.csv", index=False)
print("=== Tier-stratified EAPC comparison: China vs alternative G20 aggregates ===\n")
print(comp_df.to_string(index=False))

# Save country-level EAPC
country_eapc_df['eapc']=country_eapc_df['eapc'].round(3)
country_eapc_df['eapc_lo']=country_eapc_df['eapc_lo'].round(3)
country_eapc_df['eapc_hi']=country_eapc_df['eapc_hi'].round(3)
country_eapc_df.to_csv(TABLES / "country_eapc.csv", index=False)

# ---------- Plot: Forest plot of EAPC by country, by tier ----------
TIER_COLORS = {1: '#1B7837', 2: '#5AAE61', 3: '#F4A582', 4: '#B2182B'}
fig, axes = plt.subplots(1, 3, figsize=(15, 7))
for ax, m in zip(axes, ['ASPR','ASIR','ASDR']):
    d = country_eapc_df[country_eapc_df['measure']==m].sort_values(['tier','eapc'])
    y = np.arange(len(d))
    for i, (_, row) in enumerate(d.iterrows()):
        color = TIER_COLORS[row['tier']]
        is_china = row['country']=='China'
        ax.plot([row['eapc_lo'], row['eapc_hi']], [i,i], color=color, lw=2.5, alpha=0.9 if is_china else 0.7)
        ax.plot(row['eapc'], i, 'o' if not is_china else '*',
                color='#D62728' if is_china else color,
                markersize=15 if is_china else 9, markeredgecolor='black',
                markeredgewidth=1.5 if is_china else 0.7, zorder=10)
    ax.set_yticks(y)
    ax.set_yticklabels([f"T{int(r['tier'])} {r['country']}" for _,r in d.iterrows()], fontsize=8)
    ax.axvline(0, color='gray', lw=0.7)
    ax.set_xlabel('EAPC (% per year), 1990–2023')
    ax.set_title(m)
    ax.grid(True, axis='x', alpha=0.3)
fig.suptitle('Per-country EAPC of migraine ASR, 1990–2023 (color = data-availability Tier; China shown as red star)', fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(FIG / "fig08_country_eapc_forest.pdf", bbox_inches='tight')
plt.close()
print("\nSaved fig08_country_eapc_forest.png")

# ---------- Plot: Sensitivity bar chart ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
agg_labels_order = list(aggregates.keys())
for ax, m in zip(axes, ['ASPR','ASIR','ASDR']):
    d = comp_df[comp_df['measure']==m].set_index('aggregate').loc[agg_labels_order].reset_index()
    y = np.arange(len(d))
    ax.barh(y, d['aggregate_EAPC'], color='#1F77B4', alpha=0.7, label='Aggregate')
    china_e = float(d['china_EAPC'].iloc[0])
    ax.axvline(china_e, color='#D62728', lw=2.5, label=f'China EAPC = {china_e:.3f}', zorder=10)
    ax.set_yticks(y)
    ax.set_yticklabels(d['aggregate'], fontsize=8)
    ax.set_xlabel(f'EAPC ({m})')
    ax.set_title(m)
    ax.grid(True, axis='x', alpha=0.3)
    if m == 'ASPR':
        ax.legend(loc='lower right', fontsize=9, frameon=False)
fig.suptitle('China EAPC robustness check across alternative G20 aggregation choices', fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(FIG / "fig09_tier_sensitivity.pdf", bbox_inches='tight')
plt.close()
print("Saved fig09_tier_sensitivity.png")
