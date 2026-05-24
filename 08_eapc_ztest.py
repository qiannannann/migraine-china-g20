"""
Analysis 7: Formal Z-test of China EAPC vs G20 EAPC

Z = (beta_China - beta_G20) / sqrt(SE_China^2 + SE_G20^2)
where beta = slope of log(rate) ~ year

Also compute China vs each of the alternative G20 aggregates (IHME, pop-weighted, ex-China).
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from pathlib import Path

DATA = Path("../data")
TABLES = Path("../tables")
UPLOADS = Path("../data/raw")

df = pd.read_csv(DATA / "merged_analytic.csv")
df_sov = df[df['tier'].between(1,4)].copy()

# Build population for population-weighted aggregates
mig = pd.read_csv(UPLOADS / "IHME-GBD_2023_DATA-7500ef0b-1.csv")
NAME_MAP = {"United States of America":"United States","Republic of Korea":"South Korea",
            "Russian Federation":"Russia","Türkiye":"Turkey","European Union":"EU"}
mig['country'] = mig['location_name'].replace(NAME_MAP)
aa = mig[(mig['age_name']=='All ages') & (mig['sex_name']=='Both') &
         (mig['measure_name']=='Prevalence')]
nums = aa[aa['metric_name']=='Number'][['country','year','val']].rename(columns={'val':'n_cases'})
rates = aa[aa['metric_name']=='Rate'][['country','year','val']].rename(columns={'val':'crude_rate'})
pop = nums.merge(rates, on=['country','year'])
pop['population'] = np.where(pop['crude_rate']>0, pop['n_cases']/pop['crude_rate']*100000, 0)
df_sov = df_sov.merge(pop[['country','year','population']], on=['country','year'], how='left')

def loglin_fit(y, x):
    """Fit log(y) ~ year; return beta, SE, n."""
    logy = np.log(y)
    X = sm.add_constant(x)
    fit = sm.OLS(logy, X).fit()
    return fit.params[1], fit.bse[1], len(y)

def ztest_eapcs(beta1, se1, beta2, se2):
    """Two-sample Z-test for difference in log-linear slopes."""
    delta = beta1 - beta2
    se_delta = np.sqrt(se1**2 + se2**2)
    z = delta / se_delta
    p_two = 2 * (1 - norm.cdf(abs(z)))
    ci_lo = delta - 1.96*se_delta
    ci_hi = delta + 1.96*se_delta
    eapc_diff = (np.exp(delta) - 1) * 100
    eapc_diff_lo = (np.exp(ci_lo) - 1) * 100
    eapc_diff_hi = (np.exp(ci_hi) - 1) * 100
    return {'z': z, 'p_value': p_two, 'eapc_diff_pct': eapc_diff,
            'eapc_diff_95CI_lo': eapc_diff_lo, 'eapc_diff_95CI_hi': eapc_diff_hi}

# Aggregates to test
def alt_agg(df, countries, measure, weight='pop'):
    d = df[df['country'].isin(countries)]
    out = []
    for year, g in d.groupby('year'):
        if weight == 'pop':
            v = np.average(g[measure].values, weights=g['population'].values)
        elif weight == 'unweighted':
            v = g[measure].mean()
        elif weight == 'median':
            v = g[measure].median()
        out.append({'year':year,'value':v})
    return pd.DataFrame(out).sort_values('year').reset_index(drop=True)

all_sovs = list(df_sov['country'].unique())
ex_china = [c for c in all_sovs if c != 'China']
tier12 = list(df_sov[df_sov['tier'].between(1,2)]['country'].unique())
tier12_ex_china = [c for c in tier12 if c != 'China']

# Test specs
specs = {
    'IHME official G20': ('agg_loc', 'G20'),
    'Pop-weighted G20 (19 sovs)': ('alt', all_sovs, 'pop'),
    'Pop-weighted G20 ex-China': ('alt', ex_china, 'pop'),
    'Median across 19 G20 sovs': ('alt', all_sovs, 'median'),
    'Pop-weighted Tier 1+2': ('alt', tier12, 'pop'),
    'Pop-weighted Tier 1+2 ex-China': ('alt', tier12_ex_china, 'pop'),
}

results = []
for measure in ['ASPR_val','ASIR_val','ASDR_val']:
    measure_short = measure.replace('_val','')
    # China slope
    d_china = df_sov[df_sov['country']=='China'].sort_values('year')
    beta_c, se_c, n_c = loglin_fit(d_china[measure].values, d_china['year'].values)
    eapc_c = (np.exp(beta_c) - 1) * 100

    for label, spec in specs.items():
        if spec[0] == 'agg_loc':
            d_agg = df[df['country']==spec[1]].sort_values('year')
            y_agg = d_agg[measure].values
            x_agg = d_agg['year'].values
        else:
            _, countries, weight = spec
            agg = alt_agg(df_sov, countries, measure, weight)
            y_agg = agg['value'].values
            x_agg = agg['year'].values
        beta_a, se_a, n_a = loglin_fit(y_agg, x_agg)
        eapc_a = (np.exp(beta_a) - 1) * 100
        test = ztest_eapcs(beta_c, se_c, beta_a, se_a)
        results.append({
            'measure': measure_short,
            'aggregate': label,
            'china_EAPC_pct': round(eapc_c,3),
            'aggregate_EAPC_pct': round(eapc_a,3),
            'eapc_diff_pct': round(test['eapc_diff_pct'],3),
            'eapc_diff_95CI': f"({test['eapc_diff_95CI_lo']:.3f}, {test['eapc_diff_95CI_hi']:.3f})",
            'Z': round(test['z'],3),
            'p_value': f"{test['p_value']:.3g}",
            'n_China': n_c, 'n_aggregate': n_a,
        })

resdf = pd.DataFrame(results)
resdf.to_csv(TABLES / "eapc_ztest_results.csv", index=False)
print("=== China vs G20 aggregate EAPC differences (formal Z-test of log-linear slopes) ===\n")
print(resdf.to_string(index=False))

# Summary line
print("\n=== Summary ===")
for m in ['ASPR','ASIR','ASDR']:
    r = resdf[(resdf['measure']==m) & (resdf['aggregate']=='IHME official G20')].iloc[0]
    sig = 'YES' if float(r['p_value']) < 0.05 else 'no'
    print(f"  {m}: China EAPC {r['china_EAPC_pct']:.3f}% vs IHME G20 {r['aggregate_EAPC_pct']:.3f}% — "
          f"diff = {r['eapc_diff_pct']:+.3f}% (Z={r['Z']:.2f}, p={r['p_value']}, significant: {sig})")
