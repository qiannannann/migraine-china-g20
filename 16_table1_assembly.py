"""
Generate enhanced Table 1 — adds:
- Data-availability Tier column (per Reviewer 2/5)
- EAPC with 95% CI per measure (per Reviewer 1.4 effect sizes)
- Z-test p-value vs China for each country (per Reviewer 4.5)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from scipy.stats import norm

DATA = Path("../data")
TABLES = Path("../tables")
OUT = Path("..")

# Load
df = pd.read_csv(DATA / "merged_analytic.csv")
country_eapc = pd.read_csv(TABLES / "country_eapc.csv")

# Compute China's EAPC slopes + SEs for Z-tests
def loglin_fit(y, x):
    valid = y > 0
    if valid.sum() < 5: return None, None
    logy = np.log(y[valid])
    xs = x[valid]
    X = sm.add_constant(xs)
    fit = sm.OLS(logy, X).fit()
    return fit.params[1], fit.bse[1]

china_slopes = {}
for measure in ['ASPR_val','ASIR_val','ASDR_val']:
    d = df[df['country']=='China'].sort_values('year')
    beta, se = loglin_fit(d[measure].values, d['year'].values)
    china_slopes[measure.replace('_val','')] = (beta, se)

# For each country, compute slope + SE, then Z-test vs China
def ztest(b1, se1, b2, se2):
    delta = b1 - b2
    se = np.sqrt(se1**2 + se2**2)
    z = delta / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p

# Compose full Table 1 rows
COUNTRIES = ['China','Argentina','Australia','Brazil','Canada','France','Germany',
             'India','Indonesia','Italy','Japan','Mexico','South Korea','Russia',
             'Saudi Arabia','South Africa','Turkey','United Kingdom','United States',
             'EU','G20']

rows = []
for c in COUNTRIES:
    d = df[df['country']==c].sort_values('year').reset_index(drop=True)
    if d.empty: continue
    tier = d['tier'].iloc[0]
    tier_str = '—' if pd.isna(tier) or tier == 0 else f"Tier {int(tier)}"

    # 2023 ASR values + UIs
    d2023 = d[d['year']==2023]
    if d2023.empty: continue
    aspr  = f"{d2023['ASPR_val'].iloc[0]:,.0f}"
    aspr_ui = f"({d2023['ASPR_lower'].iloc[0]:,.0f}–{d2023['ASPR_upper'].iloc[0]:,.0f})"
    asir  = f"{d2023['ASIR_val'].iloc[0]:,.1f}"
    asir_ui = f"({d2023['ASIR_lower'].iloc[0]:,.1f}–{d2023['ASIR_upper'].iloc[0]:,.1f})"
    asdr  = f"{d2023['ASDR_val'].iloc[0]:,.1f}"
    asdr_ui = f"({d2023['ASDR_lower'].iloc[0]:,.1f}–{d2023['ASDR_upper'].iloc[0]:,.1f})"

    # EAPCs
    eapcs = country_eapc[country_eapc['country']==c]
    eapc_strs = {}
    z_p = {}
    for measure in ['ASPR','ASIR','ASDR']:
        e = eapcs[eapcs['measure']==measure]
        if not e.empty:
            v = e['eapc'].iloc[0]
            lo = e['eapc_lo'].iloc[0]
            hi = e['eapc_hi'].iloc[0]
            eapc_strs[measure] = f"{v:+.3f} ({lo:+.3f} to {hi:+.3f})"
        else:
            eapc_strs[measure] = '—'

        # Z-test vs China (only for non-China sovereigns)
        if c == 'China' or c in ['EU','G20'] or e.empty:
            z_p[measure] = '—'
        else:
            beta, se = loglin_fit(d[f'{measure}_val'].values, d['year'].values)
            if beta is None:
                z_p[measure] = '—'
            else:
                cb, cse = china_slopes[measure]
                z, p = ztest(cb, cse, beta, se)
                z_p[measure] = f"{p:.3g}"

    rows.append({
        'Country': c,
        'Tier': tier_str,
        'ASPR_2023': f"{aspr} {aspr_ui}",
        'ASPR_EAPC': eapc_strs['ASPR'],
        'ASPR_Z_p': z_p['ASPR'],
        'ASIR_2023': f"{asir} {asir_ui}",
        'ASIR_EAPC': eapc_strs['ASIR'],
        'ASIR_Z_p': z_p['ASIR'],
        'ASDR_2023': f"{asdr} {asdr_ui}",
        'ASDR_EAPC': eapc_strs['ASDR'],
        'ASDR_Z_p': z_p['ASDR'],
    })

table_df = pd.DataFrame(rows)
table_df.to_csv(TABLES / "table1_enhanced.csv", index=False)
print(f"Wrote table1_enhanced.csv ({len(table_df)} rows)")
print("\nPreview (first 5 rows, key columns):")
print(table_df[['Country','Tier','ASPR_EAPC','ASPR_Z_p','ASDR_EAPC','ASDR_Z_p']].head().to_string(index=False))
