"""
Analysis 3: SDI mixed-effects regression
- Country-year ecological association: ASPR/ASIR/ASDR ~ SDI + (1|country)
- Use only 19 G20 sovereigns (exclude G20/EU aggregates)
- Use Tier 1-3 countries only (exclude Tier 4 zero-data countries — sensitivity)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.formula.api as smf

DATA = Path("../data")
FIG = Path("../figures")
TABLES = Path("../tables")

df = pd.read_csv(DATA / "merged_analytic.csv")
df = df[df['tier'].between(1, 4)].copy()  # exclude aggregates

# Mixed-effects model with random intercept per country
results = []
for measure in ['ASPR_val','ASIR_val','ASDR_val']:
    # Full model: all 19 G20
    md = smf.mixedlm(f"{measure} ~ sdi", df, groups="country")
    fit = md.fit(method='lbfgs', maxiter=200)
    sdi_coef = fit.params['sdi']
    sdi_se = fit.bse['sdi']
    sdi_p = fit.pvalues['sdi']
    sdi_ci = (sdi_coef - 1.96*sdi_se, sdi_coef + 1.96*sdi_se)
    results.append({
        'measure': measure.replace('_val',''),
        'model': 'All G20 (Tier 1-4)',
        'n_obs': len(df),
        'n_countries': df['country'].nunique(),
        'sdi_coef': sdi_coef,
        'sdi_se': sdi_se,
        'sdi_p': sdi_p,
        'sdi_95CI_lo': sdi_ci[0],
        'sdi_95CI_hi': sdi_ci[1],
        'sigma_country': float(fit.cov_re.iloc[0,0])**0.5,
        'aic': fit.aic,
    })

    # Sensitivity: exclude Tier 4 (zero-data countries)
    d_sens = df[df['tier'] < 4]
    md2 = smf.mixedlm(f"{measure} ~ sdi", d_sens, groups="country")
    fit2 = md2.fit(method='lbfgs', maxiter=200)
    sdi_coef2 = fit2.params['sdi']
    sdi_se2 = fit2.bse['sdi']
    results.append({
        'measure': measure.replace('_val',''),
        'model': 'Tier 1-3 (exclude zero-data Tier 4)',
        'n_obs': len(d_sens),
        'n_countries': d_sens['country'].nunique(),
        'sdi_coef': sdi_coef2,
        'sdi_se': sdi_se2,
        'sdi_p': fit2.pvalues['sdi'],
        'sdi_95CI_lo': sdi_coef2 - 1.96*sdi_se2,
        'sdi_95CI_hi': sdi_coef2 + 1.96*sdi_se2,
        'sigma_country': float(fit2.cov_re.iloc[0,0])**0.5,
        'aic': fit2.aic,
    })

    # Sensitivity: Tier 1+2 only (data-rich)
    d_rich = df[df['tier'].between(1,2)]
    md3 = smf.mixedlm(f"{measure} ~ sdi", d_rich, groups="country")
    fit3 = md3.fit(method='lbfgs', maxiter=200)
    results.append({
        'measure': measure.replace('_val',''),
        'model': 'Tier 1+2 only (data-rich)',
        'n_obs': len(d_rich),
        'n_countries': d_rich['country'].nunique(),
        'sdi_coef': fit3.params['sdi'],
        'sdi_se': fit3.bse['sdi'],
        'sdi_p': fit3.pvalues['sdi'],
        'sdi_95CI_lo': fit3.params['sdi'] - 1.96*fit3.bse['sdi'],
        'sdi_95CI_hi': fit3.params['sdi'] + 1.96*fit3.bse['sdi'],
        'sigma_country': float(fit3.cov_re.iloc[0,0])**0.5,
        'aic': fit3.aic,
    })

resdf = pd.DataFrame(results)
for c in ['sdi_coef','sdi_se','sdi_95CI_lo','sdi_95CI_hi','sigma_country','aic']:
    resdf[c] = resdf[c].round(2)
resdf['sdi_p'] = resdf['sdi_p'].apply(lambda p: f"{p:.4g}")
resdf.to_csv(TABLES / "sdi_regression_results.csv", index=False)

print("=== SDI mixed-effects regression results ===\n")
print(resdf.to_string(index=False))

# ---------- Plot: scatter ASR vs SDI by country with regression overlay ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
TIER_COLORS = {1: '#1B7837', 2: '#5AAE61', 3: '#F4A582', 4: '#B2182B'}
sovs = df[df['tier'].between(1,4)]

for ax, measure, label in zip(axes,
                              ['ASPR_val','ASIR_val','ASDR_val'],
                              ['ASPR per 100,000','ASIR per 100,000','ASDR per 100,000']):
    for c in sovs['country'].unique():
        d = sovs[sovs['country']==c].sort_values('year')
        tier = int(d['tier'].iloc[0])
        color = TIER_COLORS[tier]
        ax.plot(d['sdi'], d[measure], '-', color=color, alpha=0.4, lw=0.8)
        # Highlight China
        if c == 'China':
            ax.plot(d['sdi'], d[measure], '-', color='#D62728', lw=2.5, alpha=0.95, label='China', zorder=10)
        # Mark 2023 endpoint
        d23 = d[d['year']==2023]
        if not d23.empty:
            ax.plot(d23['sdi'].iloc[0], d23[measure].iloc[0], 'o',
                    color=color, markeredgecolor='black', markersize=8, zorder=8)
            ax.annotate(c, (d23['sdi'].iloc[0], d23[measure].iloc[0]),
                       fontsize=6.5, xytext=(4,4), textcoords='offset points', alpha=0.85)
    # Marginal fit line
    res_row = resdf[(resdf['measure']==measure.replace('_val','')) & (resdf['model']=='All G20 (Tier 1-4)')].iloc[0]
    # fit line at average country intercept
    grand_intercept = sovs[measure].mean() - float(res_row['sdi_coef']) * sovs['sdi'].mean()
    xs = np.linspace(sovs['sdi'].min(), sovs['sdi'].max(), 50)
    ys = grand_intercept + float(res_row['sdi_coef']) * xs
    ax.plot(xs, ys, '--', color='black', lw=1.2, alpha=0.7,
            label=f"β_SDI = {res_row['sdi_coef']:.0f} (p={res_row['sdi_p']})")

    ax.set_xlabel('SDI (Socio-Demographic Index)')
    ax.set_ylabel(label)
    ax.set_title(measure.replace('_val',''))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left', frameon=False)

# Tier color legend
from matplotlib.patches import Patch
tier_legend = [Patch(facecolor=TIER_COLORS[t], label=f"Tier {t}") for t in [1,2,3,4]]
tier_legend.append(Patch(facecolor='#D62728', label='China'))
fig.legend(handles=tier_legend, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.suptitle('Country-year association: GBD 2023 migraine ASR vs Socio-Demographic Index (SDI), 1990–2023\nLines connect country observations across years; black dashed = mixed-effects population slope', fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(FIG / "fig06_sdi_regression.pdf", bbox_inches='tight')
plt.close()
print("\nSaved fig06_sdi_regression.png")
print(f"Saved sdi_regression_results.csv")
