"""
Analysis 4: 1990 vs 2019 HAQI endpoint delta analysis
- Cross-country: ΔHAQI (1990 → 2019) vs Δ ASR (same window)
- Pearson + Spearman correlations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

DATA = Path("../data")
FIG = Path("../figures")
TABLES = Path("../tables")

df = pd.read_csv(DATA / "merged_analytic.csv")

# Get 1990 and 2019 (HAQI available years), 19 G20 sovereigns
d90 = df[(df['year']==1990) & (df['tier'].between(1,4))].copy()
d19 = df[(df['year']==2019) & (df['tier'].between(1,4))].copy()
merged = d90[['country','tier','haqi','ASPR_val','ASIR_val','ASDR_val']].rename(columns={
    'haqi':'haqi_1990','ASPR_val':'ASPR_1990','ASIR_val':'ASIR_1990','ASDR_val':'ASDR_1990'
}).merge(d19[['country','haqi','ASPR_val','ASIR_val','ASDR_val']].rename(columns={
    'haqi':'haqi_2019','ASPR_val':'ASPR_2019','ASIR_val':'ASIR_2019','ASDR_val':'ASDR_2019'
}), on='country')

# Deltas
merged['delta_HAQI'] = merged['haqi_2019'] - merged['haqi_1990']
for m in ['ASPR','ASIR','ASDR']:
    merged[f'delta_{m}'] = merged[f'{m}_2019'] - merged[f'{m}_1990']
    merged[f'pct_change_{m}'] = (merged[f'{m}_2019'] - merged[f'{m}_1990']) / merged[f'{m}_1990'] * 100

merged.to_csv(TABLES / "haqi_delta_analysis.csv", index=False)
print(merged.round(2).to_string(index=False))

# Correlations
print("\n=== Pearson and Spearman correlations: ΔHAQI vs Δ ASR (1990 vs 2019) ===\n")
corr_results = []
for m in ['ASPR','ASIR','ASDR']:
    pr, pp = pearsonr(merged['delta_HAQI'], merged[f'delta_{m}'])
    sr, sp = spearmanr(merged['delta_HAQI'], merged[f'delta_{m}'])
    pr_pct, pp_pct = pearsonr(merged['delta_HAQI'], merged[f'pct_change_{m}'])
    sr_pct, sp_pct = spearmanr(merged['delta_HAQI'], merged[f'pct_change_{m}'])
    corr_results.append({
        'measure': m,
        'pearson_abs_change': round(pr,3),  'pearson_abs_p': f"{pp:.4g}",
        'spearman_abs_change': round(sr,3), 'spearman_abs_p': f"{sp:.4g}",
        'pearson_pct_change': round(pr_pct,3), 'pearson_pct_p': f"{pp_pct:.4g}",
        'spearman_pct_change': round(sr_pct,3),'spearman_pct_p': f"{sp_pct:.4g}",
    })
corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(TABLES / "haqi_correlation_results.csv", index=False)
print(corr_df.to_string(index=False))

# ---------- Plot: Scatter ΔHAQI vs ΔASR ----------
TIER_COLORS = {1: '#1B7837', 2: '#5AAE61', 3: '#F4A582', 4: '#B2182B'}
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
for ax, m, label in zip(axes,
                        ['ASPR','ASIR','ASDR'],
                        ['ASPR per 100,000','ASIR per 100,000','ASDR per 100,000']):
    for _, row in merged.iterrows():
        tier = int(row['tier'])
        color = TIER_COLORS[tier]
        is_china = row['country']=='China'
        ax.scatter(row['delta_HAQI'], row[f'delta_{m}'],
                   color='#D62728' if is_china else color,
                   s=120 if is_china else 80,
                   edgecolor='black', linewidth=1.5 if is_china else 0.6,
                   zorder=10 if is_china else 5)
        ax.annotate(row['country'], (row['delta_HAQI'], row[f'delta_{m}']),
                   fontsize=8, xytext=(5,5), textcoords='offset points',
                   fontweight='bold' if is_china else 'normal')

    # Fit line
    x = merged['delta_HAQI'].values
    y = merged[f'delta_{m}'].values
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(coef, xs), '--', color='black', alpha=0.6)
    pr, pp = pearsonr(x, y)
    ax.text(0.05, 0.95, f"r = {pr:.2f}, p = {pp:.3g}", transform=ax.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('Δ HAQI (1990 → 2019)')
    ax.set_ylabel(f'Δ {label}')
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Δ {m}')
fig.suptitle('Cross-country endpoint comparison: ΔHAQI vs Δ migraine ASR, 1990→2019 (G20 sovereign members)\nChina highlighted in red', fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(FIG / "fig07_haqi_endpoint.pdf", bbox_inches='tight')
plt.close()
print(f"\nSaved fig07_haqi_endpoint.png")
