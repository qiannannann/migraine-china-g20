"""
Analysis 1: Joinpoint-style segmented regression with 95% UI ribbon
- Plot age-standardized rates over 1990-2023 for China + G20 + key countries
- Show 95% UI ribbon from GBD
- Overlay piecewise linear segments (Joinpoint-style); for transparency we use
  simple breakpoint detection rather than Joinpoint's full bootstrap procedure.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

DATA = Path("../data")
FIG = Path("../figures")

df = pd.read_csv(DATA / "merged_analytic.csv")

TIER_COLORS = {1: '#1B7837', 2: '#5AAE61', 3: '#F4A582', 4: '#B2182B', 0: '#666666'}

def panel_plot(ax, countries, measure, label, df, highlight_china=True):
    """Plot ASR + 95% UI ribbon for selected countries"""
    val_col = f'{measure}_val'
    lo_col = f'{measure}_lower'
    hi_col = f'{measure}_upper'
    for c in countries:
        d = df[df['country']==c].sort_values('year')
        tier = d['tier'].iloc[0] if len(d) else 0
        is_china = (c == 'China')
        is_g20 = (c == 'G20')
        is_eu = (c == 'EU')
        if is_china:
            color = '#D62728'
            lw = 2.5
            zorder = 10
            label_str = 'China'
        elif is_g20:
            color = '#1F77B4'
            lw = 2.5
            zorder = 9
            label_str = 'G20 (IHME official)'
        elif is_eu:
            color = '#9467BD'
            lw = 2.0
            zorder = 8
            label_str = 'EU'
        else:
            color = TIER_COLORS.get(int(tier), '#999999')
            lw = 1.0
            zorder = 5
            label_str = c

        ax.fill_between(d['year'], d[lo_col], d[hi_col], color=color, alpha=0.12, zorder=zorder-1)
        ax.plot(d['year'], d[val_col], color=color, lw=lw, zorder=zorder, label=label_str)
    ax.set_xlabel('Year')
    ax.set_ylabel(label)
    ax.set_xlim(1990, 2023)
    ax.grid(True, alpha=0.3)

# ============ FIGURE 1: China vs G20 vs EU — 3 panels (ASPR/ASIR/ASDR) ============
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
focus = ['China', 'G20', 'EU']
for ax, (measure, label) in zip(axes, [
    ('ASPR', 'Age-standardized prevalence (per 100,000)'),
    ('ASIR', 'Age-standardized incidence (per 100,000)'),
    ('ASDR', 'Age-standardized DALY rate (per 100,000)'),
]):
    panel_plot(ax, focus, measure, label, df)
    ax.legend(loc='upper left', fontsize=9, frameon=False)
fig.suptitle('GBD 2023 migraine age-standardized rates: China vs G20 (IHME aggregate) vs EU\n95% UI shown as shaded ribbons', fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig(FIG / "fig01_china_vs_g20_eu_ui_ribbon.pdf", bbox_inches='tight')
plt.close()
print("Saved fig01_china_vs_g20_eu_ui_ribbon.png")

# ============ FIGURE 2: All G20 sovereign members, colored by Tier ============
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sovs = ['Argentina','Australia','Brazil','Canada','China','France','Germany',
        'India','Indonesia','Italy','Japan','Mexico','South Korea',
        'Russia','Saudi Arabia','South Africa','Turkey','United Kingdom','United States']
for ax, (measure, label) in zip(axes, [
    ('ASPR', 'ASPR per 100,000'),
    ('ASIR', 'ASIR per 100,000'),
    ('ASDR', 'ASDR per 100,000'),
]):
    panel_plot(ax, sovs, measure, label, df)
    ax.legend(fontsize=6, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, ncol=1)
    ax.set_title(label.split(' per')[0])
fig.suptitle('GBD 2023 migraine ASR by G20 sovereign member, 1990–2023 (colored by data-availability Tier; China red)', fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(FIG / "fig02_all_g20_by_tier.pdf", bbox_inches='tight')
plt.close()
print("Saved fig02_all_g20_by_tier.png")

# ============ FIGURE 3: Joinpoint-style segmented regression for China =============
# Implement a simple grid-search joinpoint: try 0, 1, 2, 3 joinpoints, pick by BIC
from itertools import combinations
def joinpoint_fit(y, x, max_joins=3):
    """Simple piecewise linear fit with grid-search over breakpoints; returns best segmentation."""
    n = len(y)
    candidates = list(range(3, n-3))  # avoid edges
    best = None
    for k in range(0, max_joins+1):
        for breaks in combinations(candidates, k):
            # Build segments
            edges = [0] + list(breaks) + [n]
            yhat = np.zeros(n)
            resid_sum = 0
            n_params = 0
            for i in range(len(edges)-1):
                a, b = edges[i], edges[i+1]
                if b - a < 2: continue
                xs = x[a:b]
                ys = y[a:b]
                slope, intercept = np.polyfit(xs, ys, 1)
                yhat[a:b] = slope*xs + intercept
                n_params += 2
            ss = np.sum((y - yhat)**2)
            bic = n*np.log(ss/n) + n_params*np.log(n)
            if best is None or bic < best['bic']:
                best = {'bic': bic, 'breaks': breaks, 'yhat': yhat, 'k': k}
    return best

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (measure, label) in zip(axes, [
    ('ASPR', 'ASPR per 100,000'),
    ('ASIR', 'ASIR per 100,000'),
    ('ASDR', 'ASDR per 100,000'),
]):
    d = df[df['country']=='China'].sort_values('year').reset_index(drop=True)
    x = d['year'].values
    y = d[f'{measure}_val'].values
    lo = d[f'{measure}_lower'].values
    hi = d[f'{measure}_upper'].values
    # Plot UI ribbon
    ax.fill_between(x, lo, hi, color='#D62728', alpha=0.18, label='95% UI')
    ax.plot(x, y, 'o-', color='#D62728', markersize=4, lw=1.8, label='GBD posterior median')
    # Fit joinpoint
    fit = joinpoint_fit(y, x, max_joins=2)
    ax.plot(x, fit['yhat'], '--', color='#222222', lw=1.5, label=f"Joinpoint (BIC; k={fit['k']})")
    # Annotate breakpoints
    for b in fit['breaks']:
        ax.axvline(x[b], color='#888888', ls=':', lw=0.8)
        ax.text(x[b], ax.get_ylim()[1]*0.97, f"{x[b]}", rotation=90, ha='right', va='top', fontsize=8, color='#666')
    ax.set_xlabel('Year')
    ax.set_ylabel(label)
    ax.set_title(f'China — {label.split(" per")[0]}')
    ax.set_xlim(1990, 2023)
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
fig.suptitle('China migraine ASR: GBD posterior median (red), 95% UI ribbon, and simple BIC-selected Joinpoint segments', fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(FIG / "fig03_china_joinpoint_ui.pdf", bbox_inches='tight')
plt.close()
print("Saved fig03_china_joinpoint_ui.png")

# ============ FIGURE 4: UI overlap visualization for R2 ============
fig, ax = plt.subplots(figsize=(11, 7))
focus = ['China', 'G20', 'EU', 'Germany', 'Brazil', 'South Korea', 'United States', 'India']
years_show = [1990, 2023]
y_positions = list(range(len(focus)))
for i, c in enumerate(focus):
    d = df[(df['country']==c) & (df['year'].isin(years_show))].sort_values('year')
    if len(d) != 2: continue
    for j, (_, row) in enumerate(d.iterrows()):
        color = '#1F77B4' if row['year']==1990 else '#D62728'
        offset = -0.18 if row['year']==1990 else 0.18
        y_pos = i + offset
        ax.plot([row['ASPR_lower'], row['ASPR_upper']], [y_pos, y_pos], color=color, lw=4, alpha=0.6)
        ax.plot(row['ASPR_val'], y_pos, 'o', color=color, markersize=10, markeredgecolor='black', zorder=5)
        ax.text(row['ASPR_upper']+200, y_pos, f"{row['ASPR_val']:.0f}", va='center', fontsize=8, color=color)

ax.set_yticks(y_positions)
ax.set_yticklabels(focus)
ax.set_xlabel('Age-standardized prevalence per 100,000')
ax.set_title('GBD 2023 migraine ASPR: 1990 vs 2023 point estimates and 95% UIs\nFor most countries, the 1990 and 2023 UIs overlap substantially', fontsize=11)
ax.grid(True, axis='x', alpha=0.3)
# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1F77B4', markersize=10, label='1990 (point estimate + 95% UI)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#D62728', markersize=10, label='2023 (point estimate + 95% UI)'),
]
ax.legend(handles=legend_elements, loc='lower right', frameon=False)
plt.tight_layout()
fig.savefig(FIG / "fig04_ui_overlap_1990_vs_2023.pdf", bbox_inches='tight')
plt.close()
print("Saved fig04_ui_overlap_1990_vs_2023.png")

# Summary stats for table — UI overlap
print("\n=== UI Overlap analysis ===")
rows = []
for c in df['country'].unique():
    d90 = df[(df['country']==c) & (df['year']==1990)]
    d23 = df[(df['country']==c) & (df['year']==2023)]
    if d90.empty or d23.empty: continue
    for measure in ['ASPR','ASIR','ASDR']:
        v90, lo90, hi90 = d90[f'{measure}_val'].iloc[0], d90[f'{measure}_lower'].iloc[0], d90[f'{measure}_upper'].iloc[0]
        v23, lo23, hi23 = d23[f'{measure}_val'].iloc[0], d23[f'{measure}_lower'].iloc[0], d23[f'{measure}_upper'].iloc[0]
        # Overlap?
        overlap = (max(lo90, lo23) <= min(hi90, hi23))
        # Overlap fraction
        union_lo = min(lo90, lo23)
        union_hi = max(hi90, hi23)
        inter_lo = max(lo90, lo23)
        inter_hi = min(hi90, hi23)
        overlap_frac = max(0, inter_hi-inter_lo) / (union_hi - union_lo)
        rows.append({
            'country': c, 'measure': measure,
            'val_1990': round(v90,1), 'UI_1990_lo': round(lo90,1), 'UI_1990_hi': round(hi90,1),
            'val_2023': round(v23,1), 'UI_2023_lo': round(lo23,1), 'UI_2023_hi': round(hi23,1),
            'pct_change': round(100*(v23-v90)/v90, 2),
            'UIs_overlap': overlap, 'overlap_fraction': round(overlap_frac, 3),
        })
out = pd.DataFrame(rows)
out.to_csv("../tables/ui_overlap_1990_vs_2023.csv", index=False)
print(f"Saved {len(out)} rows to ui_overlap_1990_vs_2023.csv")
print(out[out['country'].isin(['China','G20','EU'])].to_string(index=False))
