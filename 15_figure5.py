"""
Figure 5 — FINAL with proper spacing
- Anchored BAPC to GBD 2023
- Legend at bottom with adequate margin from x-axis "Year" labels
- Cleaner title spacing
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'legend.fontsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

UPLOADS = Path("../data/raw")
TABLES  = Path("../tables")
FIG_OUT = Path("../figures")
FIG_OUT.mkdir(parents=True, exist_ok=True)

# Load anchored BAPC (already computed in script 18)
bapc_anchored = pd.read_csv(TABLES / "bapc_projections_anchored.csv")
arima_ets = pd.read_csv(TABLES / "projections_2024_2053.csv")

# Combine
all_proj = pd.concat([bapc_anchored, arima_ets], ignore_index=True)

# Load GBD historical
raw = pd.read_csv(UPLOADS / "IHME-GBD_2023_DATA-7500ef0b-1.csv")
NAME_MAP = {"United States of America":"United States","Republic of Korea":"South Korea",
            "Russian Federation":"Russia","Türkiye":"Turkey","European Union":"EU"}
raw['country'] = raw['location_name'].replace(NAME_MAP)
MEASURE_LONG = {'ASPR':'Prevalence','ASIR':'Incidence','ASDR':'DALYs (Disability-Adjusted Life Years)'}

def get_hist(country, sex, measure):
    d = raw[(raw['country']==country) & (raw['sex_name']==sex) &
            (raw['measure_name']==MEASURE_LONG[measure]) &
            (raw['age_name']=='Age-standardized') & (raw['metric_name']=='Rate')]
    return d.sort_values('year')[['year','val','lower','upper']].reset_index(drop=True)

# Build figure with explicit spacing
fig = plt.figure(figsize=(15, 10))
# Reserve space: title top (~7%), plots middle (~75%), x-labels (~5%), gap (~3%), legend bottom (~10%)
gs = fig.add_gridspec(2, 3, top=0.88, bottom=0.18, left=0.07, right=0.97,
                       hspace=0.40, wspace=0.28)

MODEL_STYLE = {
    'BAPC':  {'color':'#C62828','ls':'-','lw':2.6,'zorder':10},
    'ARIMA': {'color':'#1565C0','ls':'--','lw':1.8,'zorder':8},
    'ETS':   {'color':'#2E7D32','ls':':','lw':1.8,'zorder':7},
}

for row_i, country in enumerate(['China','G20']):
    for col_i, measure in enumerate(['ASPR','ASIR','ASDR']):
        ax = fig.add_subplot(gs[row_i, col_i])

        # Historical
        hist = get_hist(country, 'Both', measure)
        ax.fill_between(hist['year'], hist['lower'], hist['upper'],
                        color='#9E9E9E', alpha=0.20, zorder=1)
        ax.plot(hist['year'], hist['val'], 'o-', color='#333333',
                markersize=2.5, lw=1.3, zorder=2)

        hist_2023_val = hist[hist['year']==2023]['val'].iloc[0]
        hist_2023_lo  = hist[hist['year']==2023]['lower'].iloc[0]
        hist_2023_hi  = hist[hist['year']==2023]['upper'].iloc[0]

        for model in ['BAPC','ARIMA','ETS']:
            d = all_proj[(all_proj['country']==country) & (all_proj['sex']=='Both') &
                         (all_proj['measure']==measure) & (all_proj['model']==model) &
                         (all_proj['year'] >= 2024)].sort_values('year')
            if d.empty: continue
            s = MODEL_STYLE[model]

            # Prepend 2023 GBD historical value for continuity
            x_vals = np.concatenate([[2023.0], d['year'].values])
            y_vals = np.concatenate([[hist_2023_val], d['projection'].values])
            ax.plot(x_vals, y_vals, s['ls'], color=s['color'],
                    lw=s['lw'], zorder=s['zorder'])

            if model == 'BAPC':
                lo_vals = np.concatenate([[hist_2023_lo], d['PI_lower'].values])
                hi_vals = np.concatenate([[hist_2023_hi], d['PI_upper'].values])
                ymin = hist['val'].min() * 0.4
                ymax = hist['val'].max() * 3.0
                lo_vals = np.clip(lo_vals, ymin, ymax)
                hi_vals = np.clip(hi_vals, ymin, ymax)
                ax.fill_between(x_vals, lo_vals, hi_vals,
                                color=s['color'], alpha=0.12, zorder=3)

        ax.axvline(2023.5, color='#666666', ls=':', lw=0.8, zorder=4)
        ax.set_xlim(1990, 2053)
        ax.set_xlabel('Year' if row_i == 1 else '', fontsize=10)
        ax.set_ylabel(f'{measure} per 100,000' if col_i == 0 else '', fontsize=10)
        ax.set_title(f'{country} — {measure}', pad=8)
        ax.grid(True, alpha=0.25, lw=0.5)

# Title — two lines, placed in reserved area at top
fig.text(0.5, 0.955,
    'Figure 5 — Exploratory 30-year projections of GBD 2023-modeled migraine ASR (China, G20 aggregate, both sexes)',
    ha='center', va='center', fontsize=13, fontweight='bold')
fig.text(0.5, 0.925,
    'BAPC projection (red) anchored to GBD 2023 endpoint to preserve visual continuity; '
    'BAPC integrates future demographic structure, ARIMA & ETS extrapolate historical trend',
    ha='center', va='center', fontsize=10.5, style='italic')

# Legend — in reserved area at bottom, well below the "Year" labels
handles = [
    Line2D([0],[0], color='#333333', marker='o', markersize=3, lw=1.3,
           label='GBD posterior median (1990–2023)'),
    Patch(facecolor='#9E9E9E', alpha=0.30, label='GBD 95% UI (historical)'),
    Line2D([0],[0], color=MODEL_STYLE['BAPC']['color'], lw=2.6, ls='-',
           label='BAPC (primary; anchored at 2023)'),
    Patch(facecolor=MODEL_STYLE['BAPC']['color'], alpha=0.15, label='BAPC 95% credible interval'),
    Line2D([0],[0], color=MODEL_STYLE['ARIMA']['color'], lw=1.8, ls='--',
           label='ARIMA (comparator)'),
    Line2D([0],[0], color=MODEL_STYLE['ETS']['color'],   lw=1.8, ls=':',
           label='ETS (comparator)'),
]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 0.02), fontsize=10)

fig.savefig(FIG_OUT / "figure5.pdf", bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_OUT / 'fig05_final.png'}")
