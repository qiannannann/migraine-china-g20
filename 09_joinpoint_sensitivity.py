"""
Joinpoint sensitivity analysis — 4 variants
- Vary max joinpoints (1, 2, 3)
- Vary minimum segment length (3, 5 years)
- Compare 1990-2019 vs 1990-2023
- Exclude 2020-2023 (COVID-era sensitivity)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from itertools import combinations

mpl.rcParams['font.family'] = 'DejaVu Sans'

DATA = Path("../data")
FIG = Path("../figures")
TABLES = Path("../tables")

df = pd.read_csv(DATA / "merged_analytic.csv")

def joinpoint_fit(y, x, max_joins=2, min_seg=3):
    """BIC-selected piecewise linear fit."""
    n = len(y)
    if n < 2*min_seg + 1:
        return None
    candidates = list(range(min_seg, n - min_seg))
    best = None
    for k in range(0, max_joins+1):
        for breaks in combinations(candidates, k):
            edges = [0] + list(breaks) + [n]
            # check min segment length
            seg_lengths = [edges[i+1] - edges[i] for i in range(len(edges)-1)]
            if any(s < min_seg for s in seg_lengths):
                continue
            yhat = np.zeros(n)
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
            if ss <= 0 or n_params == 0:
                continue
            bic = n*np.log(ss/n) + n_params*np.log(n)
            if best is None or bic < best['bic']:
                best = {'bic': bic, 'breaks': [x[b] for b in breaks], 'yhat': yhat, 'k': k}
    return best

# Run sensitivity for China only (most contested)
focus = 'China'
measures = ['ASPR_val','ASIR_val','ASDR_val']
sensitivity_specs = [
    {'name': 'Baseline (max 2 jp, min 3 yr)',          'max_jp': 2, 'min_seg': 3, 'years': (1990, 2023)},
    {'name': 'Max 1 joinpoint (most conservative)',    'max_jp': 1, 'min_seg': 3, 'years': (1990, 2023)},
    {'name': 'Max 3 joinpoints',                       'max_jp': 3, 'min_seg': 3, 'years': (1990, 2023)},
    {'name': 'Min segment 5 yrs',                      'max_jp': 2, 'min_seg': 5, 'years': (1990, 2023)},
    {'name': 'Pre-pandemic 1990-2019',                 'max_jp': 2, 'min_seg': 3, 'years': (1990, 2019)},
    {'name': 'Exclude 2020-2023',                      'max_jp': 2, 'min_seg': 3, 'years': (1990, 2019)},
]

results = []
for spec in sensitivity_specs:
    for measure in measures:
        d = df[(df['country']==focus) & (df['year'].between(*spec['years']))].sort_values('year').reset_index(drop=True)
        x = d['year'].values.astype(float)
        y = d[f'{measure}'].values
        fit = joinpoint_fit(y, x, max_joins=spec['max_jp'], min_seg=spec['min_seg'])
        if fit is None:
            continue
        results.append({
            'spec': spec['name'],
            'measure': measure.replace('_val',''),
            'k_joinpoints': fit['k'],
            'joinpoint_years': ', '.join([str(int(b)) for b in fit['breaks']]) if fit['breaks'] else 'none',
            'BIC': round(fit['bic'], 2),
        })

resdf = pd.DataFrame(results)
resdf.to_csv(TABLES / "joinpoint_sensitivity.csv", index=False)
print("=== Joinpoint sensitivity for China ===\n")
print(resdf.to_string(index=False))

# Plot: for each measure, show 6 specifications as separate panels
fig, axes = plt.subplots(3, 6, figsize=(22, 10), sharex='col')
for col_i, spec in enumerate(sensitivity_specs):
    for row_i, (measure, label) in enumerate(zip(measures, ['ASPR','ASIR','ASDR'])):
        ax = axes[row_i, col_i]
        d = df[(df['country']==focus) & (df['year'].between(*spec['years']))].sort_values('year').reset_index(drop=True)
        x = d['year'].values.astype(float)
        y = d[f'{measure}'].values
        lo = d[f'{measure.replace("_val","")}_lower'].values
        hi = d[f'{measure.replace("_val","")}_upper'].values
        ax.fill_between(x, lo, hi, color='#D62728', alpha=0.18)
        ax.plot(x, y, 'o-', color='#D62728', markersize=3, lw=1.4)
        fit = joinpoint_fit(y, x, max_joins=spec['max_jp'], min_seg=spec['min_seg'])
        if fit is not None:
            ax.plot(x, fit['yhat'], '--', color='black', lw=1.3)
            for b in fit['breaks']:
                ax.axvline(b, color='#888888', ls=':', lw=0.7)
        if col_i == 0:
            ax.set_ylabel(f'{label}\nper 100,000')
        if row_i == 0:
            ax.set_title(spec['name'], fontsize=9)
        ax.set_xlim(spec['years'])
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
fig.suptitle('Joinpoint sensitivity analysis — China migraine ASR\nNo p-value markers; segmentation is descriptive of the GBD posterior median, not inferential', fontsize=12, y=1.0)
plt.tight_layout()
fig.savefig(FIG / "fig11_joinpoint_sensitivity.pdf", bbox_inches='tight')
plt.close()
print("\nSaved fig11_joinpoint_sensitivity.png")
