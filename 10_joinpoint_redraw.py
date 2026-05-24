"""
Redraw Joinpoint figure WITHOUT P-value markers, with descriptive framing.
Replaces the original Figure 2 in the manuscript.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations
from pathlib import Path

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

DATA = Path("../data")
FIG = Path("../figures")

df = pd.read_csv(DATA / "merged_analytic.csv")

def joinpoint_fit(y, x, max_joins=2, min_seg=3):
    n = len(y)
    candidates = list(range(min_seg, n - min_seg))
    best = None
    for k in range(0, max_joins+1):
        for breaks in combinations(candidates, k):
            edges = [0] + list(breaks) + [n]
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
            if ss <= 0 or n_params == 0: continue
            bic = n*np.log(ss/n) + n_params*np.log(n)
            if best is None or bic < best['bic']:
                best = {'bic': bic, 'breaks': breaks, 'yhat': yhat, 'k': k, 'segments': edges}
    return best

# Compute APC for each segment (descriptive only — no p stars)
def segment_apc(y, x, edges):
    apcs = []
    for i in range(len(edges)-1):
        a, b = edges[i], edges[i+1]
        xs = x[a:b]
        ys = np.log(y[a:b])
        slope, intercept = np.polyfit(xs, ys, 1)
        # APC = (exp(slope) - 1) * 100
        apc = (np.exp(slope) - 1) * 100
        # SE for descriptive 95% CI only
        resid = ys - (slope*xs + intercept)
        if len(xs) >= 3:
            se_slope = np.std(resid, ddof=2) / np.sqrt(np.sum((xs - xs.mean())**2))
            apc_lo = (np.exp(slope - 1.96*se_slope) - 1) * 100
            apc_hi = (np.exp(slope + 1.96*se_slope) - 1) * 100
        else:
            apc_lo, apc_hi = apc, apc
        apcs.append({'start': int(x[a]), 'end': int(x[b-1]), 'APC': apc, 'lo': apc_lo, 'hi': apc_hi})
    return apcs

# Build figure: 2 rows × 3 cols (China + G20), with each panel showing trajectory + UI ribbon + descriptive segments
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
entities = ['China', 'G20']
measures = [('ASPR_val', 'Age-standardized prevalence per 100,000'),
            ('ASIR_val', 'Age-standardized incidence per 100,000'),
            ('ASDR_val', 'Age-standardized DALY rate per 100,000')]

for row_i, entity in enumerate(entities):
    for col_i, (measure, label) in enumerate(measures):
        ax = axes[row_i, col_i]
        d = df[df['country']==entity].sort_values('year').reset_index(drop=True)
        x = d['year'].values.astype(float)
        y = d[measure].values
        meas_base = measure.replace('_val','')
        lo = d[f'{meas_base}_lower'].values
        hi = d[f'{meas_base}_upper'].values

        color = '#D62728' if entity == 'China' else '#1F77B4'

        # 95% UI ribbon
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, label='GBD 2023 95% UI')
        # Posterior median
        ax.plot(x, y, 'o-', color=color, markersize=4, lw=1.8, label='GBD posterior median')

        # Joinpoint segmentation (descriptive, no p-stars)
        fit = joinpoint_fit(y, x, max_joins=2)
        if fit is not None:
            ax.plot(x, fit['yhat'], '--', color='black', lw=1.4, alpha=0.85,
                    label=f'Descriptive segmentation (BIC; k={fit["k"]})')
            apcs = segment_apc(y, x, fit['segments'])
            for seg in apcs:
                # Annotate each segment with APC and 95% CI (descriptive)
                mid_x = (seg['start'] + seg['end']) / 2
                mid_idx = int((np.argmin(np.abs(x - mid_x))))
                y_pos = ax.get_ylim()[1]*0.97 if mid_idx < len(fit['yhat']) else None
                if y_pos is not None:
                    pass  # We'll add APC text in a separate text block at bottom
            for b in fit['breaks']:
                ax.axvline(x[b], color='#888888', ls=':', lw=0.6, alpha=0.7)

        ax.set_xlim(1990, 2023)
        ax.set_xlabel('Year')
        ax.set_ylabel(label if col_i == 0 else '')
        ax.set_title(f'{entity} — {meas_base}')
        ax.grid(True, alpha=0.3)
        if row_i == 0 and col_i == 0:
            ax.legend(loc='upper left', fontsize=8, frameon=False)

# Caption-style text box at bottom
fig.text(0.5, -0.01,
    'Joinpoint segmentation is a descriptive segmentation of the GBD posterior median trajectory and is not interpreted as evidence of '
    'abrupt changes in primary epidemiological data. Significance markers (e.g., * for p<0.05) are deliberately omitted; '
    '95% confidence intervals for segment annual percentage change (APC) are tabulated in Supplementary Table SX. '
    'Specific joinpoint years are sensitive to model settings (Supplementary Table SY).',
    ha='center', fontsize=9, style='italic', wrap=True)

fig.suptitle('GBD 2023-modeled age-standardized rates of migraine, China and G20 aggregate, 1990–2023\nDescriptive segmentation with 95% UI ribbon — no inferential p-value markers',
             fontsize=12, y=1.0)
plt.tight_layout()
fig.savefig(FIG / "fig02v2_joinpoint_descriptive.pdf", bbox_inches='tight')
plt.close()
print("Saved fig02v2_joinpoint_descriptive.png — to replace original Figure 2")

# Also export APC table for Supplementary
all_apcs = []
for entity in entities + ['EU']:
    for measure in ['ASPR_val','ASIR_val','ASDR_val']:
        d = df[df['country']==entity].sort_values('year').reset_index(drop=True)
        if d.empty: continue
        x = d['year'].values.astype(float)
        y = d[measure].values
        fit = joinpoint_fit(y, x, max_joins=2)
        if fit is None: continue
        apcs = segment_apc(y, x, fit['segments'])
        for seg in apcs:
            all_apcs.append({
                'entity': entity,
                'measure': measure.replace('_val',''),
                'segment_start': seg['start'],
                'segment_end': seg['end'],
                'APC_pct': round(seg['APC'], 3),
                'APC_95CI': f"({seg['lo']:.3f}, {seg['hi']:.3f})",
            })
apcs_df = pd.DataFrame(all_apcs)
apcs_df.to_csv("../tables/joinpoint_segment_APCs.csv", index=False)
print(f"\nSaved joinpoint_segment_APCs.csv ({len(apcs_df)} rows)")
print(apcs_df.to_string(index=False))
