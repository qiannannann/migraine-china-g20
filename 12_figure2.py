"""
Simplified Main-text Figure 2: GBD posterior median + 95% UI ribbon only.
No Joinpoint segmentation, no P-value markers — pure descriptive trend display.
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

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
entities = [('China', '#D62728'), ('G20', '#1F77B4')]
measures = [('ASPR_val', 'Age-standardized prevalence per 100,000'),
            ('ASIR_val', 'Age-standardized incidence per 100,000'),
            ('ASDR_val', 'Age-standardized DALY rate per 100,000')]

for row_i, (entity, color) in enumerate(entities):
    for col_i, (measure, label) in enumerate(measures):
        ax = axes[row_i, col_i]
        d = df[df['country']==entity].sort_values('year').reset_index(drop=True)
        x = d['year'].values.astype(float)
        y = d[measure].values
        meas_base = measure.replace('_val','')
        lo = d[f'{meas_base}_lower'].values
        hi = d[f'{meas_base}_upper'].values

        ax.fill_between(x, lo, hi, color=color, alpha=0.2, label='GBD 2023 95% UI')
        ax.plot(x, y, 'o-', color=color, markersize=4, lw=2.0, label='GBD posterior median')

        ax.set_xlim(1990, 2023)
        ax.set_xlabel('Year' if row_i == 1 else '')
        ax.set_ylabel(label if col_i == 0 else '')
        ax.set_title(f'{entity} — {meas_base}')
        ax.grid(True, alpha=0.3)
        if row_i == 0 and col_i == 0:
            ax.legend(loc='upper left', fontsize=8, frameon=False)

fig.suptitle('Main-text Figure 2. GBD 2023-modeled age-standardized rates of migraine, China and IHME G20 aggregate, 1990–2023\nDescriptive trend display with 95% UI ribbons; no inferential segmentation or P-value markers.',
             fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(FIG / "figure2.pdf", bbox_inches="tight")
plt.close()
print("Saved fig02_main_simplified.png — new Main-text Figure 2 (Joinpoint demoted to Supplementary)")
