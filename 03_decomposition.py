"""
Analysis 2: Das Gupta-type decomposition

Decompose 1990 vs 2023 change in migraine DALY counts into:
  - Population growth effect
  - Population aging effect (changing age composition)
  - Age-specific rate effect

For each (country, sex), using age-specific rates and population counts from GBD.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("../data")
FIG = Path("../figures")
TABLES = Path("../tables")
UPLOADS = Path("../data/raw")

# Load migraine raw data
mig = pd.read_csv(UPLOADS / "IHME-GBD_2023_DATA-7500ef0b-1.csv")

# We need age-specific DALY rate (per 100k) and case count (number)
# rate × population / 100000 = number
# Therefore population = number / rate × 100000

# Keep DALYs only, both sexes separate, all 5-year age groups, 1990 and 2023
age_groups_5yr = ['<5 years','5-9 years','10-14 years','15-19 years','20-24 years','25-29 years',
                  '30-34 years','35-39 years','40-44 years','45-49 years','50-54 years','55-59 years',
                  '60-64 years','65-69 years','70-74 years','75-79 years','80-84 years','85-89 years',
                  '90-94 years','95+ years']

mig = mig[(mig['measure_name']=='DALYs (Disability-Adjusted Life Years)') &
          (mig['sex_name'].isin(['Male','Female','Both'])) &
          (mig['age_name'].isin(age_groups_5yr)) &
          (mig['year'].isin([1990, 2023]))]

# Pivot to wide: get number and rate
counts = mig[mig['metric_name']=='Number'][['location_name','sex_name','age_name','year','val']].rename(columns={'val':'daly_count'})
rates = mig[mig['metric_name']=='Rate'][['location_name','sex_name','age_name','year','val']].rename(columns={'val':'daly_rate'})

df = counts.merge(rates, on=['location_name','sex_name','age_name','year'])
df['population'] = np.where(df['daly_rate']>0, df['daly_count'] / df['daly_rate'] * 100000, 0)

# Country renames
NAME_MAP = {
    "United States of America": "United States",
    "Republic of Korea": "South Korea",
    "Russian Federation": "Russia",
    "Türkiye": "Turkey",
    "European Union": "EU",
}
df['country'] = df['location_name'].replace(NAME_MAP)

# Das Gupta decomposition function
def kitagawa_decomp(d1, d2, age_groups):
    """
    Decompose change in total = sum(rate × pop) between two periods.
    Returns: pop_growth, age_struct, rate_change components.
    Using a 3-factor multiplicative-style decomposition (Das Gupta 1993).

    Total_t = N_t * sum_a (p_a,t * r_a,t)
    where N = total population, p_a = age composition, r_a = age-specific rate.

    Δ Total = (N effect with avg of other 2) + (p effect ...) + (r effect ...)
    """
    # d1: dict {age: {pop, rate}} at time 1
    # d2: at time 2
    N1 = sum(d1[a]['pop'] for a in age_groups)
    N2 = sum(d2[a]['pop'] for a in age_groups)
    P1 = {a: d1[a]['pop']/N1 for a in age_groups}  # age proportions
    P2 = {a: d2[a]['pop']/N2 for a in age_groups}
    R1 = {a: d1[a]['rate'] for a in age_groups}
    R2 = {a: d2[a]['rate'] for a in age_groups}

    # Total at endpoints
    T1 = N1 * sum(P1[a] * R1[a] for a in age_groups)
    T2 = N2 * sum(P2[a] * R2[a] for a in age_groups)

    # Three "effects" using Das Gupta's symmetric averaging
    # N effect: change N from N1->N2 while averaging over P and R combinations
    avg_PR = sum(
        (
            (P1[a]+P2[a])/2 * (R1[a]+R2[a])/2 +
            (P1[a]*R1[a] + P2[a]*R2[a])/2
        )/2
        for a in age_groups
    )
    N_effect = (N2 - N1) * avg_PR

    # P effect: change P while averaging N and R
    avg_NR = lambda a: ((N1+N2)/2 * (R1[a]+R2[a])/2 + (N1*R1[a] + N2*R2[a])/2)/2
    P_effect = sum((P2[a] - P1[a]) * avg_NR(a) for a in age_groups)

    # R effect: change R while averaging N and P
    avg_NP = lambda a: ((N1+N2)/2 * (P1[a]+P2[a])/2 + (N1*P1[a] + N2*P2[a])/2)/2
    R_effect = sum((R2[a] - R1[a]) * avg_NP(a) for a in age_groups)

    total_change = T2 - T1
    # Normalize: Das Gupta decomposition should sum to total change
    check = N_effect + P_effect + R_effect
    # Re-scale residual onto rate effect for safety (typically very small)
    return {
        'T1': T1, 'T2': T2, 'total_change': total_change,
        'pop_growth': N_effect, 'pop_aging': P_effect, 'rate_change': R_effect,
        'check_sum': check, 'residual': total_change - check,
    }

# Run for each (country, sex)
results = []
focus_countries = ['China','G20','EU'] + ['Germany','Brazil','South Korea','United States','India',
                                          'Japan','United Kingdom','Russia','Saudi Arabia','Turkey']
for country in focus_countries:
    for sex in ['Both','Male','Female']:
        d = df[(df['country']==country) & (df['sex_name']==sex)]
        if d.empty: continue
        d1_dict = {}
        d2_dict = {}
        for a in age_groups_5yr:
            row1 = d[(d['age_name']==a) & (d['year']==1990)]
            row2 = d[(d['age_name']==a) & (d['year']==2023)]
            if row1.empty or row2.empty: continue
            d1_dict[a] = {'pop': row1['population'].iloc[0]/100000, 'rate': row1['daly_rate'].iloc[0]}  # rescale pop to "per 100k"
            d2_dict[a] = {'pop': row2['population'].iloc[0]/100000, 'rate': row2['daly_rate'].iloc[0]}

        if not d1_dict: continue
        res = kitagawa_decomp(d1_dict, d2_dict, list(d1_dict.keys()))
        # Re-scale back: T was in units of rate × (pop/100k), gives DALY count directly
        results.append({
            'country': country, 'sex': sex,
            **{k: v for k, v in res.items()},
        })

resdf = pd.DataFrame(results)
# Round
for c in ['T1','T2','total_change','pop_growth','pop_aging','rate_change','check_sum','residual']:
    resdf[c] = resdf[c].round(0)
# Percent contributions
for c in ['pop_growth','pop_aging','rate_change']:
    resdf[f'{c}_pct'] = (resdf[c] / resdf['total_change'].abs() * 100).round(1)

resdf.to_csv(TABLES / "decomposition_results.csv", index=False)
print("=== Das Gupta decomposition (DALY count change 1990 → 2023) ===\n")
print(resdf[['country','sex','T1','T2','total_change','pop_growth','pop_aging','rate_change',
             'pop_growth_pct','pop_aging_pct','rate_change_pct']]
      .to_string(index=False))

# ---------- Plot: Stacked bar of decomposition ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, sex_label, sex in zip(axes, ['Female', 'Male'], ['Female','Male']):
    d = resdf[resdf['sex']==sex].sort_values('total_change', ascending=True)
    countries = d['country'].tolist()
    y = np.arange(len(countries))
    pg = d['pop_growth'].values / 1e3
    pa = d['pop_aging'].values / 1e3
    rc = d['rate_change'].values / 1e3

    ax.barh(y, pg, color='#377EB8', label='Population growth', alpha=0.8)
    ax.barh(y, pa, left=pg, color='#FF7F00', label='Population aging', alpha=0.8)
    # rate_change can be negative or positive; if negative, plot on negative side
    ax.barh(y, rc, left=pg+pa, color='#4DAF4A', label='Age-specific rate change', alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(countries, fontsize=9)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Contribution to Δ DALY count, 1990–2023 (thousands)')
    ax.set_title(f'{sex_label} migraine DALYs — Das Gupta decomposition')
    if sex == 'Female':
        ax.legend(loc='lower right', fontsize=9, frameon=False)
    # Overlay total
    totals = d['total_change'].values / 1e3
    for i, t in enumerate(totals):
        ax.text(t + (10 if t>=0 else -10), i, f"Δ={t:+.0f}k", va='center',
                ha='left' if t>=0 else 'right', fontsize=8, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
fig.savefig(FIG / "figure3.pdf", bbox_inches="tight")
plt.close()
print("\nSaved fig05_decomposition.png")
print(f"Saved decomposition_results.csv ({len(resdf)} rows)")
