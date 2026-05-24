import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("data/migraine_clean.csv")

# Filter G20 aggregate, Both sexes, all-ages age-standardized (age_id=27)
g20 = df[(df["location_id"]==44586) & (df["sex_id"]==3) & (df["age_id"]==27)].copy()

# Need Rate metric (metric_id=3) for ASR
g20r = g20[g20["metric_id"]==3].copy()
print("G20 rate rows:", len(g20r), "years:", g20r["year"].min(), "to", g20r["year"].max())
print("Measures:", g20r["measure_name"].unique())

def eapc(df_sub):
    """Log-linear EAPC + 95% CI from rate series."""
    df_sub = df_sub.sort_values("year")
    y = np.log(df_sub["val"].values)
    x = df_sub["year"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    eapc_pct = (np.exp(slope) - 1) * 100
    # 95% CI for slope
    n = len(x)
    t_crit = stats.t.ppf(0.975, n-2)
    slope_lo = slope - t_crit*std_err
    slope_hi = slope + t_crit*std_err
    eapc_lo = (np.exp(slope_lo) - 1) * 100
    eapc_hi = (np.exp(slope_hi) - 1) * 100
    return eapc_pct, eapc_lo, eapc_hi

# Measures: 5=Prevalence, 6=Incidence, 2=DALYs
mapping = {5:"ASPR", 6:"ASIR", 2:"ASDR"}
for mid, lbl in mapping.items():
    sub = g20r[g20r["measure_id"]==mid]
    if len(sub)>0:
        e, lo, hi = eapc(sub)
        print(f"G20 {lbl} EAPC: {e:.3f} ({lo:.3f} to {hi:.3f}), n={len(sub)}, 2023 val={sub[sub['year']==2023]['val'].iloc[0]:.2f}")

# Also compute EU
eu = df[(df["location_id"]==4743) & (df["sex_id"]==3) & (df["age_id"]==27) & (df["metric_id"]==3)].copy()
print()
for mid, lbl in mapping.items():
    sub = eu[eu["measure_id"]==mid]
    if len(sub)>0:
        e, lo, hi = eapc(sub)
        v2023 = sub[sub['year']==2023]['val'].iloc[0]
        print(f"EU {lbl} EAPC: {e:.3f} ({lo:.3f} to {hi:.3f}), 2023 val={v2023:.2f}")
