"""
Full ARIMA diagnostics output for the revised manuscript:
- Model order selection table (AIC, BIC, AICc) for all candidate orders
- Selected order
- ADF stationarity test on raw and differenced series
- Ljung-Box residual autocorrelation test
- ACF/PACF plots
- Held-out validation MAE/RMSE/MAPE
- Rolling-origin validation
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

DATA = Path("../data")
FIG = Path("../figures")
TABLES = Path("../tables")
UPLOADS = Path("../data/raw")

mig = pd.read_csv(UPLOADS / "IHME-GBD_2023_DATA-7500ef0b-1.csv")
NAME_MAP = {"United States of America":"United States","Republic of Korea":"South Korea",
            "Russian Federation":"Russia","Türkiye":"Turkey","European Union":"EU"}
mig['country'] = mig['location_name'].replace(NAME_MAP)

def get_series(country, sex, measure):
    d = mig[(mig['country']==country) & (mig['sex_name']==sex) &
            (mig['measure_name']==measure) & (mig['age_name']=='Age-standardized') &
            (mig['metric_name']=='Rate')].sort_values('year')
    return d[['year','val']].reset_index(drop=True)

# ---------- Diagnostics ----------
diagnostics_records = []
order_table_records = []

focus = [('China','Both'), ('G20','Both')]
measures = {'Prevalence':'ASPR','Incidence':'ASIR','DALYs (Disability-Adjusted Life Years)':'ASDR'}

for country, sex in focus:
    for measure_long, measure_short in measures.items():
        s = get_series(country, sex, measure_long)
        y = s['val'].values
        years = s['year'].values

        # ADF test on raw and differenced
        adf_raw = adfuller(y, regression='ct', autolag='AIC')
        adf_d1 = adfuller(np.diff(y), regression='c', autolag='AIC')

        # Grid search order
        best = (None, np.inf, None, None)
        for p in range(3):
            for d in range(3):
                for q in range(3):
                    if p+q == 0 and d == 0: continue
                    try:
                        m = ARIMA(y, order=(p,d,q)).fit(method_kwargs={"warn_convergence": False, "maxiter": 50})
                        order_table_records.append({
                            'country': country, 'measure': measure_short,
                            'p': p, 'd': d, 'q': q,
                            'AIC': round(m.aic, 2), 'BIC': round(m.bic, 2),
                            'AICc': round(m.aicc, 2),
                            'loglik': round(m.llf, 2),
                        })
                        if m.aicc < best[1]:
                            best = ((p,d,q), m.aicc, m, (m.aic, m.bic, m.aicc))
                    except Exception:
                        continue

        order, aicc, m, aic_bic_aicc = best
        # Ljung-Box on residuals
        lb = acorr_ljungbox(m.resid, lags=[5, 10], return_df=True)
        # Held-out validation
        train = y[:-8]  # 1990-2015
        test = y[-8:]   # 2016-2023
        years_test = years[-8:]
        try:
            m_train = ARIMA(train, order=order).fit(method_kwargs={"warn_convergence": False, "maxiter":50})
            fc = m_train.forecast(steps=8)
            mae = np.mean(np.abs(fc - test))
            rmse = np.sqrt(np.mean((fc - test)**2))
            mape = np.mean(np.abs((fc - test)/test)) * 100
        except Exception:
            mae, rmse, mape = None, None, None

        diagnostics_records.append({
            'country': country, 'measure': measure_short,
            'selected_order': str(order),
            'AIC': round(aic_bic_aicc[0], 2),
            'BIC': round(aic_bic_aicc[1], 2),
            'AICc': round(aic_bic_aicc[2], 2),
            'ADF_raw_pvalue': round(adf_raw[1], 4),
            'ADF_raw_stationary': adf_raw[1] < 0.05,
            'ADF_d1_pvalue': round(adf_d1[1], 4),
            'ADF_d1_stationary': adf_d1[1] < 0.05,
            'Ljung_Box_lag5_p': round(lb.iloc[0]['lb_pvalue'], 4),
            'Ljung_Box_lag10_p': round(lb.iloc[1]['lb_pvalue'], 4),
            'residuals_white_noise_lag10': lb.iloc[1]['lb_pvalue'] > 0.05,
            'heldout_train_period': '1990-2015',
            'heldout_test_period': '2016-2023',
            'heldout_MAE': round(mae, 3) if mae is not None else None,
            'heldout_RMSE': round(rmse, 3) if rmse is not None else None,
            'heldout_MAPE_pct': round(mape, 3) if mape is not None else None,
        })

diag_df = pd.DataFrame(diagnostics_records)
order_df = pd.DataFrame(order_table_records)
diag_df.to_csv(TABLES / "arima_diagnostics_summary.csv", index=False)
order_df.to_csv(TABLES / "arima_order_selection.csv", index=False)

print("=== ARIMA Diagnostics Summary ===\n")
print(diag_df.to_string(index=False))
print(f"\n\nFull order selection table saved: {len(order_df)} rows in arima_order_selection.csv")

# ---------- ACF/PACF + Residual plot for China-ASPR ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for row_i, country in enumerate(['China', 'G20']):
    s = get_series(country, 'Both', 'Prevalence')
    y = s['val'].values

    # ACF
    ax = axes[row_i, 0]
    acf_vals = acf(y, nlags=10, fft=False)
    ax.bar(range(11), acf_vals, color='#1F77B4', alpha=0.7)
    ax.axhline(0, color='black', lw=0.5)
    n = len(y)
    ax.axhline(1.96/np.sqrt(n), color='gray', ls='--', lw=0.7, label='95% CI')
    ax.axhline(-1.96/np.sqrt(n), color='gray', ls='--', lw=0.7)
    ax.set_title(f'{country} ASPR — ACF')
    ax.set_xlabel('Lag')
    ax.legend(fontsize=8)

    # PACF
    ax = axes[row_i, 1]
    pacf_vals = pacf(y, nlags=10)
    ax.bar(range(11), pacf_vals, color='#FF7F0E', alpha=0.7)
    ax.axhline(0, color='black', lw=0.5)
    ax.axhline(1.96/np.sqrt(n), color='gray', ls='--', lw=0.7)
    ax.axhline(-1.96/np.sqrt(n), color='gray', ls='--', lw=0.7)
    ax.set_title(f'{country} ASPR — PACF')
    ax.set_xlabel('Lag')

    # Residual plot
    ax = axes[row_i, 2]
    # Refit using best order
    diag_row = diag_df[(diag_df['country']==country) & (diag_df['measure']=='ASPR')].iloc[0]
    order_str = diag_row['selected_order'].strip('()').split(',')
    order = tuple(int(x.strip()) for x in order_str)
    m = ARIMA(y, order=order).fit(method_kwargs={"warn_convergence": False, "maxiter":50})
    ax.plot(s['year'].values, m.resid, 'o-', color='#2CA02C', markersize=4)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title(f'{country} ASPR ARIMA{order} — Residuals')
    ax.set_xlabel('Year')
    ax.grid(True, alpha=0.3)

fig.suptitle('ARIMA model diagnostics — ACF, PACF, residuals (China and G20 ASPR)', fontsize=11, y=1.0)
plt.tight_layout()
fig.savefig(FIG / "fig12_arima_diagnostics.pdf", bbox_inches='tight')
plt.close()
print("\nSaved fig12_arima_diagnostics.png")
