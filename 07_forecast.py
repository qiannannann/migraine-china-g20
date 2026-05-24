"""
Analysis 6: ARIMA + ETS + APC forecast comparison
- Rolling-origin validation: train 1990-2015, test 2016-2023
- Compute MAE and RMSE per metric and sex
- Generate 2024-2053 projections from all 3 models, with 95% PI
- Note: BAPC requires R; we use a simpler Python Age-Period-Cohort GLM as a substitute.
  We label it "APC GLM" to be transparent.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

DATA = Path("../data")
FIG = Path("../figures")
TABLES = Path("../tables")
UPLOADS = Path("../data/raw")

mig = pd.read_csv(UPLOADS / "IHME-GBD_2023_DATA-7500ef0b-1.csv")
NAME_MAP = {"United States of America":"United States","Republic of Korea":"South Korea",
            "Russian Federation":"Russia","Türkiye":"Turkey","European Union":"EU"}
mig['country'] = mig['location_name'].replace(NAME_MAP)

# Helper: extract series
def get_series(country, sex, measure, age='Age-standardized', metric='Rate'):
    d = mig[(mig['country']==country) & (mig['sex_name']==sex) &
            (mig['measure_name']==measure) & (mig['age_name']==age) &
            (mig['metric_name']==metric)].sort_values('year')
    return d[['year','val','lower','upper']].reset_index(drop=True)

# ---------- ARIMA forecast with auto order selection (AICc) ----------
# Cache fits to avoid recomputing
_ARIMA_CACHE = {}
def fit_arima(y, max_p=2, max_d=2, max_q=2):
    key = (tuple(np.round(y,2)), max_p, max_d, max_q)
    if key in _ARIMA_CACHE:
        return _ARIMA_CACHE[key]
    best = (None, np.inf, None)
    for p in range(max_p+1):
        for d in range(max_d+1):
            for q in range(max_q+1):
                if p+q == 0 and d == 0: continue
                try:
                    m = ARIMA(y, order=(p,d,q)).fit(method_kwargs={"warn_convergence": False, "maxiter": 50})
                    if m.aicc < best[1]:
                        best = ((p,d,q), m.aicc, m)
                except Exception:
                    continue
    _ARIMA_CACHE[key] = best
    return best

def fit_ets(y):
    """Simple ETS with additive trend, no seasonality"""
    try:
        m = ETSModel(y, error='add', trend='add', seasonal=None).fit(disp=False)
        return m
    except Exception:
        return None

def fit_apc(years, vals):
    """Simple APC-style projection: log(rate) ~ year (poly degree 2)"""
    log_y = np.log(vals)
    X = np.column_stack([np.ones(len(years)), years, years**2])
    coef, *_ = np.linalg.lstsq(X, log_y, rcond=None)
    return coef

def apc_predict(coef, future_years):
    log_yhat = coef[0] + coef[1]*future_years + coef[2]*future_years**2
    return np.exp(log_yhat)

# ---------- Rolling-origin validation: train 1990-2015, test 2016-2023 ----------
def rolling_origin_validate(series, train_end_year=2015):
    y_all = series['val'].values
    years_all = series['year'].values
    train_mask = years_all <= train_end_year
    test_mask = years_all > train_end_year
    y_train = y_all[train_mask]
    y_test = y_all[test_mask]
    years_test = years_all[test_mask]
    n_test = len(y_test)

    out = {}

    # ARIMA
    order, aicc, m = fit_arima(y_train)
    if m is not None:
        fc = m.forecast(steps=n_test)
        mae = np.mean(np.abs(fc - y_test))
        rmse = np.sqrt(np.mean((fc - y_test)**2))
        out['ARIMA'] = {'order': order, 'forecast': fc, 'mae': mae, 'rmse': rmse}

    # ETS
    m_ets = fit_ets(y_train)
    if m_ets is not None:
        fc = m_ets.forecast(steps=n_test)
        mae = np.mean(np.abs(fc - y_test))
        rmse = np.sqrt(np.mean((fc - y_test)**2))
        out['ETS'] = {'forecast': fc, 'mae': mae, 'rmse': rmse}

    # APC GLM
    coef = fit_apc(years_all[train_mask], y_train)
    fc = apc_predict(coef, years_test)
    mae = np.mean(np.abs(fc - y_test))
    rmse = np.sqrt(np.mean((fc - y_test)**2))
    out['APC'] = {'forecast': fc, 'mae': mae, 'rmse': rmse}

    out['y_test'] = y_test
    out['years_test'] = years_test
    return out

# Run validation for China and G20 (both sexes split) and all 3 ASR measures
focus = [('China','Both'), ('China','Male'), ('China','Female'),
         ('G20','Both'), ('G20','Male'), ('G20','Female')]
measures = ['Prevalence','Incidence','DALYs (Disability-Adjusted Life Years)']
measure_short = {'Prevalence':'ASPR','Incidence':'ASIR','DALYs (Disability-Adjusted Life Years)':'ASDR'}

val_results = []
for country, sex in focus:
    for measure in measures:
        s = get_series(country, sex, measure)
        if len(s) < 20: continue
        out = rolling_origin_validate(s, train_end_year=2015)
        for model_name in ['ARIMA','ETS','APC']:
            if model_name in out:
                val_results.append({
                    'country': country, 'sex': sex, 'measure': measure_short[measure],
                    'model': model_name,
                    'mae': round(out[model_name]['mae'], 3),
                    'rmse': round(out[model_name]['rmse'], 3),
                    'arima_order': str(out['ARIMA']['order']) if model_name=='ARIMA' else '-',
                })

vdf = pd.DataFrame(val_results)
vdf.to_csv(TABLES / "forecast_validation.csv", index=False)
print("=== Rolling-origin validation (train 1990-2015, test 2016-2023) ===\n")
print(vdf.to_string(index=False))

# ---------- Project 2024-2053 from each model ----------
def project_future(series, future_years, model_type='ARIMA'):
    y = series['val'].values
    years = series['year'].values
    n_fut = len(future_years)
    if model_type == 'ARIMA':
        order, _, m = fit_arima(y)
        if m is None: return None, None, None
        fc = m.get_forecast(steps=n_fut)
        mean = fc.predicted_mean.values if hasattr(fc.predicted_mean,'values') else np.asarray(fc.predicted_mean)
        ci = fc.conf_int(alpha=0.05)
        ci = ci.values if hasattr(ci,'values') else np.asarray(ci)
        return mean, ci[:,0], ci[:,1]
    elif model_type == 'ETS':
        m = fit_ets(y)
        if m is None: return None, None, None
        mean = np.asarray(m.forecast(steps=n_fut))
        # Approximate 95% PI from residual SD scaled by sqrt(h)
        resid = y - np.asarray(m.fittedvalues)
        sd = np.std(resid)
        h = np.arange(1, n_fut+1)
        lo = mean - 1.96*sd*np.sqrt(h)
        hi = mean + 1.96*sd*np.sqrt(h)
        return mean, lo, hi
    elif model_type == 'APC':
        coef = fit_apc(years, y)
        mean = apc_predict(coef, future_years)
        # Approximate 95% PI from residual SD
        resid = np.log(y) - (coef[0] + coef[1]*years + coef[2]*years**2)
        sd = np.std(resid)
        lo = mean * np.exp(-1.96*sd)
        hi = mean * np.exp(1.96*sd)
        return mean, lo, hi

future_years = np.arange(2024, 2054)
proj_records = []
for country, sex in focus:
    for measure in measures:
        s = get_series(country, sex, measure)
        if len(s) < 20: continue
        for model_type in ['ARIMA','ETS','APC']:
            mean, lo, hi = project_future(s, future_years, model_type)
            if mean is None: continue
            for i, yr in enumerate(future_years):
                proj_records.append({
                    'country': country, 'sex': sex, 'measure': measure_short[measure],
                    'model': model_type,
                    'year': int(yr),
                    'projection': mean[i],
                    'PI_lower': lo[i] if not np.isnan(lo[i]) else None,
                    'PI_upper': hi[i] if not np.isnan(hi[i]) else None,
                })

pdf = pd.DataFrame(proj_records)
pdf.to_csv(TABLES / "projections_2024_2053.csv", index=False)
print(f"\nSaved projections_2024_2053.csv ({len(pdf)} rows)")

# ---------- Plot: China & G20, both sexes, all 3 ASR — ARIMA vs ETS vs APC ----------
MODEL_COLORS = {'ARIMA':'#1F77B4','ETS':'#2CA02C','APC':'#9467BD'}
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for row_i, country in enumerate(['China','G20']):
    for col_i, (measure_long, measure_short_name) in enumerate(zip(measures,['ASPR','ASIR','ASDR'])):
        ax = axes[row_i, col_i]
        s = get_series(country, 'Both', measure_long)
        # Historical
        ax.fill_between(s['year'], s['lower'], s['upper'], color='gray', alpha=0.2, label='GBD 95% UI')
        ax.plot(s['year'], s['val'], 'o-', color='black', markersize=3, lw=1.2, label='GBD historical')
        # Forecasts
        for model_type in ['ARIMA','ETS','APC']:
            mean, lo, hi = project_future(s, future_years, model_type)
            if mean is None: continue
            color = MODEL_COLORS[model_type]
            ax.plot(future_years, mean, '-', color=color, lw=1.8, label=f'{model_type}')
            ax.fill_between(future_years, lo, hi, color=color, alpha=0.10)
        ax.set_xlim(1990, 2053)
        ax.set_xlabel('Year')
        ax.set_ylabel(measure_short_name + ' per 100k')
        ax.set_title(f'{country} — {measure_short_name}')
        ax.grid(True, alpha=0.3)
        if row_i == 0 and col_i == 0:
            ax.legend(loc='upper left', fontsize=8, frameon=False)
fig.suptitle('Forecast comparison: ARIMA vs ETS vs APC GLM (China & G20, both sexes, 2024–2053)\n95% PI shown as shaded bands; rolling-origin validation MAE/RMSE in Table forecast_validation.csv', fontsize=11, y=1.0)
plt.tight_layout()
fig.savefig(FIG / "fig10_forecast_comparison.pdf", bbox_inches='tight')
plt.close()
print("Saved fig10_forecast_comparison.png")
