import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from scipy import stats

def _mad(arr):
    """Median Absolute Deviation (MAD). Return MAD (not scaled)."""
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return mad if mad != 0 else 0.0

def generalized_esd_test(x, max_anoms=10, alpha=0.05):
    """
    Generalized ESD for outlier detection on 1-D array x.
    Returns a boolean mask of anomalies (True = anomaly) and the test statistics.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return np.array([], dtype=bool), np.array([])
    if max_anoms is None:
        max_anoms = int(np.floor(n/2))
    max_anoms = min(max_anoms, int(np.floor(n/2)))
    R = []   # test statistics
    lam = [] # critical values
    indices = []
    x_work = x.copy()
    idx_work = np.arange(n)
    for i in range(1, max_anoms+1):
        med = np.median(x_work)
        mad = _mad(x_work)
        if mad == 0:
            # if MAD == 0 fall back to std
            mad = np.std(x_work, ddof=1)
            if mad == 0:
                break
        # compute absolute deviations from median
        abs_dev = np.abs(x_work - med)
        # find most extreme
        j = np.argmax(abs_dev)
        R_i = abs_dev[j] / mad
        R.append(R_i)
        # compute critical value lambda_i
        # degrees of freedom for t: n - i - 1
        curr_n = x_work.size
        p = 1.0 - alpha / (2.0 * (curr_n))
        df = curr_n - 2  # equivalently n-i-1 where n here is original? using current length - 1? 
        # To follow classical formulation, use df = curr_n - 2 (i.e., n-i-1 with i=1...)
        # but ensure df>0 for t.ppf
        if df <= 0:
            # cannot compute t quantile, stop
            lam.append(np.nan)
            indices.append(idx_work[j])
            break
        t_crit = stats.t.ppf(p, df)
        # formula components:
        lam_i = ( (curr_n - 1) * t_crit ) / np.sqrt( (df + t_crit**2) * curr_n )
        lam.append(lam_i)
        indices.append(idx_work[j])
        # remove the most extreme from working arrays and continue
        x_work = np.delete(x_work, j)
        idx_work = np.delete(idx_work, j)
    # Decide how many of the R exceed corresponding lam (reverse check):
    k = 0
    for i in range(len(R)):
        if R[i] > lam[i]:
            k = i + 1
    is_outlier = np.zeros(n, dtype=bool)
    for i in range(k):
        is_outlier[indices[i]] = True
    return is_outlier, np.array(R), np.array(lam), indices[:len(R)]

def shesd(series, period=None, max_anoms=None, alpha=0.05, seasonal_deg=1):
    """
    Seasonal Hybrid ESD
    - series: 1D array-like or pd.Series (if pd.Series and has DatetimeIndex, STL will use its frequency)
    - period: seasonal period (e.g., 24 for hourly daily period). If None, try to infer or skip STL
    - max_anoms: maximum anomalies to detect (if None, floor(n/2))
    - alpha: significance level for ESD (e.g., 0.05)
    - seasonal_deg: not used here but kept for API similarity
    - dates: 可选，日期序列（长度与 series 一致），用于绘图 x 轴
    Returns: dictionary with results and optionally a plot.
    """
    
    
    
    s = pd.Series(series).reset_index(drop=True)
    n = s.size
    if max_anoms is None:
        max_anoms = int(np.floor(n/2))
    # Deseasonalize / detrend via STL if period provided and feasible
    if period is not None and n >= 2*period and period >= 2:
        try:
            stl = STL(s, period=period, robust=True)
            res = stl.fit()
            resid = res.resid.values
            seasonal = res.seasonal.values
            trend = res.trend.values
            used_stl = True
        except Exception as e:
            # fallback: no decomposition
            resid = s.values - np.median(s.values)
            seasonal = np.zeros(n)
            trend = np.zeros(n)
            used_stl = False
    else:
        # no seasonal decomposition, operate on deviations from median (like the user's residual example)
        resid = s.values - np.median(s.values)
        seasonal = np.zeros(n)
        trend = np.zeros(n)
        used_stl = False

    # apply generalized ESD on residuals (work on resid directly)
    is_outlier, R, lam, cand_indices = generalized_esd_test(resid, max_anoms=max_anoms, alpha=alpha)

    results = {
        "is_outlier": is_outlier,
        "anomaly_indices": np.where(is_outlier)[0].tolist(),
        "R": R,
        "lambda": lam,
        "cand_indices": cand_indices,
        "residual": resid,
        "seasonal": seasonal,
        "trend": trend,
        "used_stl": used_stl
    }


    return results