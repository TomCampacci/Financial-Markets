# portfolio_backtest.py — Python 3.10+
import os, glob, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime

# --------- GLOBAL STYLE ---------
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 220,
    "font.size": 9, "axes.titlesize": 12, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": False,
})

# ============== COMPREHENSIVE PARAMETERS LIST ==============
"""
PORTFOLIO BACKTESTING & ANALYSIS PARAMETERS
===========================================

DATA & PATHS:
- DATA_DIR: Portfolio ETF data directory
- BENCH_DIR: Benchmark indices directory  
- RESULTS_DIR: Output charts directory

DISPLAY SETTINGS:
- SHOW_PLOTS: Display chart windows (True/False)
- PLOT_ALL_PATHS: Show all MC paths vs sample (True/False)

ANALYSIS PARAMETERS:
- ESTIMATION_YEARS: Historical data window for mu/cov estimation
- COV_METHOD: Covariance estimation method ("sample" or "ledoit")
- RF: Risk-free rate (annual)
- SEED: Random seed for reproducibility

PORTFOLIO SETTINGS:
- START_CAPITAL: Initial portfolio value (€)
- ANNUALIZATION: Trading days per year
- MONTH_FACTOR: Months per year

MONTE CARLO SIMULATION:
- MC_PATHS: Number of simulation paths
- RANDOMNESS_FACTOR: Additional randomness percentage (0.30 = 30%)
- MC_STEPS: Simulation horizon in months
- FORWARD_YEARS: Forward projection period

PORTFOLIO COMPOSITION:
- WEIGHTS_RAW: ETF allocation weights (must sum to 1.0)
- BENCH_DEF: Benchmark definitions for comparison

RISK METRICS:
- VaR confidence levels: 95%, 99%
- Expected Shortfall: 95%, 99%
- Maximum Drawdown Duration
- Calmar Ratio calculation

SECTOR ANALYSIS:
- SECTOR_MAPPING: ETF to sector classification
- SECTOR_COLORS: Color scheme for sector visualization
"""

# ============== SETTINGS ==============
DATA_DIR    = r"C:\Users\CAMPACCI\Desktop\Portefeuille"
BENCH_DIR   = os.path.join(DATA_DIR, "Benchmarks")
RESULTS_DIR = r"./results"; os.makedirs(RESULTS_DIR, exist_ok=True)

SHOW_PLOTS       = True            # Afficher les fenêtres en plus des PNG
PLOT_ALL_PATHS   = True            # Afficher les 10k chemins MC (alpha faible)
ESTIMATION_YEARS = 3               # fenêtre d’estimation mu/cov
COV_METHOD       = "ledoit"        # "sample" ou "ledoit"
RF               = 0.00
SEED             = 42

START_CAPITAL = 10_000
ANNUALIZATION = 252
MONTH_FACTOR  = 12
MC_PATHS      = 50_000
RANDOMNESS_FACTOR = 0.30  # 30% de randomness supplémentaire
MC_STEPS      = 36                  # 3 ans
FORWARD_YEARS = MC_STEPS / 12.0

# ---- Portefeuille ETF (noms = fichiers CSV dans le dossier Portefeuille) ----
# (Si un ticker manque, il est ignoré proprement et la pondération est renormalisée.)
WEIGHTS_RAW = {
    "ANXU":0.20,  # ANX - Nasdaq ETF
    "NVDA":0.07,  # NVDA
    "PLTR":0.07,  # PLTR
    "IUS2":0.06,  # IUS - US broad ETF
    "BNK": 0.13,  # BNK - Bancaires EU
    "CS1": 0.07,  # DAX - ETF Espagne
    "MIB": 0.07,  # IBEX - ETF Italie
    "CNKY":0.07,  # MIB - ETF Japon
    "GLDA":0.13,  # Gold - ETF Or
    "CG1": 0.13,  # Nikkei - ETF Japon (utilise CG1 pour Nikkei)
}

# ---- Sector Classification ----
SECTOR_MAPPING = {
    "ANXU": "Technology",      # Nasdaq ETF
    "NVDA": "Technology",      # NVIDIA
    "PLTR": "Technology",      # Palantir
    "IUS2": "Broad Market",    # US broad ETF
    "BNK":  "Financials",      # European Banks
    "CS1":  "Europe",          # Spain ETF
    "MIB":  "Europe",          # Italy ETF
    "CNKY": "Asia Pacific",    # Japan ETF
    "GLDA": "Commodities",     # Gold ETF
    "CG1":  "Europe",          # European ETF (not Japanese)
}

SECTOR_COLORS = {
    "Technology": "#1f77b4",      # Blue
    "Broad Market": "#ff7f0e",    # Orange
    "Financials": "#2ca02c",       # Green
    "Europe": "#d62728",          # Red
    "Asia Pacific": "#9467bd",    # Purple
    "Commodities": "#8c564b",     # Brown
}

# ---- Benchmarks (noms = fichiers CSV dans Portefeuille\Benchmarks) ----
# Laisse exactement ces noms si tes fichiers s’appellent ainsi.
BENCH_DEF = [
    ("US (NASDAQ)",     "NQ1!"),
    ("EU (DAX)",        "FDAX1!"),
    ("Spain (IBEX)",    "IBEX35"),
    ("Italy (MIB)",     "FTSEMIB"),
    ("Japan (Nikkei)",  "NIY1!"),
    ("Gold",            "GC1!"),
]

if SEED is not None:
    np.random.seed(SEED)

eur_fmt  = FuncFormatter(lambda x, _: f"€{x:,.0f}")
pct_fmt  = FuncFormatter(lambda x, _: f"{x*100:.0f}%")
pct1_fmt = FuncFormatter(lambda x, _: f"{x*100:.1f}%")

# ---------- helpers ----------
def add_bar_labels(ax, fmt="{:.1f}%"):
    for cont in ax.containers:
        try:
            labels = [fmt.format(v) for v in cont.datavalues]
            ax.bar_label(cont, labels=labels, padding=3, fontsize=8)
        except Exception:
            pass
    ax.margins(y=0.15)

def save_and_show(fig, name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{ts}_{name}")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    print("Saved:", path)
    if SHOW_PLOTS:
        try: plt.show()
        except Exception: pass

def placeholder_figure(title: str, subtitle: str = ""):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=12, weight="bold")
    if subtitle:
        ax.text(0.5, 0.35, subtitle, ha="center", va="center", fontsize=10)
    return fig

# ---------- IO ----------
def read_csv_file(path: str) -> pd.Series:
    """Accepte Time/Date/Datetime, Close/Adj Close (insensible à la casse)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    dt = cols.get("time") or cols.get("date") or cols.get("datetime")
    cl = cols.get("close") or cols.get("adj close") or cols.get("adj_close")
    if dt is None or cl is None:
        raise ValueError("Missing time/close columns")
    idx = pd.to_datetime(df[dt], utc=True, errors="coerce")
    s = pd.Series(df[cl].astype(float).values, index=idx)
    s = s.tz_convert(None).sort_index()
    s = s[~s.index.duplicated(keep="last")].dropna()
    return s.rename(os.path.splitext(os.path.basename(path))[0])

def load_prices_from_dir(data_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV found in {data_dir}")
    series = []
    for f in files:
        try: series.append(read_csv_file(f))
        except Exception as e:
            print(f"Skip {os.path.basename(f)}: {e}")
    return pd.concat(series, axis=1).sort_index()

def align_business_days(prices: pd.DataFrame) -> pd.DataFrame:
    idx = pd.date_range(prices.index.min(), prices.index.max(), freq="B")
    return prices.reindex(idx).ffill()

def slice_recent_safe(prices: pd.DataFrame, years: int, min_rows: int = 60) -> pd.DataFrame:
    if not years or years <= 0: return prices
    end = prices.index.max(); start = end - pd.DateOffset(years=years)
    out = prices.loc[prices.index >= start]
    if len(out) < min_rows: return prices
    return out

# ---------- CORE ----------
def ledoit_cov(X: np.ndarray):
    try:
        from sklearn.covariance import LedoitWolf
        return LedoitWolf().fit(X).covariance_
    except Exception:
        return np.cov(X, rowvar=False, ddof=1)

def compute_metrics(prices: pd.DataFrame, weights_raw: dict):
    # Utiliser uniquement les ETF présents + renormaliser les poids
    available = [c for c in weights_raw if c in prices.columns]
    missing   = [c for c in weights_raw if c not in prices.columns]
    if missing:
        print("Poids ignorés (ETF introuvables):", missing)

    w = np.array([weights_raw[c] for c in available], dtype=float)
    w = w / w.sum()
    w_series = pd.Series(w, index=available)

    rets = np.log(prices[available] / prices[available].shift(1)).dropna()
    if rets.empty:
        raise RuntimeError("Pas assez de retours pour calculer les métriques.")

    # mu, cov (journalier -> annualisé)
    cov_d = ledoit_cov(rets.values) if COV_METHOD=="ledoit" else rets.cov().values
    mu_d  = rets.mean().values
    mu_a  = mu_d * ANNUALIZATION
    cov_a = cov_d * ANNUALIZATION

    # Retour portefeuille comme Series alignée
    port_ret_d = pd.Series(rets.values @ w, index=rets.index, name="Portfolio")

    # Contributions au risque (marginales * poids)
    vol_a = float(np.sqrt(w @ cov_a @ w))
    mcr   = (cov_a @ w) / max(vol_a, 1e-12)
    cr    = w * mcr
    cr_pct = cr / max(cr.sum(), 1e-12)

    return {
        "cols": available, "w": w, "w_series": w_series,
        "rets_d": rets, "mu_a": mu_a, "cov_a": cov_a,
        "port_ret_d": port_ret_d,
        "corr": rets.corr(),
        "cr_pct": cr_pct, "vol_a": vol_a
    }

# ---------- Benchmarks utils ----------
def bench_daily(prices: pd.DataFrame, col: str) -> pd.Series:
    return np.log(prices[col] / prices[col].shift(1))

def compute_benchmark_params(bench_prices: pd.DataFrame):
    out = {}
    for label, tick in BENCH_DEF:
        if tick in bench_prices.columns:
            r = bench_daily(bench_prices, tick).dropna()
            if len(r) >= 30:
                out[label] = {
                    "mu_ann":  float(r.mean()*ANNUALIZATION),
                    "vol_ann": float(r.std(ddof=1)*np.sqrt(ANNUALIZATION))
                }
    return out

# ---------- RISK MEASURES ----------
def calculate_var(rets, confidence_level=0.95):
    """Calculate Value at Risk (VaR) at specified confidence level."""
    return np.percentile(rets, (1 - confidence_level) * 100)

def calculate_expected_shortfall(rets, confidence_level=0.95):
    """Calculate Expected Shortfall (CVaR) at specified confidence level."""
    var = calculate_var(rets, confidence_level)
    return rets[rets <= var].mean()

def calculate_max_drawdown_duration(values):
    """Calculate maximum drawdown duration in periods."""
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    underwater = drawdown < 0
    
    durations = []
    current_duration = 0
    
    for is_underwater in underwater:
        if is_underwater:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
            current_duration = 0
    
    if current_duration > 0:
        durations.append(current_duration)
    
    return max(durations) if durations else 0

def calculate_calmar_ratio(annual_return, max_drawdown):
    """Calculate Calmar Ratio: Annual Return / Maximum Drawdown."""
    if max_drawdown == 0:
        return np.inf if annual_return > 0 else 0
    return annual_return / abs(max_drawdown)

def calculate_risk_metrics(port_ret_d, paths_normal, paths_random):
    """Calculate comprehensive risk metrics for both simulations."""
    # Historical metrics
    hist_rets = port_ret_d.values
    hist_annual_ret = hist_rets.mean() * ANNUALIZATION
    hist_vol = hist_rets.std() * np.sqrt(ANNUALIZATION)
    hist_max_dd = calculate_max_drawdown_duration((1 + hist_rets).cumprod())
    
    # Normal MC metrics
    normal_end_vals = paths_normal[-1, :]
    normal_rets = (normal_end_vals / START_CAPITAL) ** (1.0 / FORWARD_YEARS) - 1.0
    normal_var_95 = calculate_var(normal_rets, 0.95)
    normal_es_95 = calculate_expected_shortfall(normal_rets, 0.95)
    normal_max_dd_duration = np.max([calculate_max_drawdown_duration(paths_normal[:, i]) for i in range(paths_normal.shape[1])])
    normal_calmar = calculate_calmar_ratio(normal_rets.mean(), normal_rets.min())
    
    # Random MC metrics
    random_end_vals = paths_random[-1, :]
    random_rets = (random_end_vals / START_CAPITAL) ** (1.0 / FORWARD_YEARS) - 1.0
    random_var_95 = calculate_var(random_rets, 0.95)
    random_es_95 = calculate_expected_shortfall(random_rets, 0.95)
    random_max_dd_duration = np.max([calculate_max_drawdown_duration(paths_random[:, i]) for i in range(paths_random.shape[1])])
    random_calmar = calculate_calmar_ratio(random_rets.mean(), random_rets.min())
    
    return {
        "historical": {
            "annual_return": hist_annual_ret,
            "volatility": hist_vol,
            "max_dd_duration": hist_max_dd,
            "var_95": calculate_var(hist_rets, 0.95),
            "es_95": calculate_expected_shortfall(hist_rets, 0.95)
        },
        "normal_mc": {
            "var_95": normal_var_95,
            "es_95": normal_es_95,
            "max_dd_duration": normal_max_dd_duration,
            "calmar_ratio": normal_calmar,
            "annual_return": normal_rets.mean(),
            "volatility": normal_rets.std()
        },
        "random_mc": {
            "var_95": random_var_95,
            "es_95": random_es_95,
            "max_dd_duration": random_max_dd_duration,
            "calmar_ratio": random_calmar,
            "annual_return": random_rets.mean(),
            "volatility": random_rets.std()
        }
    }

# ---------- MONTE CARLO ----------
def mc_gaussian(mu_a, cov_a, w, start_value, steps, paths):
    """MC multi-actifs mensuel (Normal, covariance préservée) - SANS randomness."""
    w = w.reshape(-1,1)
    mu_m  = (mu_a / MONTH_FACTOR).reshape(-1,1)
    cov_m =  cov_a / MONTH_FACTOR
    # Cholesky avec petite régularisation si nécessaire
    try:
        L = np.linalg.cholesky(cov_m + 1e-12*np.eye(cov_m.shape[0]))
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov_m + 1e-6*np.eye(cov_m.shape[0]))
    out = np.empty((steps+1, paths)); out[0] = start_value
    for t in range(1, steps+1):
        Z = np.random.randn(cov_m.shape[0], paths)
        r = mu_m + L @ Z
        pr = (w.T @ r).ravel()            # retour mensuel du portefeuille
        out[t] = out[t-1]*(1+pr)
    return out

def mc_gaussian_with_randomness(mu_a, cov_a, w, start_value, steps, paths, randomness_factor=0.30):
    """MC multi-actifs avec sauts aléatoires et volatilité stochastique."""
    w = w.reshape(-1,1)
    mu_m  = (mu_a / MONTH_FACTOR).reshape(-1,1)
    cov_m =  cov_a / MONTH_FACTOR
    
    # Cholesky avec petite régularisation si nécessaire
    try:
        L = np.linalg.cholesky(cov_m + 1e-12*np.eye(cov_m.shape[0]))
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov_m + 1e-6*np.eye(cov_m.shape[0]))
    
    out = np.empty((steps+1, paths)); out[0] = start_value
    
    for t in range(1, steps+1):
        # Simulation normale
        Z = np.random.randn(cov_m.shape[0], paths)
        r_normal = mu_m + L @ Z
        
        # Ajout de randomness avec sauts aléatoires
        jump_prob = 0.05  # 5% de chance de saut par mois
        jump_size = np.random.normal(0, randomness_factor, (cov_m.shape[0], paths))
        jump_mask = np.random.random((cov_m.shape[0], paths)) < jump_prob
        
        # Volatilité stochastique (varie dans le temps)
        vol_multiplier = 1 + np.random.normal(0, randomness_factor/2, (cov_m.shape[0], paths))
        vol_multiplier = np.clip(vol_multiplier, 0.5, 2.0)  # Limite entre 0.5x et 2x
        
        # Application des effets
        r_jumps = jump_size * jump_mask
        r_stochastic = r_normal * vol_multiplier
        
        # Retour final avec randomness
        r_final = r_stochastic + r_jumps
        pr = (w.T @ r_final).ravel()  # retour mensuel du portefeuille
        
        # Limitation des retours extrêmes
        pr = np.clip(pr, -0.5, 1.0)  # Entre -50% et +100%
        
        out[t] = out[t-1]*(1+pr)
    
    return out

def mc_single_asset(mu_ann, vol_ann, start_value, steps, paths):
    """MC mono-actif mensuel."""
    mu_m  = mu_ann / MONTH_FACTOR
    sig_m = vol_ann / np.sqrt(MONTH_FACTOR)
    out = np.empty((steps+1, paths)); out[0] = start_value
    for t in range(1, steps+1):
        r = np.random.randn(paths)*sig_m + mu_m
        out[t] = out[t-1]*(1+r)
    return out

# ---------- PLOTS ----------
def plot_allocation(w_series: pd.Series):
    fig, ax = plt.subplots(figsize=(6.2,6.2))
    labels = [f"{k} {v*100:.1f}% (€{v*START_CAPITAL:,.0f})" for k,v in w_series.items()]
    ax.pie(list(w_series.values), labels=labels, startangle=90)
    ax.set_title("Portfolio Allocation (% and €)")
    return fig

def plot_correlation_green(corr: pd.DataFrame, order_cols):
    c = corr.reindex(index=order_cols, columns=order_cols)
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    im = ax.imshow(c, vmin=-1, vmax=1, cmap="viridis")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("ρ", rotation=0, labelpad=10)
    ax.set_title("Correlation Matrix")
    ax.set_xticks(range(len(c.columns))); ax.set_xticklabels(c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c.index)));   ax.set_yticklabels(c.index)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            val = c.iloc[i, j]
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color=("white" if abs(val) > 0.55 else "black"),
                    fontsize=8)
    fig.tight_layout()
    return fig

def plot_risk_contribution(cols, cr_pct, w_series):
    s_rc = pd.Series(cr_pct, index=cols)*100
    s_w  = w_series[cols]*100
    order = s_rc.sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(11,6))
    ax.bar(np.arange(len(order))-0.18, s_rc[order].values, width=0.36, label="Risk contribution %", color="#ffb3ba")
    ax.bar(np.arange(len(order))+0.18, s_w[order].values,  width=0.36, label="Weight %",             color="#a3c4f3")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_ylabel("%"); ax.set_title("Risk Contribution vs Weight (historical)")
    add_bar_labels(ax, fmt="{:.1f}%")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig

def plot_perf_vs_benchmarks(bench_prices, port_ret_d):
    series = {"Portfolio": (1+port_ret_d).cumprod()}
    for label, tick in BENCH_DEF:
        if tick in bench_prices.columns:
            series[label] = (1+bench_daily(bench_prices, tick)).cumprod()
    df = pd.concat(series, axis=1).dropna()
    if df.empty:
        return placeholder_figure("Portfolio vs Benchmarks — Indexed to 100 (historical)",
                                  "No common dates or benchmarks available.")
    df = df/df.iloc[0]*100
    fig, ax = plt.subplots(figsize=(12,6))
    df.plot(ax=ax, linewidth=2)
    ax.set_title("Portfolio vs Benchmarks — Indexed to 100 (historical)")
    ax.set_ylabel("Index (100 = start)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig

def plot_risk_vs_indexes(bench_prices, port_ret_d):
    data = {"Portfolio": port_ret_d}
    for label, tick in BENCH_DEF:
        if tick in bench_prices.columns:
            data[label] = bench_daily(bench_prices, tick)
    df = pd.concat(data, axis=1).dropna()
    if df.empty:
        return placeholder_figure("Risk (volatility) — Portfolio vs Indexes (historical)",
                                  "No common dates or benchmarks available.")
    vol = df.std(ddof=1)*np.sqrt(ANNUALIZATION)*100
    vol = vol.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12,6))
    bars = ax.barh(vol.index[::-1], vol.values[::-1],
                   color=["#6c8cd5" if i=="Portfolio" else "lightgray" for i in vol.index[::-1]])
    for b, v in zip(bars, vol.values[::-1]):
        ax.text(v+0.3, b.get_y()+b.get_height()/2, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_title("Risk (volatility) — Portfolio vs Indexes (historical)")
    ax.set_xlabel("Annualized volatility (%)")
    ax.set_xlim(0, max(1, vol.max()*1.15))
    fig.tight_layout()
    return fig

def plot_mc_paths(paths, median_path, p10_path, p90_path, approx_days, with_randomness=False):
    fig, ax = plt.subplots(figsize=(11,6))
    
    # Color paths based on final outcome
    final_values = paths[-1, :]
    positive_mask = final_values >= START_CAPITAL
    negative_mask = final_values < START_CAPITAL
    
    if PLOT_ALL_PATHS:
        # Plot positive paths in blue
        if np.any(positive_mask):
            ax.plot(paths[:, positive_mask], alpha=0.01, color="#1f77b4", rasterized=True)
        # Plot negative paths in red
        if np.any(negative_mask):
            ax.plot(paths[:, negative_mask], alpha=0.01, color="#d62728", rasterized=True)
    else:
        # Sample paths for better visibility
        pos_idx = np.where(positive_mask)[0]
        neg_idx = np.where(negative_mask)[0]
        
        if len(pos_idx) > 0:
            sample_pos = np.random.choice(pos_idx, size=min(150, len(pos_idx)), replace=False)
            ax.plot(paths[:, sample_pos], alpha=0.1, color="#1f77b4")
        if len(neg_idx) > 0:
            sample_neg = np.random.choice(neg_idx, size=min(150, len(neg_idx)), replace=False)
            ax.plot(paths[:, sample_neg], alpha=0.1, color="#d62728")
    
    ax.plot(median_path, color="red", lw=2.2, label="Median (50th)")
    ax.plot(p10_path, "--", lw=1.2, label="10th")
    ax.plot(p90_path, "--", lw=1.2, label="90th")
    ax.axhline(START_CAPITAL, color="gray", ls="--", lw=1, label=f"Initial €{START_CAPITAL:,.0f}")
    
    # Dynamic title based on randomness
    randomness_text = "WITH 30% Randomness" if with_randomness else "Standard Gaussian"
    ax.set_title(f"Monte Carlo Simulation ({randomness_text}) — {paths.shape[1]:,} paths | {paths.shape[0]-1} months (~{approx_days} trading days)")
    ax.set_xlabel("Months"); ax.set_ylabel("Portfolio value (€)"); ax.yaxis.set_major_formatter(eur_fmt)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_final_distribution(end_vals):
    fig, ax = plt.subplots(figsize=(8,5))
    p10, p50, p90 = np.percentile(end_vals, [10,50,90])
    prob_loss = float((end_vals < START_CAPITAL).mean())
    
    # Color histogram based on positive/negative returns
    positive_mask = end_vals >= START_CAPITAL
    negative_mask = end_vals < START_CAPITAL
    
    if np.any(positive_mask):
        ax.hist(end_vals[positive_mask], bins=80, alpha=0.7, color="#1f77b4", label="Positive Returns")
    if np.any(negative_mask):
        ax.hist(end_vals[negative_mask], bins=80, alpha=0.7, color="#d62728", label="Negative Returns")
    
    ax.axvline(p50, color="red", lw=2, label=f"Median €{p50:,.0f}")
    ax.axvline(end_vals.mean(), color="black", ls=":", lw=1.5, label=f"Mean €{end_vals.mean():,.0f}")
    ax.axvline(START_CAPITAL, color="gray", ls="--", lw=1, label=f"Initial €{START_CAPITAL:,.0f}")
    ax.set_title(f"Final Value Distribution (3y) — P(loss)={prob_loss:.1%}")
    ax.set_xlabel("Final value (€)"); ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(eur_fmt); ax.legend()
    fig.tight_layout()
    return fig

def plot_forward_excess_vs_benchmarks(port_paths: np.ndarray, bench_params: dict):
    if not bench_params:
        return placeholder_figure("Forward Excess vs Benchmarks (MC)", "No benchmark parameters available.")
    V0 = port_paths[0, 0]
    Vp_end = port_paths[-1, :]
    cagr_port = (Vp_end / V0) ** (1.0 / FORWARD_YEARS) - 1.0

    stats = {}
    for label, pars in bench_params.items():
        mu_ann  = pars["mu_ann"];  vol_ann = pars["vol_ann"]
        bench_paths = mc_single_asset(mu_ann, vol_ann, START_CAPITAL, MC_STEPS, MC_PATHS)
        Ve = bench_paths[-1, :]
        cagr_bm = (Ve / START_CAPITAL) ** (1.0 / FORWARD_YEARS) - 1.0
        excess  = cagr_port - cagr_bm
        stats[label] = {
            "mean": float(np.mean(excess)),
            "p10":  float(np.percentile(excess, 10)),
            "p90":  float(np.percentile(excess, 90)),
            "p_pos": float((excess > 0).mean())
        }

    labels = list(stats.keys())
    means  = np.array([stats[k]["mean"] for k in labels]) * 100
    p10    = np.array([stats[k]["p10"]  for k in labels]) * 100
    p90    = np.array([stats[k]["p90"]  for k in labels]) * 100
    yerr   = np.vstack([means - p10, p90 - means])

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    # Updated color scheme: blue for positive, red for negative
    colors = ["#1f77b4" if m >= 0 else "#d62728" for m in means]  # Blue for positive, red for negative
    ax.bar(labels, means, yerr=yerr, capsize=5, color=colors, alpha=0.95)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Excess CAGR (Portfolio − Benchmark), %")
    ax.set_title(f"Forward Excess vs Benchmarks (WITH 30% Randomness) — {MC_PATHS:,} paths, {MC_STEPS} months (10th–90th)")
    for i, k in enumerate(labels):
        ax.text(i, means[i] + (0.5 if means[i] >= 0 else -0.5),
                f"P(>0) {stats[k]['p_pos']*100:.0f}%",
                ha="center", va="bottom" if means[i] >= 0 else "top",
                fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig

def plot_var_95_individual(risk_metrics):
    """Individual chart for VaR 95% with clear explanation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Normal MC', 'Random MC']
    values = [
        risk_metrics['normal_mc']['var_95'] * 100,
        risk_metrics['random_mc']['var_95'] * 100
    ]
    colors = ['#1f77b4', '#d62728']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
               f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title('Value at Risk (VaR) 95% - Portfolio Risk Assessment', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('VaR 95% (%)', fontsize=12)
    ax.set_xlabel('Analysis Method', fontsize=12)
    
    # Add explanation text
    explanation = ("VaR 95% represents the maximum expected loss with 95% confidence.\n"
                  "It answers: 'What is the worst loss I can expect 95% of the time?'\n"
                  "Lower VaR = Lower Risk | Higher VaR = Higher Risk")
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    return fig

def plot_expected_shortfall_individual(risk_metrics):
    """Individual chart for Expected Shortfall with clear explanation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Normal MC', 'Random MC']
    values = [
        risk_metrics['normal_mc']['es_95'] * 100,
        risk_metrics['random_mc']['es_95'] * 100
    ]
    colors = ['#1f77b4', '#d62728']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
               f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title('Expected Shortfall (ES) 95% - Tail Risk Assessment', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Expected Shortfall 95% (%)', fontsize=12)
    ax.set_xlabel('Analysis Method', fontsize=12)
    
    # Add explanation text
    explanation = ("Expected Shortfall (CVaR) is the average loss beyond VaR 95%.\n"
                  "It answers: 'If I experience losses worse than VaR, what's the average?'\n"
                  "ES captures tail risk better than VaR alone.")
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    return fig

def plot_max_dd_duration_individual(risk_metrics):
    """Individual chart for Maximum Drawdown Duration with clear explanation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Normal MC', 'Random MC']
    values = [
        risk_metrics['normal_mc']['max_dd_duration'],
        risk_metrics['random_mc']['max_dd_duration']
    ]
    colors = ['#1f77b4', '#d62728']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
               f'{val:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title('Maximum Drawdown Duration - Recovery Time Analysis', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Duration (Months)', fontsize=12)
    ax.set_xlabel('Analysis Method', fontsize=12)
    
    # Add explanation text
    explanation = ("Max DD Duration = Longest period portfolio stays underwater.\n"
                  "It answers: 'How long might I wait to recover from losses?'\n"
                  "Shorter duration = Better liquidity management")
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    return fig

def plot_calmar_ratio_individual(risk_metrics):
    """Individual chart for Calmar Ratio with clear explanation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Normal MC', 'Random MC']
    values = [
        risk_metrics['normal_mc']['calmar_ratio'],
        risk_metrics['random_mc']['calmar_ratio']
    ]
    colors = ['#1f77b4', '#d62728']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
               f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title('Calmar Ratio - Risk-Adjusted Performance', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Calmar Ratio', fontsize=12)
    ax.set_xlabel('Analysis Method', fontsize=12)
    
    # Add explanation text
    explanation = ("Calmar Ratio = Annual Return ÷ Maximum Drawdown\n"
                  "It answers: 'How much return do I get per unit of risk?'\n"
                  "Higher ratio = Better risk-adjusted performance")
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    return fig

def plot_mc_portfolio_vs_benchmarks(paths_portfolio, bench_params, with_randomness=False):
    """Monte Carlo simulation comparing portfolio vs benchmarks over 3 years."""
    if not bench_params:
        return placeholder_figure("Portfolio vs Benchmarks (MC)", "No benchmark parameters available.")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Portfolio paths
    portfolio_median = np.median(paths_portfolio, axis=1)
    portfolio_p10 = np.percentile(paths_portfolio, 10, axis=1)
    portfolio_p90 = np.percentile(paths_portfolio, 90, axis=1)
    
    # Plot portfolio paths
    ax.plot(portfolio_median, color="#1f77b4", lw=3, label="Portfolio (Median)")
    ax.fill_between(range(len(portfolio_median)), portfolio_p10, portfolio_p90, 
                   alpha=0.3, color="#1f77b4", label="Portfolio (10th-90th percentile)")
    
    # Generate and plot benchmark simulations
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for i, (label, pars) in enumerate(bench_params.items()):
        mu_ann = pars["mu_ann"]
        vol_ann = pars["vol_ann"]
        
        # Generate benchmark paths
        if with_randomness:
            # Use enhanced MC for benchmarks too
            bench_paths = mc_gaussian_with_randomness(
                np.array([mu_ann]), np.array([[vol_ann**2]]), 
                np.array([1.0]), START_CAPITAL, MC_STEPS, MC_PATHS, RANDOMNESS_FACTOR
            )
        else:
            bench_paths = mc_single_asset(mu_ann, vol_ann, START_CAPITAL, MC_STEPS, MC_PATHS)
        
        bench_median = np.median(bench_paths, axis=1)
        bench_p10 = np.percentile(bench_paths, 10, axis=1)
        bench_p90 = np.percentile(bench_paths, 90, axis=1)
        
        color = colors[i % len(colors)]
        ax.plot(bench_median, color=color, lw=2, linestyle='--', label=f"{label} (Median)")
        ax.fill_between(range(len(bench_median)), bench_p10, bench_p90, 
                       alpha=0.2, color=color, label=f"{label} (10th-90th percentile)")
    
    # Add initial capital line
    ax.axhline(START_CAPITAL, color="gray", ls=":", lw=1, label=f"Initial €{START_CAPITAL:,.0f}")
    
    # Formatting
    randomness_text = "WITH 30% Randomness" if with_randomness else "Standard Gaussian"
    ax.set_title(f"Portfolio vs Benchmarks Monte Carlo ({randomness_text}) — 3 Years", 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Months", fontsize=12)
    ax.set_ylabel("Portfolio Value (€)", fontsize=12)
    ax.yaxis.set_major_formatter(eur_fmt)
    
    # Add explanation
    explanation = ("Monte Carlo simulation comparing portfolio performance vs major benchmarks.\n"
                  "Shows median paths and 10th-90th percentile bands over 3 years.\n"
                  "Higher values = Better performance")
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def calculate_sector_decomposition(weights_series, sector_mapping):
    """Calculate portfolio sector decomposition."""
    sector_weights = {}
    
    for ticker, weight in weights_series.items():
        sector = sector_mapping.get(ticker, "Other")
        if sector in sector_weights:
            sector_weights[sector] += weight
        else:
            sector_weights[sector] = weight
    
    return pd.Series(sector_weights).sort_values(ascending=False)

def plot_sector_decomposition(sector_weights, sector_colors):
    """Plot portfolio sector decomposition."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    sectors = sector_weights.index
    weights = sector_weights.values * 100
    colors = [sector_colors.get(sector, "#cccccc") for sector in sectors]
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(weights, labels=sectors, colors=colors, autopct='%1.1f%%',
                                     startangle=90, textprops={'fontsize': 10})
    
    # Add value labels with € amounts
    for i, (wedge, weight) in enumerate(zip(wedges, weights)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = 1.2 * np.cos(np.radians(angle))
        y = 1.2 * np.sin(np.radians(angle))
        euro_amount = weight * START_CAPITAL / 100
        ax.text(x, y, f'€{euro_amount:,.0f}', ha='center', va='center', 
               fontsize=9, fontweight='bold')
    
    ax.set_title('Portfolio Sector Decomposition', fontsize=14, fontweight='bold', pad=20)
    
    # Add explanation
    explanation = ("Portfolio allocation by economic sectors.\n"
                  "Shows percentage and € amounts for each sector.\n"
                  "Helps identify sector concentration risk.")
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    fig.tight_layout()
    return fig

def plot_sector_risk_contribution(sector_weights, sector_colors, tickers, risk_contribution):
    """Plot sector risk contribution vs weight.

    Parameters
    ----------
    sector_weights : pd.Series
        Sector weights (sum to 1).
    sector_colors : dict
        Mapping from sector name to color.
    tickers : list[str]
        List of portfolio tickers in the same order as risk_contribution.
    risk_contribution : array-like
        Risk contribution per asset (as fractions that sum to 1).
    """
    # Calculate sector risk contributions
    sector_risk = {}
    # Support both dict-like and array-like inputs cleanly
    if hasattr(risk_contribution, "items"):
        iterator = risk_contribution.items()
    else:
        iterator = zip(tickers, np.asarray(risk_contribution).ravel())
    for ticker, risk_contrib in iterator:
        sector = SECTOR_MAPPING.get(ticker, "Other")
        if sector in sector_risk:
            sector_risk[sector] += float(risk_contrib)
        else:
            sector_risk[sector] = float(risk_contrib)
    
    sector_risk_series = pd.Series(sector_risk) * 100
    sector_weight_series = sector_weights * 100
    
    # Align series
    common_sectors = sector_risk_series.index.intersection(sector_weight_series.index)
    sector_risk_aligned = sector_risk_series[common_sectors]
    sector_weight_aligned = sector_weight_series[common_sectors]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(common_sectors))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sector_weight_aligned.values, width, 
                  label='Weight %', color=[sector_colors.get(s, "#cccccc") for s in common_sectors], alpha=0.8)
    bars2 = ax.bar(x + width/2, sector_risk_aligned.values, width,
                  label='Risk Contribution %', color=[sector_colors.get(s, "#cccccc") for s in common_sectors], alpha=0.6)
    
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Sector Weight vs Risk Contribution')
    ax.set_xticks(x)
    ax.set_xticklabels(common_sectors, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

# ---------- MAIN ----------
def main():
    # --- Chargement données ---
    etf_prices_raw   = load_prices_from_dir(DATA_DIR)
    bench_prices_raw = load_prices_from_dir(BENCH_DIR)

    etf_prices   = align_business_days(etf_prices_raw)
    bench_prices = align_business_days(bench_prices_raw)

    etf_prices   = slice_recent_safe(etf_prices, ESTIMATION_YEARS)
    bench_prices = slice_recent_safe(bench_prices, ESTIMATION_YEARS)

    # --- Métriques portefeuille ---
    res = compute_metrics(etf_prices, WEIGHTS_RAW)
    print(f"Poids appliqués (colonnes présentes) : {dict(res['w_series'].round(4))}")

    # 1. Allocation
    fig = plot_allocation(res["w_series"]);            save_and_show(fig, "01_portfolio_pie.png"); plt.close(fig)

    # 2. Correlation
    fig = plot_correlation_green(res["corr"], res["cols"]); save_and_show(fig, "02_correlation_matrix.png"); plt.close(fig)

    # 3. Risk contribution
    fig = plot_risk_contribution(res["cols"], res["cr_pct"], res["w_series"])
    save_and_show(fig, "03_risk_contribution.png"); plt.close(fig)

    # 3b. Sector decomposition
    sector_weights = calculate_sector_decomposition(res["w_series"], SECTOR_MAPPING)
    fig = plot_sector_decomposition(sector_weights, SECTOR_COLORS)
    save_and_show(fig, "03b_sector_decomposition.png"); plt.close(fig)

    # 3c. Sector risk contribution
    fig = plot_sector_risk_contribution(sector_weights, SECTOR_COLORS, res["cols"], res["cr_pct"])
    save_and_show(fig, "03c_sector_risk_contribution.png"); plt.close(fig)

    # 4. Perf vs benchmarks (historique indexé)
    fig = plot_perf_vs_benchmarks(bench_prices, res["port_ret_d"])
    save_and_show(fig, "04_perf_vs_benchmarks.png"); plt.close(fig)

    # 5a. Monte Carlo portefeuille SANS randomness
    paths_normal = mc_gaussian(res["mu_a"], res["cov_a"], res["w"], START_CAPITAL, MC_STEPS, MC_PATHS)
    end_vals_normal = paths_normal[-1, :]
    median_path_normal = np.median(paths_normal, axis=1)
    p10_path_normal = np.percentile(paths_normal, 10, axis=1)
    p90_path_normal = np.percentile(paths_normal, 90, axis=1)
    approx_days = int(MC_STEPS*ANNUALIZATION/12)

    fig = plot_mc_paths(paths_normal, median_path_normal, p10_path_normal, p90_path_normal, approx_days, with_randomness=False)
    save_and_show(fig, "05a_mc_paths_normal.png"); plt.close(fig)

    # 5b. Monte Carlo portefeuille AVEC randomness
    paths_random = mc_gaussian_with_randomness(res["mu_a"], res["cov_a"], res["w"], START_CAPITAL, MC_STEPS, MC_PATHS, RANDOMNESS_FACTOR)
    end_vals_random = paths_random[-1, :]
    median_path_random = np.median(paths_random, axis=1)
    p10_path_random = np.percentile(paths_random, 10, axis=1)
    p90_path_random = np.percentile(paths_random, 90, axis=1)

    fig = plot_mc_paths(paths_random, median_path_random, p10_path_random, p90_path_random, approx_days, with_randomness=True)
    save_and_show(fig, "05b_mc_paths_with_randomness.png"); plt.close(fig)

    # Calculate comprehensive risk metrics
    risk_metrics = calculate_risk_metrics(res["port_ret_d"], paths_normal, paths_random)
    
    # 11. Individual Risk Metrics Charts
    fig = plot_var_95_individual(risk_metrics)
    save_and_show(fig, "11_var_95_individual.png"); plt.close(fig)
    
    fig = plot_expected_shortfall_individual(risk_metrics)
    save_and_show(fig, "12_expected_shortfall_individual.png"); plt.close(fig)
    
    fig = plot_max_dd_duration_individual(risk_metrics)
    save_and_show(fig, "13_max_dd_duration_individual.png"); plt.close(fig)
    
    fig = plot_calmar_ratio_individual(risk_metrics)
    save_and_show(fig, "14_calmar_ratio_individual.png"); plt.close(fig)

    # 6. Distribution finale AVEC randomness (3 ans)
    fig = plot_final_distribution(end_vals_random)
    save_and_show(fig, "06_final_distribution_with_randomness.png"); plt.close(fig)

    # 7. Volatilité projetée AVEC randomness (ann.) sur 3 ans
    pr_m_random = paths_random[1:]/paths_random[:-1] - 1.0
    vol_ann_paths_random = np.std(pr_m_random, axis=0, ddof=1) * np.sqrt(12)
    fig, ax = plt.subplots(figsize=(8.6,5))
    ax.hist(vol_ann_paths_random*100, bins=60, alpha=0.9, color="#d62728")
    ax.axvline(np.median(vol_ann_paths_random*100), color="red", lw=2, label=f"Median {np.median(vol_ann_paths_random)*100:.1f}%")
    ax.set_title("Projected Volatility (WITH 30% Randomness) — next 3 years")
    ax.set_xlabel("Volatility (%)"); ax.set_ylabel("Frequency"); ax.legend()
    fig.tight_layout(); save_and_show(fig, "07_projected_volatility_with_randomness.png"); plt.close(fig)

    # 8. Max drawdown projeté AVEC randomness (3 ans)
    def path_max_dd(v):
        peak=-np.inf; mdd=0.0
        for x in v:
            peak=max(peak,x); mdd=min(mdd, x/peak-1.0)
        return mdd
    max_dd_sim_random = np.array([path_max_dd(paths_random[:, i]) for i in range(paths_random.shape[1])])
    fig, ax = plt.subplots(figsize=(8.6,5))
    ax.hist(max_dd_sim_random*100, bins=60, alpha=0.9, color="#d62728")
    ax.axvline(np.median(max_dd_sim_random*100), color="red", lw=2, label=f"Median {np.median(max_dd_sim_random)*100:.1f}%")
    ax.set_title("Projected Max Drawdown (WITH 30% Randomness) — next 3 years")
    ax.set_xlabel("Max drawdown over horizon (%)"); ax.set_ylabel("Frequency"); ax.legend()
    fig.tight_layout(); save_and_show(fig, "08_projected_maxdd_with_randomness.png"); plt.close(fig)

    # 9. Risque vs indices (historique)
    fig = plot_risk_vs_indexes(bench_prices, res["port_ret_d"])
    save_and_show(fig, "09_risk_vs_indexes.png"); plt.close(fig)

    # 10. Forward-looking Excess vs Benchmarks AVEC randomness (MC sur indices)
    bench_params = compute_benchmark_params(bench_prices)
    fig = plot_forward_excess_vs_benchmarks(paths_random, bench_params)
    save_and_show(fig, "10_forward_excess_vs_benchmarks_with_randomness_MC.png"); plt.close(fig)

    # 15. Monte Carlo Portfolio vs Benchmarks (Standard Gaussian)
    fig = plot_mc_portfolio_vs_benchmarks(paths_normal, bench_params, with_randomness=False)
    save_and_show(fig, "15_mc_portfolio_vs_benchmarks_normal.png"); plt.close(fig)

    # 16. Monte Carlo Portfolio vs Benchmarks (WITH 30% Randomness)
    fig = plot_mc_portfolio_vs_benchmarks(paths_random, bench_params, with_randomness=True)
    save_and_show(fig, "16_mc_portfolio_vs_benchmarks_with_randomness.png"); plt.close(fig)

    # Console résumé
    print("\n=== SUMMARY ===")
    print(f"Vol annualisée (hist. portefeuille): {res['vol_a']:.2%}")
    print("Figures saved in:", os.path.abspath(RESULTS_DIR))
    
    # Risk Metrics Summary
    print("\n=== RISK METRICS SUMMARY ===")
    print("Normal Monte Carlo (Standard Gaussian):")
    print(f"  VaR 95%: {risk_metrics['normal_mc']['var_95']*100:.2f}%")
    print(f"  Expected Shortfall 95%: {risk_metrics['normal_mc']['es_95']*100:.2f}%")
    print(f"  Max DD Duration: {risk_metrics['normal_mc']['max_dd_duration']} months")
    print(f"  Calmar Ratio: {risk_metrics['normal_mc']['calmar_ratio']:.2f}")
    
    print("\nEnhanced Monte Carlo (WITH 30% Randomness):")
    print(f"  VaR 95%: {risk_metrics['random_mc']['var_95']*100:.2f}%")
    print(f"  Expected Shortfall 95%: {risk_metrics['random_mc']['es_95']*100:.2f}%")
    print(f"  Max DD Duration: {risk_metrics['random_mc']['max_dd_duration']} months")
    print(f"  Calmar Ratio: {risk_metrics['random_mc']['calmar_ratio']:.2f}")
    
    # Sector Analysis Summary
    print("\n=== SECTOR ANALYSIS ===")
    sector_weights = calculate_sector_decomposition(res["w_series"], SECTOR_MAPPING)
    for sector, weight in sector_weights.items():
        euro_amount = weight * START_CAPITAL
        print(f"  {sector}: {weight*100:.1f}% (€{euro_amount:,.0f})")

if __name__ == "__main__":
    main()
