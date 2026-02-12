"""
Risk & Return Metrics
======================
Comprehensive set of financial performance and risk metrics.
Each function includes the mathematical formula as a docstring.

Metrics Included
----------------
- Annualised return, volatility
- Sharpe ratio, Sortino ratio, Information ratio, Treynor ratio
- Alpha, Beta (CAPM)
- Omega ratio, Tail ratio, Gain-to-Pain ratio
- Maximum drawdown, Drawdown duration, Calmar ratio, Ulcer Index
- Win rate, profit factor, portfolio turnover
- Value at Risk (Historical, Parametric, Cornish-Fisher)
- Conditional VaR (CVaR / Expected Shortfall)
- Return distribution: skewness, kurtosis
- Statistical tests: Jarque-Bera, ADF stationarity, Ljung-Box
- Monte Carlo simulation for backtest confidence intervals
"""

import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats as sp_stats

from ..utils.constants import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Annualised return from a series of periodic returns.

    Formula:
        R_annual = (1 + R_total)^(periods_per_year / n_periods) - 1

    Parameters
    ----------
    returns : pd.Series
        Daily (or periodic) returns

    Returns
    -------
    float
        Annualised return as a decimal
    """
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0
    return (1 + total_return) ** (periods_per_year / n_periods) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Annualised volatility (standard deviation of returns).

    Formula:
        σ_annual = σ_daily × √(periods_per_year)

    Returns
    -------
    float
        Annualised volatility as a decimal
    """
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Sharpe Ratio — risk-adjusted return using total volatility.

    Formula:
        Sharpe = (R_p - R_f) / σ_p

    Where:
        R_p = annualised portfolio return
        R_f = annualised risk-free rate
        σ_p = annualised portfolio volatility

    Interpretation:
        < 0.5: Poor
        0.5–1.0: Acceptable
        1.0–2.0: Good
        > 2.0: Excellent (possibly overfitting in backtest)

    Returns
    -------
    float
    """
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Sortino Ratio — like Sharpe but penalises only downside volatility.

    Formula:
        Sortino = (R_p - R_f) / σ_d

    Where:
        σ_d = √(mean(min(r_i, 0)²)) × √(periods_per_year)
        (downside deviation: std dev of negative returns only)

    This is more appropriate than Sharpe when return distributions
    are asymmetric (positive skew from options or momentum strategies).

    Returns
    -------
    float
    """
    ann_ret = annualized_return(returns, periods_per_year)
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return float("inf")  # No negative returns
    downside_dev = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(periods_per_year)
    if downside_dev == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / downside_dev


def compute_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Beta — sensitivity of portfolio returns to benchmark returns.

    Formula:
        β = Cov(R_p, R_m) / Var(R_m)

    Interpretation:
        β = 1: Same volatility as market
        β > 1: More volatile than market (amplifies market moves)
        β < 1: Less volatile than market (defensive)
        β < 0: Inversely correlated with market (rare, hedge-like)

    Returns
    -------
    float
    """
    # Align dates
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    aligned.columns = ["portfolio", "benchmark"]
    cov = aligned["portfolio"].cov(aligned["benchmark"])
    var_benchmark = aligned["benchmark"].var()
    if var_benchmark == 0:
        return 0.0
    return cov / var_benchmark


def compute_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Alpha (Jensen's Alpha) — excess return after adjusting for systematic risk.

    Formula (CAPM):
        α = R_p - [R_f + β × (R_m - R_f)]

    Where:
        R_p = annualised portfolio return
        R_f = risk-free rate
        β = portfolio beta
        R_m = annualised benchmark return

    Interpretation:
        α > 0: Portfolio outperforms on risk-adjusted basis (skill)
        α < 0: Portfolio underperforms (no skill, or bad luck)
        α = 0: Portfolio returns are fully explained by market risk

    Returns
    -------
    float
        Annualised alpha as a decimal
    """
    ann_port = annualized_return(portfolio_returns, periods_per_year)
    ann_bench = annualized_return(benchmark_returns, periods_per_year)
    beta = compute_beta(portfolio_returns, benchmark_returns)
    expected_return = risk_free_rate + beta * (ann_bench - risk_free_rate)
    return ann_port - expected_return


def omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0,
) -> float:
    """
    Omega Ratio — probability-weighted gains vs losses.

    Formula:
        Ω(θ) = ∫_θ^∞ [1 - F(r)] dr / ∫_{-∞}^θ F(r) dr

    In practice (discrete):
        Ω(θ) = Σ max(r_i - θ, 0) / Σ max(θ - r_i, 0)

    Unlike Sharpe (which uses only mean and std dev), Omega captures
    ALL moments of the return distribution (skewness, kurtosis, tails).

    Interpretation:
        Ω > 1: Gains outweigh losses at the threshold
        Ω < 1: Losses outweigh gains
        Ω = 1: Break-even

    Parameters
    ----------
    returns : pd.Series
        Period returns
    threshold : float
        Return threshold (default 0 = break-even)

    Returns
    -------
    float
    """
    gains = np.maximum(returns - threshold, 0).sum()
    losses = np.maximum(threshold - returns, 0).sum()
    if losses == 0:
        return float("inf")
    return gains / losses


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum Drawdown — largest peak-to-trough decline.

    Formula:
        MDD = max_t { (Peak_t - Trough_t) / Peak_t }

    This measures the worst-case capital loss an investor would have
    experienced. Critical for risk management and investor psychology.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio value over time (NOT returns)

    Returns
    -------
    float
        Max drawdown as a negative decimal (e.g., -0.15 = 15% drawdown)
    """
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()


def calmar_ratio(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calmar Ratio — annualised return / max drawdown.

    Formula:
        Calmar = R_annual / |MDD|

    Interpretation:
        > 1.0: Good
        > 3.0: Excellent
        > 5.0: Exceptional (possibly overfitting)

    Returns
    -------
    float
    """
    ann_ret = annualized_return(returns, periods_per_year)
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return ann_ret / mdd


def win_rate(returns: pd.Series) -> float:
    """
    Win Rate — percentage of positive-return trades/days.

    Formula:
        Win Rate = N(r > 0) / N(total)

    Returns
    -------
    float
        Win rate as a decimal
    """
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns)


def profit_factor(returns: pd.Series) -> float:
    """
    Profit Factor — ratio of gross profits to gross losses.

    Formula:
        PF = Σ(positive returns) / |Σ(negative returns)|

    Interpretation:
        > 1.0: Profitable
        > 1.5: Good
        > 2.0: Very good

    Returns
    -------
    float
    """
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss == 0:
        return float("inf")
    return gross_profit / gross_loss


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Value at Risk (VaR) — maximum expected loss at a confidence level.

    Formula:
        VaR_α = -Percentile(returns, 1 - α)

    Parameters
    ----------
    returns : pd.Series
        Daily returns
    confidence : float
        Confidence level (default: 0.95 → 95% VaR)

    Returns
    -------
    float
        VaR as a positive decimal (e.g., 0.02 = 2% daily VaR)
    """
    return -np.percentile(returns.dropna(), (1 - confidence) * 100)


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (CVaR / Expected Shortfall) — expected loss
    beyond VaR. More coherent risk measure than VaR.

    Formula:
        CVaR_α = -E[r | r ≤ -VaR_α]

    Returns
    -------
    float
    """
    var = value_at_risk(returns, confidence)
    beyond_var = returns[returns <= -var]
    if len(beyond_var) == 0:
        return var
    return -beyond_var.mean()


# ---------------------------------------------------------------------------
# Additional Quant Metrics — Interview-Grade
# ---------------------------------------------------------------------------

def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Information Ratio — active return per unit of tracking error.

    Formula:
        IR = (R_p - R_b) / TE
        TE = σ(R_p - R_b) × √(periods_per_year)

    The primary performance metric used by Goldman Sachs and BlackRock
    for evaluating portfolio managers. IR > 0.5 is good, > 1.0 is exceptional.

    Returns
    -------
    float
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    aligned.columns = ["portfolio", "benchmark"]
    active_return = aligned["portfolio"] - aligned["benchmark"]
    tracking_error = active_return.std() * np.sqrt(periods_per_year)
    if tracking_error == 0:
        return 0.0
    ann_active = annualized_return(aligned["portfolio"], periods_per_year) - \
                 annualized_return(aligned["benchmark"], periods_per_year)
    return ann_active / tracking_error


def treynor_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Treynor Ratio — return per unit of systematic (market) risk.

    Formula:
        Treynor = (R_p - R_f) / β

    Unlike Sharpe (which uses total risk), Treynor uses only systematic
    risk (beta). Appropriate for diversified portfolios where
    unsystematic risk has been diversified away.

    Returns
    -------
    float
    """
    beta = compute_beta(portfolio_returns, benchmark_returns)
    if beta == 0:
        return 0.0
    ann_ret = annualized_return(portfolio_returns, periods_per_year)
    return (ann_ret - risk_free_rate) / beta


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """
    Maximum Drawdown Duration — longest time underwater.

    Measures the number of trading days from peak to recovery.
    Citadel risk reports always include recovery time alongside
    drawdown magnitude.

    Returns
    -------
    int
        Number of trading days of the longest drawdown period
    """
    peak = equity_curve.cummax()
    underwater = equity_curve < peak

    if not underwater.any():
        return 0

    # Find consecutive underwater periods
    max_duration = 0
    current_duration = 0
    for is_underwater in underwater:
        if is_underwater:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration


def return_skewness(returns: pd.Series) -> float:
    """
    Skewness of return distribution.

    Formula:
        Skew = E[(r - μ)³] / σ³

    Interpretation:
        Skew > 0: Right tail heavier (desirable — large positive outliers)
        Skew < 0: Left tail heavier (risky — large negative outliers)
        Skew ≈ 0: Symmetric distribution

    Most equity returns exhibit negative skewness (crashes are larger
    than rallies), which makes positive-skew strategies valuable.

    Returns
    -------
    float
    """
    return float(returns.skew())


def return_kurtosis(returns: pd.Series) -> float:
    """
    Excess Kurtosis of return distribution.

    Formula:
        Kurt = E[(r - μ)⁴] / σ⁴ - 3

    Interpretation:
        Kurt > 0: Fat tails (more extreme events than normal)
        Kurt ≈ 0: Normal-like tails
        Kurt < 0: Thin tails (rare)

    Financial returns almost always have positive excess kurtosis
    (leptokurtic). This invalidates naive Gaussian VaR assumptions.

    Returns
    -------
    float
        Excess kurtosis (Fisher's definition, 0 for normal distribution)
    """
    return float(returns.kurtosis())


def tail_ratio(returns: pd.Series) -> float:
    """
    Tail Ratio — asymmetry of return tails.

    Formula:
        Tail Ratio = |Percentile(95)| / |Percentile(5)|

    Interpretation:
        > 1.0: Right tail is larger (more upside than downside)
        < 1.0: Left tail is larger (more downside than upside)
        = 1.0: Symmetric tails

    Returns
    -------
    float
    """
    p95 = np.percentile(returns.dropna(), 95)
    p5 = np.percentile(returns.dropna(), 5)
    if p5 == 0:
        return float("inf")
    return abs(p95 / p5)


def gain_to_pain_ratio(returns: pd.Series) -> float:
    """
    Gain-to-Pain Ratio — sum of all returns / sum of absolute losses.

    Formula:
        GtP = Σ(r_i) / Σ|min(r_i, 0)|

    Simpler than Omega ratio but captures similar risk/reward asymmetry.

    Returns
    -------
    float
    """
    total_return = returns.sum()
    total_pain = abs(returns[returns < 0].sum())
    if total_pain == 0:
        return float("inf")
    return total_return / total_pain


def ulcer_index(equity_curve: pd.Series) -> float:
    """
    Ulcer Index — time-weighted drawdown severity.

    Formula:
        UI = √(Σ(DD_i²) / N)

    Unlike max drawdown (which captures only the worst point),
    the Ulcer Index penalises persistent drawdowns over time.

    Returns
    -------
    float
        Ulcer Index as a decimal (lower is better)
    """
    peak = equity_curve.cummax()
    dd_pct = ((equity_curve - peak) / peak * 100)
    return np.sqrt(np.mean(dd_pct ** 2))


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "cornish_fisher",
) -> float:
    """
    Parametric Value at Risk with Cornish-Fisher expansion.

    The Cornish-Fisher VaR adjusts the Gaussian VaR for non-normal
    return distributions using skewness and kurtosis corrections.
    JPMorgan's RiskMetrics framework uses parametric VaR as standard.

    Formula (Cornish-Fisher):
        z_cf = z + (z² - 1)S/6 + (z³ - 3z)K/24 - (2z³ - 5z)S²/36

    Where:
        z = normal quantile at confidence level
        S = skewness of returns
        K = excess kurtosis of returns

    Parameters
    ----------
    method : str
        'gaussian' — standard normal VaR
        'cornish_fisher' — skew/kurtosis adjusted (recommended)

    Returns
    -------
    float
        VaR as a positive decimal
    """
    z = sp_stats.norm.ppf(1 - confidence)

    if method == "gaussian":
        return -(returns.mean() + z * returns.std())

    # Cornish-Fisher expansion
    s = return_skewness(returns)
    k = return_kurtosis(returns)
    z_cf = (z +
            (z**2 - 1) * s / 6 +
            (z**3 - 3 * z) * k / 24 -
            (2 * z**3 - 5 * z) * s**2 / 36)

    return -(returns.mean() + z_cf * returns.std())


def portfolio_turnover(
    trade_signals: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualised Portfolio Turnover — how frequently positions change.

    Formula:
        Turnover = (N_trades / N_days) × periods_per_year

    High turnover increases transaction costs and tax drag.
    Firms track this obsessively for cost attribution.

    Parameters
    ----------
    trade_signals : pd.Series
        Series of position signals (1=long, 0=flat, -1=short).
        Trade occurs when signal changes.

    Returns
    -------
    float
        Annualised turnover rate
    """
    if len(trade_signals) < 2:
        return 0.0
    changes = (trade_signals != trade_signals.shift(1)).sum() - 1  # -1 for initial position
    daily_turnover = changes / len(trade_signals)
    return daily_turnover * periods_per_year


# ---------------------------------------------------------------------------
# Statistical Tests — Institutional-Grade Validation
# ---------------------------------------------------------------------------

def jarque_bera_test(returns: pd.Series) -> dict:
    """
    Jarque-Bera test for normality of returns.

    Tests H₀: returns are normally distributed.
    If rejected (p < 0.05), parametric methods (Gaussian VaR, standard
    Sharpe) may be unreliable. Must use non-parametric alternatives.

    Formula:
        JB = (n/6) × [S² + K²/4]

    Returns
    -------
    dict
        statistic, p_value, is_normal (True if p > 0.05)
    """
    clean = returns.dropna()
    if len(clean) < 20:
        return {"statistic": 0, "p_value": 1.0, "is_normal": True}

    stat, p_value = sp_stats.jarque_bera(clean)
    return {
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "is_normal": bool(p_value > 0.05),
    }


def adf_stationarity_test(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.

    Tests H₀: series has a unit root (non-stationary).
    Non-stationary features cause spurious regressions — the most
    important prerequisite for any time-series model.

    Returns
    -------
    dict
        statistic, p_value, is_stationary
    """
    from statsmodels.tsa.stattools import adfuller

    clean = series.dropna()
    if len(clean) < 20:
        return {"statistic": 0, "p_value": 1.0, "is_stationary": False}

    result = adfuller(clean, autolag="AIC")
    return {
        "statistic": round(float(result[0]), 4),
        "p_value": round(float(result[1]), 6),
        "critical_values": {k: round(v, 4) for k, v in result[4].items()},
        "is_stationary": bool(result[1] < 0.05),
    }


def ljung_box_test(returns: pd.Series, lags: int = 10) -> dict:
    """
    Ljung-Box test for autocorrelation in residuals.

    Tests H₀: no autocorrelation at lags 1 through k.
    If rejected, model residuals have exploitable patterns —
    the model is systematically wrong.

    Returns
    -------
    dict
        test result with p-values for each lag
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    clean = returns.dropna()
    if len(clean) < lags + 5:
        return {"has_autocorrelation": False, "min_p_value": 1.0}

    result = acorr_ljungbox(clean, lags=lags, return_df=True)
    min_p = float(result["lb_pvalue"].min())
    return {
        "has_autocorrelation": bool(min_p < 0.05),
        "min_p_value": round(min_p, 6),
        "lag_p_values": {int(k): round(float(v), 6) for k, v in result["lb_pvalue"].items()},
    }


# ---------------------------------------------------------------------------
# Monte Carlo Simulation — Backtest Confidence Intervals
# ---------------------------------------------------------------------------

def monte_carlo_backtest(
    returns: pd.Series,
    n_simulations: int = 1000,
    n_days: Optional[int] = None,
    initial_capital: float = 100000.0,
) -> dict:
    """
    Monte Carlo simulation for backtest robustness.

    Generates n_simulations equity curves by randomly resampling
    (bootstrapping) historical returns. Provides percentile-based
    confidence intervals on terminal wealth and key metrics.

    A single backtest equity curve is statistically meaningless.
    Monte Carlo shows the RANGE of possible outcomes given the
    same return characteristics.

    Parameters
    ----------
    returns : pd.Series
        Historical daily returns
    n_simulations : int
        Number of bootstrap paths (1000 is standard)
    n_days : int, optional
        Simulation horizon (default: same as returns length)
    initial_capital : float
        Starting portfolio value

    Returns
    -------
    dict
        terminal_wealth_percentiles, max_drawdown_percentiles,
        sharpe_percentiles, probability_of_profit
    """
    clean = returns.dropna().values
    if len(clean) < 20:
        return {"error": "Insufficient return history for Monte Carlo"}

    if n_days is None:
        n_days = len(clean)

    terminal_wealths = []
    max_drawdowns = []
    sharpe_ratios_mc = []

    rng = np.random.default_rng(42)

    for _ in range(n_simulations):
        # Bootstrap resample (with replacement)
        sim_returns = rng.choice(clean, size=n_days, replace=True)
        sim_equity = initial_capital * np.cumprod(1 + sim_returns)

        terminal_wealths.append(sim_equity[-1])

        # Max drawdown of this path
        peak = np.maximum.accumulate(sim_equity)
        dd = (sim_equity - peak) / peak
        max_drawdowns.append(dd.min())

        # Sharpe of this path
        ann_ret = (sim_equity[-1] / initial_capital) ** (252 / n_days) - 1
        ann_vol = np.std(sim_returns) * np.sqrt(252)
        sr = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
        sharpe_ratios_mc.append(sr)

    tw = np.array(terminal_wealths)
    mdd = np.array(max_drawdowns)
    sr_arr = np.array(sharpe_ratios_mc)

    return {
        "n_simulations": n_simulations,
        "n_days": n_days,
        "initial_capital": initial_capital,
        "terminal_wealth": {
            "mean": round(float(tw.mean()), 2),
            "median": round(float(np.median(tw)), 2),
            "p5": round(float(np.percentile(tw, 5)), 2),
            "p25": round(float(np.percentile(tw, 25)), 2),
            "p75": round(float(np.percentile(tw, 75)), 2),
            "p95": round(float(np.percentile(tw, 95)), 2),
        },
        "max_drawdown": {
            "mean": round(float(mdd.mean()), 4),
            "p5_worst": round(float(np.percentile(mdd, 5)), 4),
            "p95_best": round(float(np.percentile(mdd, 95)), 4),
        },
        "sharpe_ratio": {
            "mean": round(float(sr_arr.mean()), 4),
            "p5": round(float(np.percentile(sr_arr, 5)), 4),
            "p95": round(float(np.percentile(sr_arr, 95)), 4),
        },
        "probability_of_profit": round(float((tw > initial_capital).mean()), 4),
        "probability_of_loss_gt_10pct": round(float((tw < initial_capital * 0.9).mean()), 4),
    }


def compute_all_metrics(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = RISK_FREE_RATE,
    trade_signals: Optional[pd.Series] = None,
) -> dict:
    """
    Compute all risk/return metrics in one call.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns
    equity_curve : pd.Series
        Portfolio value over time
    benchmark_returns : pd.Series, optional
        Benchmark daily returns (for alpha/beta/IR calculation)
    risk_free_rate : float
        Annual risk-free rate
    trade_signals : pd.Series, optional
        Position signals for turnover calculation

    Returns
    -------
    dict
        All metrics as a dictionary
    """
    metrics = {
        # ── Core Returns ──
        "Annual Return": annualized_return(portfolio_returns),
        "Annual Volatility": annualized_volatility(portfolio_returns),
        "Total Return": (1 + portfolio_returns).prod() - 1,

        # ── Risk-Adjusted Returns ──
        "Sharpe Ratio": sharpe_ratio(portfolio_returns, risk_free_rate),
        "Sortino Ratio": sortino_ratio(portfolio_returns, risk_free_rate),
        "Omega Ratio": omega_ratio(portfolio_returns),
        "Calmar Ratio": calmar_ratio(portfolio_returns, equity_curve),
        "Gain-to-Pain Ratio": gain_to_pain_ratio(portfolio_returns),

        # ── Drawdown Metrics ──
        "Max Drawdown": max_drawdown(equity_curve),
        "Max Drawdown Duration (days)": max_drawdown_duration(equity_curve),
        "Ulcer Index": ulcer_index(equity_curve),

        # ── Risk Metrics ──
        "Daily VaR (95%)": value_at_risk(portfolio_returns, 0.95),
        "Daily CVaR (95%)": conditional_var(portfolio_returns, 0.95),
        "Parametric VaR (Cornish-Fisher 95%)": parametric_var(portfolio_returns, 0.95, "cornish_fisher"),
        "Parametric VaR (Gaussian 95%)": parametric_var(portfolio_returns, 0.95, "gaussian"),

        # ── Distribution Metrics ──
        "Skewness": return_skewness(portfolio_returns),
        "Excess Kurtosis": return_kurtosis(portfolio_returns),
        "Tail Ratio": tail_ratio(portfolio_returns),

        # ── Trade Metrics ──
        "Win Rate": win_rate(portfolio_returns),
        "Profit Factor": profit_factor(portfolio_returns),
        "Downside Deviation": portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
                              if len(portfolio_returns[portfolio_returns < 0]) > 0 else 0.0,

        # ── Counts ──
        "Total Trading Days": len(portfolio_returns),
        "Positive Days": int((portfolio_returns > 0).sum()),
        "Negative Days": int((portfolio_returns < 0).sum()),

        # ── Statistical Validation ──
        "Jarque-Bera Test": jarque_bera_test(portfolio_returns),
    }

    # Turnover (if trade signals provided)
    if trade_signals is not None:
        metrics["Annual Turnover"] = portfolio_turnover(trade_signals)

    # Benchmark-relative metrics
    if benchmark_returns is not None:
        metrics["Alpha"] = compute_alpha(portfolio_returns, benchmark_returns, risk_free_rate)
        metrics["Beta"] = compute_beta(portfolio_returns, benchmark_returns)
        metrics["Information Ratio"] = information_ratio(portfolio_returns, benchmark_returns)
        metrics["Treynor Ratio"] = treynor_ratio(portfolio_returns, benchmark_returns, risk_free_rate)

        # Benchmark comparison
        metrics["Benchmark Return"] = annualized_return(benchmark_returns)
        metrics["Benchmark Volatility"] = annualized_volatility(benchmark_returns)
        metrics["Benchmark Sharpe"] = sharpe_ratio(benchmark_returns, risk_free_rate)

    return metrics
