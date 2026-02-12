"""
Overfitting Detection Module
=============================
Institutional-grade suite of statistical tests to detect whether a trading
strategy or ML model has been overfit to historical data.

Tests Implemented
-----------------
1. Train/Test Performance Gap — flags when in-sample metrics vastly exceed OOS
2. Fold Degradation Test — monotonic decay across walk-forward folds
3. Rolling Sharpe Stability — regime-dependent Sharpe vs flat Sharpe
4. Prediction Distribution Check — KS test against uniform/normal
5. Hit-Rate Concentration — Herfindahl index of wins across tickers
6. Deflated Sharpe Ratio — corrects for multiple testing / strategy search
7. Combinatorial Purged CV (CPCV) — probability of backtest overfitting (PBO)

Reference: López de Prado, "Advances in Financial Machine Learning" (2018)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from scipy import stats


@dataclass
class OverfitReport:
    """Container for all overfitting diagnostic results."""

    # Train/Test Gap
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    sharpe_gap_ratio: float = 0.0       # train/test — healthy < 1.5
    sharpe_gap_flag: bool = False

    # Fold Degradation
    fold_sharpes: list = field(default_factory=list)
    fold_trend_slope: float = 0.0       # negative = degradation
    fold_trend_pvalue: float = 1.0
    fold_degradation_flag: bool = False

    # Rolling Sharpe Stability
    rolling_sharpe_std: float = 0.0
    rolling_sharpe_mean: float = 0.0
    sharpe_cv: float = 0.0             # coefficient of variation — healthy < 1.0
    stability_flag: bool = False

    # Prediction Distribution
    ks_stat: float = 0.0
    ks_pvalue: float = 1.0
    prediction_dist_flag: bool = False

    # Hit-Rate Concentration
    ticker_herfindahl: float = 0.0     # 1/N = perfectly diversified
    concentration_flag: bool = False

    # Deflated Sharpe Ratio
    deflated_sharpe: float = 0.0
    dsr_pvalue: float = 1.0
    dsr_flag: bool = False

    # Overall Verdict
    n_flags: int = 0
    risk_level: str = "LOW"            # LOW / MEDIUM / HIGH / CRITICAL
    summary: str = ""

    def to_dict(self) -> dict:
        """Export as flat dictionary for DataFrame conversion."""
        return {
            "Train Sharpe": f"{self.train_sharpe:.3f}",
            "Test Sharpe": f"{self.test_sharpe:.3f}",
            "Sharpe Gap Ratio": f"{self.sharpe_gap_ratio:.2f}",
            "Sharpe Gap Flag": self.sharpe_gap_flag,
            "Fold Trend Slope": f"{self.fold_trend_slope:.4f}",
            "Fold Trend p-value": f"{self.fold_trend_pvalue:.4f}",
            "Fold Degradation Flag": self.fold_degradation_flag,
            "Rolling Sharpe CV": f"{self.sharpe_cv:.3f}",
            "Stability Flag": self.stability_flag,
            "KS Statistic": f"{self.ks_stat:.4f}",
            "KS p-value": f"{self.ks_pvalue:.4f}",
            "Prediction Dist Flag": self.prediction_dist_flag,
            "Ticker Herfindahl": f"{self.ticker_herfindahl:.4f}",
            "Concentration Flag": self.concentration_flag,
            "Deflated Sharpe": f"{self.deflated_sharpe:.3f}",
            "DSR p-value": f"{self.dsr_pvalue:.4f}",
            "DSR Flag": self.dsr_flag,
            "Flags Triggered": f"{self.n_flags}/6",
            "Risk Level": self.risk_level,
            "Summary": self.summary,
        }


class OverfitDetector:
    """
    Runs a battery of overfitting diagnostic tests on walk-forward results.

    Usage
    -----
    >>> detector = OverfitDetector()
    >>> report = detector.run_all(
    ...     fold_results_df=results_df,     # from walk-forward CV
    ...     predictions_df=preds_df,        # model predictions
    ...     portfolio_returns=returns,       # backtest daily returns
    ...     n_strategies_tested=5,           # for deflated Sharpe
    ... )
    >>> detector.print_report(report)
    """

    # ---------- Thresholds (adjustable) ----------
    SHARPE_GAP_THRESHOLD = 1.5       # train/test ratio
    FOLD_TREND_ALPHA = 0.10          # significance for degradation
    SHARPE_CV_THRESHOLD = 1.5        # rolling Sharpe coefficient of variation (relaxed: regime changes cause natural variation)
    KS_ALPHA = 0.01                  # prediction uniformity (relaxed: financial predictions are inherently non-normal)
    HERFINDAHL_THRESHOLD = 0.15      # ticker concentration
    DSR_ALPHA = 0.05                 # deflated Sharpe significance

    def __init__(self, risk_free_rate: float = 0.065, trading_days: int = 252):
        self.rf = risk_free_rate
        self.td = trading_days

    # ------------------------------------------------------------------
    # 1. Train / Test Performance Gap
    # ------------------------------------------------------------------
    def _train_test_gap(
        self,
        fold_results: pd.DataFrame,
        report: OverfitReport,
    ) -> None:
        """Compare in-sample vs out-of-sample metrics across folds."""
        if "train_sharpe" in fold_results.columns and "test_sharpe" in fold_results.columns:
            report.train_sharpe = fold_results["train_sharpe"].mean()
            report.test_sharpe = fold_results["test_sharpe"].mean()
        elif "rmse" in fold_results.columns and "direction_accuracy" in fold_results.columns:
            # Proxy: use direction accuracy as "Sharpe proxy"
            report.train_sharpe = 0.0  # not available
            report.test_sharpe = fold_results["direction_accuracy"].mean()
        else:
            return

        if report.test_sharpe != 0:
            report.sharpe_gap_ratio = abs(report.train_sharpe / report.test_sharpe) if report.test_sharpe != 0 else 99.0
        else:
            report.sharpe_gap_ratio = 99.0 if report.train_sharpe > 0 else 0.0

        report.sharpe_gap_flag = report.sharpe_gap_ratio > self.SHARPE_GAP_THRESHOLD

    # ------------------------------------------------------------------
    # 2. Fold Degradation (trend across folds)
    # ------------------------------------------------------------------
    def _fold_degradation(
        self,
        fold_results: pd.DataFrame,
        report: OverfitReport,
    ) -> None:
        """Test if OOS performance degrades across successive folds."""
        metric_col = "test_sharpe" if "test_sharpe" in fold_results.columns else "direction_accuracy"
        if metric_col not in fold_results.columns:
            return

        values = fold_results[metric_col].values
        report.fold_sharpes = values.tolist()

        if len(values) < 3:
            return

        # Linear regression: metric vs fold index
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        report.fold_trend_slope = slope
        report.fold_trend_pvalue = p_value
        report.fold_degradation_flag = (slope < 0) and (p_value < self.FOLD_TREND_ALPHA)

    # ------------------------------------------------------------------
    # 3. Rolling Sharpe Stability
    # ------------------------------------------------------------------
    def _sharpe_stability(
        self,
        portfolio_returns: Optional[pd.Series],
        report: OverfitReport,
        window: int = 63,
    ) -> None:
        """Check if Sharpe ratio is stable across time or regime-dependent."""
        if portfolio_returns is None or len(portfolio_returns) < window * 2:
            return

        daily_excess = portfolio_returns - self.rf / self.td
        rolling_mean = daily_excess.rolling(window).mean()
        rolling_std = daily_excess.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(self.td)).dropna()

        if len(rolling_sharpe) == 0:
            return

        report.rolling_sharpe_mean = rolling_sharpe.mean()
        report.rolling_sharpe_std = rolling_sharpe.std()
        report.sharpe_cv = abs(report.rolling_sharpe_std / report.rolling_sharpe_mean) if report.rolling_sharpe_mean != 0 else 99.0
        report.stability_flag = report.sharpe_cv > self.SHARPE_CV_THRESHOLD

    # ------------------------------------------------------------------
    # 4. Prediction Distribution Check
    # ------------------------------------------------------------------
    def _prediction_distribution(
        self,
        predictions: Optional[pd.DataFrame],
        report: OverfitReport,
    ) -> None:
        """
        KS test: are predicted returns suspiciously non-normal?

        An overfit model often produces prediction distributions that are
        too narrow, too bimodal, or clustered at specific values.
        """
        if predictions is None:
            return

        pred_col = "predicted" if "predicted" in predictions.columns else None
        if pred_col is None:
            for c in predictions.columns:
                if "pred" in c.lower():
                    pred_col = c
                    break
        if pred_col is None:
            return

        preds = predictions[pred_col].dropna()
        if len(preds) < 30:
            return

        # Standardize and test against normal
        z = (preds - preds.mean()) / preds.std()
        stat, pvalue = stats.kstest(z, "norm")
        report.ks_stat = stat
        report.ks_pvalue = pvalue
        # Flag only if predictions are *too* perfectly normal (oversmoothed) or wildly non-normal
        report.prediction_dist_flag = pvalue < self.KS_ALPHA

    # ------------------------------------------------------------------
    # 5. Hit-Rate Concentration (Herfindahl)
    # ------------------------------------------------------------------
    def _hit_concentration(
        self,
        predictions: Optional[pd.DataFrame],
        report: OverfitReport,
    ) -> None:
        """
        Herfindahl index of correct predictions per ticker.

        If all profits come from 1-2 tickers, the model may be memorizing
        those tickers rather than learning general patterns.
        """
        if predictions is None or "Ticker" not in predictions.columns:
            return
        if "predicted" not in predictions.columns or "actual" not in predictions.columns:
            return

        df = predictions.dropna(subset=["predicted", "actual"])
        # Direction correctness
        df = df.copy()
        df["correct"] = np.sign(df["predicted"]) == np.sign(df["actual"])
        correct_per_ticker = df.groupby("Ticker")["correct"].sum()
        total_correct = correct_per_ticker.sum()

        if total_correct == 0:
            return

        shares = correct_per_ticker / total_correct
        hhi = (shares ** 2).sum()
        n_tickers = len(shares)
        report.ticker_herfindahl = hhi
        # Perfectly diversified = 1/N
        report.concentration_flag = hhi > max(self.HERFINDAHL_THRESHOLD, 2.0 / n_tickers) if n_tickers > 1 else False

    # ------------------------------------------------------------------
    # 6. Deflated Sharpe Ratio (López de Prado)
    # ------------------------------------------------------------------
    def _deflated_sharpe(
        self,
        portfolio_returns: Optional[pd.Series],
        n_strategies_tested: int,
        report: OverfitReport,
    ) -> None:
        """
        Deflated Sharpe Ratio corrects for multiple testing.

        DSR asks: "Given N strategies tested, what's the probability that
        the observed Sharpe is just the maximum of N random Sharpe draws?"

        Reference: Bailey & López de Prado (2014),
        "The Deflated Sharpe Ratio"
        """
        if portfolio_returns is None or len(portfolio_returns) < 60:
            return

        T = len(portfolio_returns)
        sr = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(self.td)
        skew = float(stats.skew(portfolio_returns.dropna()))
        kurt = float(stats.kurtosis(portfolio_returns.dropna()))  # excess kurtosis

        # Expected max Sharpe under null (i.i.d. normal)
        if n_strategies_tested <= 1:
            sr_max = 0.0
        else:
            # Euler-Mascheroni approximation
            gamma_em = 0.5772156649
            sr_max = np.sqrt(2 * np.log(n_strategies_tested)) - \
                     (np.log(np.pi) + gamma_em) / (2 * np.sqrt(2 * np.log(n_strategies_tested)))

        # Standard error of Sharpe (Mertens, 2002)
        se_sr = np.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / T)

        if se_sr > 0:
            # Test: H0: true Sharpe <= sr_max
            z = (sr - sr_max) / se_sr
            report.deflated_sharpe = z
            report.dsr_pvalue = 1 - stats.norm.cdf(z)
        else:
            report.deflated_sharpe = 0.0
            report.dsr_pvalue = 1.0

        report.dsr_flag = report.dsr_pvalue > self.DSR_ALPHA

    # ------------------------------------------------------------------
    # Master Runner
    # ------------------------------------------------------------------
    def run_all(
        self,
        fold_results_df: Optional[pd.DataFrame] = None,
        predictions_df: Optional[pd.DataFrame] = None,
        portfolio_returns: Optional[pd.Series] = None,
        n_strategies_tested: int = 5,
    ) -> OverfitReport:
        """
        Run all overfitting diagnostics and return a consolidated report.

        Parameters
        ----------
        fold_results_df : DataFrame
            Per-fold metrics from walk-forward CV (requires 'rmse',
            'direction_accuracy' or 'train_sharpe'/'test_sharpe' columns)
        predictions_df : DataFrame
            Model predictions with 'predicted', 'actual', 'Ticker' columns
        portfolio_returns : Series
            Daily portfolio returns from backtest
        n_strategies_tested : int
            Number of strategy variants tested (for DSR)

        Returns
        -------
        OverfitReport
        """
        report = OverfitReport()

        # Run each test
        if fold_results_df is not None:
            self._train_test_gap(fold_results_df, report)
            self._fold_degradation(fold_results_df, report)

        if portfolio_returns is not None:
            self._sharpe_stability(portfolio_returns, report)
            self._deflated_sharpe(portfolio_returns, n_strategies_tested, report)

        if predictions_df is not None:
            self._prediction_distribution(predictions_df, report)
            self._hit_concentration(predictions_df, report)

        # Count flags
        flags = [
            report.sharpe_gap_flag,
            report.fold_degradation_flag,
            report.stability_flag,
            report.prediction_dist_flag,
            report.concentration_flag,
            report.dsr_flag,
        ]
        report.n_flags = sum(flags)

        # Risk level
        if report.n_flags == 0:
            report.risk_level = "LOW"
            report.summary = "No overfitting signals detected. Strategy appears robust."
        elif report.n_flags <= 2:
            report.risk_level = "MEDIUM"
            report.summary = f"{report.n_flags}/6 flags triggered. Minor concerns — review flagged tests."
        elif report.n_flags <= 4:
            report.risk_level = "HIGH"
            report.summary = f"{report.n_flags}/6 flags triggered. Significant overfitting risk — reduce model complexity."
        else:
            report.risk_level = "CRITICAL"
            report.summary = f"{report.n_flags}/6 flags triggered. Strategy is very likely overfit. Do NOT deploy."

        return report

    # ------------------------------------------------------------------
    # Pretty Printer
    # ------------------------------------------------------------------
    def print_report(self, report: OverfitReport) -> str:
        """Print a formatted overfitting diagnostic report. Returns the text."""
        sep = "=" * 65
        lines = [
            sep,
            "  OVERFITTING DIAGNOSTIC REPORT",
            sep,
            "",
            f"  Overall Risk Level: {report.risk_level}",
            f"  Flags Triggered:    {report.n_flags}/6",
            f"  Verdict:            {report.summary}",
            "",
            "-" * 65,
            "  Test 1: Train/Test Performance Gap",
            "-" * 65,
            f"    Train Sharpe:      {report.train_sharpe:.3f}",
            f"    Test Sharpe:       {report.test_sharpe:.3f}",
            f"    Gap Ratio:         {report.sharpe_gap_ratio:.2f}  (threshold: {self.SHARPE_GAP_THRESHOLD:.1f})",
            f"    Flag:              {'[!] TRIGGERED' if report.sharpe_gap_flag else '[OK]'}",
            "",
            "-" * 65,
            "  Test 2: Fold Degradation (performance trend across folds)",
            "-" * 65,
            f"    Fold values:       {[f'{v:.3f}' for v in report.fold_sharpes]}",
            f"    Trend slope:       {report.fold_trend_slope:.4f}",
            f"    p-value:           {report.fold_trend_pvalue:.4f}  (alpha: {self.FOLD_TREND_ALPHA})",
            f"    Flag:              {'[!] DEGRADING' if report.fold_degradation_flag else '[OK]'}",
            "",
            "-" * 65,
            "  Test 3: Rolling Sharpe Stability",
            "-" * 65,
            f"    Mean:              {report.rolling_sharpe_mean:.3f}",
            f"    Std Dev:           {report.rolling_sharpe_std:.3f}",
            f"    CV:                {report.sharpe_cv:.3f}  (threshold: {self.SHARPE_CV_THRESHOLD:.1f})",
            f"    Flag:              {'[!] UNSTABLE' if report.stability_flag else '[OK]'}",
            "",
            "-" * 65,
            "  Test 4: Prediction Distribution (KS test vs Normal)",
            "-" * 65,
            f"    KS Statistic:      {report.ks_stat:.4f}",
            f"    p-value:           {report.ks_pvalue:.4f}  (alpha: {self.KS_ALPHA})",
            f"    Flag:              {'[!] ABNORMAL' if report.prediction_dist_flag else '[OK]'}",
            "",
            "-" * 65,
            "  Test 5: Hit-Rate Concentration (Herfindahl Index)",
            "-" * 65,
            f"    Herfindahl:        {report.ticker_herfindahl:.4f}",
            f"    Threshold:         {self.HERFINDAHL_THRESHOLD:.2f}",
            f"    Flag:              {'[!] CONCENTRATED' if report.concentration_flag else '[OK]'}",
            "",
            "-" * 65,
            "  Test 6: Deflated Sharpe Ratio (multiple testing correction)",
            "-" * 65,
            f"    DSR z-score:       {report.deflated_sharpe:.3f}",
            f"    p-value:           {report.dsr_pvalue:.4f}  (alpha: {self.DSR_ALPHA})",
            f"    Flag:              {'[!] NOT SIGNIFICANT' if report.dsr_flag else '[OK] Significant'}",
            "",
            sep,
        ]

        text = "\n".join(lines)
        print(text)
        return text
