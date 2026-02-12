"""
Plotting and formatting helper functions.
Used across all notebooks for consistent visualisation.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Consistent style across all plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")

FIGSIZE_WIDE = (16, 6)
FIGSIZE_SQUARE = (10, 10)
FIGSIZE_TALL = (12, 14)


def plot_equity_curve(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Equity Curve",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot portfolio equity curve against optional benchmark."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(equity.index, equity.values, label="Strategy", linewidth=2)
    if benchmark is not None:
        # Normalize benchmark to start at same value as equity
        norm_bench = benchmark / benchmark.iloc[0] * equity.iloc[0]
        ax.plot(norm_bench.index, norm_bench.values, label="Benchmark (Nifty 50)",
                linewidth=1.5, alpha=0.7, linestyle="--")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.legend(fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_drawdown(equity: pd.Series, title: str = "Drawdown") -> plt.Figure:
    """Plot underwater (drawdown) chart."""
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.4, color="red")
    ax.plot(drawdown.index, drawdown.values, color="darkred", linewidth=1)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.tight_layout()
    return fig


def plot_monthly_returns_heatmap(
    returns: pd.Series, title: str = "Monthly Returns Heatmap"
) -> plt.Figure:
    """Plot a calendar heatmap of monthly returns."""
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_df = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values,
    })
    pivot = monthly_df.pivot_table(index="Year", columns="Month", values="Return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".1%", cmap="RdYlGn", center=0,
        linewidths=0.5, ax=ax, cbar_kws={"format": "%.0f%%"}
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 20,
    title: str = "Top Feature Importances",
) -> plt.Figure:
    """Horizontal bar chart of top N feature importances."""
    top = importance.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top)), top.values, color="steelblue")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    ax.set_xlabel("Importance")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_walk_forward_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    dates: pd.DatetimeIndex,
    title: str = "Walk-Forward CV Splits",
) -> plt.Figure:
    """Visualise train/test splits on a timeline."""
    n_folds = len(splits)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for i, (train_idx, test_idx) in enumerate(splits):
        train_start = dates[train_idx[0]]
        train_end = dates[train_idx[-1]]
        test_start = dates[test_idx[0]]
        test_end = dates[test_idx[-1]]

        y = n_folds - i
        ax.barh(y, (train_end - train_start).days, left=mdates.date2num(train_start),
                height=0.6, color="steelblue", alpha=0.7, label="Train" if i == 0 else "")
        ax.barh(y, (test_end - test_start).days, left=mdates.date2num(test_start),
                height=0.6, color="coral", alpha=0.7, label="Test" if i == 0 else "")

    ax.set_yticks(range(1, n_folds + 1))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(n_folds)][::-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def format_metrics_table(metrics: dict) -> pd.DataFrame:
    """Format a metrics dictionary as a styled DataFrame for display."""
    df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    df.index.name = "Metric"

    # Format known percentage metrics
    pct_metrics = {"Annual Return", "Annual Volatility", "Max Drawdown",
                   "Win Rate", "Daily VaR (95%)", "Downside Deviation"}
    for idx in df.index:
        if idx in pct_metrics:
            df.loc[idx, "Value"] = f"{df.loc[idx, 'Value']:.2%}"
        elif isinstance(df.loc[idx, "Value"], float):
            df.loc[idx, "Value"] = f"{df.loc[idx, 'Value']:.4f}"
    return df


def print_disclaimer():
    """Print the project disclaimer."""
    from .constants import DISCLAIMER
    print(DISCLAIMER)
