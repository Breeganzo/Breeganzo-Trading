"""
Daily Trading Report Generator (PDF)
=====================================
Generates a professional investment-banking style daily strategy report.

Includes:
  1. Executive Summary (date, capital, market outlook)
  2. Model Performance (accuracy table + charts)
  3. Today's Signals (BUY/SELL/HOLD with entry levels)
  4. Technical Analysis (RSI, MACD, Bollinger, MA levels)
  5. Position Sizing (Half-Kelly allocations)
  6. Risk Dashboard (Drawdown, Sharpe, VaR)
  7. Transaction Costs (Groww fee breakdown)
  8. Disclaimer
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt

from fpdf import FPDF


class TradingReportPDF(FPDF):
    """Custom PDF class with header/footer for trading reports."""

    # Map of unicode chars to ASCII-safe replacements
    _UNICODE_MAP = {
        "\u20b9": "Rs.",   # ₹
        "\u2014": "-",     # —
        "\u2013": "-",     # –
        "\u2018": "'",     # '
        "\u2019": "'",     # '
        "\u201c": '"',     # "
        "\u201d": '"',     # "
        "\u2026": "...",   # …
        "\u2022": "*",     # •
        "\u2713": "OK",    # ✓
        "\u2717": "X",     # ✗
        "\u26a0": "!",     # ⚠
        "\u2714": "OK",    # ✔
        "\u2716": "X",     # ✖
    }

    @staticmethod
    def _sanitize(text: str) -> str:
        """Replace non-Latin-1 characters with ASCII-safe equivalents."""
        for uchar, replacement in TradingReportPDF._UNICODE_MAP.items():
            text = text.replace(uchar, replacement)
        # Catch any remaining non-latin-1 characters
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def __init__(self, title: str = "Daily Strategy Report", date_str: str = ""):
        super().__init__()
        self.report_title = self._sanitize(title)
        self.date_str = date_str or datetime.now().strftime("%d %B %Y")

    def cell(self, *args, **kwargs):
        """Override cell to sanitize text automatically."""
        if args and len(args) >= 3 and isinstance(args[2], str):
            args = list(args)
            args[2] = self._sanitize(args[2])
        if "text" in kwargs and isinstance(kwargs["text"], str):
            kwargs["text"] = self._sanitize(kwargs["text"])
        return super().cell(*args, **kwargs)

    def multi_cell(self, *args, **kwargs):
        """Override multi_cell to sanitize text automatically."""
        if args and len(args) >= 3 and isinstance(args[2], str):
            args = list(args)
            args[2] = self._sanitize(args[2])
        if "text" in kwargs and isinstance(kwargs["text"], str):
            kwargs["text"] = self._sanitize(kwargs["text"])
        return super().multi_cell(*args, **kwargs)

    def header(self):
        self.set_fill_color(20, 30, 48)  # Dark navy
        self.rect(0, 0, 210, 22, "F")
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, self.report_title, align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(180, 200, 220)
        self.cell(0, 8, f"Generated: {self.date_str} | Masters AI Trading System | Groww Platform", align="L",
                  new_x="LMARGIN", new_y="NEXT")
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | CONFIDENTIAL - For Educational Purposes Only",
                  align="C")

    def section_header(self, title: str, number: int = 0):
        """Add a styled section header."""
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(41, 128, 185)  # Blue
        self.set_text_color(255, 255, 255)
        prefix = f"{number}. " if number else ""
        self.cell(0, 10, f"  {prefix}{title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def sub_header(self, title: str):
        """Add a sub-section header."""
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(41, 128, 185)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def body_text(self, text: str):
        """Add body text."""
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def key_value(self, key: str, value: str, bold_value: bool = False):
        """Add a key-value pair."""
        self.set_font("Helvetica", "", 10)
        self.cell(60, 6, f"  {key}:", new_x="RIGHT")
        self.set_font("Helvetica", "B" if bold_value else "", 10)
        self.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")

    def add_table(self, headers: list, data: list, col_widths: Optional[list] = None):
        """Add a formatted table."""
        if col_widths is None:
            col_widths = [190 // len(headers)] * len(headers)

        # Header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(44, 62, 80)  # Dark header
        self.set_text_color(255, 255, 255)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 8)
        self.set_text_color(0, 0, 0)
        for row_idx, row in enumerate(data):
            if row_idx % 2 == 0:
                self.set_fill_color(240, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                # Color coding for actions
                if str(val) in ("BUY", "STRONG_BUY"):
                    self.set_text_color(0, 128, 0)
                    self.set_font("Helvetica", "B", 8)
                elif str(val) in ("SELL",):
                    self.set_text_color(200, 0, 0)
                    self.set_font("Helvetica", "B", 8)
                elif str(val) in ("HOLD", "WAIT_FOR_DIP"):
                    self.set_text_color(200, 150, 0)
                    self.set_font("Helvetica", "B", 8)
                else:
                    self.set_text_color(0, 0, 0)
                    self.set_font("Helvetica", "", 8)

                self.cell(col_widths[i], 6, str(val), border=1, fill=True, align="C")
            self.ln()
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def add_image_if_exists(self, path: str, w: int = 180):
        """Add an image if the file exists."""
        if os.path.exists(path):
            self.image(path, w=w)
            self.ln(5)


class DailyReportGenerator:
    """
    Generates a comprehensive daily trading strategy PDF report.

    Usage
    -----
    >>> gen = DailyReportGenerator(capital=50000, reports_dir="reports/")
    >>> gen.generate(
    ...     allocations=allocations,
    ...     predictions=predictions_df,
    ...     price_data=price_data_dict,
    ...     model_metrics=metrics_dict,
    ...     risk_state=portfolio_state,
    ... )
    """

    def __init__(
        self,
        capital: float = 50000,
        reports_dir: str = "reports",
        broker: str = "Groww",
    ):
        self.capital = capital
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.broker = broker
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.today_display = datetime.now().strftime("%d %B %Y")

    def generate(
        self,
        allocations: list = None,
        predictions: pd.DataFrame = None,
        price_data: dict = None,
        model_metrics: dict = None,
        risk_metrics: dict = None,
        backtest_metrics: dict = None,
        technical_levels: pd.DataFrame = None,
    ) -> str:
        """Generate the full PDF report. Returns the output file path."""
        pdf = TradingReportPDF(
            title="Daily Investment Strategy Report",
            date_str=self.today_display,
        )
        pdf.alias_nb_pages()
        pdf.add_page()

        # 1. Executive Summary
        self._add_executive_summary(pdf, allocations, backtest_metrics)

        # 2. Model Performance
        pdf.add_page()
        self._add_model_performance(pdf, model_metrics)

        # 3. Today's Signals
        pdf.add_page()
        self._add_trading_signals(pdf, allocations, predictions)

        # 4. Technical Analysis Levels
        if technical_levels is not None and len(technical_levels) > 0:
            pdf.add_page()
            self._add_technical_analysis(pdf, technical_levels)

        # 5. Risk Dashboard
        pdf.add_page()
        self._add_risk_dashboard(pdf, risk_metrics, backtest_metrics)

        # 6. Transaction Costs
        self._add_transaction_costs(pdf, allocations)

        # 7. Disclaimer
        pdf.add_page()
        self._add_disclaimer(pdf)

        # Save
        output_path = self.reports_dir / f"daily_strategy_{self.today}.pdf"
        pdf.output(str(output_path))
        print(f"[OK] Report saved: {output_path}")
        return str(output_path)

    def _add_executive_summary(self, pdf: TradingReportPDF, allocations, backtest_metrics):
        pdf.section_header("Executive Summary", 1)

        pdf.key_value("Report Date", self.today_display, bold_value=True)
        pdf.key_value("Trading Capital", f"Rs.{self.capital:,.0f}", bold_value=True)
        pdf.key_value("Broker Platform", self.broker)
        pdf.key_value("Market", "NSE India (Nifty 50 + Nifty Next 50)")
        pdf.key_value("Prediction Horizon", "5 Trading Days")
        pdf.key_value("Position Sizing", "Half-Kelly Criterion")
        pdf.ln(3)

        n_buy = sum(1 for a in (allocations or []) if a.action in ("BUY", "STRONG_BUY"))
        n_sell = sum(1 for a in (allocations or []) if a.action == "SELL")
        n_hold = sum(1 for a in (allocations or []) if a.action == "HOLD")
        n_wait = sum(1 for a in (allocations or []) if a.action == "WAIT_FOR_DIP")

        pdf.sub_header("Today's Signal Summary")
        pdf.key_value("BUY Signals", str(n_buy), bold_value=True)
        pdf.key_value("SELL Signals", str(n_sell))
        pdf.key_value("HOLD Signals", str(n_hold))
        pdf.key_value("WAIT_FOR_DIP", str(n_wait))

        if backtest_metrics:
            pdf.ln(3)
            pdf.sub_header("Historical Backtest (Walk-Forward)")
            for k, v in backtest_metrics.items():
                pdf.key_value(k, str(v))

    def _add_model_performance(self, pdf: TradingReportPDF, model_metrics):
        pdf.section_header("Model Performance", 2)

        if model_metrics:
            headers = ["Model", "RMSE", "Dir. Accuracy", "IC", "Status"]
            data = []
            for name, m in model_metrics.items():
                data.append([
                    name.upper(),
                    f"{m.get('rmse', 0):.5f}",
                    f"{m.get('dir_acc', 0):.1%}",
                    f"{m.get('ic', 0):.4f}",
                    "Active" if m.get('active', True) else "Inactive",
                ])
            pdf.add_table(headers, data, col_widths=[40, 30, 35, 30, 25])
        else:
            pdf.body_text("Model metrics not available. Run notebooks 05-09 to generate.")

        # Add saved charts if they exist
        reports = str(self.reports_dir)
        pdf.add_image_if_exists(f"{reports}/algorithm_comparison.png", w=175)
        pdf.add_image_if_exists(f"{reports}/model_comparison.png", w=175)

    def _add_trading_signals(self, pdf: TradingReportPDF, allocations, predictions):
        pdf.section_header("Today's Trading Signals", 3)

        if not allocations:
            pdf.body_text("No allocations generated. Check model predictions.")
            return

        headers = ["Ticker", "Action", "Amount (Rs)", "Shares", "Price (Rs)", "Direction", "Confidence"]
        data = []
        for a in allocations:
            if a.action in ("BUY", "STRONG_BUY", "SELL"):
                data.append([
                    a.ticker.replace(".NS", ""),
                    a.action,
                    f"{a.amount_inr:,.0f}",
                    str(a.shares),
                    f"{a.current_price:,.2f}",
                    f"{a.predicted_direction:.0%}",
                    f"{a.confidence:.0%}",
                ])

        if data:
            pdf.add_table(headers, data, col_widths=[25, 28, 25, 20, 25, 25, 25])
        else:
            pdf.body_text("No actionable BUY/SELL signals today. All candidates are WAIT_FOR_DIP.")

        # Wait for dip signals
        wait_signals = [a for a in allocations if a.action == "WAIT_FOR_DIP"]
        if wait_signals:
            pdf.sub_header("Watchlist (Wait for Better Entry)")
            headers2 = ["Ticker", "Direction", "Pred. Range", "Confidence", "Rationale"]
            data2 = []
            for a in wait_signals[:10]:
                data2.append([
                    a.ticker.replace(".NS", ""),
                    f"{a.predicted_direction:.0%}",
                    f"{a.predicted_range_pct:.2%}",
                    f"{a.confidence:.0%}",
                    a.rationale[:50],
                ])
            pdf.add_table(headers2, data2, col_widths=[25, 25, 25, 25, 70])

    def _add_technical_analysis(self, pdf: TradingReportPDF, tech_df: pd.DataFrame):
        pdf.section_header("Technical Analysis - Entry/Exit Levels", 4)

        pdf.body_text(
            "For each recommended ticker, the table below shows key technical levels. "
            "Use these for intraday entry/exit decisions between 9:15 AM and 3:30 PM IST."
        )

        pdf.sub_header("Signal Interpretation Guide")
        guide_headers = ["Indicator", "Entry Signal", "Hold Signal", "Exit Signal"]
        guide_data = [
            ["RSI(14)", "< 40 (oversold)", "40-60 (neutral)", "> 70 (overbought)"],
            ["MACD", "MACD > Signal", "Histogram growing", "MACD < Signal"],
            ["Bollinger", "Near Lower Band", "Inside bands", "Near Upper Band"],
            ["SMA(20/50)", "Price > SMA20 > SMA50", "Above both SMAs", "Price < SMA20"],
            ["Volume", "> 1.5x average", "Normal", "Declining"],
        ]
        pdf.add_table(guide_headers, guide_data, col_widths=[30, 45, 45, 45])

        pdf.ln(3)
        pdf.sub_header("Current Technical Levels")

        if len(tech_df) > 0:
            cols = tech_df.columns.tolist()
            widths = [190 // len(cols)] * len(cols)
            # Adjust first column wider
            if len(cols) > 1:
                widths[0] = 25
                remaining = 165
                for i in range(1, len(cols)):
                    widths[i] = remaining // (len(cols) - 1)

            data_rows = []
            for _, row in tech_df.head(15).iterrows():
                data_rows.append([str(v) for v in row.values])
            pdf.add_table(cols, data_rows, col_widths=widths)

    def _add_risk_dashboard(self, pdf: TradingReportPDF, risk_metrics, backtest_metrics):
        pdf.section_header("Risk Dashboard", 5)

        if risk_metrics:
            pdf.sub_header("Portfolio Risk Metrics")
            risk_headers = ["Metric", "Value", "Limit", "Status"]
            risk_data = []
            for k, v in risk_metrics.items():
                if isinstance(v, dict):
                    risk_data.append([k, str(v.get("value", "")), str(v.get("limit", "")),
                                      "OK" if v.get("ok", True) else "WARNING"])
                else:
                    risk_data.append([k, str(v), "-", "OK"])
            pdf.add_table(risk_headers, risk_data, col_widths=[60, 40, 40, 30])

        # Add saved risk charts
        reports = str(self.reports_dir)
        pdf.add_image_if_exists(f"{reports}/risk_analytics.png", w=175)
        pdf.add_image_if_exists(f"{reports}/backtest_equity_curve.png", w=175)

    def _add_transaction_costs(self, pdf: TradingReportPDF, allocations):
        pdf.section_header("Transaction Cost Estimate (Groww)", 6)

        if not allocations:
            pdf.body_text("No trades to estimate costs for.")
            return

        buy_allocs = [a for a in allocations if a.action in ("BUY", "STRONG_BUY")]
        if not buy_allocs:
            pdf.body_text("No BUY signals - no transaction costs to estimate.")
            return

        try:
            from src.backtest.costs import GrowwCostCalculator
            calc = GrowwCostCalculator()

            headers = ["Ticker", "Amount (Rs)", "Brokerage", "STT", "GST", "Total Cost", "Cost %"]
            data = []
            total_cost = 0
            for a in buy_allocs:
                cost = calc.buy_cost(a.amount_inr, "equity_delivery")
                total_cost += cost.total
                data.append([
                    a.ticker.replace(".NS", ""),
                    f"{a.amount_inr:,.0f}",
                    f"Rs.{cost.brokerage:.2f}",
                    f"Rs.{cost.stt:.2f}",
                    f"Rs.{cost.gst:.2f}",
                    f"Rs.{cost.total:.2f}",
                    f"{cost.total / a.amount_inr * 100:.3f}%",
                ])
            pdf.add_table(headers, data, col_widths=[25, 27, 27, 27, 27, 27, 22])
            pdf.body_text(f"Total estimated transaction costs: Rs.{total_cost:,.2f}")
        except Exception as e:
            pdf.body_text(f"Cost calculation error: {e}")

    def _add_disclaimer(self, pdf: TradingReportPDF):
        pdf.section_header("Important Disclaimer", 7)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_fill_color(255, 240, 240)
        disclaimer = (
            "DISCLAIMER: This report is generated by an AI-based trading system for EDUCATIONAL "
            "and RESEARCH purposes only. It does NOT constitute financial advice, investment "
            "recommendations, or solicitation to buy or sell any securities.\n\n"
            "RISK WARNING: Trading in equities and derivatives involves substantial risk of loss. "
            "Past performance, whether actual or simulated, does not guarantee future results. "
            "The models used in this system have directional accuracy of approximately 52-55%, "
            "which means roughly half of all predictions may be incorrect.\n\n"
            "IMPORTANT: Always consult a SEBI-registered financial advisor before making "
            "investment decisions. Never invest more than you can afford to lose. The authors, "
            "developers, and distributors of this system accept NO LIABILITY for any trading "
            "losses incurred.\n\n"
            "This system is designed as a master's thesis project demonstrating the application "
            "of machine learning to Indian equity markets. It should be used for paper trading "
            "and backtesting only until thoroughly validated."
        )
        pdf.multi_cell(0, 5, disclaimer, fill=True)


def compute_technical_levels(price_data: dict, tickers: list = None) -> pd.DataFrame:
    """
    Compute current technical indicator levels for each ticker.

    Returns a DataFrame with columns:
    Ticker, Price, RSI, MACD_Signal, BB_Position, SMA20, SMA50,
    Entry_Price, Stop_Loss, Target, Signal
    """
    rows = []
    for ticker, df in price_data.items():
        if tickers and ticker not in tickers:
            continue
        if "Close" not in df.columns or len(df) < 60:
            continue

        close = df["Close"]
        price = close.iloc[-1]

        # RSI
        rsi_val = df["RSI"].iloc[-1] if "RSI" in df.columns else np.nan

        # MACD
        macd_val = df.get("MACD", pd.Series(dtype=float)).iloc[-1] if "MACD" in df.columns else np.nan
        macd_sig = df.get("MACD_Signal", pd.Series(dtype=float)).iloc[-1] if "MACD_Signal" in df.columns else np.nan
        macd_hist = df.get("MACD_Histogram", pd.Series(dtype=float)).iloc[-1] if "MACD_Histogram" in df.columns else np.nan

        # Bollinger Band position
        bb_pos = df.get("BB_Position", pd.Series(dtype=float)).iloc[-1] if "BB_Position" in df.columns else np.nan
        bb_upper = df.get("BB_Upper", pd.Series(dtype=float)).iloc[-1] if "BB_Upper" in df.columns else np.nan
        bb_lower = df.get("BB_Lower", pd.Series(dtype=float)).iloc[-1] if "BB_Lower" in df.columns else np.nan

        # SMAs
        sma20 = df.get("SMA_20", pd.Series(dtype=float)).iloc[-1] if "SMA_20" in df.columns else np.nan
        sma50 = df.get("SMA_50", pd.Series(dtype=float)).iloc[-1] if "SMA_50" in df.columns else np.nan

        # ATR for stop-loss/target
        atr = df.get("ATR", pd.Series(dtype=float)).iloc[-1] if "ATR" in df.columns else price * 0.02

        # Signal determination
        signal = "NEUTRAL"
        if not np.isnan(rsi_val) and not np.isnan(macd_hist):
            if rsi_val < 40 and macd_hist > 0:
                signal = "ENTRY"
            elif rsi_val > 70 or macd_hist < 0:
                signal = "EXIT"
            elif rsi_val < 50 and macd_hist > 0:
                signal = "ENTRY"
            elif rsi_val > 60 and macd_hist < 0:
                signal = "EXIT"

        # Entry/Stop/Target
        entry_price = round(price * 0.998, 2)  # Slightly below current
        stop_loss = round(price - 2 * atr, 2)
        target = round(price + 3 * atr, 2)

        rows.append({
            "Ticker": ticker.replace(".NS", ""),
            "Price": f"Rs.{price:,.2f}",
            "RSI": f"{rsi_val:.1f}" if not np.isnan(rsi_val) else "-",
            "MACD_Hist": f"{macd_hist:.2f}" if not np.isnan(macd_hist) else "-",
            "BB_Pos": f"{bb_pos:.2f}" if not np.isnan(bb_pos) else "-",
            "SMA20": f"Rs.{sma20:,.0f}" if not np.isnan(sma20) else "-",
            "SMA50": f"Rs.{sma50:,.0f}" if not np.isnan(sma50) else "-",
            "Entry": f"Rs.{entry_price:,.2f}",
            "StopLoss": f"Rs.{stop_loss:,.2f}",
            "Target": f"Rs.{target:,.2f}",
            "Signal": signal,
        })

    return pd.DataFrame(rows)
