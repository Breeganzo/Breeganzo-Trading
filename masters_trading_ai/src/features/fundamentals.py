"""
Fundamental Analysis Module
=============================
Fetches and scores fundamental data from yfinance for live predictions.

Uses PE ratio, PB ratio, dividend yield, ROE, debt-to-equity,
earnings growth, free cash flow, and analyst targets to compute:
  - Value Score (is it cheap?)
  - Quality Score (is it healthy?)
  - Growth Score (is it growing?)
  - Composite Fundamental Score (overall attractiveness)

IMPORTANT: This module is for LIVE predictions only. Historical fundamental
data from yfinance is point-in-time current data and cannot be used for
backtesting without introducing look-ahead bias.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

try:
    import yfinance as yf
except ImportError:
    yf = None


class FundamentalAnalyzer:
    """
    Fetches and analyzes fundamental data from yfinance.
    
    Caches results to avoid repeated API calls (6-hour TTL).
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(__file__).resolve().parent.parent.parent / "cache" / "fundamentals"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "fundamental_cache.json"
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if self._cache_file.exists():
            try:
                with open(self._cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_cache(self):
        with open(self._cache_file, "w") as f:
            json.dump(self._cache, f, indent=2, default=str)

    def _is_cache_valid(self, ticker: str) -> bool:
        entry = self._cache.get(ticker)
        if entry is None:
            return False
        cached_time = datetime.fromisoformat(entry.get("_cached_at", "2000-01-01"))
        return datetime.now() - cached_time < timedelta(hours=6)

    def fetch_single(self, ticker: str, use_cache: bool = True) -> dict:
        """
        Fetch fundamental data for a single ticker.

        Returns dict with PE, PB, dividend yield, ROE, etc.
        """
        if use_cache and self._is_cache_valid(ticker):
            return self._cache[ticker]

        if yf is None:
            return {"Ticker": ticker, "Error": "yfinance not installed"}

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            result = {
                "Ticker": ticker,
                "Name": info.get("shortName", ticker.replace(".NS", "")),
                "Sector": info.get("sector", "Unknown"),
                "Industry": info.get("industry", "Unknown"),
                "MarketCap": info.get("marketCap", 0),
                "PE_Ratio": info.get("trailingPE"),
                "Forward_PE": info.get("forwardPE"),
                "PB_Ratio": info.get("priceToBook"),
                "Dividend_Yield": info.get("dividendYield", 0) or 0,
                "ROE": info.get("returnOnEquity"),
                "ROA": info.get("returnOnAssets"),
                "Debt_to_Equity": info.get("debtToEquity"),
                "Revenue_Growth": info.get("revenueGrowth"),
                "Earnings_Growth": info.get("earningsGrowth"),
                "Profit_Margin": info.get("profitMargins"),
                "Operating_Margin": info.get("operatingMargins"),
                "Current_Price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "Target_Price": info.get("targetMeanPrice"),
                "FiftyTwo_Week_High": info.get("fiftyTwoWeekHigh"),
                "FiftyTwo_Week_Low": info.get("fiftyTwoWeekLow"),
                "Beta": info.get("beta"),
                "Free_Cash_Flow": info.get("freeCashflow"),
                "Book_Value": info.get("bookValue"),
                "EPS_Trailing": info.get("trailingEps"),
                "EPS_Forward": info.get("forwardEps"),
                "_cached_at": datetime.now().isoformat(),
            }

            self._cache[ticker] = result
            self._save_cache()
            return result

        except Exception as e:
            return {"Ticker": ticker, "Error": str(e)}

    def fetch_batch(self, tickers: list[str], use_cache: bool = True) -> pd.DataFrame:
        """Fetch fundamentals for multiple tickers. Returns DataFrame."""
        results = []
        for ticker in tickers:
            data = self.fetch_single(ticker, use_cache=use_cache)
            if "Error" not in data:
                results.append(data)
        return pd.DataFrame(results)

    def compute_scores(self, fundamentals: pd.DataFrame) -> pd.DataFrame:
        """
        Compute composite fundamental scores.

        Returns DataFrame with Value_Score, Quality_Score, Growth_Score,
        and Fundamental_Score (composite).
        """
        df = fundamentals.copy()

        # --- Value Score (lower PE/PB = better value) ---
        # Rank within universe: lower PE → higher score
        pe = df["PE_Ratio"].clip(0, 200)  # Cap extreme PEs
        df["PE_Score"] = 100 - pe.rank(pct=True, na_option="keep") * 100

        pb = df["PB_Ratio"].clip(0, 50)
        df["PB_Score"] = 100 - pb.rank(pct=True, na_option="keep") * 100

        div = df["Dividend_Yield"].fillna(0)
        df["Div_Score"] = div.rank(pct=True, na_option="keep") * 100

        df["Value_Score"] = (
            df["PE_Score"].fillna(50) * 0.40 +
            df["PB_Score"].fillna(50) * 0.35 +
            df["Div_Score"].fillna(50) * 0.25
        )

        # --- Quality Score (profitability + low debt) ---
        roe = df["ROE"].fillna(0).clip(-1, 1) * 100
        df["ROE_Score"] = roe.rank(pct=True, na_option="keep") * 100

        # Lower debt = better quality
        dte = df["Debt_to_Equity"].fillna(100).clip(0, 500)
        df["Debt_Score"] = 100 - dte.rank(pct=True, na_option="keep") * 100

        margin = df["Profit_Margin"].fillna(0).clip(-1, 1) * 100
        df["Margin_Score"] = margin.rank(pct=True, na_option="keep") * 100

        df["Quality_Score"] = (
            df["ROE_Score"].fillna(50) * 0.40 +
            df["Debt_Score"].fillna(50) * 0.30 +
            df["Margin_Score"].fillna(50) * 0.30
        )

        # --- Growth Score ---
        rev_g = df["Revenue_Growth"].fillna(0).clip(-1, 5)
        df["RevGrowth_Score"] = rev_g.rank(pct=True, na_option="keep") * 100

        earn_g = df["Earnings_Growth"].fillna(0).clip(-2, 10)
        df["EarnGrowth_Score"] = earn_g.rank(pct=True, na_option="keep") * 100

        df["Growth_Score"] = (
            df["RevGrowth_Score"].fillna(50) * 0.50 +
            df["EarnGrowth_Score"].fillna(50) * 0.50
        )

        # --- 52-Week Position (momentum/value indicator) ---
        if "Current_Price" in df.columns and "FiftyTwo_Week_High" in df.columns:
            df["FiftyTwo_Position"] = (
                (df["Current_Price"] - df["FiftyTwo_Week_Low"])
                / (df["FiftyTwo_Week_High"] - df["FiftyTwo_Week_Low"] + 1e-10)
            ).clip(0, 1) * 100

        # --- Analyst Upside ---
        if "Target_Price" in df.columns and "Current_Price" in df.columns:
            df["Analyst_Upside"] = (
                (df["Target_Price"] - df["Current_Price"])
                / (df["Current_Price"] + 1e-10) * 100
            ).clip(-50, 100)

        # --- Composite Fundamental Score ---
        df["Fundamental_Score"] = (
            df["Value_Score"].fillna(50) * 0.30 +
            df["Quality_Score"].fillna(50) * 0.35 +
            df["Growth_Score"].fillna(50) * 0.35
        )

        return df

    def get_recommendation(self, ticker: str) -> dict:
        """
        Get a fundamental-based recommendation for a single ticker.

        Returns dict with scores and recommendation.
        """
        data = self.fetch_single(ticker)
        if "Error" in data:
            return {"ticker": ticker, "error": data["Error"]}

        # Need at least PE or PB for meaningful scoring
        df = pd.DataFrame([data])
        df = self.compute_scores(df)
        row = df.iloc[0]

        fund_score = row.get("Fundamental_Score", 50)
        value_score = row.get("Value_Score", 50)
        quality_score = row.get("Quality_Score", 50)
        growth_score = row.get("Growth_Score", 50)
        upside = row.get("Analyst_Upside", 0)

        # Classification
        if fund_score >= 70 and upside > 10:
            recommendation = "STRONG_BUY"
        elif fund_score >= 60 or upside > 15:
            recommendation = "BUY"
        elif fund_score <= 30 or upside < -10:
            recommendation = "SELL"
        elif fund_score <= 20 and upside < -5:
            recommendation = "STRONG_SELL"
        else:
            recommendation = "HOLD"

        return {
            "ticker": ticker,
            "name": data.get("Name", ticker),
            "sector": data.get("Sector", "Unknown"),
            "pe_ratio": data.get("PE_Ratio"),
            "pb_ratio": data.get("PB_Ratio"),
            "dividend_yield": round((data.get("Dividend_Yield", 0) or 0) * 100, 2),
            "roe": round((data.get("ROE", 0) or 0) * 100, 2),
            "debt_to_equity": data.get("Debt_to_Equity"),
            "revenue_growth": round((data.get("Revenue_Growth", 0) or 0) * 100, 2),
            "earnings_growth": round((data.get("Earnings_Growth", 0) or 0) * 100, 2),
            "profit_margin": round((data.get("Profit_Margin", 0) or 0) * 100, 2),
            "current_price": data.get("Current_Price"),
            "target_price": data.get("Target_Price"),
            "fifty_two_high": data.get("FiftyTwo_Week_High"),
            "fifty_two_low": data.get("FiftyTwo_Week_Low"),
            "analyst_upside": None if pd.isna(upside) else round(upside, 2),
            "value_score": round(value_score, 1),
            "quality_score": round(quality_score, 1),
            "growth_score": round(growth_score, 1),
            "fundamental_score": round(fund_score, 1),
            "recommendation": recommendation,
            "market_cap": data.get("MarketCap", 0),
            "beta": data.get("Beta"),
        }

    def get_daily_picks(self, tickers: list[str], top_n: int = 10) -> pd.DataFrame:
        """
        Scan a universe of tickers and return top fundamental picks.

        Combines fundamental scoring with analyst targets to find
        the most attractive stocks for the day.
        """
        fundamentals = self.fetch_batch(tickers)
        if fundamentals.empty:
            return pd.DataFrame()

        scored = self.compute_scores(fundamentals)

        # Rank by composite score
        scored = scored.sort_values("Fundamental_Score", ascending=False)

        # Select columns for display
        display_cols = [
            "Ticker", "Name", "Sector", "Current_Price",
            "PE_Ratio", "PB_Ratio", "Dividend_Yield", "ROE",
            "Revenue_Growth", "Earnings_Growth",
            "Value_Score", "Quality_Score", "Growth_Score",
            "Fundamental_Score", "Analyst_Upside",
        ]
        available_cols = [c for c in display_cols if c in scored.columns]
        return scored[available_cols].head(top_n)
