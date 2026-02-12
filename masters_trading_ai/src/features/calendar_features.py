"""
Calendar Feature Engineering
=============================
Time-based features that capture seasonality, expiry effects,
and day-of-week patterns in Indian equity markets.

All features are deterministic based on the date — no look-ahead bias.
"""

import pandas as pd
import numpy as np
from datetime import datetime


class CalendarFeatures:
    """
    Add calendar/time-based features to a stock DataFrame.

    These capture well-documented effects in Indian markets:
    - Monday effect (historically negative)
    - Expiry week volatility (monthly F&O settlement)
    - Month-end rebalancing
    - Budget-day effects (typically Feb 1)
    - Quarter-end effects
    """

    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar features to DataFrame.

        Features Added
        --------------
        - DayOfWeek: 0=Monday ... 4=Friday (one-hot encoded)
        - Month: 1-12 (sine/cosine encoded for cyclicality)
        - WeekOfYear: 1-52 (sine/cosine encoded)
        - IsMonthEnd: last 3 trading days of month
        - IsMonthStart: first 3 trading days of month
        - IsQuarterEnd: last 5 trading days of quarter
        - DaysToExpiry: approximate days to monthly F&O expiry (last Thursday)
        - IsExpiryWeek: 1 if within expiry week
        - IsBudgetWeek: 1 if within first week of February

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with DatetimeIndex

        Returns
        -------
        pd.DataFrame
            With calendar features added
        """
        df = df.copy()
        dates = df.index

        # --- Day of Week (one-hot) ---
        dow = dates.dayofweek  # 0=Monday, 4=Friday
        df["DayOfWeek_Mon"] = (dow == 0).astype(int)
        df["DayOfWeek_Tue"] = (dow == 1).astype(int)
        df["DayOfWeek_Wed"] = (dow == 2).astype(int)
        df["DayOfWeek_Thu"] = (dow == 3).astype(int)
        df["DayOfWeek_Fri"] = (dow == 4).astype(int)

        # --- Month (sine/cosine encoding for cyclicality) ---
        month = dates.month
        df["Month_sin"] = np.sin(2 * np.pi * month / 12)
        df["Month_cos"] = np.cos(2 * np.pi * month / 12)

        # --- Week of Year (sine/cosine encoding) ---
        # Use isocalendar for correct week numbers
        week = dates.isocalendar().week.values.astype(float)
        df["WeekOfYear_sin"] = np.sin(2 * np.pi * week / 52)
        df["WeekOfYear_cos"] = np.cos(2 * np.pi * week / 52)

        # --- Month boundaries ---
        df["IsMonthEnd"] = CalendarFeatures._is_near_month_boundary(dates, end=True, days=3)
        df["IsMonthStart"] = CalendarFeatures._is_near_month_boundary(dates, end=False, days=3)

        # --- Quarter end ---
        df["IsQuarterEnd"] = CalendarFeatures._is_quarter_end(dates, days=5)

        # --- F&O Expiry (approximate: last Thursday of month) ---
        df["DaysToExpiry"] = CalendarFeatures._days_to_expiry(dates)
        df["IsExpiryWeek"] = (df["DaysToExpiry"] <= 5).astype(int)

        # --- Budget week (typically Feb 1) ---
        df["IsBudgetWeek"] = ((month == 2) & (dates.day <= 7)).astype(int)

        # --- Year-end effect (last 5 days of December) ---
        df["IsYearEnd"] = ((month == 12) & (dates.day >= 25)).astype(int)

        return df

    @staticmethod
    def _is_near_month_boundary(dates: pd.DatetimeIndex, end: bool = True, days: int = 3) -> pd.Series:
        """Check if dates are near month start or end."""
        result = pd.Series(0, index=dates)
        if end:
            # Group by year-month and find last N trading days
            for name, group in pd.Series(range(len(dates)), index=dates).groupby(
                [dates.year, dates.month]
            ):
                if len(group) >= days:
                    result.iloc[group.iloc[-days:].values] = 1
        else:
            for name, group in pd.Series(range(len(dates)), index=dates).groupby(
                [dates.year, dates.month]
            ):
                if len(group) >= days:
                    result.iloc[group.iloc[:days].values] = 1
        return result

    @staticmethod
    def _is_quarter_end(dates: pd.DatetimeIndex, days: int = 5) -> pd.Series:
        """Check if dates are near quarter end (March, June, Sept, Dec)."""
        result = pd.Series(0, index=dates)
        quarter_months = {3, 6, 9, 12}
        for name, group in pd.Series(range(len(dates)), index=dates).groupby(
            [dates.year, dates.month]
        ):
            year, month = name
            if month in quarter_months and len(group) >= days:
                result.iloc[group.iloc[-days:].values] = 1
        return result

    @staticmethod
    def _days_to_expiry(dates: pd.DatetimeIndex) -> pd.Series:
        """
        Approximate days to next monthly F&O expiry.
        Indian F&O monthly expiry = last Thursday of the month.
        """
        result = []
        for dt in dates:
            # Find last Thursday of current month
            last_day = pd.Timestamp(dt.year, dt.month, 1) + pd.offsets.MonthEnd(0)
            # Go back to last Thursday
            offset = (last_day.weekday() - 3) % 7
            expiry = last_day - pd.Timedelta(days=offset)

            if dt.date() > expiry.date():
                # Past this month's expiry — find next month's
                next_month = dt + pd.offsets.MonthEnd(1)
                last_day_next = next_month
                offset_next = (last_day_next.weekday() - 3) % 7
                expiry = last_day_next - pd.Timedelta(days=offset_next)

            days_left = (expiry.date() - dt.date()).days
            result.append(max(days_left, 0))

        return pd.Series(result, index=dates)
