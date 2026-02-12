"""
Walk-Forward Cross-Validation with Purging & Embargo
=====================================================
Implements temporal cross-validation that respects the time-series nature
of financial data, preventing look-ahead bias.

Key Features
------------
1. **Temporal ordering**: Train set ALWAYS precedes test set.
2. **Purging**: Removes training samples whose labels overlap with test period.
3. **Embargo**: Adds a gap between train end and test start to prevent
   serial correlation leakage.
4. **Sliding window**: Fixed training window to adapt to regime changes.
5. **Expanding window**: Growing training window option for comparison.

Reference: López de Prado, "Advances in Financial Machine Learning" (2018)
"""

import numpy as np
import pandas as pd
from typing import Optional


class WalkForwardCV:
    """
    Walk-Forward Cross-Validation splitter.

    This is the most important class for preventing look-ahead bias.
    It ensures that at every fold:
    - Training data is strictly BEFORE the test data
    - A gap (embargo) exists between train and test
    - Training samples with overlapping labels near the boundary are purged

    Parameters
    ----------
    n_folds : int
        Number of walk-forward folds (default: 8)
    train_window : int
        Number of days in training window (default: 504 ≈ 2 years)
    test_window : int
        Number of days in test window (default: 63 ≈ 1 quarter)
    embargo_days : int
        Gap days between train end and test start (default: 5)
    purge_days : int
        Number of days to purge from train end (default: 5)
    expanding : bool
        If True, use expanding window (train grows from start).
        If False, use sliding window (fixed train size). Default: False.

    Example
    -------
    >>> wfcv = WalkForwardCV(n_folds=8, train_window=504, test_window=63)
    >>> for fold, (train_idx, test_idx) in enumerate(wfcv.split(X)):
    ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    ...     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    ...     # Train model on X_train, evaluate on X_test
    """

    def __init__(
        self,
        n_folds: int = 18,
        train_window: int = 504,
        test_window: int = 63,
        embargo_days: int = 5,
        purge_days: int = 5,
        expanding: bool = True,
    ):
        self.n_folds = n_folds
        self.train_window = train_window
        self.test_window = test_window
        self.embargo_days = embargo_days
        self.purge_days = purge_days
        self.expanding = expanding

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test index pairs for walk-forward CV.

        Timeline visualisation (sliding window):
        ```
        |====TRAIN====|--embargo--|==TEST==|
              |====TRAIN====|--embargo--|==TEST==|
                    |====TRAIN====|--embargo--|==TEST==|
        ```

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with DatetimeIndex
        y : pd.Series, optional
            Not used, but kept for sklearn API compatibility

        Returns
        -------
        list of (train_indices, test_indices) tuples
            Indices are positional (integer-based), not label-based.
        """
        n_samples = len(X)
        splits = []

        # Calculate total space needed
        total_needed = self.train_window + self.embargo_days + self.test_window
        if n_samples < total_needed:
            raise ValueError(
                f"Not enough data ({n_samples} rows) for even 1 fold. "
                f"Need at least {total_needed} rows "
                f"(train={self.train_window} + embargo={self.embargo_days} + test={self.test_window})"
            )

        # Calculate the starting position for the last fold's test end
        # Work backwards from the end to place n_folds
        step = self.test_window  # Each fold steps forward by test_window days

        # First fold starts here
        if self.expanding:
            # Expanding: train always starts at 0
            first_test_start = self.train_window + self.embargo_days
        else:
            # Sliding: calculate start so last fold ends near data end
            last_test_end = n_samples
            first_test_end = last_test_end - (self.n_folds - 1) * step
            first_test_start = first_test_end - self.test_window

            # Ensure first fold has enough training data
            if first_test_start - self.embargo_days < self.train_window:
                # Adjust: reduce number of folds
                available_folds = (n_samples - self.train_window - self.embargo_days - self.test_window) // step + 1
                if available_folds < 2:
                    raise ValueError(
                        f"Not enough data for {self.n_folds} folds. "
                        f"Max possible: {available_folds}"
                    )
                print(f"  ⚠ Reduced to {available_folds} folds (data too short for {self.n_folds})")
                self.n_folds = available_folds
                first_test_end = last_test_end - (self.n_folds - 1) * step
                first_test_start = first_test_end - self.test_window

        for fold in range(self.n_folds):
            # Test window
            test_start = first_test_start + fold * step
            test_end = min(test_start + self.test_window, n_samples)

            if test_end > n_samples:
                break

            # Train window (with purging and embargo)
            if self.expanding:
                train_start = 0
            else:
                train_start = test_start - self.embargo_days - self.train_window

            train_end = test_start - self.embargo_days

            # Apply purging: remove training samples near boundary
            train_end_purged = train_end - self.purge_days

            if train_start < 0:
                train_start = 0
            if train_end_purged <= train_start:
                continue

            train_idx = np.arange(train_start, train_end_purged)
            test_idx = np.arange(test_start, test_end)

            splits.append((train_idx, test_idx))

        if not splits:
            raise ValueError("No valid splits generated. Check data length and parameters.")

        return splits

    def get_fold_dates(
        self, X: pd.DataFrame
    ) -> list[dict]:
        """
        Return human-readable date ranges for each fold.

        Returns
        -------
        list of dict
            Each dict: {fold, train_start, train_end, test_start, test_end,
                         train_days, test_days, embargo_days}
        """
        splits = self.split(X)
        fold_info = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            fold_info.append({
                "Fold": fold + 1,
                "Train Start": X.index[train_idx[0]].strftime("%Y-%m-%d"),
                "Train End": X.index[train_idx[-1]].strftime("%Y-%m-%d"),
                "Test Start": X.index[test_idx[0]].strftime("%Y-%m-%d"),
                "Test End": X.index[test_idx[-1]].strftime("%Y-%m-%d"),
                "Train Days": len(train_idx),
                "Test Days": len(test_idx),
                "Embargo": self.embargo_days,
            })

        return fold_info

    def summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a summary DataFrame of all folds."""
        return pd.DataFrame(self.get_fold_dates(X))

    def __repr__(self) -> str:
        mode = "expanding" if self.expanding else "sliding"
        return (
            f"WalkForwardCV(n_folds={self.n_folds}, "
            f"train={self.train_window}d, test={self.test_window}d, "
            f"embargo={self.embargo_days}d, purge={self.purge_days}d, "
            f"mode={mode})"
        )
