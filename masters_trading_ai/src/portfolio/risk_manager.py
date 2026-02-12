"""
Risk Manager
=============
Pre-trade and portfolio-level risk checks.

Enforces:
- Maximum drawdown limits (3% daily, 15% total)
- Position concentration limits (10% per ticker)
- Sector concentration limits (30% per sector)
- Maximum number of concurrent positions (15)
- Minimum model confidence (0.55)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from ..utils.constants import CONFIG_DIR


@dataclass
class RiskCheck:
    """Result of a risk check."""
    passed: bool
    reason: str
    level: str = "INFO"  # INFO, WARNING, BLOCKED


@dataclass
class PortfolioState:
    """Current state of the portfolio for risk tracking."""
    capital: float = 100_000.0
    peak_capital: float = 100_000.0
    daily_start: float = 100_000.0
    positions: Dict[str, float] = field(default_factory=dict)  # ticker → invested ₹
    sector_map: Dict[str, str] = field(default_factory=dict)   # ticker → sector


class RiskManager:
    """
    Centralised risk manager.

    Checks every proposed trade against portfolio-level risk limits
    before allowing execution. Returns RiskCheck objects describing
    pass/fail and reasons.

    Parameters
    ----------
    config_path : str, optional
        Path to settings.yaml for risk parameters
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = CONFIG_DIR / "settings.yaml"

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        risk_cfg = cfg.get("risk_management", {})
        self.max_daily_drawdown = risk_cfg.get("max_daily_drawdown", 0.03)
        self.max_total_drawdown = risk_cfg.get("max_total_drawdown", 0.15)
        self.max_position_pct = risk_cfg.get("max_position_pct", 0.10)
        self.max_sector_pct = risk_cfg.get("max_sector_pct", 0.30)
        self.max_concurrent = risk_cfg.get("max_concurrent_positions", 15)
        self.min_confidence = risk_cfg.get("min_confidence", 0.55)

    # ------------------------------------------------------------------ #
    #  Individual checks                                                  #
    # ------------------------------------------------------------------ #
    def check_daily_drawdown(self, state: PortfolioState) -> RiskCheck:
        """Check if daily drawdown limit is breached."""
        dd = (state.daily_start - state.capital) / state.daily_start
        if dd >= self.max_daily_drawdown:
            return RiskCheck(
                passed=False,
                reason=f"Daily drawdown {dd:.2%} >= limit {self.max_daily_drawdown:.2%}. "
                       "No new trades today.",
                level="BLOCKED",
            )
        return RiskCheck(passed=True, reason=f"Daily DD {dd:.2%} OK", level="INFO")

    def check_total_drawdown(self, state: PortfolioState) -> RiskCheck:
        """Check if total drawdown from peak is breached."""
        dd = (state.peak_capital - state.capital) / state.peak_capital
        if dd >= self.max_total_drawdown:
            return RiskCheck(
                passed=False,
                reason=f"Total drawdown {dd:.2%} >= limit {self.max_total_drawdown:.2%}. "
                       "Portfolio frozen — reduce exposure.",
                level="BLOCKED",
            )
        if dd >= self.max_total_drawdown * 0.75:
            return RiskCheck(
                passed=True,
                reason=f"Total DD warning: {dd:.2%} approaching limit.",
                level="WARNING",
            )
        return RiskCheck(passed=True, reason=f"Total DD {dd:.2%} OK", level="INFO")

    def check_position_concentration(
        self, ticker: str, amount: float, state: PortfolioState
    ) -> RiskCheck:
        """Check if a single position exceeds max %."""
        current = state.positions.get(ticker, 0.0)
        total_after = current + amount
        pct = total_after / state.capital if state.capital > 0 else 1.0

        if pct > self.max_position_pct:
            return RiskCheck(
                passed=False,
                reason=f"{ticker} would be {pct:.1%} of portfolio "
                       f"(limit {self.max_position_pct:.1%}).",
                level="BLOCKED",
            )
        return RiskCheck(passed=True, reason=f"{ticker} concentration OK", level="INFO")

    def check_sector_concentration(
        self, ticker: str, amount: float, state: PortfolioState
    ) -> RiskCheck:
        """Check if sector exposure exceeds max %."""
        sector = state.sector_map.get(ticker, "Unknown")
        sector_exposure = sum(
            v for t, v in state.positions.items()
            if state.sector_map.get(t, "Unknown") == sector
        )
        after = sector_exposure + amount
        pct = after / state.capital if state.capital > 0 else 1.0

        if pct > self.max_sector_pct:
            return RiskCheck(
                passed=False,
                reason=f"Sector '{sector}' would be {pct:.1%} "
                       f"(limit {self.max_sector_pct:.1%}).",
                level="BLOCKED",
            )
        return RiskCheck(passed=True, reason=f"Sector {sector} OK", level="INFO")

    def check_concurrent_positions(self, state: PortfolioState) -> RiskCheck:
        """Check if max concurrent positions is reached."""
        n = len(state.positions)
        if n >= self.max_concurrent:
            return RiskCheck(
                passed=False,
                reason=f"Already at {n} positions (limit {self.max_concurrent}).",
                level="BLOCKED",
            )
        return RiskCheck(passed=True, reason=f"{n} positions OK", level="INFO")

    def check_confidence(self, confidence: float) -> RiskCheck:
        """Check if model confidence meets minimum threshold."""
        if confidence < self.min_confidence:
            return RiskCheck(
                passed=False,
                reason=f"Model confidence {confidence:.3f} < minimum "
                       f"{self.min_confidence:.3f}.",
                level="BLOCKED",
            )
        return RiskCheck(passed=True, reason=f"Confidence {confidence:.3f} OK", level="INFO")

    # ------------------------------------------------------------------ #
    #  Composite check                                                    #
    # ------------------------------------------------------------------ #
    def approve_trade(
        self,
        ticker: str,
        amount: float,
        confidence: float,
        state: PortfolioState,
    ) -> List[RiskCheck]:
        """
        Run all risk checks for a proposed trade.

        Returns list of RiskCheck objects. Trade should only proceed
        if all checks have passed=True.
        """
        checks = [
            self.check_daily_drawdown(state),
            self.check_total_drawdown(state),
            self.check_position_concentration(ticker, amount, state),
            self.check_sector_concentration(ticker, amount, state),
            self.check_concurrent_positions(state),
            self.check_confidence(confidence),
        ]
        return checks

    def is_trade_allowed(
        self,
        ticker: str,
        amount: float,
        confidence: float,
        state: PortfolioState,
    ) -> bool:
        """Convenience: True if all checks pass."""
        checks = self.approve_trade(ticker, amount, confidence, state)
        return all(c.passed for c in checks)

    def format_risk_report(self, state: PortfolioState) -> str:
        """
        Generate a formatted risk status report.

        Returns
        -------
        str
            Human-readable risk summary
        """
        daily_dd = (state.daily_start - state.capital) / state.daily_start
        total_dd = (state.peak_capital - state.capital) / state.peak_capital
        n_pos = len(state.positions)
        invested = sum(state.positions.values())
        cash_pct = 1.0 - (invested / state.capital) if state.capital > 0 else 0.0

        lines = [
            "╔══════════════ Risk Dashboard ══════════════╗",
            f"  Capital:   ₹{state.capital:>12,.0f}",
            f"  Peak:      ₹{state.peak_capital:>12,.0f}",
            f"  Daily DD:  {daily_dd:>7.2%}  (limit {self.max_daily_drawdown:.2%})",
            f"  Total DD:  {total_dd:>7.2%}  (limit {self.max_total_drawdown:.2%})",
            f"  Positions: {n_pos:>4d}       (limit {self.max_concurrent})",
            f"  Cash:      {cash_pct:>7.2%}",
            "╚════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)
