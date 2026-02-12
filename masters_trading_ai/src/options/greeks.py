"""
Black-Scholes Greeks Calculator
================================
Computes option prices and Greeks using the Black-Scholes model.

This is for educational / signal generation purposes —
real options pricing would need implied vol surfaces,
dividend adjustments, and American option handling.

Black-Scholes Formula
---------------------
Call price:
    C = S × N(d₁) − K × e^{-rT} × N(d₂)

Put price:
    P = K × e^{-rT} × N(−d₂) − S × N(−d₁)

Where:
    d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d₂ = d₁ − σ√T
"""

import numpy as np
from scipy.stats import norm


class BlackScholesGreeks:
    """
    Black-Scholes option pricing and Greeks.

    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate (default 6.5% for India)
    """

    def __init__(self, risk_free_rate: float = 0.065):
        self.r = risk_free_rate

    def _d1_d2(
        self, S: float, K: float, T: float, sigma: float
    ) -> tuple:
        """Calculate d1 and d2."""
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        European call option price.

        Parameters
        ----------
        S : spot price
        K : strike price
        T : time to expiry in years
        sigma : annualised volatility
        """
        d1, d2 = self._d1_d2(S, K, T, sigma)
        return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)

    def put_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """European put option price."""
        d1, d2 = self._d1_d2(S, K, T, sigma)
        return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def delta(self, S: float, K: float, T: float, sigma: float, option_type: str = "call") -> float:
        """
        Delta: ∂C/∂S — rate of change of option price w.r.t. spot.

        Call delta ∈ [0, 1], Put delta ∈ [-1, 0]
        """
        d1, _ = self._d1_d2(S, K, T, sigma)
        if option_type == "call":
            return norm.cdf(d1)
        return norm.cdf(d1) - 1

    def gamma(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Gamma: ∂²C/∂S² — rate of change of delta w.r.t. spot.

        Same for calls and puts.
        """
        d1, _ = self._d1_d2(S, K, T, sigma)
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def theta(self, S: float, K: float, T: float, sigma: float, option_type: str = "call") -> float:
        """
        Theta: ∂C/∂T — time decay (per calendar day).

        Negative for long options (time decay hurts buyers).
        """
        d1, d2 = self._d1_d2(S, K, T, sigma)
        if T <= 0 or sigma <= 0:
            return 0.0
        first = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type == "call":
            second = -self.r * K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            second = self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)
        return (first + second) / 365  # Per calendar day

    def vega(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Vega: ∂C/∂σ — sensitivity to volatility (per 1% move).

        Same for calls and puts.
        """
        d1, _ = self._d1_d2(S, K, T, sigma)
        if T <= 0:
            return 0.0
        return S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change

    def rho(self, S: float, K: float, T: float, sigma: float, option_type: str = "call") -> float:
        """
        Rho: ∂C/∂r — sensitivity to risk-free rate (per 1% move).
        """
        _, d2 = self._d1_d2(S, K, T, sigma)
        if option_type == "call":
            return K * T * np.exp(-self.r * T) * norm.cdf(d2) / 100
        return -K * T * np.exp(-self.r * T) * norm.cdf(-d2) / 100

    def all_greeks(
        self, S: float, K: float, T: float, sigma: float, option_type: str = "call"
    ) -> dict:
        """
        Compute all Greeks at once.

        Returns
        -------
        dict
            price, delta, gamma, theta, vega, rho
        """
        price = self.call_price(S, K, T, sigma) if option_type == "call" else self.put_price(S, K, T, sigma)
        return {
            "price": round(price, 2),
            "delta": round(self.delta(S, K, T, sigma, option_type), 4),
            "gamma": round(self.gamma(S, K, T, sigma), 6),
            "theta": round(self.theta(S, K, T, sigma, option_type), 4),
            "vega": round(self.vega(S, K, T, sigma), 4),
            "rho": round(self.rho(S, K, T, sigma, option_type), 4),
        }

    def implied_volatility(
        self, market_price: float, S: float, K: float, T: float,
        option_type: str = "call", tol: float = 1e-5, max_iter: int = 100,
    ) -> float:
        """
        Implied volatility via Newton-Raphson.

        Finds σ such that BS_price(σ) = market_price.
        """
        sigma = 0.3  # Initial guess
        for _ in range(max_iter):
            if option_type == "call":
                price = self.call_price(S, K, T, sigma)
            else:
                price = self.put_price(S, K, T, sigma)
            diff = price - market_price
            if abs(diff) < tol:
                return sigma
            v = self.vega(S, K, T, sigma) * 100  # Undo the /100
            if v < 1e-12:
                break
            sigma -= diff / v
            sigma = max(0.01, min(sigma, 5.0))
        return sigma
