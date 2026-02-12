from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from groq import Groq

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ── System Prompt ────────────────────────────────────────────────────────
# Hard-coded guardrail: Groq may ONLY explain risk metrics and suggest
# improvements.  It must NEVER generate trade signals or buy/sell
# recommendations.

SYSTEM_PROMPT = (
    "You are a risk analytics assistant for an institutional trading platform. "
    "You explain what financial metrics mean and suggest risk improvements. "
    "You MUST NOT suggest specific trades, buy/sell signals, or recommend any "
    "specific stocks. Focus only on risk management education and portfolio "
    "risk assessment."
)

# ── Metric Definitions (fallback when Groq is unavailable) ───────────────

METRIC_DEFINITIONS: dict[str, dict[str, str]] = {
    "sharpe_ratio": {
        "name": "Sharpe Ratio",
        "description": (
            "The Sharpe Ratio measures risk-adjusted return by comparing "
            "excess return (over the risk-free rate) to the portfolio's "
            "standard deviation. A ratio above 1.0 is generally considered "
            "acceptable; above 2.0 is very good."
        ),
        "suggestion": (
            "To improve the Sharpe Ratio, consider reducing portfolio "
            "volatility through diversification or rebalancing toward "
            "lower-volatility asset classes."
        ),
    },
    "sortino_ratio": {
        "name": "Sortino Ratio",
        "description": (
            "The Sortino Ratio is similar to the Sharpe Ratio but only "
            "penalises downside volatility, making it a better measure "
            "when return distributions are asymmetric. Higher values "
            "indicate better risk-adjusted performance."
        ),
        "suggestion": (
            "Focus on strategies that reduce downside risk, such as "
            "protective hedging or stop-loss discipline, to improve the "
            "Sortino Ratio."
        ),
    },
    "beta": {
        "name": "Beta",
        "description": (
            "Beta measures a portfolio's sensitivity to market movements. "
            "A beta of 1.0 moves in line with the market; below 1.0 is "
            "defensive; above 1.0 is aggressive."
        ),
        "suggestion": (
            "If beta is higher than your risk tolerance, consider adding "
            "low-correlation or defensive positions to reduce overall "
            "market sensitivity."
        ),
    },
    "max_drawdown": {
        "name": "Maximum Drawdown",
        "description": (
            "Maximum Drawdown is the largest peak-to-trough decline in "
            "portfolio value over a given period. It represents the "
            "worst-case loss scenario an investor would have experienced."
        ),
        "suggestion": (
            "Implement drawdown limits and consider position-sizing rules "
            "to cap exposure. Diversification across uncorrelated assets "
            "can also reduce maximum drawdown."
        ),
    },
    "var_95": {
        "name": "Value at Risk (95%)",
        "description": (
            "VaR at the 95th percentile estimates the maximum expected "
            "loss over a given time horizon with 95%% confidence. "
            "For example, a daily VaR of 2%% means there is only a 5%% "
            "chance the portfolio loses more than 2%% in a single day."
        ),
        "suggestion": (
            "Reduce VaR by lowering position concentrations or adding "
            "hedging instruments. Regularly stress-test the portfolio "
            "against extreme scenarios."
        ),
    },
    "volatility": {
        "name": "Volatility",
        "description": (
            "Volatility measures the standard deviation of portfolio "
            "returns, quantifying the degree of variation in value over "
            "time. Higher volatility means greater uncertainty."
        ),
        "suggestion": (
            "Reduce volatility through broader diversification, "
            "rebalancing schedules, or allocation to less volatile "
            "instruments."
        ),
    },
    "correlation": {
        "name": "Correlation",
        "description": (
            "Correlation measures how closely two assets or a portfolio "
            "and a benchmark move together, ranging from -1 (perfectly "
            "inverse) to +1 (perfectly aligned). Low or negative "
            "correlations improve diversification benefits."
        ),
        "suggestion": (
            "Seek assets with low or negative correlation to your "
            "existing holdings to improve diversification and reduce "
            "overall portfolio risk."
        ),
    },
}


class AIService:
    """Groq AI integration for risk metric explanation and portfolio
    risk assessment.

    STRICT CONSTRAINT: Groq is used ONLY to explain risk metrics and
    suggest risk-management improvements.  It must NEVER generate trade
    signals or buy/sell recommendations.
    """

    # Rate-limit tracking for Groq free tier
    _last_request_ts: float = 0.0
    _min_request_interval: float = 1.0  # seconds between requests

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: Groq | None = None

        if self._settings.GROQ_API_KEY:
            try:
                self._client = Groq(api_key=self._settings.GROQ_API_KEY)
                logger.info("Groq AI client initialised successfully")
            except Exception as exc:
                logger.error("Failed to initialise Groq client: %s", exc)
                self._client = None
        else:
            logger.warning(
                "GROQ_API_KEY not configured; AI explanations will use "
                "fallback definitions"
            )

    # ── Private Helpers ──────────────────────────────────────────────

    def _wait_for_rate_limit(self) -> None:
        """Simple rate-limit guard for the Groq free tier.

        Ensures at least ``_min_request_interval`` seconds pass between
        consecutive API calls.
        """
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        AIService._last_request_ts = time.monotonic()

    def _call_groq(self, user_prompt: str) -> str:
        """Synchronous, rate-limited call to the Groq chat completions
        endpoint.  Meant to be invoked via ``asyncio.to_thread``.
        """
        if self._client is None:
            raise RuntimeError("Groq client is not available")

        self._wait_for_rate_limit()

        response = self._client.chat.completions.create(
            model=self._settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        return response.choices[0].message.content

    def _fallback_metric_explanation(
        self, metric: str, value: Any
    ) -> dict[str, Any]:
        """Return a pre-written explanation when Groq is unavailable."""
        defn = METRIC_DEFINITIONS.get(metric)
        if defn is None:
            return {
                "metric": metric,
                "value": value,
                "explanation": (
                    f"'{metric}' is a financial risk metric. "
                    "Detailed AI explanation is currently unavailable."
                ),
                "suggestions": [
                    "Consult your risk management documentation for "
                    "details on this metric."
                ],
                "source": "fallback",
            }

        return {
            "metric": metric,
            "value": value,
            "explanation": defn["description"],
            "suggestions": [defn["suggestion"]],
            "source": "fallback",
        }

    def _fallback_portfolio_risk(
        self,
        risk_metrics: dict[str, Any],
        holdings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a basic portfolio risk assessment when Groq is
        unavailable."""
        explanations: list[dict[str, str]] = []
        for metric, value in risk_metrics.items():
            defn = METRIC_DEFINITIONS.get(metric)
            if defn:
                explanations.append(
                    {"metric": defn["name"], "explanation": defn["description"]}
                )

        return {
            "explanation": (
                "Portfolio risk assessment is based on the provided "
                "metrics. Detailed AI analysis is currently unavailable."
            ),
            "risk_assessment": {
                "metrics_analyzed": list(risk_metrics.keys()),
                "holdings_count": len(holdings),
                "metric_explanations": explanations,
            },
            "suggestions": [
                "Ensure adequate diversification across sectors and "
                "asset classes.",
                "Review position sizing relative to overall portfolio "
                "value.",
                "Monitor correlation between holdings to avoid "
                "concentration risk.",
            ],
            "source": "fallback",
        }

    # ── Public API ───────────────────────────────────────────────────

    async def explain_metric(
        self,
        metric: str,
        value: float | dict,
        portfolio_context: dict | None = None,
    ) -> dict[str, Any]:
        """Ask Groq to explain a single risk metric and suggest
        improvements.

        Falls back to pre-written definitions if the Groq API is
        unavailable or errors out.

        Parameters
        ----------
        metric:
            One of the supported risk metric keys (e.g. ``sharpe_ratio``).
        value:
            The current value of the metric (scalar or dict).
        portfolio_context:
            Optional extra context (e.g. sector weights) to tailor the
            explanation.
        """
        if self._client is None:
            return self._fallback_metric_explanation(metric, value)

        metric_name = METRIC_DEFINITIONS.get(metric, {}).get("name", metric)

        context_block = ""
        if portfolio_context:
            context_block = (
                f"\n\nAdditional portfolio context:\n"
                f"{_format_dict(portfolio_context)}"
            )

        user_prompt = (
            f"Explain the following risk metric for a portfolio:\n\n"
            f"Metric: {metric_name}\n"
            f"Current value: {value}\n"
            f"{context_block}\n\n"
            f"Provide:\n"
            f"1. A clear, concise explanation of what this metric means "
            f"in plain language.\n"
            f"2. Whether this value is healthy, concerning, or critical.\n"
            f"3. Two or three actionable suggestions to improve this "
            f"metric through risk management (do NOT recommend specific "
            f"stocks or trades)."
        )

        try:
            raw_response = await asyncio.to_thread(self._call_groq, user_prompt)

            return {
                "metric": metric,
                "value": value,
                "explanation": raw_response,
                "suggestions": _extract_suggestions(raw_response),
                "source": "groq",
            }
        except Exception as exc:
            logger.warning(
                "Groq API call failed for metric '%s': %s", metric, exc
            )
            return self._fallback_metric_explanation(metric, value)

    async def explain_portfolio_risk(
        self,
        risk_metrics: dict[str, Any],
        holdings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Ask Groq for a holistic portfolio risk assessment.

        Falls back to pre-written analysis if the Groq API is
        unavailable or errors out.

        Parameters
        ----------
        risk_metrics:
            Dictionary of metric name -> value (e.g.
            ``{"sharpe_ratio": 1.2, "beta": 0.85}``).
        holdings:
            List of holding dicts with at least ``symbol`` and
            ``weight`` keys.
        """
        if self._client is None:
            return self._fallback_portfolio_risk(risk_metrics, holdings)

        holdings_summary = "\n".join(
            f"  - {h.get('ticker', 'N/A')}: "
            f"weight {h.get('weight', 'N/A')}"
            for h in holdings[:20]  # cap to avoid token overflow
        )
        if len(holdings) > 20:
            holdings_summary += f"\n  ... and {len(holdings) - 20} more"

        metrics_summary = "\n".join(
            f"  - {METRIC_DEFINITIONS.get(k, {}).get('name', k)}: {v}"
            for k, v in risk_metrics.items()
        )

        user_prompt = (
            f"Analyze the following portfolio risk profile:\n\n"
            f"Risk Metrics:\n{metrics_summary}\n\n"
            f"Holdings ({len(holdings)} positions):\n{holdings_summary}\n\n"
            f"Provide:\n"
            f"1. A plain-language summary of the portfolio's overall "
            f"risk posture.\n"
            f"2. Key risk concerns based on the metrics.\n"
            f"3. Three to five actionable risk-management suggestions "
            f"(do NOT recommend buying or selling specific stocks)."
        )

        try:
            raw_response = await asyncio.to_thread(self._call_groq, user_prompt)

            return {
                "explanation": raw_response,
                "risk_assessment": {
                    "metrics_analyzed": list(risk_metrics.keys()),
                    "holdings_count": len(holdings),
                },
                "suggestions": _extract_suggestions(raw_response),
                "source": "groq",
            }
        except Exception as exc:
            logger.warning(
                "Groq API call failed for portfolio risk assessment: %s", exc
            )
            return self._fallback_portfolio_risk(risk_metrics, holdings)


# ── Module-Level Helpers ─────────────────────────────────────────────────


def _format_dict(d: dict, indent: int = 2) -> str:
    """Format a dictionary into a readable multi-line string for
    inclusion in prompts."""
    lines: list[str] = []
    prefix = " " * indent
    for key, val in d.items():
        lines.append(f"{prefix}{key}: {val}")
    return "\n".join(lines)


def _extract_suggestions(text: str) -> list[str]:
    """Attempt to pull numbered suggestion lines from the Groq response.

    Falls back to returning the full text as a single suggestion if
    parsing does not find clearly numbered items.
    """
    suggestions: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # Match lines starting with a digit followed by . or )
        if stripped and len(stripped) > 3 and stripped[0].isdigit() and stripped[1] in ".)" :
            suggestions.append(stripped[2:].strip().lstrip(" "))
        elif stripped and len(stripped) > 4 and stripped[:2].rstrip(".)" ).isdigit():
            idx = stripped.index(".") if "." in stripped[:3] else stripped.index(")")
            suggestions.append(stripped[idx + 1 :].strip())
    return suggestions if suggestions else [text.strip()]


# ── Module-Level Singleton ───────────────────────────────────────────────

ai_service = AIService()
