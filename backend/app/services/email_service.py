from __future__ import annotations

import asyncio
import logging
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _smtp_ready(settings) -> bool:
    return bool(
        settings.TRADE_EMAIL_ENABLED
        and settings.SMTP_HOST
        and settings.SMTP_PORT > 0
        and settings.SMTP_FROM
        and settings.TRADE_EMAIL_TO
    )


def _send_sync(subject: str, body: str, recipient: str | None = None) -> tuple[bool, str]:
    settings = get_settings()
    to_addr = (recipient or settings.TRADE_EMAIL_TO or "").strip()
    if not to_addr:
        return False, "No recipient configured"
    if not _smtp_ready(settings):
        return False, "SMTP is not configured"

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = settings.SMTP_FROM
        msg["To"] = to_addr
        msg.set_content(body)

        if settings.SMTP_USE_SSL:
            with smtplib.SMTP_SSL(
                settings.SMTP_HOST,
                settings.SMTP_PORT,
                timeout=30,
            ) as smtp:
                if settings.SMTP_USER and settings.SMTP_PASSWORD:
                    smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(
                settings.SMTP_HOST,
                settings.SMTP_PORT,
                timeout=30,
            ) as smtp:
                if settings.SMTP_STARTTLS:
                    smtp.starttls()
                if settings.SMTP_USER and settings.SMTP_PASSWORD:
                    smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                smtp.send_message(msg)
        return True, "sent"
    except Exception as exc:
        return False, str(exc)


async def send_trade_email(
    *,
    action: str,
    ticker: str,
    quantity: float,
    price: float,
    total_amount: float,
    total_cost: float,
    net_amount: float,
    user_email: str,
    source: str,
    executed_at: datetime | None = None,
) -> None:
    """
    Best-effort email notification for executed trades.
    Designed to never raise into API paths.
    """
    settings = get_settings()
    if not settings.TRADE_EMAIL_ENABLED:
        return

    ts = executed_at or datetime.now(timezone.utc)
    ts_iso = ts.astimezone(timezone.utc).isoformat()
    action_norm = str(action or "").strip().upper()
    if action_norm not in {"BUY", "SELL"}:
        return

    subject = (
        f"[QuantDesk] {action_norm} {str(ticker).upper()} "
        f"qty={float(quantity):g} @ ₹{float(price):.2f}"
    )
    body = (
        "Trade executed.\n\n"
        f"Timestamp (UTC): {ts_iso}\n"
        f"User: {user_email}\n"
        f"Source: {source}\n"
        f"Action: {action_norm}\n"
        f"Ticker: {str(ticker).upper()}\n"
        f"Quantity: {float(quantity):g}\n"
        f"Price: ₹{float(price):.2f}\n"
        f"Notional: ₹{float(total_amount):.2f}\n"
        f"Transaction Cost: ₹{float(total_cost):.2f}\n"
        f"Net Amount: ₹{float(net_amount):.2f}\n"
    )

    ok, msg = await asyncio.to_thread(_send_sync, subject, body)
    if not ok:
        logger.warning("Trade email notification failed: %s", msg)

