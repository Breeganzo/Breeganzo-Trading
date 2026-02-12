# ============================================================================
# QuantDesk Pro — Portfolio Seed Script
# Seeds the database with initial holdings
# Run: python -m scripts.seed_portfolio
# ============================================================================

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import init_db, get_db, close_db
from app.models.models import User, Portfolio, Trade
from app.engines.transaction_cost import TransactionCostEngine
from sqlalchemy import select
from datetime import datetime, timezone


# ==========================================================================
# INITIAL PORTFOLIO — Edit this to match your actual holdings
# ==========================================================================

INITIAL_HOLDINGS = [
    {
        "ticker": "KTKBANK.NS",
        "exchange": "NSE",
        "quantity": 196,
        "avg_buy_price": 211.48,
        "sector": "Banking & Finance",
    },
]

ALLOWED_EMAIL = "anthonybreeganzo02@gmail.com"


async def seed():
    """Seed the database with initial portfolio holdings."""
    await init_db()

    cost_engine = TransactionCostEngine()

    async for db in get_db():
        # Find or create the user
        result = await db.execute(
            select(User).where(User.email == ALLOWED_EMAIL)
        )
        user = result.scalar_one_or_none()

        if not user:
            from uuid import uuid4
            user = User(
                id=uuid4(),
                email=ALLOWED_EMAIL,
                name="Anthony Breeganzo",
                is_active=True,
            )
            db.add(user)
            await db.flush()
            print(f"✅ Created user: {user.email} (ID: {user.id})")
        else:
            print(f"✅ Found existing user: {user.email} (ID: {user.id})")

        for holding in INITIAL_HOLDINGS:
            # Check if position already exists
            existing = await db.execute(
                select(Portfolio).where(
                    Portfolio.user_id == user.id,
                    Portfolio.ticker == holding["ticker"],
                )
            )
            if existing.scalar_one_or_none():
                print(f"⏭️  Position already exists: {holding['ticker']}")
                continue

            # Calculate transaction costs for the original purchase
            costs = cost_engine.calculate_costs(
                price=holding["avg_buy_price"],
                quantity=holding["quantity"],
                trade_type="BUY",
            )

            total_invested = holding["avg_buy_price"] * holding["quantity"]

            # Create portfolio entry
            position = Portfolio(
                user_id=user.id,
                ticker=holding["ticker"],
                exchange=holding["exchange"],
                quantity=holding["quantity"],
                avg_buy_price=holding["avg_buy_price"],
                total_invested=round(total_invested, 2),
                realized_pnl=0.0,
                total_buy_costs=costs["total_cost"],
                total_sell_costs=0.0,
                sector=holding["sector"],
            )
            db.add(position)

            # Create a matching trade record
            trade = Trade(
                user_id=user.id,
                ticker=holding["ticker"],
                exchange=holding["exchange"],
                trade_type="BUY",
                quantity=holding["quantity"],
                price=holding["avg_buy_price"],
                total_amount=round(total_invested, 2),
                brokerage=costs["brokerage"],
                stt=costs["stt"],
                exchange_charges=costs["exchange_charges"],
                gst=costs["gst"],
                sebi_charges=costs["sebi_charges"],
                stamp_duty=costs["stamp_duty"],
                slippage_cost=costs["slippage"],
                total_cost=costs["total_cost"],
                net_amount=costs["net_amount"],
                notes="Initial portfolio seed",
                executed_at=datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            )
            db.add(trade)

            print(
                f"✅ Seeded: {holding['ticker']} — "
                f"{holding['quantity']} shares @ ₹{holding['avg_buy_price']} "
                f"(costs: ₹{costs['total_cost']:.2f})"
            )

        await db.commit()
        print("\n✅ Portfolio seeding complete!")

    await close_db()


if __name__ == "__main__":
    asyncio.run(seed())
