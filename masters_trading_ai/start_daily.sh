#!/bin/bash
# =============================================================================
# DAILY TRADING AI STARTER
# Run this every morning for fresh predictions
# =============================================================================

echo "🚀 Starting Daily Trading AI..."
echo "================================="

# Go to project folder
cd /Users/anto/Trading_Project/masters_trading_ai

# Run the daily script
echo "📊 Generating fresh predictions..."
./run_daily.sh

echo ""
echo "✅ Done! Open your browser to: http://localhost:5001"
echo "📈 Check today's top buy/sell opportunities!"
echo ""
echo "Press Ctrl+C to stop the server when done."