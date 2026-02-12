#!/bin/bash
# =============================================================================
# Masters AI Trading Bot — Daily Run Script
# =============================================================================
# Usage:
#   chmod +x run_daily.sh    (first time only)
#   ./run_daily.sh            (start the webapp)
#   ./run_daily.sh --retrain  (retrain models first, then start)
#
# The webapp runs at http://localhost:5001
# Press Ctrl+C to stop.
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Masters AI Trading Bot"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Python environment
if [ ! -f "$PYTHON" ]; then
    echo "❌ Virtual environment not found at $PROJECT_DIR/.venv"
    echo "   Run: python3 -m venv .venv && pip install -r requirements.txt"
    exit 1
fi

# Check .env file
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "⚠️  No .env file found. Copy .env.example and fill in your keys:"
    echo "   cp .env.example .env"
fi

cd "$PROJECT_DIR"

# Optional: retrain models
if [ "$1" = "--retrain" ]; then
    echo ""
    echo "🔄 Retraining models..."
    echo "   This may take 10-30 minutes depending on your hardware."
    echo ""
    # Run the ensemble notebook non-interactively
    $PYTHON -c "
import subprocess, sys
notebooks = [
    'notebooks/12_ensemble.ipynb',
]
for nb in notebooks:
    print(f'  Running {nb}...')
    subprocess.run([sys.executable, '-m', 'jupyter', 'nbconvert',
                    '--to', 'notebook', '--execute', '--inplace',
                    '--ExecutePreprocessor.timeout=600', nb],
                   check=True)
    print(f'  ✅ {nb} complete')
print('✅ Retraining complete!')
"
fi

# Clear stale prediction cache
echo ""
echo "🗑️  Clearing stale prediction cache..."
$PYTHON -c "
import json
from pathlib import Path
from datetime import datetime, timedelta
cache_file = Path('cache/predictions.json')
if cache_file.exists():
    try:
        data = json.loads(cache_file.read_text())
        stale = [k for k, v in data.items()
                 if datetime.now() - datetime.fromisoformat(v.get('timestamp', '2000-01-01')) > timedelta(hours=8)]
        for k in stale:
            del data[k]
        cache_file.write_text(json.dumps(data, indent=2, default=str))
        print(f'   Removed {len(stale)} stale entries, {len(data)} fresh entries remain')
    except Exception as e:
        print(f'   Cache clear: {e}')
else:
    print('   No cache file found (fresh start)')
"

# Start the webapp
echo ""
echo "🚀 Starting webapp at http://localhost:5001"
echo "   Press Ctrl+C to stop"
echo ""

$PYTHON webapp/server.py
