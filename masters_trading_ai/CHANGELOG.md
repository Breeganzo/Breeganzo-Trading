# Changelog

## 2026-02-16 (UI/Risk/Premarket hardening)

### Added
- `tests/test_premarket_snapshot.py` for market-open snapshot typing/timing validation.
- `tests/test_risk_monte_carlo.py` for equity-curve Monte Carlo and portfolio-only API validation.
- `tests/test_tooltip_popover.py` for popover accessibility + scrollability checks.
- `tests/test_artifact_cleanup.py` for tracked-file cleanup guardrails.
- `DAILY_OPERATIONS.md` runbook with market-open and after-hours verification steps.

### Changed
- Market-open snapshot flow in `webapp/server.py`:
  - new `snapshot_type` lifecycle (`pending_market_open`, `market_open`, `near_open_fallback`, `premarket_preview`),
  - no reuse of stale/open-mismatched snapshots,
  - richer `/api/premarket-outlook` metadata.
- `PredictionTracker` schema upgraded to `SCHEMA_VERSION = 4` with persisted `snapshot_type`.
- `/api/expected-vs-actual` now:
  - prefers close-price path for same-day after-hours mode,
  - uses market-open snapshot rows for `strategy_price_at_open` when available,
  - returns `after_hours_mode` and `snapshot_type`.
- Dashboard and risk popovers are now persistent, keyboard accessible, and scrollable.
- Risk Monte Carlo now uses portfolio equity-curve-derived returns with minimum history checks.
- Feature engineering lag/zscore generation now batches columns to avoid pandas fragmentation warnings.

### Removed
- `__pycache__`/temp/demo artifacts from tracked files are now blocked by test coverage.

## 2026-02-16

### Added
- Premarket snapshot pipeline with configurable `PREMARKET_MAX_BUFFER_MINUTES` and `/api/premarket-outlook` endpoint.
- Dashboard tables for `Premarket vs Current` and `Current Second Snapshot`.
- `/api/training-feedback` alias endpoint for retraining export.
- CI workflow `.github/workflows/ci.yml` for pytest + black + flake8 + mypy.
- New tests:
  - `tests/test_premarket_outlook.py`
  - `tests/test_expected_vs_actual.py`
  - `tests/test_backtest_costs.py`

### Changed
- Prediction sanitization contract tightened in `src/inference/predictor.py`:
  - decimal internal unit only,
  - percent-to-decimal fix when `abs(raw) > 2.0`,
  - hard cap at ±50%.
- `predict_single` now exits early when `current_price <= 0` and asserts predicted-price math in cached and live paths.
- `PredictionTracker` schema upgraded to persist:
  - `strategy_price_at_open`
  - `ai_last_prediction`
  - `actual_close`
  - `direction_comparison` and related direction fields.
- `/api/expected-vs-actual` now returns upgraded tracking fields and schema version.
- Risk analytics endpoint now uses portfolio-only tickers and ignores non-portfolio requests.
- UI now renders `—` instead of placeholder `₹0` values for invalid/zero prices.

### Removed
- `reports/technical_features_demo.png` (demo artifact cleanup).
