# Changelog

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
