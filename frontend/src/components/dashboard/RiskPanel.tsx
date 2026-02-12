'use client';

import { useState, useCallback } from 'react';
import { useRiskMetrics, useRegime } from '@/hooks/useData';
import { api } from '@/lib/api';
import {
  formatNumber,
  formatPercent,
  pnlColor,
  regimeColor,
  regimeLabel,
} from '@/lib/utils';
import type { AIExplanation } from '@/types';

// ── Color helpers for risk metrics ──

function sharpeColor(value: number | null | undefined): string {
  if (value == null) return 'text-text-secondary';
  if (value > 1) return 'pnl-positive';
  if (value >= 0) return 'text-warning';
  return 'pnl-negative';
}

function betaColor(value: number | null | undefined): string {
  if (value == null) return 'text-text-secondary';
  if (value < 0.9) return 'pnl-positive';
  if (value <= 1.1) return 'text-text-primary';
  return 'text-warning';
}

function drawdownColor(value: number | null | undefined): string {
  if (value == null) return 'text-text-secondary';
  // More negative = more red intensity
  if (value < -0.2) return 'text-loss font-bold';
  if (value < -0.1) return 'text-loss';
  return 'text-loss/80';
}

function regimeBadgeBg(regime: string): string {
  switch (regime) {
    case 'bull':
      return 'bg-profit/15 border-profit/30';
    case 'bear':
      return 'bg-loss/15 border-loss/30';
    case 'high_vol':
      return 'bg-warning/15 border-warning/30';
    case 'low_vol':
      return 'bg-accent-blue/15 border-accent-blue/30';
    default:
      return 'bg-neutral/15 border-neutral/30';
  }
}

// ── AI Explain Button Component ──

interface ExplainButtonProps {
  metricName: string;
  metricValue: number | null | undefined;
  context?: Record<string, any>;
}

function ExplainButton({ metricName, metricValue, context }: ExplainButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [explanation, setExplanation] = useState<AIExplanation | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleExplain = useCallback(async () => {
    if (isOpen) {
      setIsOpen(false);
      return;
    }

    setIsOpen(true);

    // Skip fetch if we already have the explanation cached
    if (explanation) return;

    setIsLoading(true);
    setError(null);

    try {
      const result = await api.explainMetric(metricName, {
        value: metricValue,
        ...context,
      });
      setExplanation(result);
    } catch {
      setError('Unable to fetch explanation.');
    } finally {
      setIsLoading(false);
    }
  }, [isOpen, explanation, metricName, metricValue, context]);

  return (
    <div className="relative inline-block">
      <button
        onClick={handleExplain}
        className="ml-1.5 w-4 h-4 rounded-full bg-bg-tertiary border border-border/50 text-text-muted
                   hover:text-accent-blue hover:border-accent-blue/40 transition-colors
                   flex items-center justify-center text-[9px] font-bold leading-none"
        title={`Explain ${metricName}`}
        aria-label={`Explain ${metricName}`}
      >
        ?
      </button>

      {isOpen && (
        <div className="absolute z-50 left-0 top-6 w-72 max-w-[calc(100vw-2rem)] bg-bg-elevated border border-border rounded-lg shadow-lg p-3">
          {/* Close button */}
          <button
            onClick={() => setIsOpen(false)}
            className="absolute top-1.5 right-2 text-text-muted hover:text-text-primary text-xs"
            aria-label="Close explanation"
          >
            {'\u2715'}
          </button>

          {isLoading && (
            <div className="flex items-center gap-2 text-text-muted text-xs py-2">
              <div className="w-3 h-3 border-2 border-accent-blue/30 border-t-accent-blue rounded-full animate-spin" />
              Analyzing...
            </div>
          )}

          {error && (
            <p className="text-loss text-xs">{error}</p>
          )}

          {explanation && !isLoading && (
            <div className="space-y-2">
              <p className="text-xs text-text-secondary leading-relaxed">
                {explanation.explanation}
              </p>
              {explanation.suggestions && explanation.suggestions.length > 0 && (
                <div>
                  <div className="text-label text-text-muted uppercase tracking-wider mb-1">
                    Suggestions
                  </div>
                  <ul className="space-y-0.5">
                    {explanation.suggestions.map((s, i) => (
                      <li key={i} className="text-xs text-text-secondary flex gap-1.5">
                        <span className="text-accent-blue shrink-0">{'\u2022'}</span>
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Metric Card Sub-component ──

interface MetricCardProps {
  label: string;
  value: number | null | undefined;
  formatter: (v: number | null | undefined) => string;
  colorFn: (v: number | null | undefined) => string;
  explainContext?: Record<string, any>;
}

function MetricCard({ label, value, formatter, colorFn, explainContext }: MetricCardProps) {
  return (
    <div className="metric-card">
      <div className="flex items-center justify-between">
        <div className="metric-label">{label}</div>
        <ExplainButton metricName={label} metricValue={value} context={explainContext} />
      </div>
      <div className={`metric-value ${colorFn(value)}`}>
        {formatter(value)}
      </div>
    </div>
  );
}

// ── Main Component ──

export default function RiskPanel() {
  const { data: riskMetrics, isLoading: riskLoading, error: riskError } = useRiskMetrics();
  const { data: regime, isLoading: regimeLoading, error: regimeError } = useRegime();

  const isLoading = riskLoading || regimeLoading;
  const hasError = riskError || regimeError;

  // ── Loading state ──
  if (isLoading) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Risk Analytics</h2>
        </div>
        <div className="panel-body">
          <div className="flex items-center justify-center py-12">
            <div className="flex items-center gap-3 text-text-muted text-sm">
              <div className="w-4 h-4 border-2 border-accent-blue/30 border-t-accent-blue rounded-full animate-spin" />
              Loading risk metrics...
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── Error state ──
  if (hasError) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Risk Analytics</h2>
        </div>
        <div className="panel-body">
          <div className="flex items-center justify-center py-12 text-loss text-sm">
            Failed to load risk data.
          </div>
        </div>
      </div>
    );
  }

  // ── Empty state ──
  if (!riskMetrics) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Risk Analytics</h2>
        </div>
        <div className="panel-body">
          <div className="flex flex-col items-center justify-center py-12 gap-2">
            <span className="text-text-muted text-sm">No risk data available.</span>
            <span className="text-text-muted/60 text-xs">
              Risk metrics require portfolio holdings and market data.
            </span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      {/* ── Panel Header ── */}
      <div className="panel-header">
        <h2 className="panel-title">Risk Analytics</h2>
        {riskMetrics.last_updated && (
          <span className="text-label text-text-muted font-mono tabular-nums">
            Updated: {new Date(riskMetrics.last_updated).toLocaleTimeString('en-IN', {
              hour12: false,
              timeZone: 'Asia/Kolkata',
            })}
          </span>
        )}
      </div>

      <div className="panel-body space-y-5">
        {/* ── 1. Risk Metrics Cards (2x3 grid) ── */}
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          <MetricCard
            label="Sharpe Ratio"
            value={riskMetrics.sharpe_ratio}
            formatter={(v) => formatNumber(v)}
            colorFn={sharpeColor}
          />
          <MetricCard
            label="Sortino Ratio"
            value={riskMetrics.sortino_ratio}
            formatter={(v) => formatNumber(v)}
            colorFn={sharpeColor}
          />
          <MetricCard
            label="Portfolio Beta"
            value={riskMetrics.portfolio_beta}
            formatter={(v) => formatNumber(v)}
            colorFn={betaColor}
          />
          <MetricCard
            label="Max Drawdown"
            value={riskMetrics.max_drawdown}
            formatter={(v) => formatPercent(v)}
            colorFn={drawdownColor}
          />
          <MetricCard
            label="VaR 95%"
            value={riskMetrics.var_95}
            formatter={(v) => formatPercent(v)}
            colorFn={() => 'pnl-negative'}
          />
          <MetricCard
            label="Rolling Return 30d"
            value={riskMetrics.rolling_return_30d}
            formatter={(v) => formatPercent(v)}
            colorFn={(v) => pnlColor(v)}
          />
        </div>

        {/* ── 2. Rolling Returns Section ── */}
        <div>
          <div className="text-label text-text-muted uppercase tracking-wider mb-2">
            Rolling Returns
          </div>
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-bg-tertiary rounded px-3 py-2 border border-border/30">
              <div className="text-label text-text-muted">30 Day</div>
              <div className={`text-sm font-mono font-semibold tabular-nums ${pnlColor(riskMetrics.rolling_return_30d)}`}>
                {formatPercent(riskMetrics.rolling_return_30d)}
              </div>
            </div>
            <div className="bg-bg-tertiary rounded px-3 py-2 border border-border/30">
              <div className="text-label text-text-muted">90 Day</div>
              <div className={`text-sm font-mono font-semibold tabular-nums ${pnlColor(riskMetrics.rolling_return_90d)}`}>
                {formatPercent(riskMetrics.rolling_return_90d)}
              </div>
            </div>
            <div className="bg-bg-tertiary rounded px-3 py-2 border border-border/30">
              <div className="text-label text-text-muted">1 Year</div>
              <div className={`text-sm font-mono font-semibold tabular-nums ${pnlColor(riskMetrics.rolling_return_1y)}`}>
                {formatPercent(riskMetrics.rolling_return_1y)}
              </div>
            </div>
          </div>
        </div>

        {/* ── 3. Regime Section ── */}
        {regime && (
          <div>
            <div className="text-label text-text-muted uppercase tracking-wider mb-2">
              Market Regime
            </div>
            <div className="bg-bg-tertiary rounded-md border border-border/50 p-4 space-y-3">
              {/* Regime badge */}
              <div className="flex items-center gap-3">
                <span
                  className={`badge border text-sm font-semibold ${regimeBadgeBg(regime.regime)} ${regimeColor(regime.regime)}`}
                >
                  {regimeLabel(regime.regime)}
                </span>
                <ExplainButton
                  metricName="Market Regime"
                  metricValue={null}
                  context={{
                    regime: regime.regime,
                    confidence: regime.confidence,
                    ma_signal: regime.ma_signal,
                    vol_regime: regime.vol_regime,
                    breadth_signal: regime.breadth_signal,
                  }}
                />
              </div>

              {/* Regime details grid */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                {/* MA 50 */}
                <div>
                  <div className="text-label text-text-muted">MA 50</div>
                  <div className="text-data font-mono tabular-nums text-text-primary">
                    {formatNumber(regime.ma_50)}
                  </div>
                </div>
                {/* MA 200 */}
                <div>
                  <div className="text-label text-text-muted">MA 200</div>
                  <div className="text-data font-mono tabular-nums text-text-primary">
                    {formatNumber(regime.ma_200)}
                  </div>
                </div>
                {/* Vol Ratio */}
                <div>
                  <div className="text-label text-text-muted">Vol Ratio</div>
                  <div className="text-data font-mono tabular-nums text-text-primary">
                    {formatNumber(regime.vol_ratio)}
                  </div>
                </div>
                {/* Breadth */}
                <div>
                  <div className="text-label text-text-muted">Breadth</div>
                  <div className="text-data font-mono tabular-nums text-text-primary">
                    {regime.breadth_pct != null
                      ? `${(regime.breadth_pct * 100).toFixed(1)}%`
                      : '--'}
                  </div>
                </div>
              </div>

              {/* Confidence meter */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-label text-text-muted">Confidence</span>
                  <span className="text-label font-mono tabular-nums text-text-secondary">
                    {(regime.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-bg-primary rounded-full overflow-hidden border border-border/30">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      regime.confidence >= 0.7
                        ? 'bg-profit'
                        : regime.confidence >= 0.4
                          ? 'bg-warning'
                          : 'bg-loss'
                    }`}
                    style={{ width: `${(regime.confidence * 100).toFixed(0)}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
