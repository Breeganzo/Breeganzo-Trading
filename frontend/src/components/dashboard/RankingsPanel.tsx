'use client';

import { useState, useCallback, useEffect } from 'react';
import { useRankings } from '@/hooks/useData';
import { api } from '@/lib/api';
import { formatNumber, formatPercent, formatCurrency, pnlColor } from '@/lib/utils';
import type { RankingEntry, RankingCategory } from '@/types';

// ── Category options ──

interface CategoryOption {
  value: RankingCategory;
  label: string;
}

const CATEGORIES: CategoryOption[] = [
  { value: 'overall', label: 'Overall' },
  { value: 'top_buy', label: 'Top 10 Buy' },
  { value: 'top_sell', label: 'Top 10 Sell' },
  { value: 'banking', label: 'Banking' },
  { value: 'large_cap', label: 'Large Cap' },
  { value: 'small_cap', label: 'Small Cap' },
  { value: 'high_vol', label: 'High Volatility' },
];

// ── Score color helper ──

function scoreColor(score: number): string {
  if (score >= 80) return 'text-profit font-semibold';
  if (score >= 60) return 'text-profit/80';
  if (score >= 40) return 'text-accent-blue';
  if (score >= 20) return 'text-text-secondary';
  return 'text-text-muted';
}

// ── Main Component ──

export default function RankingsPanel() {
  const [category, setCategory] = useState<RankingCategory>('overall');
  const [isComputing, setIsComputing] = useState(false);
  const [autoComputeAttempted, setAutoComputeAttempted] = useState(false);

  const { data: rankings, isLoading, error, mutate } = useRankings(category);

  const handleCompute = useCallback(async () => {
    setIsComputing(true);
    try {
      await api.triggerRankingCompute();
      await mutate();
    } catch {
      // Compute failed silently; data will reflect stale state
    } finally {
      setIsComputing(false);
    }
  }, [mutate]);

  // ── Loading state ──
  if (isLoading) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Ranking Engine</h2>
        </div>
        <div className="panel-body">
          <div className="flex items-center justify-center py-12">
            <div className="flex items-center gap-3 text-text-muted text-sm">
              <div className="w-4 h-4 border-2 border-accent-blue/30 border-t-accent-blue rounded-full animate-spin" />
              Loading rankings...
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── Error state ──
  if (error) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Ranking Engine</h2>
        </div>
        <div className="panel-body">
          <div className="flex items-center justify-center py-12 text-loss text-sm">
            Failed to load ranking data.
          </div>
        </div>
      </div>
    );
  }

  const entries: RankingEntry[] = rankings?.entries ?? [];
  const computedAt = rankings?.computed_at ?? null;

  useEffect(() => {
    if (isLoading || isComputing || autoComputeAttempted) return;
    if (!entries.length) {
      setAutoComputeAttempted(true);
      void handleCompute();
    }
  }, [isLoading, isComputing, autoComputeAttempted, entries.length, handleCompute]);

  return (
    <div className="panel">
      {/* ── Panel Header ── */}
      <div className="panel-header">
        <h2 className="panel-title">Ranking Engine</h2>
        <div className="flex items-center gap-3">
          {computedAt && (
            <span className="text-label text-text-muted font-mono tabular-nums">
              Last computed:{' '}
              {new Date(computedAt).toLocaleString('en-IN', {
                hour12: false,
                timeZone: 'Asia/Kolkata',
              })}
            </span>
          )}
          <button
            onClick={handleCompute}
            disabled={isComputing}
            className="btn btn-sm flex items-center gap-1.5 text-xs px-2.5 py-1 rounded
                       bg-accent-blue/10 text-accent-blue border border-accent-blue/30
                       hover:bg-accent-blue/20 disabled:opacity-50 disabled:cursor-not-allowed
                       transition-colors"
          >
            {isComputing && (
              <div className="w-3 h-3 border-2 border-accent-blue/30 border-t-accent-blue rounded-full animate-spin" />
            )}
            {isComputing ? 'Computing...' : 'Recompute'}
          </button>
        </div>
      </div>

      <div className="panel-body space-y-4">
        {/* ── 1. Category Selector ── */}
        <div>
          <select
            className="select"
            value={category}
            onChange={(e) => setCategory(e.target.value as RankingCategory)}
          >
            {CATEGORIES.map((cat) => (
              <option key={cat.value} value={cat.value}>
                {cat.label}
              </option>
            ))}
          </select>
        </div>

        {/* ── 2. Rankings Table ── */}
        {entries.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 gap-2">
            <span className="text-text-muted text-sm">No rankings available for this category.</span>
            <span className="text-text-muted/60 text-xs">
              Trigger a recompute or select a different category.
            </span>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="text-right w-12">Rank</th>
                  <th className="text-left">Ticker</th>
                  <th className="text-right">Score</th>
                  <th className="text-right">Expected Return</th>
                  <th className="text-right">30d Momentum</th>
                  <th className="text-right">Volatility</th>
                  <th className="text-right">Liquidity</th>
                  <th className="text-right">Price</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => (
                  <tr key={`${entry.ticker}-${entry.rank_position}`}>
                    {/* Rank */}
                    <td className="text-right text-text-muted font-mono tabular-nums">
                      {entry.rank_position}
                    </td>
                    {/* Ticker */}
                    <td className="text-left font-semibold text-text-primary">
                      {entry.ticker}
                    </td>
                    {/* Score */}
                    <td className={`text-right font-mono tabular-nums ${scoreColor(entry.score)}`}>
                      {formatNumber(entry.score, 1)}
                    </td>
                    {/* Expected Return */}
                    <td className={`text-right font-mono tabular-nums ${pnlColor(entry.expected_return)}`}>
                      {formatPercent(entry.expected_return)}
                    </td>
                    {/* 30d Momentum */}
                    <td className={`text-right font-mono tabular-nums ${pnlColor(entry.momentum_30d)}`}>
                      {formatPercent(entry.momentum_30d)}
                    </td>
                    {/* Volatility */}
                    <td className="text-right text-text-secondary font-mono tabular-nums">
                      {formatPercent(entry.volatility)}
                    </td>
                    {/* Liquidity */}
                    <td className="text-right text-text-secondary font-mono tabular-nums">
                      {formatNumber(entry.liquidity_score, 1)}
                    </td>
                    {/* Price */}
                    <td className="text-right text-text-primary font-mono tabular-nums">
                      {formatCurrency(entry.current_price)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
