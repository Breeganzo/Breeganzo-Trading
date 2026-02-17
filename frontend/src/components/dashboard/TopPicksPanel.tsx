'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useTopPicks } from '@/hooks/useData';
import { formatCurrency, formatPercentRaw, formatTimestamp, pnlColor } from '@/lib/utils';
import type { TopPickItem } from '@/types';

const SIGNALS: Array<'BUY' | 'SELL' | 'HOLD' | 'ALL'> = ['ALL', 'BUY', 'SELL', 'HOLD'];

export default function TopPicksPanel() {
  const [source, setSource] = useState<'strategy' | 'ai'>('strategy');
  const [signal, setSignal] = useState<'BUY' | 'SELL' | 'HOLD' | 'ALL'>('ALL');

  const signalFilter = signal === 'ALL' ? undefined : signal;
  const { data, isLoading, error } = useTopPicks(source, signalFilter, 10);

  const items: TopPickItem[] = data?.items ?? [];

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="panel-title">Top Picks</h2>
        <div className="flex items-center gap-2">
          <select
            className="select"
            value={source}
            onChange={(e) => setSource(e.target.value as 'strategy' | 'ai')}
          >
            <option value="strategy">Strategy Picks</option>
            <option value="ai">AI Picks</option>
          </select>
          <select
            className="select"
            value={signal}
            onChange={(e) => setSignal(e.target.value as 'BUY' | 'SELL' | 'HOLD' | 'ALL')}
          >
            {SIGNALS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="panel-body space-y-3">
        <p className="text-label text-text-muted">
          {source === 'strategy'
            ? 'Strategy-first ranked picks with confidence and agreement.'
            : 'AI picks blend strategy with same-day news sentiment.'}
        </p>
        <p className="text-label text-text-muted">
          Captured: {formatTimestamp(data?.captured_at)}
        </p>

        {isLoading && (
          <div className="text-sm text-text-muted py-8 text-center">Loading top picks...</div>
        )}
        {error && (
          <div className="text-sm text-loss py-8 text-center">Failed to load top picks.</div>
        )}

        {!isLoading && !error && (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th className="text-right">Current</th>
                  <th className="text-right">Strategy</th>
                  <th className="text-right">AI</th>
                  <th className="text-right">Return</th>
                  <th className="text-right">Signal</th>
                  <th className="text-right">Confidence</th>
                  <th className="text-right">Agreement</th>
                  <th className="text-right">Score</th>
                </tr>
              </thead>
              <tbody>
                {items.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="text-center text-text-muted py-8">
                      No picks available for this filter.
                    </td>
                  </tr>
                ) : (
                  items.map((item) => {
                    const activeReturn = source === 'ai'
                      ? (item.ai_return_pct ?? item.strategy_return_pct)
                      : item.strategy_return_pct;
                    const activeSignal = source === 'ai'
                      ? (item.signal_ai ?? item.signal_strategy)
                      : item.signal_strategy;
                    return (
                      <tr key={`${item.ticker}-${source}`}>
                        <td>
                          <Link className="text-accent-blue hover:underline" href={`/stock/${encodeURIComponent(item.ticker)}`}>
                            {item.ticker}
                          </Link>
                        </td>
                        <td className="text-right">{formatCurrency(item.current_price)}</td>
                        <td className="text-right">{formatCurrency(item.strategy_price)}</td>
                        <td className="text-right">{formatCurrency(item.ai_price)}</td>
                        <td className={`text-right ${pnlColor(activeReturn)}`}>
                          {formatPercentRaw(activeReturn)}
                        </td>
                        <td className={`text-right ${
                          activeSignal === 'BUY' ? 'text-profit' : activeSignal === 'SELL' ? 'text-loss' : 'text-text-secondary'
                        }`}>
                          {activeSignal}
                        </td>
                        <td className="text-right">{item.confidence.toFixed(1)}%</td>
                        <td className="text-right">{item.agreement.toFixed(1)}%</td>
                        <td className="text-right">{item.score.toFixed(2)}</td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
