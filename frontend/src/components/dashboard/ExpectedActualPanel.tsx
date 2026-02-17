'use client';

import { useState } from 'react';
import { useExpectedVsActual } from '@/hooks/useData';
import { formatCurrency, formatPercentRaw, formatTimestamp, pnlColor } from '@/lib/utils';
import type { ExpectedActualRow } from '@/types';

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

export default function ExpectedActualPanel() {
  const [snapshotDate, setSnapshotDate] = useState(todayIso());
  const { data, isLoading, error } = useExpectedVsActual(snapshotDate);

  const rows: ExpectedActualRow[] = data?.items ?? [];

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="panel-title">Expected vs Actual</h2>
        <input
          type="date"
          className="input max-w-[180px]"
          value={snapshotDate}
          onChange={(e) => setSnapshotDate(e.target.value)}
        />
      </div>

      <div className="panel-body space-y-3">
        <p className="text-label text-text-muted">
          Daily open-based scorecard for strategy and AI projections.
        </p>

        {isLoading && <div className="text-sm text-text-muted py-8 text-center">Loading expected vs actual...</div>}
        {error && <div className="text-sm text-loss py-8 text-center">Failed to load expected vs actual.</div>}

        {!isLoading && !error && (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th className="text-right">Open</th>
                  <th className="text-right">Current/Close</th>
                  <th className="text-right">Strategy @ Open</th>
                  <th className="text-right">AI @ Open</th>
                  <th className="text-right">Actual Return</th>
                  <th className="text-right">Strategy Return</th>
                  <th className="text-right">AI Return</th>
                  <th className="text-right">Alpha</th>
                  <th className="text-right">Direction</th>
                </tr>
              </thead>
              <tbody>
                {rows.length === 0 ? (
                  <tr>
                    <td colSpan={10} className="text-center text-text-muted py-8">
                      No rows available for this date. Capture a daily snapshot first.
                    </td>
                  </tr>
                ) : (
                  rows.map((r) => (
                    <tr key={r.ticker}>
                      <td>{r.ticker}</td>
                      <td className="text-right">{formatCurrency(r.open_price)}</td>
                      <td className="text-right">{formatCurrency(r.current_price)}</td>
                      <td className="text-right">{formatCurrency(r.strategy_price_at_open)}</td>
                      <td className="text-right">{formatCurrency(r.ai_price_at_open)}</td>
                      <td className={`text-right ${pnlColor(r.actual_return_pct)}`}>{formatPercentRaw(r.actual_return_pct)}</td>
                      <td className={`text-right ${pnlColor(r.strategy_return_pct)}`}>{formatPercentRaw(r.strategy_return_pct)}</td>
                      <td className={`text-right ${pnlColor(r.ai_return_pct)}`}>{formatPercentRaw(r.ai_return_pct)}</td>
                      <td className={`text-right ${pnlColor(r.alpha_pct)}`}>{formatPercentRaw(r.alpha_pct)}</td>
                      <td className={`text-right ${r.direction_comparison ? 'text-profit' : 'text-loss'}`}>
                        {r.direction_comparison ? 'Aligned' : 'Mismatch'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}

        {rows.length > 0 && (
          <p className="text-label text-text-muted">
            Last row capture time: {formatTimestamp(rows[0]?.captured_at)}
          </p>
        )}
      </div>
    </div>
  );
}
