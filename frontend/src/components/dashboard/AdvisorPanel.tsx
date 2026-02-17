'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useAdvisorOpenBuyList } from '@/hooks/useData';
import { formatCurrency, formatTimestamp } from '@/lib/utils';
import type { AdvisorPick } from '@/types';

export default function AdvisorPanel() {
  const [budget, setBudget] = useState(40000);
  const [count, setCount] = useState(10);
  const { data, isLoading, error, mutate } = useAdvisorOpenBuyList(count, budget);

  const picks: AdvisorPick[] = data?.picks ?? [];

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="panel-title">Strategy Advisor (Simulation)</h2>
        <button
          className="btn btn-secondary"
          onClick={() => void mutate()}
        >
          Refresh
        </button>
      </div>

      <div className="panel-body space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="metric-card">
            <div className="metric-label">Budget</div>
            <div className="metric-value">{formatCurrency(data?.budget ?? budget)}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Est. Cost</div>
            <div className="metric-value">{formatCurrency(data?.estimated_total_cost)}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Remaining Cash</div>
            <div className="metric-value">{formatCurrency(data?.remaining_cash)}</div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <label className="text-label text-text-muted">Budget</label>
          <input
            className="input max-w-[160px]"
            type="number"
            min={1000}
            value={budget}
            onChange={(e) => setBudget(Number(e.target.value || 40000))}
          />
          <label className="text-label text-text-muted">Picks</label>
          <input
            className="input max-w-[90px]"
            type="number"
            min={1}
            max={25}
            value={count}
            onChange={(e) => setCount(Number(e.target.value || 10))}
          />
        </div>

        <p className="text-label text-text-muted">
          Captured: {formatTimestamp(data?.captured_at)}. This advisor is strategy-led and fee-aware.
        </p>

        {isLoading && <div className="text-sm text-text-muted py-8 text-center">Loading advisor picks...</div>}
        {error && <div className="text-sm text-loss py-8 text-center">Failed to load advisor picks.</div>}

        {!isLoading && !error && (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th className="text-right">Entry Range</th>
                  <th className="text-right">Strategy Price</th>
                  <th className="text-right">Qty</th>
                  <th className="text-right">Trade Cost</th>
                  <th className="text-right">Fee</th>
                  <th className="text-right">Stop Loss</th>
                  <th className="text-right">Target</th>
                  <th className="text-right">R:R</th>
                </tr>
              </thead>
              <tbody>
                {picks.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="text-center text-text-muted py-8">
                      No advisor picks available right now.
                    </td>
                  </tr>
                ) : (
                  picks.map((p) => (
                    <tr key={p.ticker}>
                      <td>
                        <Link className="text-accent-blue hover:underline" href={`/stock/${encodeURIComponent(p.ticker)}`}>
                          {p.ticker}
                        </Link>
                      </td>
                      <td className="text-right">
                        {formatCurrency(p.entry_range_low)} - {formatCurrency(p.entry_range_high)}
                      </td>
                      <td className="text-right">{formatCurrency(p.strategy_price_at_open)}</td>
                      <td className="text-right">{p.suggested_qty}</td>
                      <td className="text-right">{formatCurrency(p.est_trade_cost)}</td>
                      <td className="text-right">{formatCurrency(p.estimated_fee)}</td>
                      <td className="text-right text-loss">{formatCurrency(p.stop_loss_price)}</td>
                      <td className="text-right text-profit">{formatCurrency(p.target_price)}</td>
                      <td className="text-right">{p.risk_reward.toFixed(2)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
