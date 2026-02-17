'use client';

import Link from 'next/link';
import { useStocksOverview } from '@/hooks/useData';
import { formatCurrency, formatPercentRaw, formatVolume, formatTimestamp, pnlColor } from '@/lib/utils';
import type { StocksOverviewItem } from '@/types';

const SECTOR_ORDER = ['large_cap', 'banking', 'commodity', 'small_cap', 'other'];

function sectorLabel(name: string): string {
  switch (name) {
    case 'large_cap':
      return 'Large Cap';
    case 'small_cap':
      return 'Small Cap';
    case 'banking':
      return 'Banking';
    case 'commodity':
      return 'Commodity';
    default:
      return 'Other';
  }
}

export default function StocksPanel() {
  const { data, isLoading, error } = useStocksOverview(150, false);

  const grouped = (data?.grouped ?? {}) as Record<string, StocksOverviewItem[]>;
  const sectors = SECTOR_ORDER.filter((s) => (grouped[s] ?? []).length > 0);

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="panel-title">All Stocks Overview</h2>
      </div>

      <div className="panel-body space-y-4">
        <p className="text-label text-text-muted">
          Live sector buckets with strategy signal context. Click a ticker to open stock detail.
        </p>
        <p className="text-label text-text-muted">
          Captured: {formatTimestamp(data?.captured_at)}
        </p>

        {isLoading && <div className="text-sm text-text-muted py-8 text-center">Loading stocks...</div>}
        {error && <div className="text-sm text-loss py-8 text-center">Failed to load stock overview.</div>}

        {!isLoading && !error && sectors.map((sector) => (
          <div key={sector} className="border border-border rounded-md overflow-hidden">
            <div className="px-3 py-2 border-b border-border bg-bg-tertiary text-xs font-semibold uppercase tracking-wider text-text-secondary">
              {sectorLabel(sector)} ({grouped[sector]?.length ?? 0})
            </div>
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Ticker</th>
                    <th className="text-right">Current</th>
                    <th className="text-right">Open</th>
                    <th className="text-right">Change</th>
                    <th className="text-right">Signal</th>
                    <th className="text-right">Volume</th>
                  </tr>
                </thead>
                <tbody>
                  {(grouped[sector] ?? []).map((row) => (
                    <tr key={`${sector}-${row.ticker}`}>
                      <td>
                        <Link className="text-accent-blue hover:underline" href={`/stock/${encodeURIComponent(row.ticker)}`}>
                          {row.ticker}
                        </Link>
                      </td>
                      <td className="text-right">{formatCurrency(row.current_price)}</td>
                      <td className="text-right">{formatCurrency(row.open_price)}</td>
                      <td className={`text-right ${pnlColor(row.change_pct)}`}>{formatPercentRaw(row.change_pct)}</td>
                      <td className={`text-right ${
                        row.signal === 'BUY' ? 'text-profit' : row.signal === 'SELL' ? 'text-loss' : 'text-text-secondary'
                      }`}>
                        {row.signal}
                      </td>
                      <td className="text-right">{formatVolume(row.volume)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
