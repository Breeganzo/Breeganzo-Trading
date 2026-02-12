'use client';

import { useState, useMemo } from 'react';
import { usePortfolio } from '@/hooks/useData';
import { formatCurrency, formatPercent, formatNumber, pnlColor } from '@/lib/utils';
import type { PortfolioHolding, SortDirection } from '@/types';

// ── Sector color palette ──
const SECTOR_COLORS: Record<string, string> = {
  'Financial Services': 'bg-blue-500/70',
  'Information Technology': 'bg-violet-500/70',
  'Healthcare': 'bg-emerald-500/70',
  'Consumer Goods': 'bg-amber-500/70',
  'Energy': 'bg-orange-500/70',
  'Automobile': 'bg-cyan-500/70',
  'Metals & Mining': 'bg-slate-400/70',
  'Telecom': 'bg-rose-500/70',
  'Infrastructure': 'bg-lime-500/70',
  'Pharma': 'bg-teal-500/70',
  'FMCG': 'bg-yellow-500/70',
  'Chemicals': 'bg-indigo-500/70',
  'Cement': 'bg-stone-400/70',
  'Realty': 'bg-pink-500/70',
  'Media': 'bg-fuchsia-500/70',
};

function getSectorColor(sector: string): string {
  return SECTOR_COLORS[sector] || 'bg-neutral/70';
}

// ── Sort columns ──
type SortKey = keyof PortfolioHolding;

interface SortConfig {
  key: SortKey;
  direction: SortDirection;
}

function sortHoldings(holdings: PortfolioHolding[], config: SortConfig): PortfolioHolding[] {
  return [...holdings].sort((a, b) => {
    const aVal = a[config.key];
    const bVal = b[config.key];

    // Handle null/undefined
    if (aVal == null && bVal == null) return 0;
    if (aVal == null) return 1;
    if (bVal == null) return -1;

    let cmp = 0;
    if (typeof aVal === 'string' && typeof bVal === 'string') {
      cmp = aVal.localeCompare(bVal);
    } else if (typeof aVal === 'number' && typeof bVal === 'number') {
      cmp = aVal - bVal;
    }

    return config.direction === 'asc' ? cmp : -cmp;
  });
}

// ── Column definitions ──
interface ColumnDef {
  key: SortKey;
  label: string;
  align: 'left' | 'right';
}

const COLUMNS: ColumnDef[] = [
  { key: 'ticker', label: 'Ticker', align: 'left' },
  { key: 'quantity', label: 'Qty', align: 'right' },
  { key: 'avg_buy_price', label: 'Avg Price', align: 'right' },
  { key: 'current_price', label: 'LTP', align: 'right' },
  { key: 'unrealized_pnl', label: 'Unrealized P&L', align: 'right' },
  { key: 'pnl_pct', label: '% Change', align: 'right' },
  { key: 'beta', label: 'Beta', align: 'right' },
  { key: 'volatility', label: 'Volatility', align: 'right' },
];

// ── Sort arrow indicator ──
function SortArrow({ column, sortConfig }: { column: SortKey; sortConfig: SortConfig }) {
  if (sortConfig.key !== column) {
    return <span className="text-text-muted/40 ml-1">{'\u2195'}</span>;
  }
  return (
    <span className="text-accent-blue ml-1">
      {sortConfig.direction === 'asc' ? '\u2191' : '\u2193'}
    </span>
  );
}

// ── Main Component ──
export default function PortfolioPanel() {
  const { data: portfolio, isLoading, error } = usePortfolio();

  const [sortConfig, setSortConfig] = useState<SortConfig>({
    key: 'unrealized_pnl',
    direction: 'desc',
  });

  const handleSort = (key: SortKey) => {
    setSortConfig((prev) => ({
      key,
      direction: prev.key === key && prev.direction === 'desc' ? 'asc' : 'desc',
    }));
  };

  const sortedHoldings = useMemo(() => {
    if (!portfolio?.holdings) return [];
    return sortHoldings(portfolio.holdings, sortConfig);
  }, [portfolio?.holdings, sortConfig]);

  // Compute transaction cost totals from holdings
  const costTotals = useMemo(() => {
    if (!portfolio?.holdings) return { buy: 0, sell: 0, total: 0 };
    const buy = portfolio.holdings.reduce((acc: number, h: PortfolioHolding) => acc + (h.total_buy_costs ?? 0), 0);
    const sell = portfolio.holdings.reduce((acc: number, h: PortfolioHolding) => acc + (h.total_sell_costs ?? 0), 0);
    return { buy, sell, total: buy + sell };
  }, [portfolio?.holdings]);

  // Sector exposure entries sorted by weight
  const sectorEntries = useMemo(() => {
    if (!portfolio?.sector_exposure) return [];
    return Object.entries(portfolio.sector_exposure).sort(
      ([, a], [, b]) => (b as number) - (a as number)
    );
  }, [portfolio?.sector_exposure]);

  // ── Loading state ──
  if (isLoading) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Portfolio Overview</h2>
        </div>
        <div className="panel-body">
          <div className="flex items-center justify-center py-12">
            <div className="flex items-center gap-3 text-text-muted text-sm">
              <div className="w-4 h-4 border-2 border-accent-blue/30 border-t-accent-blue rounded-full animate-spin" />
              Loading portfolio data...
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
          <h2 className="panel-title">Portfolio Overview</h2>
        </div>
        <div className="panel-body">
          <div className="flex items-center justify-center py-12 text-loss text-sm">
            Failed to load portfolio data.
          </div>
        </div>
      </div>
    );
  }

  // ── Empty state ──
  if (!portfolio || !portfolio.holdings || portfolio.holdings.length === 0) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Portfolio Overview</h2>
        </div>
        <div className="panel-body">
          <div className="flex flex-col items-center justify-center py-12 gap-2">
            <span className="text-text-muted text-sm">No holdings found.</span>
            <span className="text-text-muted/60 text-xs">
              Execute trades to build your portfolio.
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
        <h2 className="panel-title">Portfolio Overview</h2>
        <span className="text-label text-text-muted font-mono tabular-nums">
          {portfolio.holdings.length} holding{portfolio.holdings.length !== 1 ? 's' : ''}
        </span>
      </div>

      <div className="panel-body space-y-5">
        {/* ── 1. Summary Cards Row ── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {/* Total Portfolio Value */}
          <div className="metric-card">
            <div className="metric-label">Total Value</div>
            <div className="metric-value text-text-primary">
              {formatCurrency(portfolio.total_value)}
            </div>
          </div>

          {/* Unrealized P&L */}
          <div className="metric-card">
            <div className="metric-label">Unrealized P&L</div>
            <div className={`metric-value ${pnlColor(portfolio.total_unrealized_pnl)}`}>
              {formatCurrency(portfolio.total_unrealized_pnl)}
            </div>
            <div className={`text-label font-mono tabular-nums mt-0.5 ${pnlColor(portfolio.total_pnl_pct)}`}>
              {formatPercent(portfolio.total_pnl_pct)}
            </div>
          </div>

          {/* Realized P&L */}
          <div className="metric-card">
            <div className="metric-label">Realized P&L</div>
            <div className={`metric-value ${pnlColor(portfolio.total_realized_pnl)}`}>
              {formatCurrency(portfolio.total_realized_pnl)}
            </div>
          </div>

          {/* Day P&L */}
          <div className="metric-card">
            <div className="metric-label">Day P&L</div>
            <div className={`metric-value ${pnlColor(portfolio.day_pnl)}`}>
              {formatCurrency(portfolio.day_pnl)}
            </div>
            <div className={`text-label font-mono tabular-nums mt-0.5 ${pnlColor(portfolio.day_pnl_pct)}`}>
              {formatPercent(portfolio.day_pnl_pct)}
            </div>
          </div>
        </div>

        {/* ── 2. Sector Exposure Bar ── */}
        {sectorEntries.length > 0 && (
          <div>
            <div className="text-label text-text-muted uppercase tracking-wider mb-2">
              Sector Exposure
            </div>
            {/* Stacked bar */}
            <div className="flex h-5 rounded overflow-hidden border border-border/50">
              {sectorEntries.map(([sector, weight]) => (
                <div
                  key={sector}
                  className={`${getSectorColor(sector)} transition-all duration-300`}
                  style={{ width: `${((weight as number) * 100).toFixed(1)}%` }}
                  title={`${sector}: ${((weight as number) * 100).toFixed(1)}%`}
                />
              ))}
            </div>
            {/* Legend */}
            <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
              {sectorEntries.map(([sector, weight]) => (
                <div key={sector} className="flex items-center gap-1.5 text-label">
                  <span className={`w-2.5 h-2.5 rounded-sm ${getSectorColor(sector)}`} />
                  <span className="text-text-secondary">{sector}</span>
                  <span className="text-text-muted font-mono tabular-nums">
                    {((weight as number) * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── 3. Holdings Table ── */}
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                {COLUMNS.map((col) => (
                  <th
                    key={col.key}
                    className={`sortable ${col.align === 'right' ? 'text-right' : 'text-left'}`}
                    onClick={() => handleSort(col.key)}
                  >
                    <span className="inline-flex items-center">
                      {col.label}
                      <SortArrow column={col.key} sortConfig={sortConfig} />
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedHoldings.map((h) => (
                <tr key={h.id}>
                  {/* Ticker */}
                  <td className="font-semibold text-text-primary">
                    <div className="flex items-center gap-2">
                      <span>{h.ticker}</span>
                      {h.sector && (
                        <span className="text-label text-text-muted font-normal hidden xl:inline">
                          {h.sector}
                        </span>
                      )}
                    </div>
                  </td>
                  {/* Qty */}
                  <td className="text-right text-text-primary">
                    {formatNumber(h.quantity, 0)}
                  </td>
                  {/* Avg Price */}
                  <td className="text-right text-text-secondary">
                    {formatCurrency(h.avg_buy_price)}
                  </td>
                  {/* LTP */}
                  <td className="text-right text-text-primary">
                    {formatCurrency(h.current_price)}
                  </td>
                  {/* Unrealized P&L */}
                  <td className={`text-right ${pnlColor(h.unrealized_pnl)}`}>
                    {formatCurrency(h.unrealized_pnl)}
                  </td>
                  {/* % Change */}
                  <td className={`text-right ${pnlColor(h.pnl_pct)}`}>
                    {formatPercent(h.pnl_pct)}
                  </td>
                  {/* Beta */}
                  <td className="text-right text-text-secondary">
                    {formatNumber(h.beta)}
                  </td>
                  {/* Volatility */}
                  <td className="text-right text-text-secondary">
                    {formatPercent(h.volatility)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* ── 4. Transaction Costs Summary ── */}
        <div className="border-t border-border/50 pt-3">
          <div className="text-label text-text-muted uppercase tracking-wider mb-2">
            Transaction Costs
          </div>
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-bg-tertiary rounded px-3 py-2 border border-border/30">
              <div className="text-label text-text-muted">Buy Costs</div>
              <div className="text-data font-mono tabular-nums text-text-secondary">
                {formatCurrency(costTotals.buy)}
              </div>
            </div>
            <div className="bg-bg-tertiary rounded px-3 py-2 border border-border/30">
              <div className="text-label text-text-muted">Sell Costs</div>
              <div className="text-data font-mono tabular-nums text-text-secondary">
                {formatCurrency(costTotals.sell)}
              </div>
            </div>
            <div className="bg-bg-tertiary rounded px-3 py-2 border border-border/30">
              <div className="text-label text-text-muted">Total Costs</div>
              <div className="text-data font-mono tabular-nums text-text-primary font-semibold">
                {formatCurrency(costTotals.total)}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
