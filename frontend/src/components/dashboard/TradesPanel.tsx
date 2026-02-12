'use client';

import { useState, useCallback, useEffect, Fragment } from 'react';
import { useTrades } from '@/hooks/useData';
import { api } from '@/lib/api';
import { formatCurrency, formatNumber, formatTimestamp } from '@/lib/utils';
import type { Trade, TradeCreate, CostPreview } from '@/types';

// ── Slippage options ──

const SLIPPAGE_OPTIONS = [
  { value: 0.1, label: '0.1%' },
  { value: 0.2, label: '0.2%' },
  { value: 0.3, label: '0.3%' },
];

// ── Main Component ──

export default function TradesPanel() {
  // ── Data hooks ──
  const { data: trades, isLoading, error, mutate: mutateTrades } = useTrades();

  // ── Form state ──
  const [formTicker, setFormTicker] = useState('');
  const [formType, setFormType] = useState<'BUY' | 'SELL'>('BUY');
  const [formQuantity, setFormQuantity] = useState('');
  const [formPrice, setFormPrice] = useState('');
  const [formSlippage, setFormSlippage] = useState<number>(0.1);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executeError, setExecuteError] = useState<string | null>(null);

  // ── Cost preview state ──
  const [costPreview, setCostPreview] = useState<CostPreview | null>(null);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);

  // ── Expanded row for cost breakdown ──
  const [expandedTradeId, setExpandedTradeId] = useState<string | null>(null);

  // ── Fetch cost preview when form fields are complete ──
  useEffect(() => {
    const ticker = formTicker.trim().toUpperCase();
    const quantity = parseInt(formQuantity, 10);
    const price = parseFloat(formPrice);

    if (!ticker || isNaN(quantity) || quantity <= 0 || isNaN(price) || price <= 0) {
      setCostPreview(null);
      return;
    }

    const controller = new AbortController();
    let cancelled = false;

    const fetchPreview = async () => {
      setIsPreviewLoading(true);
      try {
        const preview = await api.getCostPreview(ticker, formType, quantity, price, formSlippage);
        if (!cancelled) {
          setCostPreview(preview);
        }
      } catch {
        if (!cancelled) {
          setCostPreview(null);
        }
      } finally {
        if (!cancelled) {
          setIsPreviewLoading(false);
        }
      }
    };

    // Debounce the preview fetch
    const timeout = setTimeout(fetchPreview, 400);

    return () => {
      cancelled = true;
      controller.abort();
      clearTimeout(timeout);
    };
  }, [formTicker, formType, formQuantity, formPrice, formSlippage]);

  // ── Execute trade ──
  const handleExecute = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setExecuteError(null);

    const ticker = formTicker.trim().toUpperCase();
    const quantity = parseInt(formQuantity, 10);
    const price = parseFloat(formPrice);

    if (!ticker) {
      setExecuteError('Ticker is required.');
      return;
    }
    if (isNaN(quantity) || quantity <= 0) {
      setExecuteError('Valid quantity is required.');
      return;
    }
    if (isNaN(price) || price <= 0) {
      setExecuteError('Valid price is required.');
      return;
    }

    const payload: TradeCreate = {
      ticker,
      trade_type: formType,
      quantity,
      price,
      slippage_pct: formSlippage,
    };

    setIsExecuting(true);
    try {
      await api.executeTrade(payload);
      // Reset form
      setFormTicker('');
      setFormQuantity('');
      setFormPrice('');
      setCostPreview(null);
      // Refresh trade history
      await mutateTrades();
    } catch (err: any) {
      setExecuteError(err?.message || 'Failed to execute trade.');
    } finally {
      setIsExecuting(false);
    }
  }, [formTicker, formType, formQuantity, formPrice, formSlippage, mutateTrades]);

  // ── Toggle expanded row ──
  const toggleExpand = useCallback((tradeId: string) => {
    setExpandedTradeId((prev) => (prev === tradeId ? null : tradeId));
  }, []);

  // ── Resolve trade list ──
  const tradeList: Trade[] = (() => {
    if (!trades) return [];
    const list: Trade[] = Array.isArray(trades) ? trades : (trades as any)?.trades ?? [];
    // Sort by date descending
    return [...list].sort(
      (a, b) => new Date(b.executed_at).getTime() - new Date(a.executed_at).getTime()
    );
  })();

  // ── Loading state ──
  if (isLoading) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Trades</h2>
        </div>
        <div className="panel-body">
          <div className="flex items-center justify-center py-12">
            <div className="flex items-center gap-3 text-text-muted text-sm">
              <div className="w-4 h-4 border-2 border-accent-blue/30 border-t-accent-blue rounded-full animate-spin" />
              Loading trades...
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
          <h2 className="panel-title">Trades</h2>
        </div>
        <div className="panel-body">
          <div className="flex items-center justify-center py-12 text-loss text-sm">
            Failed to load trade history.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      {/* ── Panel Header ── */}
      <div className="panel-header">
        <h2 className="panel-title">Trades</h2>
        <span className="text-label text-text-muted font-mono tabular-nums">
          {tradeList.length} trade{tradeList.length !== 1 ? 's' : ''}
        </span>
      </div>

      <div className="panel-body space-y-5">
        {/* ── 1. New Trade Form ── */}
        <div>
          <div className="text-label text-text-muted uppercase tracking-wider mb-3">
            Execute Trade
          </div>
          <form onSubmit={handleExecute} className="space-y-3">
            {/* Ticker */}
            <input
              type="text"
              className="input"
              placeholder="Ticker (e.g. INFY)"
              value={formTicker}
              onChange={(e) => setFormTicker(e.target.value)}
            />

            {/* Trade Type Toggle */}
            <div className="flex gap-1">
              <button
                type="button"
                onClick={() => setFormType('BUY')}
                className={`btn flex-1 py-1.5 text-xs font-semibold rounded transition-colors ${
                  formType === 'BUY'
                    ? 'bg-profit/20 text-profit border border-profit/40'
                    : 'bg-bg-tertiary text-text-muted border border-border/30 hover:text-text-secondary'
                }`}
              >
                BUY
              </button>
              <button
                type="button"
                onClick={() => setFormType('SELL')}
                className={`btn flex-1 py-1.5 text-xs font-semibold rounded transition-colors ${
                  formType === 'SELL'
                    ? 'bg-loss/20 text-loss border border-loss/40'
                    : 'bg-bg-tertiary text-text-muted border border-border/30 hover:text-text-secondary'
                }`}
              >
                SELL
              </button>
            </div>

            {/* Quantity & Price */}
            <div className="grid grid-cols-2 gap-3">
              <input
                type="number"
                className="input"
                placeholder="Quantity"
                min="1"
                step="1"
                value={formQuantity}
                onChange={(e) => setFormQuantity(e.target.value)}
              />
              <input
                type="number"
                className="input"
                placeholder="Price"
                min="0.01"
                step="0.01"
                value={formPrice}
                onChange={(e) => setFormPrice(e.target.value)}
              />
            </div>

            {/* Slippage Radio Buttons */}
            <div>
              <div className="text-label text-text-muted mb-1.5">Slippage</div>
              <div className="flex gap-3">
                {SLIPPAGE_OPTIONS.map((opt) => (
                  <label
                    key={opt.value}
                    className="flex items-center gap-1.5 cursor-pointer text-xs text-text-secondary"
                  >
                    <input
                      type="radio"
                      name="slippage"
                      value={opt.value}
                      checked={formSlippage === opt.value}
                      onChange={() => setFormSlippage(opt.value)}
                      className="accent-accent-blue"
                    />
                    {opt.label}
                  </label>
                ))}
              </div>
            </div>

            {/* ── 2. Cost Preview ── */}
            {isPreviewLoading && (
              <div className="flex items-center gap-2 text-text-muted text-xs py-2">
                <div className="w-3 h-3 border-2 border-accent-blue/30 border-t-accent-blue rounded-full animate-spin" />
                Computing costs...
              </div>
            )}

            {costPreview && !isPreviewLoading && (
              <div className="bg-bg-tertiary rounded-md border border-border/50 p-3">
                <div className="text-label text-text-muted uppercase tracking-wider mb-2">
                  Cost Preview
                </div>
                <table className="w-full text-xs">
                  <tbody className="divide-y divide-border/20">
                    <tr>
                      <td className="py-1 text-text-muted">Brokerage</td>
                      <td className="py-1 text-right font-mono tabular-nums text-text-secondary">
                        {formatCurrency(costPreview.brokerage)}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-1 text-text-muted">STT</td>
                      <td className="py-1 text-right font-mono tabular-nums text-text-secondary">
                        {formatCurrency(costPreview.stt)}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-1 text-text-muted">Exchange Charges</td>
                      <td className="py-1 text-right font-mono tabular-nums text-text-secondary">
                        {formatCurrency(costPreview.exchange_charges)}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-1 text-text-muted">GST</td>
                      <td className="py-1 text-right font-mono tabular-nums text-text-secondary">
                        {formatCurrency(costPreview.gst)}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-1 text-text-muted">SEBI Charges</td>
                      <td className="py-1 text-right font-mono tabular-nums text-text-secondary">
                        {formatCurrency(costPreview.sebi_charges)}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-1 text-text-muted">Stamp Duty</td>
                      <td className="py-1 text-right font-mono tabular-nums text-text-secondary">
                        {formatCurrency(costPreview.stamp_duty)}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-1 text-text-muted">Slippage</td>
                      <td className="py-1 text-right font-mono tabular-nums text-text-secondary">
                        {formatCurrency(costPreview.slippage_cost)}
                      </td>
                    </tr>
                    <tr className="border-t border-border/40">
                      <td className="py-1.5 text-text-secondary font-semibold">Total Cost</td>
                      <td className="py-1.5 text-right font-mono tabular-nums text-loss font-semibold">
                        {formatCurrency(costPreview.total_cost)}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-1.5 text-text-primary font-semibold">Net Amount</td>
                      <td className="py-1.5 text-right font-mono tabular-nums text-text-primary font-semibold">
                        {formatCurrency(costPreview.net_amount)}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}

            {/* Execute Error */}
            {executeError && (
              <div className="text-loss text-xs">{executeError}</div>
            )}

            {/* Execute Button */}
            <button
              type="submit"
              disabled={isExecuting}
              className={`btn w-full py-2 text-sm font-semibold rounded transition-colors
                         flex items-center justify-center gap-2
                         disabled:opacity-50 disabled:cursor-not-allowed
                         ${formType === 'BUY'
                           ? 'bg-profit text-white hover:bg-profit/90'
                           : 'bg-loss text-white hover:bg-loss/90'
                         }`}
            >
              {isExecuting && (
                <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              )}
              {isExecuting ? 'Executing...' : `Execute ${formType}`}
            </button>
          </form>
        </div>

        {/* ── 3. Trade History Table ── */}
        <div className="border-t border-border/50 pt-4">
          <div className="text-label text-text-muted uppercase tracking-wider mb-3">
            Trade History
          </div>

          {tradeList.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 gap-2">
              <span className="text-text-muted text-sm">No trades yet.</span>
              <span className="text-text-muted/60 text-xs">
                Execute a trade above to get started.
              </span>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th className="text-left">Date</th>
                    <th className="text-left">Ticker</th>
                    <th className="text-center">Type</th>
                    <th className="text-right">Qty</th>
                    <th className="text-right">Price</th>
                    <th className="text-right">Amount</th>
                    <th className="text-right">Total Cost</th>
                    <th className="text-right">Net Amount</th>
                  </tr>
                </thead>
                <tbody>
                  {tradeList.map((trade) => {
                    const isExpanded = expandedTradeId === trade.id;
                    return (
                      <Fragment key={trade.id}>
                        <tr
                          onClick={() => toggleExpand(trade.id)}
                          className="cursor-pointer hover:bg-bg-tertiary/50"
                        >
                          {/* Date */}
                          <td className="text-text-muted text-xs font-mono tabular-nums whitespace-nowrap">
                            {formatTimestamp(trade.executed_at)}
                          </td>
                          {/* Ticker */}
                          <td className="font-semibold text-text-primary">
                            {trade.ticker}
                          </td>
                          {/* Type */}
                          <td className="text-center">
                            <span className={trade.trade_type === 'BUY' ? 'badge badge-buy' : 'badge badge-sell'}>
                              {trade.trade_type}
                            </span>
                          </td>
                          {/* Qty */}
                          <td className="text-right text-text-primary font-mono tabular-nums">
                            {trade.quantity.toLocaleString('en-IN')}
                          </td>
                          {/* Price */}
                          <td className="text-right text-text-primary font-mono tabular-nums">
                            {formatCurrency(trade.price)}
                          </td>
                          {/* Amount */}
                          <td className="text-right text-text-secondary font-mono tabular-nums">
                            {formatCurrency(trade.total_amount)}
                          </td>
                          {/* Total Cost */}
                          <td className="text-right text-loss/80 font-mono tabular-nums">
                            {formatCurrency(trade.total_cost)}
                          </td>
                          {/* Net Amount */}
                          <td className="text-right text-text-primary font-mono tabular-nums font-semibold">
                            {formatCurrency(trade.net_amount)}
                          </td>
                        </tr>

                        {/* ── Expanded Cost Breakdown Row ── */}
                        {isExpanded && (
                          <tr key={`${trade.id}-details`}>
                            <td colSpan={8} className="bg-bg-tertiary/30 px-4 py-3">
                              <div className="text-label text-text-muted uppercase tracking-wider mb-2">
                                Cost Breakdown
                              </div>
                              <div className="grid grid-cols-3 lg:grid-cols-4 gap-x-6 gap-y-1.5 text-xs">
                                <div className="flex justify-between">
                                  <span className="text-text-muted">Brokerage</span>
                                  <span className="font-mono tabular-nums text-text-secondary">
                                    {formatCurrency(trade.brokerage)}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-text-muted">STT</span>
                                  <span className="font-mono tabular-nums text-text-secondary">
                                    {formatCurrency(trade.stt)}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-text-muted">Exchange</span>
                                  <span className="font-mono tabular-nums text-text-secondary">
                                    {formatCurrency(trade.exchange_charges)}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-text-muted">GST</span>
                                  <span className="font-mono tabular-nums text-text-secondary">
                                    {formatCurrency(trade.gst)}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-text-muted">SEBI</span>
                                  <span className="font-mono tabular-nums text-text-secondary">
                                    {formatCurrency(trade.sebi_charges)}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-text-muted">Stamp Duty</span>
                                  <span className="font-mono tabular-nums text-text-secondary">
                                    {formatCurrency(trade.stamp_duty)}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-text-muted">Slippage</span>
                                  <span className="font-mono tabular-nums text-text-secondary">
                                    {formatCurrency(trade.slippage_cost)}
                                  </span>
                                </div>
                                <div className="flex justify-between border-t border-border/30 pt-1">
                                  <span className="text-text-secondary font-semibold">Total</span>
                                  <span className="font-mono tabular-nums text-loss font-semibold">
                                    {formatCurrency(trade.total_cost)}
                                  </span>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
