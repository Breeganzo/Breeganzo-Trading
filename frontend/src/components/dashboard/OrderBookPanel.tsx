'use client';

import { useState, useCallback, useEffect } from 'react';
import { useAutoSignals, useOrders, useOrderSummary } from '@/hooks/useData';
import { api } from '@/lib/api';
import { formatCurrency, formatTimestamp } from '@/lib/utils';
import type { AutoSignal, AutoSignalCreate, Order, OrderCreate } from '@/types';

// ── Status filter tabs ──

type StatusFilter = 'all' | 'DRAFT' | 'CONFIRMED' | 'CANCELLED';

const STATUS_TABS: { value: StatusFilter; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'DRAFT', label: 'Draft' },
  { value: 'CONFIRMED', label: 'Confirmed' },
  { value: 'CANCELLED', label: 'Cancelled' },
];

// ── Status badge styling ──

function statusBadgeClass(status: string): string {
  switch (status) {
    case 'DRAFT':
      return 'badge badge-draft';
    case 'CONFIRMED':
      return 'badge badge-buy';
    case 'CANCELLED':
      return 'badge badge-sell';
    default:
      return 'badge';
  }
}

// ── Order type badge ──

function orderTypeBadge(type: string): string {
  return type === 'BUY' ? 'badge badge-buy' : 'badge badge-sell';
}

// ── Main Component ──

export default function OrderBookPanel() {
  // ── Data hooks ──
  const { data: orders, isLoading: ordersLoading, error: ordersError, mutate: mutateOrders } = useOrders();
  const { data: summary, isLoading: summaryLoading, mutate: mutateSummary } = useOrderSummary();
  const { data: autoSignals, mutate: mutateAutoSignals } = useAutoSignals('PENDING');

  // ── Filter state ──
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');

  // ── Form state ──
  const [formTicker, setFormTicker] = useState('');
  const [formType, setFormType] = useState<'BUY' | 'SELL'>('BUY');
  const [formQuantity, setFormQuantity] = useState('');
  const [formPrice, setFormPrice] = useState('');
  const [formNotes, setFormNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // ── Action loading states ──
  const [actionLoadingId, setActionLoadingId] = useState<string | null>(null);
  const [signalTicker, setSignalTicker] = useState('');
  const [signalAction, setSignalAction] = useState<'BUY' | 'SELL' | 'HOLD'>('BUY');
  const [signalQuantity, setSignalQuantity] = useState('1');
  const [signalLow, setSignalLow] = useState('');
  const [signalHigh, setSignalHigh] = useState('');
  const [signalSentimentMin, setSignalSentimentMin] = useState('');
  const [signalSentimentMax, setSignalSentimentMax] = useState('');
  const [signalNotes, setSignalNotes] = useState('');
  const [signalError, setSignalError] = useState<string | null>(null);
  const [signalBusy, setSignalBusy] = useState(false);

  // ── Form submit handler ──
  const handleSubmitOrder = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitError(null);

    const ticker = formTicker.trim().toUpperCase();
    const quantity = parseInt(formQuantity, 10);
    const targetPrice = parseFloat(formPrice);

    if (!ticker) {
      setSubmitError('Ticker is required.');
      return;
    }
    if (isNaN(quantity) || quantity <= 0) {
      setSubmitError('Valid quantity is required.');
      return;
    }
    if (isNaN(targetPrice) || targetPrice <= 0) {
      setSubmitError('Valid target price is required.');
      return;
    }

    const payload: OrderCreate = {
      ticker,
      order_type: formType,
      quantity,
      target_price: targetPrice,
    };
    if (formNotes.trim()) {
      payload.notes = formNotes.trim();
    }

    setIsSubmitting(true);
    try {
      await api.createOrder(payload);
      // Reset form
      setFormTicker('');
      setFormQuantity('');
      setFormPrice('');
      setFormNotes('');
      // Refresh data
      await Promise.all([mutateOrders(), mutateSummary()]);
    } catch (err: any) {
      setSubmitError(err?.message || 'Failed to create order.');
    } finally {
      setIsSubmitting(false);
    }
  }, [formTicker, formType, formQuantity, formPrice, formNotes, mutateOrders, mutateSummary]);

  // ── Confirm order ──
  const handleConfirm = useCallback(async (orderId: string) => {
    setActionLoadingId(orderId);
    try {
      await api.confirmOrder(orderId);
      await Promise.all([mutateOrders(), mutateSummary()]);
    } catch {
      // Silent fail; UI will reflect stale state
    } finally {
      setActionLoadingId(null);
    }
  }, [mutateOrders, mutateSummary]);

  // ── Cancel order ──
  const handleCancel = useCallback(async (orderId: string) => {
    setActionLoadingId(orderId);
    try {
      await api.cancelOrder(orderId);
      await Promise.all([mutateOrders(), mutateSummary()]);
    } catch {
      // Silent fail
    } finally {
      setActionLoadingId(null);
    }
  }, [mutateOrders, mutateSummary]);

  const handleCreateSignal = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setSignalError(null);
    const ticker = signalTicker.trim().toUpperCase();
    const low = signalLow.trim() ? Number(signalLow) : undefined;
    const high = signalHigh.trim() ? Number(signalHigh) : undefined;
    const qty = Math.max(1, parseInt(signalQuantity || '1', 10) || 1);
    const sentimentMin = signalSentimentMin.trim() ? Number(signalSentimentMin) : undefined;
    const sentimentMax = signalSentimentMax.trim() ? Number(signalSentimentMax) : undefined;

    if (!ticker) {
      setSignalError('Ticker is required for auto signal.');
      return;
    }
    if (!Number.isFinite(low as number) && !Number.isFinite(high as number)) {
      setSignalError('Set trigger low or trigger high.');
      return;
    }
    if (Number.isFinite(low as number) && Number(low) <= 0) {
      setSignalError('Trigger low must be > 0.');
      return;
    }
    if (Number.isFinite(high as number) && Number(high) <= 0) {
      setSignalError('Trigger high must be > 0.');
      return;
    }

    const payload: AutoSignalCreate = {
      ticker,
      action: signalAction,
      quantity: signalAction === 'HOLD' ? 1 : qty,
      trigger_price_low: Number.isFinite(low as number) ? low : undefined,
      trigger_price_high: Number.isFinite(high as number) ? high : undefined,
      sentiment_min: Number.isFinite(sentimentMin as number) ? sentimentMin : undefined,
      sentiment_max: Number.isFinite(sentimentMax as number) ? sentimentMax : undefined,
      notes: signalNotes.trim() || undefined,
      source: 'dashboard_manual',
    };

    setSignalBusy(true);
    try {
      await api.createAutoSignal(payload);
      setSignalTicker('');
      setSignalQuantity('1');
      setSignalLow('');
      setSignalHigh('');
      setSignalSentimentMin('');
      setSignalSentimentMax('');
      setSignalNotes('');
      await mutateAutoSignals();
    } catch (err: any) {
      setSignalError(err?.message || 'Failed to create auto signal.');
    } finally {
      setSignalBusy(false);
    }
  }, [
    mutateAutoSignals,
    signalAction,
    signalHigh,
    signalLow,
    signalNotes,
    signalQuantity,
    signalSentimentMax,
    signalSentimentMin,
    signalTicker,
  ]);

  const handleDeleteSignal = useCallback(async (signalId: string) => {
    setSignalBusy(true);
    try {
      await api.deleteAutoSignal(signalId);
      await mutateAutoSignals();
    } catch (err: any) {
      setSignalError(err?.message || 'Failed to delete auto signal.');
    } finally {
      setSignalBusy(false);
    }
  }, [mutateAutoSignals]);

  const handleProcessSignals = useCallback(async () => {
    setSignalBusy(true);
    setSignalError(null);
    try {
      await api.processAutoSignals(50);
      await Promise.all([mutateAutoSignals(), mutateOrders(), mutateSummary()]);
    } catch (err: any) {
      setSignalError(err?.message || 'Failed to process auto signals.');
    } finally {
      setSignalBusy(false);
    }
  }, [mutateAutoSignals, mutateOrders, mutateSummary]);

  // ── Filter orders ──
  const filteredOrders: Order[] = (() => {
    if (!orders) return [];
    const list: Order[] = Array.isArray(orders) ? orders : (orders as any)?.orders ?? [];
    if (statusFilter === 'all') return list;
    return list.filter((o) => o.status === statusFilter);
  })();
  const signalRows: AutoSignal[] = Array.isArray(autoSignals)
    ? autoSignals
    : ((autoSignals as any)?.signals ?? []);

  const isLoading = ordersLoading || summaryLoading;

  useEffect(() => {
    if (!signalRows.length) return;
    const timer = setInterval(() => {
      if (!signalBusy) {
        void handleProcessSignals();
      }
    }, 10000);
    return () => clearInterval(timer);
  }, [handleProcessSignals, signalBusy, signalRows.length]);

  // ── Loading state ──
  if (isLoading) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Order Book</h2>
        </div>
        <div className="panel-body">
          <div className="flex items-center justify-center py-12">
            <div className="flex items-center gap-3 text-text-muted text-sm">
              <div className="w-4 h-4 border-2 border-accent-blue/30 border-t-accent-blue rounded-full animate-spin" />
              Loading orders...
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── Error state ──
  if (ordersError) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Order Book</h2>
        </div>
        <div className="panel-body">
          <div className="flex items-center justify-center py-12 text-loss text-sm">
            Failed to load orders.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      {/* ── Panel Header ── */}
      <div className="panel-header">
        <h2 className="panel-title">Order Book</h2>
      </div>

      <div className="panel-body space-y-5">
        {/* ── 1. Order Summary Row ── */}
        {summary && (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="metric-card">
              <div className="metric-label">Draft Orders</div>
              <div className="metric-value text-text-primary">
                {summary.draft_count}
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Confirmed Orders</div>
              <div className="metric-value text-profit">
                {summary.confirmed_count}
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Pending Buy Value</div>
              <div className="metric-value text-profit">
                {formatCurrency(summary.total_pending_buy_value)}
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Pending Sell Value</div>
              <div className="metric-value text-loss">
                {formatCurrency(summary.total_pending_sell_value)}
              </div>
            </div>
          </div>
        )}

        {/* ── 2. New Order Form ── */}
        <div className="border-t border-border/50 pt-4">
          <div className="text-label text-text-muted uppercase tracking-wider mb-3">
            New Order
          </div>
          <form onSubmit={handleSubmitOrder} className="space-y-3">
            {/* Ticker */}
            <div>
              <input
                type="text"
                className="input"
                placeholder="Ticker (e.g. RELIANCE)"
                value={formTicker}
                onChange={(e) => setFormTicker(e.target.value)}
              />
            </div>

            {/* Order Type Tabs */}
            <div className="flex gap-1">
              <button
                type="button"
                onClick={() => setFormType('BUY')}
                className={`flex-1 py-1.5 text-xs font-semibold rounded transition-colors ${
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
                className={`flex-1 py-1.5 text-xs font-semibold rounded transition-colors ${
                  formType === 'SELL'
                    ? 'bg-loss/20 text-loss border border-loss/40'
                    : 'bg-bg-tertiary text-text-muted border border-border/30 hover:text-text-secondary'
                }`}
              >
                SELL
              </button>
            </div>

            {/* Quantity & Target Price */}
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
                placeholder="Target Price"
                min="0.01"
                step="0.01"
                value={formPrice}
                onChange={(e) => setFormPrice(e.target.value)}
              />
            </div>

            {/* Notes */}
            <textarea
              className="input"
              placeholder="Notes (optional)"
              rows={2}
              value={formNotes}
              onChange={(e) => setFormNotes(e.target.value)}
            />

            {/* Submit Error */}
            {submitError && (
              <div className="text-loss text-xs">{submitError}</div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isSubmitting}
              className="btn w-full py-2 text-sm font-semibold rounded
                         bg-accent-blue text-white hover:bg-accent-blue/90
                         disabled:opacity-50 disabled:cursor-not-allowed transition-colors
                         flex items-center justify-center gap-2"
            >
              {isSubmitting && (
                <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              )}
              {isSubmitting ? 'Creating...' : 'Create Draft Order'}
            </button>
          </form>
        </div>

        {/* ── 3. Auto Signal Queue ── */}
        <div className="border-t border-border/50 pt-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-label text-text-muted uppercase tracking-wider">
              Auto Signal Queue (DB Trigger)
            </div>
            <button
              type="button"
              onClick={handleProcessSignals}
              disabled={signalBusy}
              className="px-3 py-1 text-xs font-semibold rounded
                         bg-accent-blue/15 text-accent-blue border border-accent-blue/30
                         hover:bg-accent-blue/25 disabled:opacity-50"
            >
              {signalBusy ? 'Processing...' : 'Process Triggers'}
            </button>
          </div>

          <form onSubmit={handleCreateSignal} className="grid grid-cols-1 md:grid-cols-4 gap-2">
            <input
              type="text"
              className="input"
              placeholder="Ticker (e.g. INFY)"
              value={signalTicker}
              onChange={(e) => setSignalTicker(e.target.value)}
            />
            <select
              className="input"
              value={signalAction}
              onChange={(e) => setSignalAction(e.target.value as 'BUY' | 'SELL' | 'HOLD')}
            >
              <option value="BUY">BUY</option>
              <option value="SELL">SELL</option>
              <option value="HOLD">HOLD</option>
            </select>
            <input
              type="number"
              className="input"
              placeholder="Qty"
              min="1"
              step="1"
              value={signalQuantity}
              onChange={(e) => setSignalQuantity(e.target.value)}
              disabled={signalAction === 'HOLD'}
            />
            <input
              type="text"
              className="input"
              placeholder="Notes (optional)"
              value={signalNotes}
              onChange={(e) => setSignalNotes(e.target.value)}
            />
            <input
              type="number"
              className="input"
              placeholder="Trigger Low"
              min="0.01"
              step="0.01"
              value={signalLow}
              onChange={(e) => setSignalLow(e.target.value)}
            />
            <input
              type="number"
              className="input"
              placeholder="Trigger High"
              min="0.01"
              step="0.01"
              value={signalHigh}
              onChange={(e) => setSignalHigh(e.target.value)}
            />
            <input
              type="number"
              className="input"
              placeholder="Sentiment Min (-1..1)"
              min="-1"
              max="1"
              step="0.01"
              value={signalSentimentMin}
              onChange={(e) => setSignalSentimentMin(e.target.value)}
            />
            <div className="flex gap-2">
              <input
                type="number"
                className="input"
                placeholder="Sentiment Max (-1..1)"
                min="-1"
                max="1"
                step="0.01"
                value={signalSentimentMax}
                onChange={(e) => setSignalSentimentMax(e.target.value)}
              />
              <button
                type="submit"
                disabled={signalBusy}
                className="px-3 py-1 text-xs font-semibold rounded
                           bg-profit/15 text-profit border border-profit/30
                           hover:bg-profit/25 disabled:opacity-50"
              >
                Add Trigger
              </button>
            </div>
          </form>
          {signalError && <div className="text-loss text-xs">{signalError}</div>}

          {signalRows.length === 0 ? (
            <div className="text-text-muted text-sm">No pending auto signals.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th className="text-left">Ticker</th>
                    <th className="text-center">Action</th>
                    <th className="text-right">Qty</th>
                    <th className="text-right">Trigger Low</th>
                    <th className="text-right">Trigger High</th>
                    <th className="text-right">Sentiment</th>
                    <th className="text-left">Created</th>
                    <th className="text-center">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {signalRows.map((row) => (
                    <tr key={row.id}>
                      <td className="font-semibold text-text-primary">{row.ticker}</td>
                      <td className="text-center">
                        <span className={row.action === 'BUY' ? 'badge badge-buy' : row.action === 'SELL' ? 'badge badge-sell' : 'badge badge-draft'}>
                          {row.action}
                        </span>
                      </td>
                      <td className="text-right font-mono tabular-nums">{row.quantity}</td>
                      <td className="text-right font-mono tabular-nums">{formatCurrency(row.trigger_price_low)}</td>
                      <td className="text-right font-mono tabular-nums">{formatCurrency(row.trigger_price_high)}</td>
                      <td className="text-right font-mono tabular-nums">
                        {row.sentiment_min != null || row.sentiment_max != null
                          ? `[${row.sentiment_min ?? '-1'} .. ${row.sentiment_max ?? '1'}]`
                          : 'Any'}
                      </td>
                      <td className="text-text-muted text-xs font-mono tabular-nums">{formatTimestamp(row.created_at)}</td>
                      <td className="text-center">
                        <button
                          type="button"
                          onClick={() => handleDeleteSignal(row.id)}
                          disabled={signalBusy}
                          className="px-2 py-0.5 text-xs font-medium rounded
                                     bg-loss/10 text-loss border border-loss/30
                                     hover:bg-loss/20 disabled:opacity-50"
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* ── 4. Orders Table with Tabs ── */}
        <div className="border-t border-border/50 pt-4">
          {/* Tab Buttons */}
          <div className="flex gap-1 mb-3">
            {STATUS_TABS.map((tab) => (
              <button
                key={tab.value}
                onClick={() => setStatusFilter(tab.value)}
                className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
                  statusFilter === tab.value
                    ? 'bg-accent-blue/15 text-accent-blue border border-accent-blue/30'
                    : 'bg-bg-tertiary text-text-muted border border-border/30 hover:text-text-secondary'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Orders Table */}
          {filteredOrders.length === 0 ? (
            <div className="flex items-center justify-center py-8">
              <span className="text-text-muted text-sm">No orders found.</span>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th className="text-left">Ticker</th>
                    <th className="text-center">Type</th>
                    <th className="text-right">Qty</th>
                    <th className="text-right">Target Price</th>
                    <th className="text-center">Status</th>
                    <th className="text-left">Created</th>
                    <th className="text-center">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredOrders.map((order) => {
                    const isCancelled = order.status === 'CANCELLED';
                    const isDraft = order.status === 'DRAFT';
                    const isActionLoading = actionLoadingId === order.id;
                    const rowClass = isCancelled ? 'opacity-50' : '';

                    return (
                      <tr key={order.id} className={rowClass}>
                        {/* Ticker */}
                        <td className="font-semibold text-text-primary">
                          {order.ticker}
                        </td>
                        {/* Type */}
                        <td className="text-center">
                          <span className={orderTypeBadge(order.order_type)}>
                            {order.order_type}
                          </span>
                        </td>
                        {/* Qty */}
                        <td className="text-right text-text-primary font-mono tabular-nums">
                          {order.quantity.toLocaleString('en-IN')}
                        </td>
                        {/* Target Price */}
                        <td className="text-right text-text-primary font-mono tabular-nums">
                          {formatCurrency(order.target_price)}
                        </td>
                        {/* Status */}
                        <td className="text-center">
                          <span className={statusBadgeClass(order.status)}>
                            {order.status}
                          </span>
                        </td>
                        {/* Created */}
                        <td className="text-text-muted text-xs font-mono tabular-nums">
                          {formatTimestamp(order.created_at)}
                          {order.status === 'CONFIRMED' && order.confirmed_at && (
                            <div className="text-profit/80 mt-0.5">
                              Confirmed: {formatTimestamp(order.confirmed_at)}
                            </div>
                          )}
                        </td>
                        {/* Actions */}
                        <td className="text-center">
                          {isDraft && (
                            <div className="flex items-center justify-center gap-1.5">
                              <button
                                onClick={() => handleConfirm(order.id)}
                                disabled={isActionLoading}
                                className="px-2 py-0.5 text-xs font-medium rounded
                                           bg-profit/10 text-profit border border-profit/30
                                           hover:bg-profit/20 disabled:opacity-50
                                           disabled:cursor-not-allowed transition-colors"
                              >
                                {isActionLoading ? '...' : 'Confirm'}
                              </button>
                              <button
                                onClick={() => handleCancel(order.id)}
                                disabled={isActionLoading}
                                className="px-2 py-0.5 text-xs font-medium rounded
                                           bg-loss/10 text-loss border border-loss/30
                                           hover:bg-loss/20 disabled:opacity-50
                                           disabled:cursor-not-allowed transition-colors"
                              >
                                {isActionLoading ? '...' : 'Cancel'}
                              </button>
                            </div>
                          )}
                          {!isDraft && (
                            <span className="text-text-muted text-xs">--</span>
                          )}
                        </td>
                      </tr>
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
