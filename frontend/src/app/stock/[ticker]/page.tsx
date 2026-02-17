'use client';

import Link from 'next/link';
import { useParams, useRouter } from 'next/navigation';
import { useEffect } from 'react';
import DashboardLayout from '@/components/layout/DashboardLayout';
import { api } from '@/lib/api';
import { useStockDetail } from '@/hooks/useData';
import { formatCurrency, formatPercentRaw, formatTimestamp, pnlColor } from '@/lib/utils';

export default function StockDetailPage() {
  const params = useParams<{ ticker: string }>();
  const router = useRouter();
  const authBypass = process.env.NEXT_PUBLIC_AUTH_BYPASS_LOCAL === 'true';
  const ticker = decodeURIComponent(String(params?.ticker ?? '')).toUpperCase();
  const { data, isLoading, error, mutate } = useStockDetail(ticker || null);

  useEffect(() => {
    const token = api.getToken();
    if (!token && !authBypass) {
      router.push('/login');
    }
  }, [authBypass, router]);

  return (
    <DashboardLayout>
      <div className="space-y-4">
        <div className="panel">
          <div className="panel-header">
            <h2 className="panel-title">Stock Detail · {ticker}</h2>
            <div className="flex items-center gap-2">
              <button className="btn btn-secondary" onClick={() => void mutate()}>Refresh</button>
              <Link className="btn btn-secondary" href="/dashboard">Back</Link>
            </div>
          </div>
          <div className="panel-body">
            {isLoading && <div className="text-sm text-text-muted py-8 text-center">Loading stock detail...</div>}
            {error && <div className="text-sm text-loss py-8 text-center">Failed to load stock detail.</div>}

            {!isLoading && !error && data && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="metric-card">
                    <div className="metric-label">Current Price</div>
                    <div className="metric-value">{formatCurrency(data.current_price)}</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Open Price</div>
                    <div className="metric-value">{formatCurrency(data.open_price)}</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Strategy @ Open</div>
                    <div className="metric-value">{formatCurrency(data.strategy_price_at_open)}</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">AI Predicted</div>
                    <div className="metric-value">{formatCurrency(data.ai_predicted_price)}</div>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="metric-card">
                    <div className="metric-label">Strategy Return</div>
                    <div className={`metric-value ${pnlColor(data.strategy_return_pct)}`}>
                      {formatPercentRaw(data.strategy_return_pct)}
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">AI Return</div>
                    <div className={`metric-value ${pnlColor(data.ai_return_pct)}`}>
                      {formatPercentRaw(data.ai_return_pct)}
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Confidence</div>
                    <div className="metric-value">{Number(data.confidence).toFixed(1)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Agreement</div>
                    <div className="metric-value">{Number(data.agreement).toFixed(1)}%</div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div className="metric-card">
                    <div className="metric-label">Entry Range</div>
                    <div className="metric-value">
                      {formatCurrency(data.entry_range_low)} - {formatCurrency(data.entry_range_high)}
                    </div>
                    <div className="text-label text-text-muted mt-1">
                      Strategy Signal: {data.strategy_signal} · AI Signal: {data.ai_signal}
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Session Stats</div>
                    <div className="text-data font-mono tabular-nums text-text-primary">
                      Prev Close: {formatCurrency(data.prev_close)}<br />
                      Day High: {formatCurrency(data.day_high)}<br />
                      Day Low: {formatCurrency(data.day_low)}<br />
                      Volume: {Number(data.volume || 0).toLocaleString('en-IN')}
                    </div>
                  </div>
                </div>

                <div className="metric-card">
                  <div className="metric-label">Indicators</div>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-data font-mono tabular-nums">
                    {Object.entries(data.indicators || {}).map(([k, v]) => (
                      <div key={k} className="text-text-primary">
                        {k}: {String(v)}
                      </div>
                    ))}
                  </div>
                </div>

                <p className="text-label text-text-muted">
                  Captured: {formatTimestamp(data.captured_at)} · Sentiment Score: {Number(data.sentiment_score || 0).toFixed(3)}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
