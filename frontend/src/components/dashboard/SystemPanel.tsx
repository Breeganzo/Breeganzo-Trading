'use client';

import { useSystemHealth } from '@/hooks/useData';
import { formatTimestamp, timeAgo } from '@/lib/utils';
import { api } from '@/lib/api';
import { useState } from 'react';
import {
  Database,
  HardDrive,
  Wifi,
  Clock,
  RefreshCw,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Server,
  Activity,
  BarChart3 as BarChart,
} from 'lucide-react';

// ── Status indicator component ──
function StatusDot({ status }: { status: string }) {
  const color =
    status === 'healthy' || status === 'connected' || status === 'ok'
      ? 'bg-profit'
      : status === 'degraded' || status === 'stale'
        ? 'bg-warning'
        : 'bg-loss';

  return (
    <span className="relative flex h-2.5 w-2.5">
      <span className={`absolute inline-flex h-full w-full rounded-full ${color} opacity-40 animate-ping`} />
      <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${color}`} />
    </span>
  );
}

function StatusIcon({ status }: { status: string }) {
  if (status === 'healthy' || status === 'connected' || status === 'ok')
    return <CheckCircle2 size={16} className="text-profit" />;
  if (status === 'degraded' || status === 'stale')
    return <AlertTriangle size={16} className="text-warning" />;
  return <XCircle size={16} className="text-loss" />;
}

export default function SystemPanel() {
  const { data: health, error, isLoading, mutate } = useSystemHealth();
  const [validating, setValidating] = useState(false);
  const [validationResult, setValidationResult] = useState<any>(null);

  const handleValidate = async () => {
    setValidating(true);
    try {
      const result = await api.validateSystem();
      setValidationResult(result);
    } catch (e) {
      setValidationResult({ error: 'Validation failed' });
    } finally {
      setValidating(false);
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="panel">
          <div className="panel-header">
            <h2 className="panel-title">System Status</h2>
          </div>
          <div className="panel-body">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="h-24 bg-bg-tertiary rounded animate-pulse" />
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error || !health) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h2 className="panel-title">System Status</h2>
        </div>
        <div className="panel-body">
          <div className="flex items-center gap-3 text-loss">
            <XCircle size={20} />
            <span className="text-sm">Unable to reach backend. Is the server running?</span>
          </div>
        </div>
      </div>
    );
  }

  const overallStatus = health.status || 'unknown';

  return (
    <div className="space-y-4">
      {/* ── Header ── */}
      <div className="panel">
        <div className="panel-header flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Server size={18} className="text-accent-blue" />
            <h2 className="panel-title">System Status</h2>
            <StatusDot status={overallStatus} />
            <span className="text-label text-text-muted uppercase tracking-wider">
              {overallStatus}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => mutate()}
              className="btn btn-secondary text-label flex items-center gap-1.5"
            >
              <RefreshCw size={13} />
              Refresh
            </button>
            <button
              onClick={handleValidate}
              disabled={validating}
              className="btn btn-primary text-label flex items-center gap-1.5"
            >
              <Activity size={13} />
              {validating ? 'Validating...' : 'Deep Validate'}
            </button>
          </div>
        </div>
      </div>

      {/* ── Component Status Grid ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {/* Database */}
        <div className="panel">
          <div className="panel-body p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Database size={16} className="text-accent-cyan" />
                <span className="text-sm font-medium text-text-primary">Database</span>
              </div>
              <StatusIcon status={health.database || 'unknown'} />
            </div>
            <div className="space-y-1.5">
              <div className="flex justify-between text-label">
                <span className="text-text-muted">Status</span>
                <span className="text-text-primary capitalize">{health.database || 'Unknown'}</span>
              </div>
              <div className="flex justify-between text-label">
                <span className="text-text-muted">Provider</span>
                <span className="text-text-secondary">Supabase PostgreSQL</span>
              </div>
            </div>
          </div>
        </div>

        {/* Redis */}
        <div className="panel">
          <div className="panel-body p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <HardDrive size={16} className="text-loss" />
                <span className="text-sm font-medium text-text-primary">Redis Cache</span>
              </div>
              <StatusIcon status={health.redis || 'unknown'} />
            </div>
            <div className="space-y-1.5">
              <div className="flex justify-between text-label">
                <span className="text-text-muted">Status</span>
                <span className="text-text-primary capitalize">{health.redis || 'Unknown'}</span>
              </div>
              <div className="flex justify-between text-label">
                <span className="text-text-muted">Provider</span>
                <span className="text-text-secondary">Upstash</span>
              </div>
            </div>
          </div>
        </div>

        {/* Data Feed */}
        <div className="panel">
          <div className="panel-body p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Wifi size={16} className="text-profit" />
                <span className="text-sm font-medium text-text-primary">Market Data Feed</span>
              </div>
              <StatusIcon status={health.data_feed || 'unknown'} />
            </div>
            <div className="space-y-1.5">
              <div className="flex justify-between text-label">
                <span className="text-text-muted">Status</span>
                <span className="text-text-primary capitalize">{health.data_feed || 'Unknown'}</span>
              </div>
              <div className="flex justify-between text-label">
                <span className="text-text-muted">Source</span>
                <span className="text-text-secondary">Yahoo Finance</span>
              </div>
            </div>
          </div>
        </div>

        {/* Uptime */}
        <div className="panel">
          <div className="panel-body p-4">
            <div className="flex items-center gap-2 mb-3">
              <Clock size={16} className="text-warning" />
              <span className="text-sm font-medium text-text-primary">Uptime</span>
            </div>
            <div className="space-y-1.5">
              <div className="flex justify-between text-label">
                <span className="text-text-muted">Uptime</span>
                <span className="text-text-primary font-mono">
                  {health.uptime_seconds
                    ? `${Math.floor(health.uptime_seconds / 3600)}h ${Math.floor((health.uptime_seconds % 3600) / 60)}m`
                    : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between text-label">
                <span className="text-text-muted">Version</span>
                <span className="text-text-secondary font-mono">{health.version || '1.0.0'}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Model Freshness */}
        <div className="panel">
          <div className="panel-body p-4">
            <div className="flex items-center gap-2 mb-3">
              <Activity size={16} className="text-accent-indigo" />
              <span className="text-sm font-medium text-text-primary">Rankings</span>
            </div>
            <div className="space-y-1.5">
              <div className="flex justify-between text-label">
                <span className="text-text-muted">Last Computed</span>
                <span className="text-text-primary">
                  {health.rankings_last_computed
                    ? timeAgo(health.rankings_last_computed)
                    : 'Never'}
                </span>
              </div>
              <div className="flex justify-between text-label">
                <span className="text-text-muted">Freshness</span>
                <span className="text-text-secondary capitalize">
                  {health.model_freshness || 'Unknown'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Correlation Matrix */}
        <div className="panel">
          <div className="panel-body p-4">
            <div className="flex items-center gap-2 mb-3">
              <BarChart size={16} className="text-accent-blue" />
              <span className="text-sm font-medium text-text-primary">Correlation</span>
            </div>
            <div className="space-y-1.5">
              <div className="flex justify-between text-label">
                <span className="text-text-muted">Last Calculated</span>
                <span className="text-text-primary">
                  {health.correlation_last_computed
                    ? timeAgo(health.correlation_last_computed)
                    : 'Never'}
                </span>
              </div>
              <div className="flex justify-between text-label">
                <span className="text-text-muted">Freshness</span>
                <span className="text-text-secondary capitalize">
                  {health.correlation_freshness || 'Unknown'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Validation Results ── */}
      {validationResult && (
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Deep Validation Results</h3>
          </div>
          <div className="panel-body">
            {validationResult.error ? (
              <div className="text-loss text-sm">{validationResult.error}</div>
            ) : (
              <div className="space-y-2">
                {Object.entries(validationResult.checks || validationResult).map(
                  ([key, val]: [string, any]) => (
                    <div
                      key={key}
                      className="flex items-center justify-between py-2 border-b border-border last:border-0"
                    >
                      <span className="text-sm text-text-secondary capitalize">
                        {key.replace(/_/g, ' ')}
                      </span>
                      <div className="flex items-center gap-2">
                        {typeof val === 'object' ? (
                          <span className="text-sm text-text-primary font-mono">
                            {val.status || JSON.stringify(val)}
                          </span>
                        ) : (
                          <span className="text-sm text-text-primary">{String(val)}</span>
                        )}
                      </div>
                    </div>
                  )
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Architecture Info ── */}
      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">Infrastructure</h3>
        </div>
        <div className="panel-body">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-2">
            {[
              ['Frontend', 'Vercel (Next.js)'],
              ['Backend', 'Render (FastAPI)'],
              ['Database', 'Supabase PostgreSQL'],
              ['Cache', 'Upstash Redis'],
              ['Cron', 'GitHub Actions'],
              ['AI Engine', 'Groq (Llama 3.3 70B)'],
              ['Market Data', 'Yahoo Finance'],
              ['Auth', 'Google OAuth + TOTP 2FA'],
            ].map(([label, value]) => (
              <div key={label} className="flex justify-between py-1.5 border-b border-border/50">
                <span className="text-label text-text-muted">{label}</span>
                <span className="text-label text-text-primary font-mono">{value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
