import { clsx, type ClassValue } from 'clsx';

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function formatCurrency(value: number | null | undefined, decimals: number = 2): string {
  if (value == null || isNaN(value)) return '--';
  const absVal = Math.abs(value);
  if (absVal >= 10000000) return `${(value / 10000000).toFixed(2)} Cr`;
  if (absVal >= 100000) return `${(value / 100000).toFixed(2)} L`;
  return `₹${value.toLocaleString('en-IN', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
}

export function formatNumber(value: number | null | undefined, decimals: number = 2): string {
  if (value == null || isNaN(value)) return '--';
  return value.toFixed(decimals);
}

export function formatPercent(value: number | null | undefined, decimals: number = 2): string {
  if (value == null || isNaN(value)) return '--';
  const sign = value > 0 ? '+' : '';
  return `${sign}${(value * 100).toFixed(decimals)}%`;
}

export function formatPercentRaw(value: number | null | undefined, decimals: number = 2): string {
  if (value == null || isNaN(value)) return '--';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(decimals)}%`;
}

export function formatVolume(value: number | null | undefined): string {
  if (value == null || isNaN(value)) return '--';
  if (value >= 10000000) return `${(value / 10000000).toFixed(2)} Cr`;
  if (value >= 100000) return `${(value / 100000).toFixed(1)} L`;
  if (value >= 1000) return `${(value / 1000).toFixed(1)} K`;
  return value.toString();
}

export function pnlColor(value: number | null | undefined): string {
  if (value == null || isNaN(value) || value === 0) return 'text-text-secondary';
  return value > 0 ? 'pnl-positive' : 'pnl-negative';
}

export function regimeColor(regime: string): string {
  switch (regime) {
    case 'bull': return 'text-profit';
    case 'bear': return 'text-loss';
    case 'high_vol': return 'text-warning';
    case 'low_vol': return 'text-accent-blue';
    default: return 'text-text-muted';
  }
}

export function regimeLabel(regime: string): string {
  switch (regime) {
    case 'bull': return 'BULL';
    case 'bear': return 'BEAR';
    case 'high_vol': return 'HIGH VOL';
    case 'low_vol': return 'LOW VOL';
    default: return 'UNKNOWN';
  }
}

export function formatTimestamp(ts: string | null | undefined): string {
  if (!ts) return '--';
  const d = new Date(ts);
  return d.toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false });
}

export function timeAgo(ts: string | null | undefined): string {
  if (!ts) return '--';
  const now = Date.now();
  const then = new Date(ts).getTime();
  const diff = Math.floor((now - then) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}
