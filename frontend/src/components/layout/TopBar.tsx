'use client';

import { useMemo } from 'react';
import Link from 'next/link';
import { Menu } from 'lucide-react';
import { useStore } from '@/lib/store';
import { formatCurrency, pnlColor, regimeColor, regimeLabel } from '@/lib/utils';

// ── Regime badge background mapping ──
function regimeBgClass(regime: string): string {
  switch (regime) {
    case 'bull':
      return 'bg-regime-bull/15 border-regime-bull/30';
    case 'bear':
      return 'bg-regime-bear/15 border-regime-bear/30';
    case 'high_vol':
      return 'bg-regime-high_vol/15 border-regime-high_vol/30';
    case 'low_vol':
      return 'bg-regime-low_vol/15 border-regime-low_vol/30';
    default:
      return 'bg-neutral/15 border-neutral/30';
  }
}

// ── System health dot color ──
function healthDotClass(status: string | undefined): string {
  switch (status) {
    case 'healthy':
      return 'status-dot status-healthy';
    case 'degraded':
      return 'status-dot status-degraded';
    case 'down':
    case 'unhealthy':
      return 'status-dot status-down';
    default:
      return 'status-dot bg-neutral';
  }
}

export default function TopBar() {
  const user = useStore((s) => s.user);
  const portfolio = useStore((s) => s.portfolio);
  const regime = useStore((s) => s.regime);
  const systemHealth = useStore((s) => s.systemHealth);
  const tickerData = useStore((s) => s.tickerData);
  const isMobileMenuOpen = useStore((s) => s.isMobileMenuOpen);
  const setMobileMenuOpen = useStore((s) => s.setMobileMenuOpen);

  // Build ticker items array; duplicate for seamless infinite scroll
  const tickerItems = useMemo(() => {
    const items = Object.values(tickerData);
    if (items.length === 0) return [];
    // Duplicate the list so the animation can loop without a gap
    return [...items, ...items];
  }, [tickerData]);

  const userInitial = user?.name
    ? user.name.charAt(0).toUpperCase()
    : user?.email
      ? user.email.charAt(0).toUpperCase()
      : 'U';

  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-12 bg-bg-secondary border-b border-border flex items-center select-none">
      {/* ── Left: Logo + Mobile hamburger ── */}
      <div className="flex items-center gap-2 px-3 shrink-0">
        {/* Mobile hamburger */}
        <button
          className="lg:hidden p-1 rounded hover:bg-bg-hover text-text-secondary transition-colors"
          onClick={() => setMobileMenuOpen(!isMobileMenuOpen)}
          aria-label="Toggle sidebar"
        >
          <Menu size={18} />
        </button>

        {/* Logo */}
        <div className="flex items-center gap-1.5">
          <span className="font-mono font-bold text-accent-blue text-base tracking-tight">
            QD
          </span>
          <span className="hidden md:inline text-text-secondary text-xs font-medium tracking-wide">
            QuantDesk Pro
          </span>
        </div>
        <div className="hidden xl:flex items-center gap-1 ml-2">
          <Link href="/stocks" className="text-label text-text-muted hover:text-text-primary px-2 py-1 rounded hover:bg-bg-hover">
            Stocks
          </Link>
          <Link href="/top-picks" className="text-label text-text-muted hover:text-text-primary px-2 py-1 rounded hover:bg-bg-hover">
            Top Picks
          </Link>
          <Link href="/advisor" className="text-label text-text-muted hover:text-text-primary px-2 py-1 rounded hover:bg-bg-hover">
            Advisor
          </Link>
          <Link href="/expected-vs-actual" className="text-label text-text-muted hover:text-text-primary px-2 py-1 rounded hover:bg-bg-hover">
            Expected vs Actual
          </Link>
        </div>
      </div>

      {/* ── Center: Live ticker strip ── */}
      <div className="flex-1 overflow-hidden mx-3">
        {tickerItems.length > 0 ? (
          <div className="animate-ticker ticker-strip">
            {tickerItems.map((t, idx) => {
              const isPositive = t.change_pct >= 0;
              const colorClass = isPositive ? 'text-profit' : 'text-loss';
              const arrow = isPositive ? '\u25B2' : '\u25BC';

              return (
                <div key={`${t.ticker}-${idx}`} className="ticker-item">
                  <span className="text-text-primary font-semibold">
                    {t.ticker}
                  </span>
                  <span className="text-text-secondary tabular-nums">
                    {t.price.toLocaleString('en-IN', {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2,
                    })}
                  </span>
                  <span className={`${colorClass} tabular-nums`}>
                    {arrow} {Math.abs(t.change_pct).toFixed(2)}%
                  </span>
                </div>
              );
            })}
          </div>
        ) : (
          <span className="text-text-muted text-data font-mono">
            Awaiting market data...
          </span>
        )}
      </div>

      {/* ── Right: Status indicators ── */}
      <div className="flex items-center gap-3 px-3 shrink-0">
        {/* Market regime badge */}
        {regime && (
          <span
            className={`badge border text-label ${regimeBgClass(regime.regime)} ${regimeColor(regime.regime)}`}
          >
            {regimeLabel(regime.regime)}
          </span>
        )}

        {/* Portfolio value */}
        <div className="hidden sm:flex flex-col items-end leading-none">
          <span className="text-label text-text-muted">Portfolio</span>
          <span className="text-data font-mono font-semibold text-text-primary tabular-nums">
            {formatCurrency(portfolio?.total_value)}
          </span>
        </div>

        {/* Day PnL */}
        <div className="hidden sm:flex flex-col items-end leading-none">
          <span className="text-label text-text-muted">Day P&L</span>
          <span
            className={`text-data font-mono font-semibold tabular-nums ${pnlColor(portfolio?.day_pnl)}`}
          >
            {formatCurrency(portfolio?.day_pnl)}
          </span>
        </div>

        {/* System health dot */}
        <div className="flex items-center" title={`System: ${systemHealth?.status ?? 'unknown'}`}>
          <span className={healthDotClass(systemHealth?.status)} />
        </div>

        {/* User avatar */}
        <div
          className="w-7 h-7 rounded-full bg-accent-blue/20 border border-accent-blue/40 flex items-center justify-center text-accent-blue text-label font-semibold cursor-pointer hover:bg-accent-blue/30 transition-colors"
          title={user?.name ?? user?.email ?? 'User'}
        >
          {userInitial}
        </div>
      </div>
    </header>
  );
}
