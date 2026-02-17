'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import DashboardLayout from '@/components/layout/DashboardLayout';
import PortfolioPanel from '@/components/dashboard/PortfolioPanel';
import RankingsPanel from '@/components/dashboard/RankingsPanel';
import RiskPanel from '@/components/dashboard/RiskPanel';
import TradesPanel from '@/components/dashboard/TradesPanel';
import OrderBookPanel from '@/components/dashboard/OrderBookPanel';
import SystemPanel from '@/components/dashboard/SystemPanel';
import StocksPanel from '@/components/dashboard/StocksPanel';
import TopPicksPanel from '@/components/dashboard/TopPicksPanel';
import AdvisorPanel from '@/components/dashboard/AdvisorPanel';
import ExpectedActualPanel from '@/components/dashboard/ExpectedActualPanel';
import { useStore } from '@/lib/store';
import { api } from '@/lib/api';
import {
  usePortfolio,
  useRegime,
  useSystemHealth,
  useTickerStream,
} from '@/hooks/useData';

export default function DashboardPage() {
  const activePanel = useStore((s) => s.activePanel);
  const setUser = useStore((s) => s.setUser);
  const router = useRouter();
  const authBypass = process.env.NEXT_PUBLIC_AUTH_BYPASS_LOCAL === 'true';

  // Bootstrap global market/system streams so top bar always has live values.
  useTickerStream();
  usePortfolio();
  useRegime();
  useSystemHealth();

  // Verify auth on mount
  useEffect(() => {
    const token = api.getToken();
    if (!token && !authBypass) {
      router.push('/login');
      return;
    }
    api
      .getMe()
      .then((data) => setUser(data))
      .catch(() => {
        if (!authBypass) {
          api.clearToken();
          router.push('/login');
        }
      });
  }, [setUser, router, authBypass]);

  const renderPanel = () => {
    switch (activePanel) {
      case 'portfolio':
        return <PortfolioPanel />;
      case 'stocks':
        return <StocksPanel />;
      case 'top_picks':
        return <TopPicksPanel />;
      case 'advisor':
        return <AdvisorPanel />;
      case 'expected_actual':
        return <ExpectedActualPanel />;
      case 'rankings':
        return <RankingsPanel />;
      case 'risk':
        return <RiskPanel />;
      case 'trades':
        return <TradesPanel />;
      case 'orders':
        return <OrderBookPanel />;
      case 'system':
        return <SystemPanel />;
      default:
        return <PortfolioPanel />;
    }
  };

  return (
    <DashboardLayout>
      {renderPanel()}
    </DashboardLayout>
  );
}
