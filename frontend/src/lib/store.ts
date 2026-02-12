import { create } from 'zustand';
import type { User, PortfolioSummary, RiskMetrics, RegimeData, SystemHealth, TickerData } from '@/types';

interface AppState {
  // Auth
  user: User | null;
  isAuthenticated: boolean;
  setUser: (user: User | null) => void;
  logout: () => void;

  // Portfolio
  portfolio: PortfolioSummary | null;
  setPortfolio: (data: PortfolioSummary | null) => void;

  // Risk
  riskMetrics: RiskMetrics | null;
  setRiskMetrics: (data: RiskMetrics | null) => void;

  // Regime
  regime: RegimeData | null;
  setRegime: (data: RegimeData | null) => void;

  // System
  systemHealth: SystemHealth | null;
  setSystemHealth: (data: SystemHealth | null) => void;

  // Ticker
  tickerData: Record<string, TickerData>;
  setTickerData: (data: Record<string, TickerData>) => void;
  updateTicker: (ticker: string, data: TickerData) => void;

  // UI
  activePanel: string;
  setActivePanel: (panel: string) => void;
  isMobileMenuOpen: boolean;
  setMobileMenuOpen: (open: boolean) => void;
}

export const useStore = create<AppState>((set) => ({
  // Auth
  user: null,
  isAuthenticated: false,
  setUser: (user) => set({ user, isAuthenticated: !!user }),
  logout: () => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('quantdesk_token');
    }
    set({ user: null, isAuthenticated: false });
  },

  // Portfolio
  portfolio: null,
  setPortfolio: (portfolio) => set({ portfolio }),

  // Risk
  riskMetrics: null,
  setRiskMetrics: (riskMetrics) => set({ riskMetrics }),

  // Regime
  regime: null,
  setRegime: (regime) => set({ regime }),

  // System
  systemHealth: null,
  setSystemHealth: (systemHealth) => set({ systemHealth }),

  // Ticker
  tickerData: {},
  setTickerData: (tickerData) => set({ tickerData }),
  updateTicker: (ticker, data) =>
    set((state) => ({
      tickerData: { ...state.tickerData, [ticker]: data },
    })),

  // UI
  activePanel: 'portfolio',
  setActivePanel: (activePanel) => set({ activePanel }),
  isMobileMenuOpen: false,
  setMobileMenuOpen: (isMobileMenuOpen) => set({ isMobileMenuOpen }),
}));
