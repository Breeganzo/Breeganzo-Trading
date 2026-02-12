'use client';

import { useEffect, useRef, useCallback } from 'react';
import useSWR from 'swr';
import { api } from '@/lib/api';
import { useStore } from '@/lib/store';
import type { TickerData } from '@/types';

// ── SWR Fetcher Wrappers ──

export function usePortfolio() {
  const { setPortfolio } = useStore();
  const result = useSWR('portfolio', () => api.getPortfolio(), {
    refreshInterval: 30000,
    onSuccess: (data) => setPortfolio(data),
    revalidateOnFocus: false,
  });
  return result;
}

export function useRiskMetrics() {
  const { setRiskMetrics } = useStore();
  const result = useSWR('risk', () => api.getRiskMetrics(), {
    refreshInterval: 120000,
    onSuccess: (data) => setRiskMetrics(data),
    revalidateOnFocus: false,
  });
  return result;
}

export function useRegime() {
  const { setRegime } = useStore();
  const result = useSWR('regime', () => api.getRegime(), {
    refreshInterval: 300000,
    onSuccess: (data) => setRegime(data),
    revalidateOnFocus: false,
  });
  return result;
}

export function useCorrelation() {
  return useSWR('correlation', () => api.getCorrelation(), {
    refreshInterval: 3600000,
    revalidateOnFocus: false,
  });
}

export function useRankings(category?: string) {
  const key = category ? `rankings-${category}` : 'rankings-all';
  const fetcher = category
    ? () => api.getRankingsByCategory(category)
    : () => api.getAllRankings();
  return useSWR(key, fetcher, {
    refreshInterval: 600000,
    revalidateOnFocus: false,
  });
}

export function useTrades(ticker?: string) {
  return useSWR(
    ticker ? `trades-${ticker}` : 'trades',
    () => api.getTrades(ticker),
    { revalidateOnFocus: false }
  );
}

export function useOrders(status?: string) {
  return useSWR(
    status ? `orders-${status}` : 'orders',
    () => api.getOrders(status),
    { revalidateOnFocus: false }
  );
}

export function useOrderSummary() {
  return useSWR('order-summary', () => api.getOrderSummary(), {
    refreshInterval: 30000,
    revalidateOnFocus: false,
  });
}

export function useSystemHealth() {
  const { setSystemHealth } = useStore();
  return useSWR('system-health', () => api.getSystemHealth(), {
    refreshInterval: 60000,
    onSuccess: (data) => setSystemHealth(data),
    revalidateOnFocus: false,
  });
}

export function useMarketStatus() {
  return useSWR('market-status', () => api.getMarketStatus(), {
    refreshInterval: 60000,
    revalidateOnFocus: false,
  });
}

export function useDailyReturns(days: number = 90) {
  return useSWR(`daily-returns-${days}`, () => api.getDailyReturns(days), {
    revalidateOnFocus: false,
  });
}

// ── WebSocket / Polling Ticker ──

export function useTickerStream() {
  const { setTickerData } = useStore();
  const wsRef = useRef<WebSocket | null>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  const startPolling = useCallback(() => {
    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      if (!mountedRef.current) return;
      try {
        const prices: TickerData[] = await api.getTickerPrices();
        const mapped: Record<string, TickerData> = {};
        prices.forEach((p: TickerData) => { mapped[p.ticker] = p; });
        setTickerData(mapped);
      } catch {
        // silent fail on polling
      }
    }, 5000);
  }, [setTickerData]);

  const connectWs = useCallback(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/api/v1/ticker/ws';
    const token = api.getToken();
    if (!token) {
      startPolling();
      return;
    }

    try {
      const ws = new WebSocket(`${wsUrl}?token=${token}`);
      wsRef.current = ws;

      ws.onopen = () => {
        if (pollRef.current) {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'price_update' && msg.data) {
            const mapped: Record<string, TickerData> = {};
            msg.data.forEach((p: TickerData) => { mapped[p.ticker] = p; });
            setTickerData(mapped);
          }
        } catch {
          // ignore parse errors
        }
      };

      ws.onclose = () => {
        wsRef.current = null;
        if (mountedRef.current) {
          startPolling();
          reconnectRef.current = setTimeout(connectWs, 10000);
        }
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch {
      startPolling();
    }
  }, [setTickerData, startPolling]);

  useEffect(() => {
    mountedRef.current = true;
    connectWs();

    return () => {
      mountedRef.current = false;
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      if (reconnectRef.current) {
        clearTimeout(reconnectRef.current);
        reconnectRef.current = null;
      }
    };
  }, [connectWs]);
}
