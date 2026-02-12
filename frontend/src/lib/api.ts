const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

class ApiClient {
  private baseUrl: string;
  private token: string | null = null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
    if (typeof window !== 'undefined') {
      this.token = localStorage.getItem('quantdesk_token');
    }
  }

  setToken(token: string) {
    this.token = token;
    if (typeof window !== 'undefined') {
      localStorage.setItem('quantdesk_token', token);
    }
  }

  clearToken() {
    this.token = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('quantdesk_token');
    }
  }

  getToken(): string | null {
    return this.token;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string>),
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers,
    });

    if (response.status === 401) {
      this.clearToken();
      if (typeof window !== 'undefined') {
        window.location.href = '/login';
      }
      throw new Error('Unauthorized');
    }

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // ── Auth ──
  async getLoginUrl(): Promise<{ auth_url: string }> {
    return this.request('/auth/login');
  }

  async handleCallback(code: string): Promise<any> {
    return this.request(`/auth/callback?code=${encodeURIComponent(code)}`);
  }

  async setupTotp(): Promise<{ secret: string; uri: string; qr_code: string }> {
    return this.request('/auth/totp/setup', { method: 'POST' });
  }

  async verifyTotp(code: string): Promise<any> {
    return this.request('/auth/totp/verify', {
      method: 'POST',
      body: JSON.stringify({ code }),
    });
  }

  async getMe(): Promise<any> {
    return this.request('/auth/me');
  }

  // ── Portfolio ──
  async getPortfolio(): Promise<any> {
    return this.request('/portfolio/');
  }

  async getHoldings(): Promise<any> {
    return this.request('/portfolio/holdings');
  }

  async getDailyReturns(days: number = 90): Promise<any> {
    return this.request(`/portfolio/daily-returns?days=${days}`);
  }

  async seedPortfolio(holdings: any[]): Promise<any> {
    return this.request('/portfolio/seed', {
      method: 'POST',
      body: JSON.stringify({ holdings }),
    });
  }

  // ── Trades ──
  async executeTrade(trade: any): Promise<any> {
    return this.request('/trades/', {
      method: 'POST',
      body: JSON.stringify(trade),
    });
  }

  async getTrades(ticker?: string, limit: number = 50, offset: number = 0): Promise<any> {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (ticker) params.set('ticker', ticker);
    return this.request(`/trades/?${params}`);
  }

  async getCostPreview(
    ticker: string,
    tradeType: string,
    quantity: number,
    price: number,
    slippagePct?: number
  ): Promise<any> {
    const params = new URLSearchParams({
      ticker,
      trade_type: tradeType,
      quantity: String(quantity),
      price: String(price),
    });
    if (slippagePct) params.set('slippage_pct', String(slippagePct));
    return this.request(`/trades/cost-preview?${params}`);
  }

  // ── Orders ──
  async createOrder(order: any): Promise<any> {
    return this.request('/orders/', {
      method: 'POST',
      body: JSON.stringify(order),
    });
  }

  async getOrders(status?: string, limit: number = 50): Promise<any> {
    const params = new URLSearchParams({ limit: String(limit) });
    if (status) params.set('status', status);
    return this.request(`/orders/?${params}`);
  }

  async confirmOrder(orderId: string): Promise<any> {
    return this.request(`/orders/${orderId}/confirm`, { method: 'POST' });
  }

  async cancelOrder(orderId: string): Promise<any> {
    return this.request(`/orders/${orderId}`, { method: 'DELETE' });
  }

  async getOrderSummary(): Promise<any> {
    return this.request('/orders/summary');
  }

  // ── Rankings ──
  async getAllRankings(): Promise<any> {
    return this.request('/rankings/');
  }

  async getRankingsByCategory(category: string): Promise<any> {
    return this.request(`/rankings/${category}`);
  }

  async triggerRankingCompute(): Promise<any> {
    return this.request('/rankings/compute', { method: 'POST' });
  }

  // ── Risk ──
  async getRiskMetrics(): Promise<any> {
    return this.request('/risk/');
  }

  async getCorrelation(): Promise<any> {
    return this.request('/risk/correlation');
  }

  async getRegime(): Promise<any> {
    return this.request('/risk/regime');
  }

  async recalculateRisk(): Promise<any> {
    return this.request('/risk/recalculate', { method: 'POST' });
  }

  // ── Ticker ──
  async getTickerPrices(): Promise<any> {
    return this.request('/ticker/prices');
  }

  async getTickerPrice(ticker: string): Promise<any> {
    return this.request(`/ticker/price/${ticker}`);
  }

  async getMarketStatus(): Promise<any> {
    return this.request('/ticker/market-status');
  }

  // ── System ──
  async getSystemHealth(): Promise<any> {
    return this.request('/system/health');
  }

  async validateSystem(): Promise<any> {
    return this.request('/system/validate', { method: 'POST' });
  }

  // ── AI ──
  async explainMetric(metric: string, context?: Record<string, any>): Promise<any> {
    return this.request('/ai/explain', {
      method: 'POST',
      body: JSON.stringify({ metric, context }),
    });
  }

  async getPortfolioAnalysis(): Promise<any> {
    return this.request('/ai/portfolio-analysis', { method: 'POST' });
  }
}

export const api = new ApiClient(API_URL);
export default api;
