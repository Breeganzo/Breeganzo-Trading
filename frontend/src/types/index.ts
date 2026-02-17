// ── Core Types ──

export interface User {
  id: string;
  email: string;
  name: string | null;
  picture: string | null;
  totp_enabled: boolean;
  is_active: boolean;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  requires_totp: boolean;
  totp_setup_uri?: string;
}

// ── Portfolio ──

export interface PortfolioHolding {
  id: string;
  ticker: string;
  exchange: string;
  quantity: number;
  avg_buy_price: number;
  total_invested: number;
  realized_pnl: number;
  current_price: number | null;
  unrealized_pnl: number | null;
  pnl_pct: number | null;
  day_change_pct: number | null;
  total_buy_costs: number;
  total_sell_costs: number;
  sector: string | null;
  beta: number | null;
  volatility: number | null;
}

export interface PortfolioSummary {
  total_value: number;
  total_invested: number;
  total_unrealized_pnl: number;
  total_realized_pnl: number;
  total_pnl: number;
  total_pnl_pct: number;
  total_transaction_costs: number;
  day_pnl: number;
  day_pnl_pct: number;
  holdings: PortfolioHolding[];
  sector_exposure: Record<string, number>;
  beta_exposure: number;
}

export interface DailyReturn {
  date: string;
  portfolio_value: number;
  daily_return_pct: number;
  total_invested: number;
  total_pnl: number;
}

// ── Trades ──

export interface TradeCreate {
  ticker: string;
  exchange?: string;
  trade_type: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  slippage_pct?: number;
  notes?: string;
  executed_at?: string;
}

export interface Trade {
  id: string;
  ticker: string;
  exchange: string;
  trade_type: string;
  quantity: number;
  price: number;
  total_amount: number;
  brokerage: number;
  stt: number;
  exchange_charges: number;
  gst: number;
  sebi_charges: number;
  stamp_duty: number;
  slippage_cost: number;
  total_cost: number;
  net_amount: number;
  notes: string | null;
  executed_at: string;
}

export interface CostPreview {
  ticker: string;
  trade_type: string;
  quantity: number;
  price: number;
  total_amount: number;
  brokerage: number;
  stt: number;
  exchange_charges: number;
  gst: number;
  sebi_charges: number;
  stamp_duty: number;
  slippage_cost: number;
  total_cost: number;
  net_amount: number;
}

// ── Orders ──

export interface OrderCreate {
  ticker: string;
  exchange?: string;
  order_type: 'BUY' | 'SELL';
  quantity: number;
  target_price: number;
  notes?: string;
}

export interface Order {
  id: string;
  ticker: string;
  exchange: string;
  order_type: string;
  quantity: number;
  target_price: number;
  status: 'DRAFT' | 'CONFIRMED' | 'CANCELLED';
  notes: string | null;
  created_at: string;
  confirmed_at: string | null;
}

export interface OrderSummary {
  draft_count: number;
  confirmed_count: number;
  cancelled_count: number;
  total_pending_buy_value: number;
  total_pending_sell_value: number;
}

export interface AutoSignalCreate {
  ticker: string;
  exchange?: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  quantity?: number;
  trigger_price_low?: number;
  trigger_price_high?: number;
  sentiment_min?: number;
  sentiment_max?: number;
  source?: string;
  notes?: string;
}

export interface AutoSignal {
  id: string;
  ticker: string;
  exchange: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  quantity: number;
  trigger_price_low: number | null;
  trigger_price_high: number | null;
  sentiment_min: number | null;
  sentiment_max: number | null;
  sentiment_last: number | null;
  status: 'PENDING' | 'SKIPPED' | 'CANCELLED';
  source: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string | null;
  triggered_at: string | null;
}

// ── Rankings ──

export interface RankingEntry {
  ticker: string;
  rank_position: number;
  score: number;
  expected_return: number | null;
  momentum_30d: number | null;
  volatility: number | null;
  liquidity_score: number | null;
  current_price: number | null;
  computed_at: string;
}

export interface RankingResponse {
  category: string;
  entries: RankingEntry[];
  computed_at: string | null;
}

export type RankingCategory =
  | 'top_buy'
  | 'top_sell'
  | 'banking'
  | 'large_cap'
  | 'small_cap'
  | 'high_vol'
  | 'overall';

// ── Risk ──

export interface RiskMetrics {
  sharpe_ratio: number | null;
  sortino_ratio: number | null;
  portfolio_beta: number | null;
  max_drawdown: number | null;
  var_95: number | null;
  rolling_return_30d: number | null;
  rolling_return_90d: number | null;
  rolling_return_1y: number | null;
  regime: string;
  regime_details: Record<string, any>;
  correlation_matrix: {
    tickers: string[];
    matrix: number[][];
  } | null;
  last_updated: string;
}

export interface RegimeData {
  regime: string;
  ma_50: number | null;
  ma_200: number | null;
  ma_signal: string;
  current_vol: number | null;
  avg_vol: number | null;
  vol_ratio: number | null;
  vol_regime: string;
  breadth_pct: number | null;
  breadth_signal: string;
  confidence: number;
  detected_at: string;
}

// ── Ticker ──

export interface TickerData {
  ticker: string;
  price: number;
  change: number;
  change_pct: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  prev_close: number;
  timestamp: string;
}

export interface MarketStatus {
  is_open: boolean;
  next_open: string;
  current_time: string;
}

// ── System ──

export interface SystemHealth {
  status: string;
  database: string;
  redis: string;
  data_feed: string;
  model_last_updated: string | null;
  correlation_last_calculated: string | null;
  uptime_seconds: number;
  version: string;
}

// ── AI ──

export interface AIExplainRequest {
  metric: string;
  context?: Record<string, any>;
}

export interface AIExplanation {
  explanation: string;
  suggestions: string[];
  metric: string;
}

export interface PortfolioAnalysis {
  explanation: string;
  risk_assessment: string;
  suggestions: string[];
  source: string;
}

// ── UI State ──

export type SortDirection = 'asc' | 'desc';

export interface SortConfig {
  key: string;
  direction: SortDirection;
}

export interface PanelConfig {
  id: string;
  title: string;
  visible: boolean;
  order: number;
}
