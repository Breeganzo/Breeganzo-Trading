/* ============================================================
   Masters AI Trading Bot — Stock Detail Page JS
   ============================================================ */

// TICKER and STOCK_NAME are injected via template

let chart = null;
let lineSeries = null;
let volumeSeries = null;
let predictionData = null;
let autoRefreshInterval = null;
let latestLivePrice = 0;
let chartRecords = [];
let indicatorSeries = {};
let drawMode = null;
let pendingDrawPoint = null;
let drawingSeries = [];

// ── Init ──────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    initChart();
    loadChartData('1d', '5m');
    loadLivePrice();
    loadPrediction();
    loadPriceTracker();
    loadGroqForecast();
    loadFeatureImportance();
    loadNews();
    refreshPortfolioStatus();

    // Auto-refresh live price every 5 seconds
    autoRefreshInterval = setInterval(() => {
        loadLivePrice();
        loadPriceTracker();
    }, 5000);
});

// ── Status ────────────────────────────────────────
async function checkStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const mktBadge = document.getElementById('market-status');
        if (data.market.status === 'market_open') {
            mktBadge.textContent = '🟢 Market Open';
            mktBadge.className = 'status-badge open';
        } else {
            mktBadge.textContent = '🔴 ' + data.market.description;
            mktBadge.className = 'status-badge closed';
        }
    } catch (e) {
        console.error('Status check failed:', e);
    }
}

// ── Chart ─────────────────────────────────────────
function initChart() {
    const container = document.getElementById('chart-container');
    chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: 400,
        layout: {
            background: { type: 'solid', color: '#0d1117' },
            textColor: '#8b949e',
            fontFamily: 'Inter, sans-serif',
        },
        grid: {
            vertLines: { color: 'rgba(48, 54, 61, 0.5)' },
            horzLines: { color: 'rgba(48, 54, 61, 0.5)' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        rightPriceScale: {
            borderColor: '#30363d',
        },
        timeScale: {
            borderColor: '#30363d',
            timeVisible: true,
            secondsVisible: false,
        },
    });

    // Area/line series for price
    lineSeries = chart.addAreaSeries({
        topColor: 'rgba(0, 208, 156, 0.3)',
        bottomColor: 'rgba(0, 208, 156, 0.0)',
        lineColor: '#00d09c',
        lineWidth: 2,
        priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    });

    // Volume series
    volumeSeries = chart.addHistogramSeries({
        color: 'rgba(91, 141, 239, 0.3)',
        priceFormat: { type: 'volume' },
        priceScaleId: '',
    });

    chart.priceScale('').applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
    });

    // Responsive
    window.addEventListener('resize', () => {
        chart.applyOptions({ width: container.clientWidth });
    });

    chart.subscribeClick(handleChartClick);
}

function clearIndicatorSeries() {
    for (const key of Object.keys(indicatorSeries)) {
        try { chart.removeSeries(indicatorSeries[key]); } catch (_) {}
    }
    indicatorSeries = {};
}

function calcSMA(values, n) {
    const out = [];
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
        sum += values[i];
        if (i >= n) sum -= values[i - n];
        out.push(i >= n - 1 ? sum / n : null);
    }
    return out;
}

function calcEMA(values, n) {
    const out = [];
    const k = 2 / (n + 1);
    let ema = null;
    for (let i = 0; i < values.length; i++) {
        if (ema === null) ema = values[i];
        else ema = values[i] * k + ema * (1 - k);
        out.push(ema);
    }
    return out;
}

function calcVWAP(records) {
    let cumPV = 0;
    let cumV = 0;
    return records.map(r => {
        const tp = ((r.high + r.low + r.close) / 3);
        const vol = Math.max(r.volume || 0, 0);
        cumPV += tp * vol;
        cumV += vol;
        return cumV > 0 ? (cumPV / cumV) : r.close;
    });
}

function toggleIndicator() {
    if (!chartRecords.length) return;
    clearIndicatorSeries();

    const closes = chartRecords.map(r => r.close);
    const times = chartRecords.map(r => r.time);

    if (document.getElementById('ind-sma20')?.checked) {
        const sma = calcSMA(closes, 20);
        const s = chart.addLineSeries({ color: '#f0b90b', lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
        s.setData(times.map((t, i) => ({ time: t, value: sma[i] })).filter(x => x.value != null));
        indicatorSeries.sma20 = s;
    }
    if (document.getElementById('ind-ema20')?.checked) {
        const ema = calcEMA(closes, 20);
        const s = chart.addLineSeries({ color: '#a78bfa', lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
        s.setData(times.map((t, i) => ({ time: t, value: ema[i] })));
        indicatorSeries.ema20 = s;
    }
    if (document.getElementById('ind-vwap')?.checked) {
        const vwap = calcVWAP(chartRecords);
        const s = chart.addLineSeries({ color: '#5b8def', lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
        s.setData(times.map((t, i) => ({ time: t, value: vwap[i] })));
        indicatorSeries.vwap = s;
    }
}

function applyIndicatorPreset(preset) {
    const presets = {
        custom: { sma20: false, ema20: false, vwap: false },
        intraday: { sma20: true, ema20: true, vwap: true },
        swing: { sma20: true, ema20: false, vwap: false },
        trend: { sma20: true, ema20: true, vwap: false },
    };
    const sel = presets[preset] || presets.custom;
    const m = {
        sma20: document.getElementById('ind-sma20'),
        ema20: document.getElementById('ind-ema20'),
        vwap: document.getElementById('ind-vwap'),
    };
    if (m.sma20) m.sma20.checked = sel.sma20;
    if (m.ema20) m.ema20.checked = sel.ema20;
    if (m.vwap) m.vwap.checked = sel.vwap;
    toggleIndicator();
}

function setDrawMode(mode) {
    drawMode = mode;
    pendingDrawPoint = null;
    document.querySelectorAll('#draw-trendline, #draw-support, #draw-resistance')
        .forEach(btn => btn?.classList.remove('active'));
    const activeBtn = document.getElementById(`draw-${mode}`);
    if (activeBtn) activeBtn.classList.add('active');
    const status = document.getElementById('draw-status');
    if (!status) return;
    if (mode === 'trendline') status.textContent = 'Trendline: click 2 points on chart.';
    if (mode === 'support') status.textContent = 'Support: click a price level.';
    if (mode === 'resistance') status.textContent = 'Resistance: click a price level.';
}

function clearDrawings() {
    for (const s of drawingSeries) {
        try { chart.removeSeries(s); } catch (_) {}
    }
    drawingSeries = [];
    pendingDrawPoint = null;
    const status = document.getElementById('draw-status');
    if (status) status.textContent = 'Drawings cleared.';
}

function handleChartClick(param) {
    if (!drawMode || !param || !param.time) return;
    let price = param.seriesData?.get?.(lineSeries)?.value;
    if (price == null && chartRecords.length) {
        // Fallback if direct series data is unavailable.
        price = chartRecords[chartRecords.length - 1].close;
    }
    if (price == null) return;

    if (drawMode === 'trendline') {
        if (!pendingDrawPoint) {
            pendingDrawPoint = { time: param.time, value: price };
            const status = document.getElementById('draw-status');
            if (status) status.textContent = 'Trendline: select second point.';
            return;
        }
        const s = chart.addLineSeries({
            color: '#00c2ff',
            lineWidth: 2,
            priceLineVisible: false,
            lastValueVisible: false,
        });
        s.setData([pendingDrawPoint, { time: param.time, value: price }]);
        drawingSeries.push(s);
        pendingDrawPoint = null;
    } else if (drawMode === 'support' || drawMode === 'resistance') {
        if (!chartRecords.length) return;
        const firstTime = chartRecords[0].time;
        const lastTime = chartRecords[chartRecords.length - 1].time;
        const s = chart.addLineSeries({
            color: drawMode === 'support' ? '#2ecc71' : '#ff6b6b',
            lineWidth: 2,
            lineStyle: 2,
            priceLineVisible: false,
            lastValueVisible: false,
        });
        s.setData([{ time: firstTime, value: price }, { time: lastTime, value: price }]);
        drawingSeries.push(s);
    }
    const status = document.getElementById('draw-status');
    if (status) status.textContent = 'Drawing added.';
}

async function loadChartData(period, interval) {
    try {
        let url;
        if (interval === '1d') {
            url = `/api/history/${encodeURIComponent(TICKER)}?period=${period}`;
        } else {
            url = `/api/intraday/${encodeURIComponent(TICKER)}?period=${period}&interval=${interval}`;
        }

        const res = await fetch(url);
        const records = await res.json();

        if (records.error) {
            console.error('Chart data error:', records.error);
            return;
        }

        // Convert to Lightweight Charts format
        const priceData = [];
        const volumeData = [];

        const IST_OFFSET = 5.5 * 60 * 60; // 19800s — shifts UTC→IST for chart labels
        chartRecords = [];
        for (const r of records) {
            const time = r.time.includes('T')
                ? Math.floor(new Date(r.time).getTime() / 1000) + IST_OFFSET
                : r.time;  // Already YYYY-MM-DD

            chartRecords.push({
                time,
                open: Number(r.open || 0),
                high: Number(r.high || r.close || 0),
                low: Number(r.low || r.close || 0),
                close: Number(r.close || 0),
                volume: Number(r.volume || 0),
            });
            priceData.push({ time, value: r.close });
            volumeData.push({
                time,
                value: r.volume,
                color: r.close >= r.open
                    ? 'rgba(0, 208, 156, 0.4)'
                    : 'rgba(235, 87, 87, 0.4)',
            });
        }

        lineSeries.setData(priceData);
        volumeSeries.setData(volumeData);
        toggleIndicator();
        chart.timeScale().fitContent();

        // Color the line based on overall direction
        if (priceData.length >= 2) {
            const first = priceData[0].value;
            const last = priceData[priceData.length - 1].value;
            const isUp = last >= first;

            lineSeries.applyOptions({
                topColor: isUp ? 'rgba(0, 208, 156, 0.3)' : 'rgba(235, 87, 87, 0.3)',
                bottomColor: isUp ? 'rgba(0, 208, 156, 0.0)' : 'rgba(235, 87, 87, 0.0)',
                lineColor: isUp ? '#00d09c' : '#eb5757',
            });
        }

        // Add prediction marker if available
        if (predictionData && interval === '1d') {
            addPredictionLine(priceData);
        }
    } catch (e) {
        console.error('Chart load failed:', e);
    }
}

function addPredictionLine(priceData) {
    if (!predictionData || !priceData.length) return;

    const lastTime = priceData[priceData.length - 1].time;
    const predPrice = predictionData.predicted_price;

    // Add a marker on the chart
    lineSeries.setMarkers([
        {
            time: lastTime,
            position: 'aboveBar',
            color: '#5b8def',
            shape: 'arrowDown',
            text: `AI: ₹${predPrice}`,
        },
    ]);
}

function changeTimeframe(btn) {
    // Update active button
    document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const period = btn.dataset.period;
    const interval = btn.dataset.interval;
    loadChartData(period, interval);
}

// ── Live Price ────────────────────────────────────
async function loadLivePrice() {
    try {
        const res = await fetch(`/api/prices?tickers=${encodeURIComponent(TICKER)}`);
        const data = await res.json();
        const price = data[TICKER];

        if (!price) return;
        latestLivePrice = Number(price.price || 0);

        // Update header
        document.getElementById('live-price').textContent = `₹${formatN(price.price)}`;

        const changeEl = document.getElementById('live-change');
        const sign = price.change >= 0 ? '+' : '';
        changeEl.textContent = `${sign}${price.change.toFixed(2)} (${price.change_pct.toFixed(2)}%)`;
        changeEl.className = `stock-change-big ${price.change >= 0 ? 'up-color' : 'down-color'}`;

        // Also update the big price color
        document.getElementById('live-price').className = `stock-price-big`;

        // Performance stats
        document.getElementById('today-low').textContent = `₹${formatN(price.low)}`;
        document.getElementById('today-high').textContent = `₹${formatN(price.high)}`;
        document.getElementById('stat-open').textContent = `₹${formatN(price.open)}`;
        document.getElementById('stat-prev-close').textContent = `₹${formatN(price.prev_close)}`;
        document.getElementById('stat-volume').textContent = formatVolume(price.volume);

        // Position marker on range bar
        if (price.low > 0 && price.high > price.low) {
            const pct = ((price.price - price.low) / (price.high - price.low)) * 100;
            document.getElementById('price-marker').style.left = `${Math.max(0, Math.min(100, pct))}%`;
        }
    } catch (e) {
        console.error('Live price failed:', e);
    }
}

// ── Prediction ────────────────────────────────────
async function loadPrediction() {
    const loading = document.getElementById('pred-loading');
    const content = document.getElementById('pred-content');

    try {
        const res = await fetch(`/api/predict/${encodeURIComponent(TICKER)}`);

        if (res.status === 503) {
            loading.innerHTML = '<div class="spinner"></div><p>Models still loading... will retry in 5s</p>';
            setTimeout(loadPrediction, 5000);
            return;
        }

        const data = await res.json();

        if (data.error) {
            loading.innerHTML = `<p class="muted-text">⚠️ ${data.error}</p>`;
            return;
        }

        predictionData = data;
        loading.classList.add('hidden');
        content.classList.remove('hidden');

        // Signal badge
        const badge = document.getElementById('signal-badge');
        badge.textContent = data.signal;
        badge.className = `signal-badge-large ${data.signal}`;

        // Prediction values
        const predReturn = data.predicted_return;
        const retSign = predReturn >= 0 ? '+' : '';
        const retColor = predReturn >= 0 ? 'up-color' : 'down-color';

        document.getElementById('pred-return').innerHTML = `<span class="${retColor}">${retSign}${predReturn?.toFixed(3)}%</span>`;
        document.getElementById('pred-price').textContent = `₹${formatN(data.predicted_price)}`;
        document.getElementById('pred-confidence').textContent = `${data.confidence?.toFixed(0)}%`;
        document.getElementById('pred-agreement').textContent = `${data.model_agreement?.toFixed(0)}%`;

        // Show generated-at timestamp
        const genEl = document.getElementById('pred-generated-at');
        if (genEl && data.generated_at) {
            genEl.textContent = data.generated_at;
        }

        // Model breakdown — sorted by weight (highest first)
        const breakdown = document.getElementById('model-breakdown');
        let bhtml = '';
        const weights = data.ensemble_weights || {};
        const models = Object.entries(data.model_predictions || {});
        models.sort((a, b) => (weights[b[0]] || 0) - (weights[a[0]] || 0));

        // Show ensemble strategy
        const stratEl = document.getElementById('ensemble-strategy');
        if (stratEl && data.ensemble_strategy) {
            const stratLabel = data.ensemble_strategy.replace(/_/g, ' ');
            stratEl.textContent = stratLabel;
        }

        for (const [model, pred] of models) {
            const sign = pred >= 0 ? '+' : '';
            const color = pred >= 0 ? 'up-color' : 'down-color';
            const wVal = weights[model] != null ? weights[model] : 0;
            const wPct = (wVal * 100).toFixed(1) + '%';
            const barW = Math.max(wVal * 100, 2);  // min 2% for visibility
            bhtml += `
            <div class="model-chip clickable-metric" onclick="explainModel('${model}')">
                <span class="model-name">${model}</span>
                <span class="model-pred ${color}">${sign}${pred?.toFixed(3)}%</span>
                <span class="model-weight">${wPct}</span>
                <div class="weight-bar" style="width:${barW}%"></div>
            </div>`;
        }
        breakdown.innerHTML = bhtml;

        // Trade levels
        document.getElementById('level-entry').textContent = `₹${formatN(data.entry_price)}`;
        document.getElementById('level-target').textContent = `₹${formatN(data.target_price)}`;
        document.getElementById('level-sl').textContent = `₹${formatN(data.stop_loss)}`;
        document.getElementById('level-rr').textContent = data.risk_reward?.toFixed(1);

        // Fundamentals
        if (data.fundamentals) {
            const fundSection = document.getElementById('fundamentals-section');
            fundSection.classList.remove('hidden');
            const fundGrid = document.getElementById('fund-grid');
            const f = data.fundamentals;
            const fundItems = [
                ['Market Cap', f.market_cap ? `₹${formatN(f.market_cap)}` : '—', 'market_cap', f.market_cap],
                ['P/E Ratio', f.pe_ratio?.toFixed(2) || '—', 'pe_ratio', f.pe_ratio],
                ['P/B Ratio', f.pb_ratio?.toFixed(2) || '—', 'pb_ratio', f.pb_ratio],
                ['ROE', f.roe ? (f.roe * 100).toFixed(1) + '%' : '—', 'roe', f.roe],
                ['Dividend Yield', f.dividend_yield ? (f.dividend_yield * 100).toFixed(2) + '%' : '—', 'dividend_yield', f.dividend_yield],
                ['Debt/Equity', f.debt_to_equity?.toFixed(2) || '—', 'debt_to_equity', f.debt_to_equity],
                ['Revenue Growth', f.revenue_growth ? (f.revenue_growth * 100).toFixed(1) + '%' : '—', 'revenue_growth', f.revenue_growth],
                ['Earnings Growth', f.earnings_growth ? (f.earnings_growth * 100).toFixed(1) + '%' : '—', 'earnings_growth', f.earnings_growth],
                ['Profit Margin', f.profit_margin ? (f.profit_margin * 100).toFixed(1) + '%' : '—', 'profit_margin', f.profit_margin],
                ['Analyst Target', f.target_price_analyst ? `₹${formatN(f.target_price_analyst)}` : '—', 'analyst_target', f.target_price_analyst],
                ['52W High', f.fifty_two_high ? `₹${formatN(f.fifty_two_high)}` : '—', 'fifty_two_high', f.fifty_two_high],
                ['52W Low', f.fifty_two_low ? `₹${formatN(f.fifty_two_low)}` : '—', 'fifty_two_low', f.fifty_two_low],
                ['Beta', f.beta?.toFixed(2) || '—', 'beta', f.beta],
                ['Sector', f.sector || '—', 'sector', f.sector],
                ['Value Score', f.value_score ? (f.value_score * 100).toFixed(0) : '—', 'value_score', f.value_score],
                ['Quality Score', f.quality_score ? (f.quality_score * 100).toFixed(0) : '—', 'quality_score', f.quality_score],
            ];

            fundGrid.innerHTML = fundItems.map(([label, value, key, rawVal]) => `
                <div class="fund-item clickable-metric" onclick="explainFundamental('${label}', '${value}')">
                    <span class="fund-label">${label}</span>
                    <span class="fund-value">${value}</span>
                    <span class="explain-icon">ℹ️</span>
                </div>
            `).join('');
        }

        // Technical Indicators
        if (data.indicators) {
            renderIndicators(data.indicators);
        }

        // Options Greeks
        if (data.greeks) {
            renderGreeks(data.greeks);
        }

        // Load strategies
        loadStrategies(data);

        // Load prediction tracking
        loadTracking();

        // Update chart with prediction marker
        const activeTf = document.querySelector('.tf-btn.active');
        if (activeTf && activeTf.dataset.interval === '1d') {
            // Re-trigger chart to add marker
        }

        // Load expected vs actual for this stock
        loadStockEVA();

    } catch (e) {
        loading.innerHTML = `<p class="muted-text">⚠️ Prediction error: ${e.message}</p>`;
    }
}

// ── Expected vs Actual for this stock ─────────────
async function loadStockEVA() {
    const container = document.getElementById('stock-eva-content');

    try {
        const datesRes = await fetch('/api/prediction-dates');
        const dates = await datesRes.json();

        if (!dates.length) {
            container.innerHTML = '<p class="muted-text">No past predictions logged yet. Predictions made today will be compared with actual close at end of day.</p>';
            return;
        }

        // Check today's prediction and actual
        let html = '<div style="margin-top: 12px;">';

        // Show current prediction vs current price
        if (predictionData) {
            const predRet = predictionData.predicted_return;
            const predPrice = predictionData.predicted_price;
            const currPrice = predictionData.current_price;

            html += `
            <div class="pred-grid">
                <div class="pred-card">
                    <span class="pred-label">Price When Predicted</span>
                    <span class="pred-value">₹${formatN(currPrice)}</span>
                </div>
                <div class="pred-card">
                    <span class="pred-label">AI Predicted Price</span>
                    <span class="pred-value ${predRet >= 0 ? 'up-color' : 'down-color'}">₹${formatN(predPrice)}</span>
                </div>
                <div class="pred-card">
                    <span class="pred-label">Expected Return</span>
                    <span class="pred-value ${predRet >= 0 ? 'up-color' : 'down-color'}">${predRet >= 0 ? '+' : ''}${predRet?.toFixed(3)}%</span>
                </div>
            </div>
            <p class="muted-text" style="margin-top: 12px;">
                At end of day, this will show: Actual Close → Actual Return → Alpha generated
            </p>`;
        }

        // Show historical comparison if we have past data
        const latestDate = dates[0];
        try {
            const evaRes = await fetch(`/api/expected-vs-actual?date=${latestDate}`);
            const evaData = await evaRes.json();

            if (!evaData.error && evaData.results) {
                const stockResult = evaData.results.find(r => r.ticker === TICKER);
                if (stockResult) {
                    const dirClass = stockResult.direction_correct ? 'correct' : 'wrong';
                    html += `
                    <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--border);">
                        <h4 class="sub-heading">Last Prediction (${latestDate})</h4>
                        <div class="pred-grid">
                            <div class="pred-card">
                                <span class="pred-label">Predicted</span>
                                <span class="pred-value ${stockResult.predicted_return_pct >= 0 ? 'up-color' : 'down-color'}">
                                    ${stockResult.predicted_return_pct >= 0 ? '+' : ''}${stockResult.predicted_return_pct}%
                                </span>
                            </div>
                            <div class="pred-card">
                                <span class="pred-label">Actual</span>
                                <span class="pred-value ${stockResult.actual_return_pct >= 0 ? 'up-color' : 'down-color'}">
                                    ${stockResult.actual_return_pct >= 0 ? '+' : ''}${stockResult.actual_return_pct}%
                                </span>
                            </div>
                            <div class="pred-card">
                                <span class="pred-label">Direction</span>
                                <span class="pred-value"><span class="direction-badge ${dirClass}">${stockResult.direction_correct ? '✓ Correct' : '✗ Wrong'}</span></span>
                            </div>
                            <div class="pred-card">
                                <span class="pred-label">Alpha</span>
                                <span class="pred-value ${stockResult.alpha_pct >= 0 ? 'up-color' : 'down-color'}">
                                    ${stockResult.alpha_pct >= 0 ? '+' : ''}${stockResult.alpha_pct}%
                                </span>
                            </div>
                        </div>
                    </div>`;
                }
            }
        } catch (e) {
            // Ignore — no past data for this stock
        }

        html += '</div>';
        container.innerHTML = html;
    } catch (e) {
        container.innerHTML = `<p class="muted-text">⚠️ ${e.message}</p>`;
    }
}

// ── Helpers ───────────────────────────────────────
function formatN(n) {
    if (n === undefined || n === null) return '---';
    if (typeof n === 'string') n = parseFloat(n);
    if (isNaN(n)) return '---';
    if (n >= 10000000) return (n / 10000000).toFixed(2) + ' Cr';
    if (n >= 100000) return (n / 100000).toFixed(2) + ' L';
    return n.toLocaleString('en-IN', { maximumFractionDigits: 2 });
}

function formatVolume(v) {
    if (!v) return '---';
    if (v >= 10000000) return (v / 10000000).toFixed(2) + ' Cr';
    if (v >= 100000) return (v / 100000).toFixed(2) + ' L';
    if (v >= 1000) return (v / 1000).toFixed(1) + 'K';
    return v.toLocaleString('en-IN');
}

// ── Technical Indicators ──────────────────────────
function renderIndicators(indicators) {
    const grid = document.getElementById('indicator-grid');
    if (!grid || !indicators) return;

    const items = [
        { name: 'RSI (14)', key: 'rsi', format: v => v?.toFixed(1) },
        { name: 'MACD', key: 'macd', format: v => v?.toFixed(3) },
        { name: 'MACD Signal', key: 'macd_signal', format: v => v?.toFixed(3) },
        { name: 'SMA 20', key: 'sma_20', format: v => `₹${formatN(v)}` },
        { name: 'SMA 50', key: 'sma_50', format: v => `₹${formatN(v)}` },
        { name: 'EMA 12', key: 'ema_12', format: v => `₹${formatN(v)}` },
        { name: 'EMA 26', key: 'ema_26', format: v => `₹${formatN(v)}` },
        { name: 'Bollinger Upper', key: 'bb_upper', format: v => `₹${formatN(v)}` },
        { name: 'Bollinger Lower', key: 'bb_lower', format: v => `₹${formatN(v)}` },
        { name: 'ATR %', key: 'atr_pct', format: v => (v * 100).toFixed(2) + '%' },
        { name: 'Volume Ratio', key: 'volume_ratio', format: v => v?.toFixed(2) + 'x' },
        { name: 'ADX', key: 'adx', format: v => v?.toFixed(1) },
    ];

    let html = '';
    for (const item of items) {
        const val = indicators[item.key];
        if (val === undefined || val === null) continue;
        const display = item.format(val);
        const signal = getIndicatorSignal(item.key, val, indicators);
        html += `
        <div class="indicator-item clickable-metric" onclick="explainIndicator('${item.name}', '${display}')">
            <span class="indicator-name">${item.name}</span>
            <span class="indicator-value">${display}</span>
            <span class="indicator-signal ${signal.class}">${signal.text}</span>
            <span class="explain-icon">ℹ️</span>
        </div>`;
    }
    grid.innerHTML = html;
}

function getIndicatorSignal(key, val, indicators) {
    if (key === 'rsi') {
        if (val > 70) return { text: 'Overbought', class: 'signal-sell' };
        if (val < 30) return { text: 'Oversold', class: 'signal-buy' };
        return { text: 'Neutral', class: 'signal-neutral' };
    }
    if (key === 'macd') {
        const signal = indicators.macd_signal || 0;
        if (val > signal) return { text: 'Bullish', class: 'signal-buy' };
        return { text: 'Bearish', class: 'signal-sell' };
    }
    if (key === 'adx') {
        if (val > 25) return { text: 'Trending', class: 'signal-buy' };
        return { text: 'Ranging', class: 'signal-neutral' };
    }
    if (key === 'volume_ratio') {
        if (val > 1.5) return { text: 'High Vol', class: 'signal-buy' };
        if (val < 0.5) return { text: 'Low Vol', class: 'signal-sell' };
        return { text: 'Normal', class: 'signal-neutral' };
    }
    return { text: '', class: '' };
}

// ── Options Greeks ────────────────────────────────
function renderGreeks(greeks) {
    const section = document.getElementById('greeks-section');
    const grid = document.getElementById('greeks-grid');
    if (!grid || !greeks) return;

    section.classList.remove('hidden');

    const items = [
        { name: 'Delta', key: 'delta', desc: 'Price sensitivity' },
        { name: 'Gamma', key: 'gamma', desc: 'Delta change rate' },
        { name: 'Theta', key: 'theta', desc: 'Time decay' },
        { name: 'Vega', key: 'vega', desc: 'Volatility sensitivity' },
        { name: 'Implied Vol', key: 'iv', desc: 'Market expected vol' },
    ];

    let html = '';
    for (const item of items) {
        const val = greeks[item.key];
        if (val === undefined || val === null) continue;
        const display = typeof val === 'number' ? val.toFixed(4) : val;
        html += `
        <div class="greek-item clickable-metric" onclick="explainGreek('${item.name}', '${display}')">
            <span class="greek-name">${item.name}</span>
            <span class="greek-value">${display}</span>
            <span class="greek-desc">${item.desc}</span>
            <span class="explain-icon">ℹ️</span>
        </div>`;
    }
    grid.innerHTML = html;
}

// ── Groq Explanation Modal ────────────────────────
function showExplanation(title) {
    const modal = document.getElementById('explanation-modal');
    const titleEl = document.getElementById('explanation-title');
    const bodyEl = document.getElementById('explanation-body');
    titleEl.textContent = title;
    bodyEl.innerHTML = '<div class="spinner"></div><p>Getting AI explanation...</p>';
    modal.classList.remove('hidden');
}

function closeExplanation() {
    document.getElementById('explanation-modal').classList.add('hidden');
}

async function explainFundamental(name, value) {
    showExplanation(name);
    try {
        const res = await fetch('/api/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: 'fundamental',
                metric: name,
                value: value,
                ticker: TICKER,
                stock_name: STOCK_NAME,
            }),
        });
        const data = await res.json();
        document.getElementById('explanation-body').innerHTML = `<p>${data.explanation || data.error}</p>`;
    } catch (e) {
        document.getElementById('explanation-body').innerHTML = `<p>Error: ${e.message}</p>`;
    }
}

async function explainIndicator(name, value) {
    showExplanation(name);
    try {
        const res = await fetch('/api/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: 'indicator',
                metric: name,
                value: value,
                ticker: TICKER,
                stock_name: STOCK_NAME,
            }),
        });
        const data = await res.json();
        document.getElementById('explanation-body').innerHTML = `<p>${data.explanation || data.error}</p>`;
    } catch (e) {
        document.getElementById('explanation-body').innerHTML = `<p>Error: ${e.message}</p>`;
    }
}

async function explainGreek(name, value) {
    showExplanation(name);
    try {
        const res = await fetch('/api/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: 'greek',
                metric: name,
                value: value,
                ticker: TICKER,
                stock_name: STOCK_NAME,
            }),
        });
        const data = await res.json();
        document.getElementById('explanation-body').innerHTML = `<p>${data.explanation || data.error}</p>`;
    } catch (e) {
        document.getElementById('explanation-body').innerHTML = `<p>Error: ${e.message}</p>`;
    }
}

// ── Strategy Loading ──────────────────────────────
async function loadStrategies(predData) {
    // ML Strategy — immediate (from prediction data)
    const mlEl = document.getElementById('ml-strategy');
    if (predData) {
        const signal = predData.signal || 'HOLD';
        const ret = predData.predicted_return || 0;
        const conf = predData.confidence || 50;
        const agreement = predData.model_agreement || 0;
        mlEl.innerHTML = `
            <div class="strategy-signal ${signal}">${signal}</div>
            <p>Predicted return: <strong class="${ret >= 0 ? 'up-color' : 'down-color'}">${ret >= 0 ? '+' : ''}${ret.toFixed(3)}%</strong> 
               with ${conf.toFixed(0)}% confidence and ${agreement.toFixed(0)}% model agreement.</p>
            <p>Entry: ₹${formatN(predData.entry_price)} → Target: ₹${formatN(predData.target_price)} | SL: ₹${formatN(predData.stop_loss)}</p>
            <p>Risk:Reward = ${predData.risk_reward?.toFixed(1)}</p>
        `;
    }

    // Groq Strategy — async API call
    try {
        const res = await fetch(`/api/strategy/${encodeURIComponent(TICKER)}`);
        const data = await res.json();

        if (data.groq_strategy) {
            document.getElementById('groq-strategy').innerHTML = `<p>${data.groq_strategy}</p>`;
        }
        if (data.combined_strategy) {
            document.getElementById('combined-strategy').innerHTML = `<p>${data.combined_strategy}</p>`;
        }
    } catch (e) {
        document.getElementById('groq-strategy').innerHTML = `<p class="muted-text">Strategy analysis unavailable: ${e.message}</p>`;
        document.getElementById('combined-strategy').innerHTML = `<p class="muted-text">Combined analysis unavailable</p>`;
    }
}

// ── Prediction Tracking ───────────────────────────
async function loadTracking() {
    const summary = document.getElementById('tracking-summary');
    const detail = document.getElementById('tracking-detail');

    try {
        const res = await fetch(`/api/tracking/daily`);
        const data = await res.json();

        if (data.total === 0) {
            summary.innerHTML = '<p class="muted-text">No predictions tracked yet. Predictions made today will be checked at end of day.</p>';
            return;
        }

        // Find this stock in today's tracking
        const stockPred = data.predictions?.find(p => p.ticker === TICKER);

        let html = '<div class="tracking-cards">';

        // Overall summary
        html += `
        <div class="tracking-card">
            <span class="tracking-label">Today's Predictions</span>
            <span class="tracking-value">${data.total}</span>
        </div>`;

        if (data.evaluated > 0) {
            html += `
            <div class="tracking-card">
                <span class="tracking-label">Checked</span>
                <span class="tracking-value">${data.evaluated}</span>
            </div>
            <div class="tracking-card">
                <span class="tracking-label">Hits</span>
                <span class="tracking-value up-color">${data.hits}</span>
            </div>
            <div class="tracking-card">
                <span class="tracking-label">Misses</span>
                <span class="tracking-value down-color">${data.misses}</span>
            </div>`;
            if (data.accuracy_pct !== null) {
                html += `
                <div class="tracking-card">
                    <span class="tracking-label">Accuracy</span>
                    <span class="tracking-value ${data.accuracy_pct >= 50 ? 'up-color' : 'down-color'}">${data.accuracy_pct}%</span>
                </div>`;
            }
        }
        html += '</div>';

        // This stock's tracking
        if (stockPred) {
            const outcomeClass = stockPred.outcome === 'HIT' ? 'hit' : stockPred.outcome === 'MISS' ? 'miss' : 'pending';
            const outcomeText = stockPred.outcome || 'PENDING';
            html += `
            <div class="stock-tracking-result ${outcomeClass}">
                <strong>${STOCK_NAME}</strong> prediction: 
                <span class="${stockPred.predicted_return_pct >= 0 ? 'up-color' : 'down-color'}">
                    ${stockPred.predicted_return_pct >= 0 ? '+' : ''}${stockPred.predicted_return_pct?.toFixed(3)}%
                </span>
                → Outcome: <span class="outcome-badge ${outcomeClass}">${outcomeText}</span>
                ${stockPred.actual_close ? ` | Actual Close: ₹${formatN(stockPred.actual_close)}` : ''}
                ${stockPred.actual_return_pct !== null && stockPred.actual_return_pct !== undefined ? 
                    ` | Actual Return: <span class="${stockPred.actual_return_pct >= 0 ? 'up-color' : 'down-color'}">${stockPred.actual_return_pct >= 0 ? '+' : ''}${stockPred.actual_return_pct?.toFixed(3)}%</span>` : ''}
            </div>`;
        }

        summary.innerHTML = html;

        // Monthly report link
        const monthRes = await fetch('/api/tracking/monthly');
        const monthData = await monthRes.json();
        if (monthData.total_predictions > 0) {
            detail.innerHTML = `
            <div class="monthly-report">
                <h4 class="sub-heading">Monthly Report (${monthData.period})</h4>
                <div class="tracking-cards">
                    <div class="tracking-card"><span class="tracking-label">Total</span><span class="tracking-value">${monthData.total_predictions}</span></div>
                    <div class="tracking-card"><span class="tracking-label">Evaluated</span><span class="tracking-value">${monthData.evaluated}</span></div>
                    <div class="tracking-card"><span class="tracking-label">Accuracy</span><span class="tracking-value ${monthData.accuracy_pct >= 50 ? 'up-color' : 'down-color'}">${monthData.accuracy_pct}%</span></div>
                    <div class="tracking-card"><span class="tracking-label">Avg Predicted</span><span class="tracking-value">${monthData.avg_predicted_return?.toFixed(3)}%</span></div>
                    <div class="tracking-card"><span class="tracking-label">Avg Actual</span><span class="tracking-value">${monthData.avg_actual_return?.toFixed(3)}%</span></div>
                </div>
            </div>`;
        }
    } catch (e) {
        summary.innerHTML = `<p class="muted-text">Tracking unavailable: ${e.message}</p>`;
    }
}
// ── Stock Overview (AI-powered) ───────────────────
let overviewLoaded = false;

function toggleOverview() {
    const content = document.getElementById('overview-content');
    const hint = document.querySelector('.toggle-hint');
    
    if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        hint.textContent = '(click to collapse)';
        if (!overviewLoaded) {
            loadStockOverview();
        }
    } else {
        content.classList.add('hidden');
        hint.textContent = '(click to expand)';
    }
}

async function loadStockOverview() {
    const loading = document.getElementById('overview-loading');
    const body = document.getElementById('overview-body');
    const companyEl = document.getElementById('company-overview');
    const sentimentEl = document.getElementById('news-sentiment');

    try {
        const res = await fetch(`/api/overview/${TICKER}`);
        const data = await res.json();

        if (data.error) {
            companyEl.innerHTML = `<p class="error-text">${data.error}</p>`;
            sentimentEl.innerHTML = '';
        } else {
            // Format the overview text with markdown-like styling
            companyEl.innerHTML = formatAIText(data.overview);
            sentimentEl.innerHTML = formatAIText(data.sentiment);
            overviewLoaded = true;
        }

        loading.classList.add('hidden');
        body.classList.remove('hidden');
    } catch (e) {
        companyEl.innerHTML = `<p class="error-text">Failed to load overview: ${e.message}</p>`;
        loading.classList.add('hidden');
        body.classList.remove('hidden');
    }
}

// Format AI text with basic markdown support
function formatAIText(text) {
    if (!text) return '';
    // Convert **bold** to <strong>
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // Convert • bullets to list items
    text = text.replace(/•\s*/g, '<br>• ');
    // Convert numbered lists
    text = text.replace(/(\d+)\.\s+/g, '<br>$1. ');
    // Convert newlines to breaks
    text = text.replace(/\n/g, '<br>');
    return `<p>${text}</p>`;
}

// ── Price Tracker (Opening → Predicted → Current) ─
async function loadPriceTracker() {
    try {
        const res = await fetch(`/api/price-tracker/${encodeURIComponent(TICKER)}`);
        if (!res.ok) return;
        const stock = await res.json();
        if (stock.error) return;

        const openPrice = stock.open_price;
        const predPrice = stock.strategy_predicted_price || stock.predicted_price;
        const currPrice = stock.current_price;
        const aiPrice = stock.ai_predicted_price || predPrice;

        // Update tracker cards
        document.getElementById('tracker-open').textContent = `₹${formatN(openPrice)}`;
        document.getElementById('tracker-predicted').textContent = `₹${formatN(predPrice)}`;
        document.getElementById('tracker-current').textContent = `₹${formatN(currPrice)}`;
        const aiEl = document.getElementById('tracker-ai');
        if (aiEl) aiEl.textContent = `₹${formatN(aiPrice)}`;

        // Percentage changes from open
        const predPct = stock.open_to_predicted_pct;
        const currPct = stock.open_to_current_pct;
        const aiPct = stock.open_to_ai_predicted_pct ?? ((aiPrice - openPrice) / (openPrice || 1) * 100);

        const predEl = document.getElementById('tracker-predicted-pct');
        predEl.textContent = `${predPct >= 0 ? '+' : ''}${predPct}% from open`;
        predEl.className = `tracker-change ${predPct >= 0 ? 'up-color' : 'down-color'}`;

        const currEl = document.getElementById('tracker-current-pct');
        currEl.textContent = `${currPct >= 0 ? '+' : ''}${currPct}% from open`;
        currEl.className = `tracker-change ${currPct >= 0 ? 'up-color' : 'down-color'}`;

        const aiPctEl = document.getElementById('tracker-ai-pct');
        if (aiPctEl) {
            aiPctEl.textContent = `${aiPct >= 0 ? '+' : ''}${Number(aiPct).toFixed(3)}% from open`;
            aiPctEl.className = `tracker-change ${aiPct >= 0 ? 'up-color' : 'down-color'}`;
        }

        // Progress bar: how far current is toward predicted
        const range = Math.abs(predPrice - openPrice);
        if (range > 0) {
            const progress = ((currPrice - openPrice) / (predPrice - openPrice)) * 100;
            const clampedProgress = Math.max(0, Math.min(100, progress));

            const fillEl = document.getElementById('progress-fill');
            fillEl.style.width = `${clampedProgress}%`;
            fillEl.className = `progress-fill ${progress >= 0 ? '' : 'negative'}`;

            // Position markers
            document.getElementById('predicted-marker').style.left = '100%';
            document.getElementById('current-marker').style.left = `${clampedProgress}%`;

            // Status text
            const statusEl = document.getElementById('progress-status');
            if (progress >= 100) {
                statusEl.textContent = '✅ Target Reached!';
                statusEl.className = 'up-color';
            } else if (progress >= 50) {
                statusEl.textContent = `${clampedProgress.toFixed(0)}% to target`;
                statusEl.className = 'up-color';
            } else if (progress > 0) {
                statusEl.textContent = `${clampedProgress.toFixed(0)}% to target`;
                statusEl.className = '';
            } else {
                statusEl.textContent = 'Moving opposite to prediction';
                statusEl.className = 'down-color';
            }
        }
    } catch (e) {
        console.error('Price tracker failed:', e);
    }
}

async function loadGroqForecast() {
    const container = document.getElementById('groq-forecast-content');
    if (!container) return;
    try {
        const res = await fetch(`/api/groq-price-forecast/${encodeURIComponent(TICKER)}`);
        const data = await res.json();
        if (!res.ok || data.error) {
            if (res.status === 503) {
                setTimeout(loadGroqForecast, 8000);
            }
            container.innerHTML = `<p class="muted-text">Groq AI forecast unavailable: ${data.error || res.status}</p>`;
            return;
        }
        const aiPrice = Number(data.ai_predicted_price || 0);
        const aiPct = Number(data.open_to_ai_predicted_pct || 0);
        container.innerHTML = `
            <div class="news-card">
                <div class="news-body">
                    <p><strong>Groq AI Predicted Price:</strong> ₹${formatN(aiPrice)}
                    <span class="${aiPct >= 0 ? 'up-color' : 'down-color'}">(${aiPct >= 0 ? '+' : ''}${aiPct.toFixed(3)}% vs open)</span></p>
                    <p><strong>Outlook:</strong> ${data.outlook || 'Neutral'}</p>
                    <p>${formatAIText(data.rationale || '').replace(/^<p>|<\/p>$/g, '')}</p>
                    <p class="muted-text">Generated: ${data.generated_at || 'now'}</p>
                </div>
            </div>`;
        const aiTracker = document.getElementById('tracker-ai');
        if (aiTracker && aiPrice > 0) aiTracker.textContent = `₹${formatN(aiPrice)}`;
    } catch (e) {
        container.innerHTML = `<p class="muted-text">Groq AI forecast failed: ${e.message}</p>`;
    }
}

async function addToPortfolio() {
    const qtyEl = document.getElementById('portfolio-qty');
    const statusEl = document.getElementById('portfolio-add-status');
    const qty = Number(qtyEl?.value || 0);
    const custom = Number(document.getElementById('portfolio-price')?.value || 0);
    const entryPrice = Number(custom || latestLivePrice || predictionData?.current_price || 0);
    if (!qty || qty <= 0) {
        if (statusEl) statusEl.textContent = 'Enter valid quantity (>0)';
        return;
    }
    if (!entryPrice || entryPrice <= 0) {
        if (statusEl) statusEl.textContent = 'Live price not available yet';
        return;
    }
    try {
        const res = await fetch('/api/portfolio/trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: TICKER, side: 'BUY', quantity: qty, price: entryPrice }),
        });
        const data = await res.json();
        if (!res.ok || data.error) {
            if (statusEl) statusEl.textContent = `Portfolio update failed: ${data.error || res.status}`;
            return;
        }
        if (statusEl) statusEl.textContent = `Added ${qty} @ ₹${formatN(entryPrice)} to portfolio`;
        refreshPortfolioStatus();
        loadPortfolioSuggestion();
    } catch (e) {
        if (statusEl) statusEl.textContent = `Portfolio update failed: ${e.message}`;
    }
}

async function refreshPortfolioStatus() {
    const statusEl = document.getElementById('portfolio-add-status');
    if (!statusEl) return;
    try {
        const res = await fetch(`/api/portfolio?ticker=${encodeURIComponent(TICKER)}`);
        const data = await res.json();
        const row = (data.holdings || [])[0];
        if (!row) {
            statusEl.textContent = 'Not in portfolio yet';
            return;
        }
        statusEl.textContent = `Holding: ${row.quantity} shares @ avg ₹${formatN(row.avg_buy_price || row.entry_price)}`;
    } catch (e) {
        statusEl.textContent = 'Portfolio status unavailable';
    }
}

async function sellFromPortfolio() {
    const qty = Number(document.getElementById('portfolio-sell-qty')?.value || 0);
    const custom = Number(document.getElementById('portfolio-sell-price')?.value || 0);
    const sellPrice = Number(custom || latestLivePrice || predictionData?.current_price || 0);
    const statusEl = document.getElementById('portfolio-add-status');
    if (!qty || qty <= 0 || !sellPrice || sellPrice <= 0) {
        if (statusEl) statusEl.textContent = 'Enter valid sell qty and price';
        return;
    }
    try {
        const res = await fetch('/api/portfolio/trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: TICKER, side: 'SELL', quantity: qty, price: sellPrice }),
        });
        const data = await res.json();
        if (!res.ok || data.error) {
            if (statusEl) statusEl.textContent = `Sell failed: ${data.error || res.status}`;
            return;
        }
        const pnl = data.summary?.realized_pnl ?? 0;
        if (statusEl) statusEl.textContent = `Recorded SELL ${qty} @ ₹${formatN(sellPrice)} | Realized P&L: ₹${formatN(pnl)}`;
        refreshPortfolioStatus();
        loadPortfolioSuggestion();
    } catch (e) {
        if (statusEl) statusEl.textContent = `Sell failed: ${e.message}`;
    }
}

async function explainModel(modelName) {
    showExplanation(`${modelName} Model`);
    try {
        const res = await fetch(`/api/explain-model?model=${encodeURIComponent(modelName)}`);
        const data = await res.json();
        document.getElementById('explanation-body').innerHTML = `<p>${data.explanation || data.error}</p>`;
    } catch (e) {
        document.getElementById('explanation-body').innerHTML = `<p>Error: ${e.message}</p>`;
    }
}

async function askStockAssistant() {
    const input = document.getElementById('stock-chat-input');
    const output = document.getElementById('stock-chat-output');
    const q = (input?.value || '').trim();
    if (!q) return;
    output.textContent = 'Thinking...';
    try {
        const res = await fetch(`/api/stock-chat/${encodeURIComponent(TICKER)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q }),
        });
        const data = await res.json();
        output.innerHTML = formatAIText(data.answer || data.error || 'No response');
    } catch (e) {
        output.textContent = `Assistant error: ${e.message}`;
    }
}

async function loadPortfolioSuggestion() {
    const output = document.getElementById('stock-chat-output');
    if (!output) return;
    try {
        const res = await fetch('/api/portfolio/summary?suggest=true');
        const data = await res.json();
        if (!data.strategy_suggestion) return;
        const cur = output.innerHTML || '';
        output.innerHTML = `${cur}<hr style="border-color:var(--border);margin:10px 0;"><strong>Portfolio Suggestion:</strong>${formatAIText(data.strategy_suggestion)}`;
    } catch (_) {
        // best effort
    }
}

// ── Feature Importance ────────────────────────────
async function loadFeatureImportance() {
    const container = document.getElementById('feature-list');
    try {
        const res = await fetch(`/api/feature-importance/${encodeURIComponent(TICKER)}`);
        if (!res.ok) {
            container.innerHTML = '<p class="muted-text">Feature analysis will be available after predictions are generated.</p>';
            return;
        }
        const data = await res.json();
        if (data.error) {
            container.innerHTML = `<p class="muted-text">${data.error}</p>`;
            return;
        }

        const factors = data.key_factors || [];
        if (!factors.length) {
            container.innerHTML = '<p class="muted-text">No key factors identified yet.</p>';
            return;
        }

        let html = '<div class="feature-cards">';
        for (const f of factors) {
            const impactClass = f.impact === 'bullish' ? 'bullish' : f.impact === 'bearish' ? 'bearish' : 'neutral-impact';
            const impactIcon = f.impact === 'bullish' ? '↑' : f.impact === 'bearish' ? '↓' : '—';
            const barWidth = Math.min(f.weight * 100 * 6, 100);  // Scale for visual

            html += `
            <div class="feature-card ${impactClass}">
                <div class="feature-header">
                    <span class="feature-name">${f.factor}</span>
                    <span class="feature-impact ${impactClass}">${impactIcon} ${f.impact}</span>
                </div>
                <div class="feature-value">${f.value}</div>
                <div class="feature-bar-track">
                    <div class="feature-bar-fill ${impactClass}" style="width:${barWidth}%"></div>
                </div>
            </div>`;
        }
        html += '</div>';

        container.innerHTML = html;
    } catch (e) {
        container.innerHTML = '<p class="muted-text">Feature analysis unavailable.</p>';
    }
}

// ── News Section ──────────────────────────────────
async function loadNews() {
    const container = document.getElementById('news-content');
    try {
        const res = await fetch(`/api/news/${encodeURIComponent(TICKER)}`);
        if (!res.ok) {
            container.innerHTML = '<p class="muted-text">News sentiment unavailable.</p>';
            return;
        }
        const data = await res.json();
        if (data.error) {
            container.innerHTML = `<p class="muted-text">${data.error}</p>`;
            return;
        }

        if (data.sentiment) {
            container.innerHTML = `
            <div class="news-card">
                <div class="news-body">${formatAIText(data.sentiment)}</div>
            </div>`;
        } else {
            container.innerHTML = '<p class="muted-text">No news sentiment available.</p>';
        }
    } catch (e) {
        container.innerHTML = '<p class="muted-text">News loading failed.</p>';
    }
}
