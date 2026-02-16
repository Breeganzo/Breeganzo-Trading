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
let currentPeriod = '1d';
let currentInterval = '1m';
let chartRefreshTimer = null;
let metricExplainCache = {};
let metricTooltipHideTimer = null;

// ── Init ──────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    initChart();
    loadChartData(currentPeriod, currentInterval);
    loadLivePrice();
    loadPrediction();
    loadPriceTracker();
    loadGroqForecast();
    loadFeatureImportance();
    loadNews();
    refreshPortfolioStatus();
    bindPredictionMetricExplainers();

    // Auto-refresh live quote every 3 seconds.
    autoRefreshInterval = setInterval(() => {
        loadLivePrice();
        loadPriceTracker();
    }, 3000);

    // Refresh visible chart data window so graph keeps moving.
    chartRefreshTimer = setInterval(() => {
        if (document.visibilityState !== 'visible') return;
        if (currentInterval === '1d') return;
        loadChartData(currentPeriod, currentInterval);
    }, 15000);

    // Refresh AI narrative blocks periodically (news + Groq outlook).
    setInterval(() => {
        loadNews();
        loadGroqForecast();
    }, 15 * 60 * 1000);
    document.addEventListener('keydown', (evt) => {
        if (evt.key === 'Escape') hideMetricTooltip(true);
    });
    document.addEventListener('click', (evt) => {
        const tip = document.getElementById('metric-tooltip');
        if (!tip) return;
        if (tip.contains(evt.target)) return;
        if (evt.target.closest('.clickable-metric') || evt.target.closest('.strategy-evidence-explain')) {
            return;
        }
        hideMetricTooltip(true);
    });
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

function formatTimestamp(value) {
    if (!value) return '—';
    const d = new Date(value);
    if (!Number.isNaN(d.getTime())) {
        return d.toLocaleString('en-IN', {
            day: '2-digit',
            month: 'short',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: true,
            timeZone: 'Asia/Kolkata',
        }) + ' IST';
    }
    return String(value);
}

function toNum(value, fallback = 0) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
}

function istDateStrNow() {
    return new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Kolkata' });
}

function openWindowFallbackIso(dateStr = '', minute = 20) {
    const day = dateStr || istDateStrNow();
    const mm = String(Math.max(15, Math.min(30, minute))).padStart(2, '0');
    return `${day}T09:${mm}:00+05:30`;
}

function formatOpenWindowTime(value, dateStr = '', minute = 20) {
    return formatTimestamp(value || openWindowFallbackIso(dateStr, minute));
}

function moveMetricTooltip(evt) {
    const tip = document.getElementById('metric-tooltip');
    if (!tip || !evt || tip.classList.contains('hidden')) return;
    tip.style.left = `${evt.pageX + 14}px`;
    tip.style.top = `${evt.pageY + 14}px`;
}

function clearMetricTooltipHideTimer() {
    if (metricTooltipHideTimer) {
        clearTimeout(metricTooltipHideTimer);
        metricTooltipHideTimer = null;
    }
}

function scheduleMetricTooltipHide() {
    clearMetricTooltipHideTimer();
    metricTooltipHideTimer = setTimeout(() => {
        const tip = document.getElementById('metric-tooltip');
        if (!tip) return;
        if (tip.matches(':hover') || tip.matches(':focus-within')) return;
        tip.classList.add('hidden');
    }, 180);
}

function bindMetricTooltipInteractions() {
    const tip = document.getElementById('metric-tooltip');
    if (!tip || tip.dataset.bound === '1') return;
    tip.dataset.bound = '1';
    tip.addEventListener('mouseenter', clearMetricTooltipHideTimer);
    tip.addEventListener('mouseleave', scheduleMetricTooltipHide);
    tip.addEventListener('focusin', clearMetricTooltipHideTimer);
    tip.addEventListener('focusout', scheduleMetricTooltipHide);
}

function renderExplainPopover(bodyEl, text) {
    const full = String(text || '').trim();
    bodyEl.innerHTML = '';
    if (full.length <= 260) {
        bodyEl.textContent = full || 'No explanation available';
        return;
    }
    const summary = document.createElement('div');
    summary.className = 'explain-summary';
    summary.textContent = `${full.slice(0, 260).trim()}...`;

    const readMore = document.createElement('button');
    readMore.type = 'button';
    readMore.className = 'tf-btn explain-read-more';
    readMore.textContent = 'Read more';

    const fullText = document.createElement('div');
    fullText.className = 'explain-full hidden';
    fullText.textContent = full;

    readMore.addEventListener('click', () => {
        const expanded = !fullText.classList.contains('hidden');
        fullText.classList.toggle('hidden', expanded);
        summary.classList.toggle('hidden', !expanded);
        readMore.textContent = expanded ? 'Read more' : 'Show less';
    });

    bodyEl.append(summary, readMore, fullText);
}

async function showMetricTooltip(evt, term, context = '') {
    const tip = document.getElementById('metric-tooltip');
    const titleEl = document.getElementById('metric-tooltip-title');
    const bodyEl = document.getElementById('metric-tooltip-body');
    if (!tip || !titleEl || !bodyEl) return;

    bindMetricTooltipInteractions();
    clearMetricTooltipHideTimer();
    titleEl.textContent = term;
    bodyEl.textContent = 'Loading explanation...';
    if (evt && Number.isFinite(evt.pageX) && Number.isFinite(evt.pageY)) {
        moveMetricTooltip(evt);
    }
    tip.classList.remove('hidden');

    const key = `${term}::${context}`;
    if (metricExplainCache[key]) {
        renderExplainPopover(bodyEl, metricExplainCache[key]);
        return;
    }

    try {
        const res = await fetch(`/api/explain-risk-term?term=${encodeURIComponent(term)}&context=${encodeURIComponent(context || 'stock page metric')}`);
        const data = await res.json();
        const text = data.explanation || data.error || 'No explanation';
        metricExplainCache[key] = text;
        renderExplainPopover(bodyEl, text);
    } catch (e) {
        bodyEl.textContent = `Explanation unavailable: ${e.message}`;
    }
}

function hideMetricTooltip(force = false) {
    const tip = document.getElementById('metric-tooltip');
    if (!tip) return;
    if (force) {
        clearMetricTooltipHideTimer();
        tip.classList.add('hidden');
        return;
    }
    scheduleMetricTooltipHide();
}

function handleMetricLabelKey(evt, term, context = '') {
    if (!evt) return;
    if (evt.key === 'Enter' || evt.key === ' ') {
        evt.preventDefault();
        showMetricTooltip(evt, term, context);
    } else if (evt.key === 'Escape') {
        hideMetricTooltip(true);
    }
}

function bindPredictionMetricExplainers() {
    const bindings = [
        {
            id: 'pred-confidence-label',
            term: 'Confidence',
            context: 'Stock AI prediction confidence',
        },
        {
            id: 'pred-agreement-label',
            term: 'Model Agreement',
            context: 'Stock AI prediction model agreement',
        },
    ];

    for (const b of bindings) {
        const el = document.getElementById(b.id);
        if (!el) continue;
        el.setAttribute('tabindex', '0');
        el.setAttribute('role', 'button');
        el.addEventListener('mouseenter', (evt) => showMetricTooltip(evt, b.term, b.context));
        el.addEventListener('mousemove', moveMetricTooltip);
        el.addEventListener('mouseleave', hideMetricTooltip);
        el.addEventListener('click', (evt) => showMetricTooltip(evt, b.term, b.context));
        el.addEventListener('focus', (evt) => showMetricTooltip(evt, b.term, b.context));
        el.addEventListener('blur', hideMetricTooltip);
        el.addEventListener('keydown', (evt) => handleMetricLabelKey(evt, b.term, b.context));
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
    lineSeries = chart.addCandlestickSeries({
        upColor: '#00d09c',
        downColor: '#eb5757',
        borderUpColor: '#00d09c',
        borderDownColor: '#eb5757',
        wickUpColor: '#00d09c',
        wickDownColor: '#eb5757',
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
    const point = param.seriesData?.get?.(lineSeries);
    let price = point?.close ?? point?.value;
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
    currentPeriod = period;
    currentInterval = interval;
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
            priceData.push({
                time,
                open: Number(r.open || r.close || 0),
                high: Number(r.high || r.close || 0),
                low: Number(r.low || r.close || 0),
                close: Number(r.close || 0),
            });
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
    currentPeriod = period;
    currentInterval = interval;
    loadChartData(period, interval);
}

// ── Live Price ────────────────────────────────────
async function loadLivePrice() {
    const statusEl = document.getElementById('quote-status');
    const setStatus = (txt) => { if (statusEl) statusEl.textContent = txt; };
    try {
        const res = await fetch(`/api/prices?tickers=${encodeURIComponent(TICKER)}`);
        const data = await res.json();
        let price = data[TICKER];

        // Fallback to price-tracker snapshot if batch quote is missing.
        if (!price) {
            const tRes = await fetch(`/api/price-tracker/${encodeURIComponent(TICKER)}`);
            const tData = await tRes.json();
            if (!tRes.ok || tData.error) {
                setStatus('Quote unavailable (retrying)');
                return;
            }
            price = {
                price: Number(tData.current_price || 0),
                prev_close: Number(tData.prev_close || 0),
                open: Number(tData.open_price || 0),
                high: Number(tData.high || tData.current_price || 0),
                low: Number(tData.low || tData.current_price || 0),
                volume: Number(tData.volume || 0),
                change: Number(tData.change || 0),
                change_pct: Number(tData.change_pct || 0),
            };
            setStatus('Live quote fallback');
        } else {
            setStatus('Live quote');
        }

        latestLivePrice = Number(price.price || 0);
        if (!(latestLivePrice > 0)) {
            setStatus('Quote unavailable');
            return;
        }

        document.getElementById('live-price').textContent = `₹${formatN(price.price)}`;
        const changeEl = document.getElementById('live-change');
        const sign = Number(price.change || 0) >= 0 ? '+' : '';
        changeEl.textContent = `${sign}${Number(price.change || 0).toFixed(2)} (${Number(price.change_pct || 0).toFixed(2)}%)`;
        changeEl.className = `stock-change-big ${Number(price.change || 0) >= 0 ? 'up-color' : 'down-color'}`;

        const low = Number(price.low || 0) > 0 ? Number(price.low) : latestLivePrice;
        const high = Number(price.high || 0) > 0 ? Number(price.high) : latestLivePrice;
        const open = Number(price.open || 0) > 0 ? Number(price.open) : latestLivePrice;
        const prevClose = Number(price.prev_close || 0) > 0 ? Number(price.prev_close) : open;

        document.getElementById('today-low').textContent = `₹${formatN(low)}`;
        document.getElementById('today-high').textContent = `₹${formatN(high)}`;
        document.getElementById('stat-open').textContent = `₹${formatN(open)}`;
        document.getElementById('stat-prev-close').textContent = `₹${formatN(prevClose)}`;
        document.getElementById('stat-volume').textContent = formatVolume(price.volume);

        if (high > low) {
            const pct = ((latestLivePrice - low) / (high - low)) * 100;
            document.getElementById('price-marker').style.left = `${Math.max(0, Math.min(100, pct))}%`;
        }
    } catch (e) {
        console.error('Live price failed:', e);
        setStatus('Quote error');
    }
}

// ── Prediction ────────────────────────────────────
async function loadPrediction(options = {}) {
    const loading = document.getElementById('pred-loading');
    const content = document.getElementById('pred-content');
    const force = Boolean(options.force);
    const useLatestStored = Boolean(options.useLatestStored);

    loading.classList.remove('hidden');
    loading.innerHTML = '<div class="spinner"></div><p>Running ML models...</p>';
    content.classList.add('hidden');

    try {
        const params = new URLSearchParams({
            force: force ? 'true' : 'false',
            use_latest_stored: useLatestStored ? 'true' : 'false',
        });
        const res = await fetch(`/api/predict/${encodeURIComponent(TICKER)}?${params.toString()}`);

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
        const confVal = Number(data.confidence);
        const agreementVal = Number(data.model_agreement);
        document.getElementById('pred-confidence').textContent = Number.isFinite(confVal) ? `${confVal.toFixed(0)}%` : '—';
        document.getElementById('pred-agreement').textContent = Number.isFinite(agreementVal) ? `${agreementVal.toFixed(0)}%` : '—';

        // Show generated-at timestamp
        const genEl = document.getElementById('pred-generated-at');
        if (genEl) {
            const mode = data.cache_policy || 'fresh_inference';
            genEl.textContent = data.generated_at ? `${data.generated_at} (${mode})` : mode;
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

        // Trade levels are strategy-only; pull from strategy endpoint.
        await loadStrategyTradeLevels({ force, useLatestStored });

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

async function loadStrategyTradeLevels(options = {}) {
    const force = Boolean(options.force);
    const useLatestStored = Boolean(options.useLatestStored);
    try {
        const params = new URLSearchParams({
            force: force ? 'true' : 'false',
            use_latest_stored: useLatestStored ? 'true' : 'false',
        });
        const res = await fetch(`/api/strategy-price/${encodeURIComponent(TICKER)}?${params.toString()}`);
        const data = await res.json();
        if (!res.ok || data.error) {
            document.getElementById('level-entry').textContent = '—';
            document.getElementById('level-target').textContent = '—';
            document.getElementById('level-sl').textContent = '—';
            document.getElementById('level-rr').textContent = '—';
            return;
        }
        const current = Number(data.current_price || predictionData?.current_price || latestLivePrice || 0);
        const target = Number(data.strategy_price || 0);
        const rr = Number(data.rr_ratio || 0);
        const entry = current > 0 ? current : Number(data.open_price || 0);
        let stopLoss = 0;
        if (entry > 0 && target > 0 && rr > 0) {
            const risk = Math.abs(target - entry) / rr;
            stopLoss = target >= entry ? entry - risk : entry + risk;
        }
        document.getElementById('level-entry').textContent = entry > 0 ? `₹${formatN(entry)}` : '—';
        document.getElementById('level-target').textContent = target > 0 ? `₹${formatN(target)}` : '—';
        document.getElementById('level-sl').textContent = stopLoss > 0 ? `₹${formatN(stopLoss)}` : '—';
        document.getElementById('level-rr').textContent = rr > 0 ? rr.toFixed(2) : '—';
    } catch (_) {
        document.getElementById('level-entry').textContent = '—';
        document.getElementById('level-target').textContent = '—';
        document.getElementById('level-sl').textContent = '—';
        document.getElementById('level-rr').textContent = '—';
    }
}

function refreshPredictionNow() {
    loadPrediction({ force: true, useLatestStored: false });
}

// ── Expected vs Actual for this stock ─────────────
async function loadStockEVA() {
    const container = document.getElementById('stock-eva-content');

    try {
        const trackerRes = await fetch(`/api/price-tracker/${encodeURIComponent(TICKER)}`);
        const trackerData = trackerRes.ok ? await trackerRes.json() : null;
        const datesRes = await fetch('/api/prediction-dates');
        const dates = await datesRes.json();
        let html = '<div style="margin-top: 12px;">';

        if (trackerData && !trackerData.error) {
            const openPx = toNum(trackerData.open_price || predictionData?.current_price, 0);
            const strategyOpenPx = toNum(
                trackerData.strategy_predicted_price || trackerData.predicted_price || predictionData?.predicted_price,
                openPx
            );
            const currentPx = toNum(trackerData.current_price || predictionData?.current_price, 0);
            const label = trackerData.display_price_label || 'Current Price';
            const direction = strategyOpenPx > openPx ? 'UP' : strategyOpenPx < openPx ? 'DOWN' : 'FLAT';

            html += `
            <div class="pred-grid">
                <div class="pred-card">
                    <span class="pred-label">Market Open Price</span>
                    <span class="pred-value">₹${formatN(openPx)}</span>
                    <span class="muted-text">${trackerData.open_price_captured_at ? formatTimestamp(trackerData.open_price_captured_at) : 'Market Open (09:15 IST)'}</span>
                </div>
                <div class="pred-card">
                    <span class="pred-label">Strategy @ Open</span>
                    <span class="pred-value ${strategyOpenPx >= openPx ? 'up-color' : 'down-color'}">₹${formatN(strategyOpenPx)}</span>
                    <span class="muted-text">${formatOpenWindowTime(trackerData.strategy_predicted_at_open, istDateStrNow(), 20)}</span>
                </div>
                <div class="pred-card">
                    <span class="pred-label">${label}</span>
                    <span class="pred-value ${currentPx >= openPx ? 'up-color' : 'down-color'}">₹${formatN(currentPx)}</span>
                    <span class="muted-text">${formatTimestamp(trackerData.current_snapshot_at)}</span>
                </div>
                <div class="pred-card">
                    <span class="pred-label">Strategy Direction</span>
                    <span class="pred-value ${direction === 'UP' ? 'up-color' : direction === 'DOWN' ? 'down-color' : ''}">${direction}</span>
                    <span class="muted-text">Based on Open vs Strategy@Open</span>
                </div>
            </div>
            <p class="muted-text" style="margin-top: 12px;">
                Expected vs Actual is evaluated as Strategy-at-Open direction versus actual market direction.
            </p>`;
        }

        if (dates.length) {
            const latestDate = dates[0];
            try {
                const evaRes = await fetch(`/api/expected-vs-actual?date=${latestDate}`);
                const evaData = await evaRes.json();

                if (!evaData.error && evaData.results) {
                    const stockResult = evaData.results.find(r => r.ticker === TICKER);
                    if (stockResult) {
                        const dirClass = stockResult.direction_correct ? 'correct' : 'wrong';
                        const todayIst = new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Kolkata' });
                        const closeLabel = latestDate === todayIst
                            ? (stockResult.actual_close ? 'Current / Close Price' : 'Current Price')
                            : 'Close Price';
                        const openPx = toNum(stockResult.market_open_price ?? stockResult.open_price ?? predictionData?.current_price, 0);
                        const strategyOpenPx = toNum(stockResult.strategy_price_at_open ?? stockResult.predicted_price ?? openPx, openPx);
                        const actualClosePx = toNum(stockResult.actual_close ?? stockResult.actual_price ?? 0, 0);
                        const diffPx = Number.isFinite(Number(stockResult.strategy_vs_actual_price_diff))
                            ? Number(stockResult.strategy_vs_actual_price_diff)
                            : (actualClosePx > 0 && strategyOpenPx > 0 ? actualClosePx - strategyOpenPx : 0);
                        const alphaPct = toNum(stockResult.alpha_pct, 0);
                        html += `
                        <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--border);">
                            <h4 class="sub-heading">Last Prediction (${latestDate})</h4>
                            <div class="pred-grid">
                                <div class="pred-card">
                                    <span class="pred-label">Market Open Price</span>
                                    <span class="pred-value">₹${formatN(openPx)}</span>
                                </div>
                                <div class="pred-card">
                                    <span class="pred-label">Strategy @ Open</span>
                                    <span class="pred-value ${strategyOpenPx >= openPx ? 'up-color' : 'down-color'}">₹${formatN(strategyOpenPx)}</span>
                                    <span class="muted-text">${formatOpenWindowTime(stockResult.strategy_predicted_at_open, latestDate, 20)}</span>
                                </div>
                                <div class="pred-card">
                                    <span class="pred-label">${closeLabel}</span>
                                    <span class="pred-value ${actualClosePx >= openPx ? 'up-color' : 'down-color'}">₹${formatN(actualClosePx)}</span>
                                </div>
                                <div class="pred-card">
                                    <span class="pred-label">Actual vs Strategy Price</span>
                                    <span class="pred-value ${diffPx >= 0 ? 'up-color' : 'down-color'}">
                                        ${diffPx >= 0 ? '+' : ''}₹${formatN(diffPx)}
                                    </span>
                                </div>
                                <div class="pred-card">
                                    <span class="pred-label">Direction Check</span>
                                    <span class="pred-value"><span class="direction-badge ${dirClass}">${stockResult.direction_correct ? '✓ Strategy Correct' : '✗ Strategy Wrong'}</span></span>
                                </div>
                                <div class="pred-card">
                                    <span class="pred-label">Alpha Generated</span>
                                    <span class="pred-value ${alphaPct >= 0 ? 'up-color' : 'down-color'}">
                                        ${alphaPct >= 0 ? '+' : ''}${alphaPct.toFixed(3)}%
                                    </span>
                                </div>
                            </div>
                        </div>`;
                    }
                }
            } catch (e) {
                // Ignore — no past data for this stock
            }
        } else {
            html += '<p class="muted-text">No past predictions logged yet. End-of-day close and alpha will appear after market data is available.</p>';
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

function classifyStrategyScore(score) {
    if (score >= 1) return { cls: 'bullish', label: 'Bullish' };
    if (score <= -1) return { cls: 'bearish', label: 'Bearish' };
    return { cls: 'neutral', label: 'Neutral' };
}

function renderStrategyEvidenceCard(title, rows) {
    if (!rows.length) {
        return `
        <div class="strategy-evidence-card">
            <div class="strategy-evidence-title">${title}</div>
            <div class="muted-text">Not enough live values.</div>
        </div>`;
    }
    return `
    <div class="strategy-evidence-card">
        <div class="strategy-evidence-title">${title}</div>
        ${rows.map((row) => {
            const tag = classifyStrategyScore(row.score);
            const term = row.term || row.key;
            return `
            <div class="strategy-evidence-row clickable-metric strategy-evidence-explain"
                data-strategy-term="${term}"
                data-strategy-context="${title}"
                tabindex="0"
                role="button">
                <span class="strategy-evidence-key">${row.key}</span>
                <span class="strategy-evidence-value">${row.value}</span>
                <span class="strategy-evidence-tag ${tag.cls}">${tag.label}</span>
            </div>`;
        }).join('')}
    </div>`;
}

function buildStrategyEvidence(predData) {
    const indicators = predData?.indicators || {};
    const fundamentals = predData?.fundamentals || {};
    const greeks = predData?.greeks || {};

    const indicatorRows = [];
    const rsi = toNum(indicators.rsi, NaN);
    if (Number.isFinite(rsi)) {
        const score = rsi < 35 ? 1 : rsi > 70 ? -1 : 0;
        indicatorRows.push({ key: 'RSI (14)', value: rsi.toFixed(1), score });
    }
    const macd = toNum(indicators.macd, NaN);
    const macdSignal = toNum(indicators.macd_signal, NaN);
    if (Number.isFinite(macd) && Number.isFinite(macdSignal)) {
        indicatorRows.push({
            key: 'MACD vs Signal',
            value: `${macd.toFixed(3)} / ${macdSignal.toFixed(3)}`,
            score: macd > macdSignal ? 1 : -1,
        });
    }
    const adx = toNum(indicators.adx, NaN);
    if (Number.isFinite(adx)) {
        indicatorRows.push({
            key: 'ADX',
            value: adx.toFixed(1),
            score: adx >= 25 ? 1 : 0,
        });
    }
    const volRatio = toNum(indicators.volume_ratio, NaN);
    if (Number.isFinite(volRatio)) {
        indicatorRows.push({
            key: 'Volume Ratio',
            value: `${volRatio.toFixed(2)}x`,
            score: volRatio >= 1.2 ? 1 : volRatio < 0.8 ? -1 : 0,
        });
    }

    const fundamentalRows = [];
    const pe = toNum(fundamentals.pe_ratio, NaN);
    if (Number.isFinite(pe)) {
        fundamentalRows.push({
            key: 'P/E Ratio',
            value: pe.toFixed(2),
            score: pe > 0 && pe <= 25 ? 1 : pe > 45 ? -1 : 0,
        });
    }
    const roe = toNum(fundamentals.roe, NaN);
    if (Number.isFinite(roe)) {
        fundamentalRows.push({
            key: 'ROE',
            value: `${(roe * 100).toFixed(1)}%`,
            score: roe >= 0.15 ? 1 : roe < 0.08 ? -1 : 0,
        });
    }
    const debtToEquity = toNum(fundamentals.debt_to_equity, NaN);
    if (Number.isFinite(debtToEquity)) {
        fundamentalRows.push({
            key: 'Debt / Equity',
            value: debtToEquity.toFixed(2),
            score: debtToEquity <= 1.0 ? 1 : debtToEquity >= 2.0 ? -1 : 0,
        });
    }
    const earningsGrowth = toNum(fundamentals.earnings_growth, NaN);
    if (Number.isFinite(earningsGrowth)) {
        fundamentalRows.push({
            key: 'Earnings Growth',
            value: `${(earningsGrowth * 100).toFixed(1)}%`,
            score: earningsGrowth > 0 ? 1 : -1,
        });
    }

    const greekRows = [];
    const delta = toNum(greeks.delta, NaN);
    if (Number.isFinite(delta)) {
        greekRows.push({
            key: 'Delta',
            value: delta.toFixed(3),
            score: delta >= 0.55 ? 1 : delta <= 0.45 ? -1 : 0,
        });
    }
    const gamma = toNum(greeks.gamma, NaN);
    if (Number.isFinite(gamma)) {
        greekRows.push({
            key: 'Gamma',
            value: gamma.toFixed(4),
            score: gamma >= 0.03 ? 1 : 0,
        });
    }
    const theta = toNum(greeks.theta, NaN);
    if (Number.isFinite(theta)) {
        greekRows.push({
            key: 'Theta',
            value: theta.toFixed(3),
            score: theta >= 0 ? 0 : -1,
        });
    }
    const iv = toNum(greeks.iv, NaN);
    if (Number.isFinite(iv)) {
        greekRows.push({
            key: 'Implied Vol',
            value: `${(iv * 100).toFixed(1)}%`,
            score: iv <= 0.25 ? 1 : iv >= 0.45 ? -1 : 0,
        });
    }

    return `
    <div class="strategy-evidence">
        <div class="strategy-evidence-grid">
            ${renderStrategyEvidenceCard('Strategy Indicators', indicatorRows)}
            ${renderStrategyEvidenceCard('Strategy Fundamentals', fundamentalRows)}
            ${renderStrategyEvidenceCard('Strategy Greeks', greekRows)}
        </div>
    </div>`;
}

function bindStrategyEvidenceExplainers() {
    const container = document.getElementById('ml-strategy');
    if (!container) return;
    const nodes = container.querySelectorAll('.strategy-evidence-explain');
    nodes.forEach((el) => {
        if (el.dataset.explainBound === '1') return;
        el.dataset.explainBound = '1';
        const term = el.dataset.strategyTerm || 'Strategy Metric';
        const context = `${el.dataset.strategyContext || 'Strategy evidence'} for ${TICKER}`;
        el.addEventListener('mouseenter', (evt) => showMetricTooltip(evt, term, context));
        el.addEventListener('mousemove', moveMetricTooltip);
        el.addEventListener('mouseleave', hideMetricTooltip);
        el.addEventListener('focus', (evt) => showMetricTooltip(evt, term, context));
        el.addEventListener('blur', hideMetricTooltip);
        el.addEventListener('click', (evt) => showMetricTooltip(evt, term, context));
        el.addEventListener('keydown', (evt) => handleMetricLabelKey(evt, term, context));
    });
}

// ── Strategy Loading ──────────────────────────────
async function loadStrategies(predData) {
    // ML Strategy — immediate (from prediction data)
    const mlEl = document.getElementById('ml-strategy');
    if (predData) {
        const signal = predData.signal || 'HOLD';
        const ret = toNum(predData.predicted_return, 0);
        const conf = toNum(predData.confidence, 50);
        const agreement = toNum(predData.model_agreement, 0);
        const strategyEvidence = buildStrategyEvidence(predData);
        mlEl.innerHTML = `
            <div class="strategy-signal ${signal}">${signal}</div>
            <p>Predicted return: <strong class="${ret >= 0 ? 'up-color' : 'down-color'}">${ret >= 0 ? '+' : ''}${ret.toFixed(3)}%</strong> 
               with ${conf.toFixed(0)}% confidence and ${agreement.toFixed(0)}% model agreement.</p>
            <p>Entry: ₹${formatN(predData.entry_price)} → Target: ₹${formatN(predData.target_price)} | SL: ₹${formatN(predData.stop_loss)}</p>
            <p>Risk:Reward = ${predData.risk_reward?.toFixed(1)}</p>
            <p class="muted-text">Strategy evidence below uses live model inputs (indicators, fundamentals, and option greeks) with rule tags.</p>
            ${strategyEvidence}
        `;
        bindStrategyEvidenceExplainers();
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

        let html = `
        <p class="section-help-text">
            This table tracks how many strategy predictions were logged today, how many already had outcome data, and how often strategy direction matched actual market direction.
        </p>
        <div class="tracking-cards">`;

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

        const openPrice = Number(stock.open_price || 0);
        const nextDayMode = stock.prediction_mode === 'next_day_after_close';
        const predPrice = Number(
            (nextDayMode
                ? stock.next_day_strategy_predicted_price
                : stock.strategy_predicted_price)
            || stock.strategy_predicted_price
            || stock.predicted_price
            || 0
        );
        const currPrice = Number(stock.current_price || 0);
        const aiPrice = Number(
            (nextDayMode
                ? stock.next_day_ai_predicted_price
                : stock.ai_predicted_price)
            || stock.ai_predicted_price
            || 0
        );
        const currentStrategyPrice = Number(
            stock.current_strategy_predicted_price
            || stock.next_day_strategy_predicted_price
            || predPrice
            || 0
        );
        const currentAiPrice = Number(
            stock.current_ai_predicted_price
            || stock.next_day_ai_predicted_price
            || aiPrice
            || 0
        );
        const aiAvailable = aiPrice > 0;
        const currentAiAvailable = currentAiPrice > 0;

        // Update tracker cards
        document.getElementById('tracker-open').textContent = `₹${formatN(openPrice)}`;
        document.getElementById('tracker-predicted').textContent = `₹${formatN(predPrice)}`;
        document.getElementById('tracker-current').textContent = `₹${formatN(currPrice)}`;
        const strategyLabelEl = document.getElementById('tracker-strategy-label');
        if (strategyLabelEl) {
            strategyLabelEl.textContent = nextDayMode ? `Strategy Prediction (Next Day ${stock.predicted_for_date || ''})` : 'Strategy Predicted';
        }
        const aiLabelEl = document.getElementById('tracker-ai-label');
        if (aiLabelEl) {
            aiLabelEl.textContent = nextDayMode ? `AI Prediction (Next Day ${stock.predicted_for_date || ''})` : 'AI Predicted (Groq)';
        }
        const currentLabelEl = document.getElementById('tracker-current-label');
        if (currentLabelEl) currentLabelEl.textContent = stock.display_price_label || 'Current Price';
        const aiEl = document.getElementById('tracker-ai');
        if (aiEl) aiEl.textContent = aiAvailable ? `₹${formatN(aiPrice)}` : '—';
        const openTimeEl = document.getElementById('tracker-open-time');
        if (openTimeEl) {
            openTimeEl.textContent = stock.open_price_captured_at
                ? `Captured: ${formatTimestamp(stock.open_price_captured_at)}`
                : 'Market Open (09:15 IST)';
        }
        const strategyOpenTimeEl = document.getElementById('tracker-strategy-open-time');
        if (strategyOpenTimeEl) {
            strategyOpenTimeEl.textContent = nextDayMode
                ? `Predicted: ${formatTimestamp(stock.next_day_predicted_at || stock.current_strategy_predicted_at)}`
                : `Predicted: ${formatOpenWindowTime(stock.strategy_predicted_at_open, istDateStrNow(), 20)}`;
        }
        const aiOpenTimeEl = document.getElementById('tracker-ai-open-time');
        if (aiOpenTimeEl) {
            aiOpenTimeEl.textContent = nextDayMode
                ? `Predicted: ${formatTimestamp(stock.next_day_predicted_at || stock.current_ai_predicted_at)}`
                : `Predicted: ${formatOpenWindowTime(stock.ai_predicted_at_open, istDateStrNow(), 22)}`;
        }
        const currentTimeEl = document.getElementById('tracker-current-time');
        if (currentTimeEl) {
            currentTimeEl.textContent = `Updated: ${formatTimestamp(stock.current_snapshot_at)}`;
        }

        const currentStrategyPriceEl = document.getElementById('current-strategy-price');
        if (currentStrategyPriceEl) {
            currentStrategyPriceEl.textContent = currentStrategyPrice > 0 ? `₹${formatN(currentStrategyPrice)}` : '—';
        }
        const currentStrategyTimeEl = document.getElementById('current-strategy-time');
        if (currentStrategyTimeEl) {
            currentStrategyTimeEl.textContent = formatTimestamp(stock.current_strategy_predicted_at);
        }
        const currentAiPriceEl = document.getElementById('current-ai-price');
        if (currentAiPriceEl) {
            currentAiPriceEl.textContent = currentAiAvailable ? `₹${formatN(currentAiPrice)}` : '—';
        }
        const currentAiTimeEl = document.getElementById('current-ai-time');
        if (currentAiTimeEl) {
            currentAiTimeEl.textContent = formatTimestamp(stock.current_ai_predicted_at);
        }

        // Percentage changes from open
        const predPct = Number(
            (nextDayMode ? stock.close_to_strategy_pct : stock.open_to_predicted_pct) || 0
        );
        const currPct = Number(stock.open_to_current_pct || 0);
        const aiPct = aiAvailable
            ? Number(
                nextDayMode
                    ? (stock.close_to_ai_pct ?? ((aiPrice - currPrice) / (currPrice || 1) * 100))
                    : (stock.open_to_ai_predicted_pct ?? ((aiPrice - openPrice) / (openPrice || 1) * 100))
            )
            : null;

        const predEl = document.getElementById('tracker-predicted-pct');
        predEl.textContent = `${predPct >= 0 ? '+' : ''}${predPct.toFixed(3)}% ${nextDayMode ? 'vs close' : 'from open'}`;
        predEl.className = `tracker-change ${predPct >= 0 ? 'up-color' : 'down-color'}`;

        const currEl = document.getElementById('tracker-current-pct');
        currEl.textContent = `${currPct >= 0 ? '+' : ''}${currPct.toFixed(3)}% from open`;
        currEl.className = `tracker-change ${currPct >= 0 ? 'up-color' : 'down-color'}`;

        const aiPctEl = document.getElementById('tracker-ai-pct');
        if (aiPctEl) {
            if (aiAvailable && Number.isFinite(aiPct)) {
                aiPctEl.textContent = `${aiPct >= 0 ? '+' : ''}${Number(aiPct).toFixed(3)}% ${nextDayMode ? 'vs close' : 'from open'}`;
                aiPctEl.className = `tracker-change ${aiPct >= 0 ? 'up-color' : 'down-color'}`;
            } else {
                aiPctEl.textContent = 'Awaiting Groq forecast';
                aiPctEl.className = 'tracker-change';
            }
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
        const aiAvailable = Boolean(data.ai_available) && aiPrice > 0;
        const aiPctRaw = data.open_to_ai_predicted_pct;
        const aiPct = Number.isFinite(Number(aiPctRaw))
            ? Number(aiPctRaw)
            : ((aiPrice - Number(data.open_price || 0)) / Math.max(Number(data.open_price || 1), 1)) * 100;
        if (!aiAvailable) {
            container.innerHTML = `
                <div class="news-card">
                    <div class="news-body">
                        <p><strong>Groq AI price forecast:</strong> unavailable right now.</p>
                        <p><strong>Outlook:</strong> ${data.outlook || 'Unavailable'}</p>
                        <p>${formatAIText(data.rationale || '').replace(/^<p>|<\/p>$/g, '')}</p>
                        <p class="muted-text">Source: ${data.ai_source || 'fallback'} | Generated: ${data.generated_at || 'now'}</p>
                    </div>
                </div>`;
            return;
        }
        container.innerHTML = `
            <div class="news-card">
                <div class="news-body">
                    <p><strong>Groq AI Predicted Price:</strong> ₹${formatN(aiPrice)}
                    <span class="${aiPct >= 0 ? 'up-color' : 'down-color'}">(${aiPct >= 0 ? '+' : ''}${aiPct.toFixed(3)}% vs open)</span></p>
                    <p><strong>Outlook:</strong> ${data.outlook || 'Neutral'}</p>
                    <p>${formatAIText(data.rationale || '').replace(/^<p>|<\/p>$/g, '')}</p>
                    <p class="muted-text">Source: ${data.ai_source || 'groq'} | Generated: ${data.generated_at || 'now'}</p>
                </div>
            </div>`;
        const aiTracker = document.getElementById('tracker-ai');
        if (aiTracker) aiTracker.textContent = `₹${formatN(aiPrice)}`;
        const currentAiEl = document.getElementById('current-ai-price');
        if (currentAiEl) currentAiEl.textContent = `₹${formatN(aiPrice)}`;
        const currentAiTimeEl = document.getElementById('current-ai-time');
        if (currentAiTimeEl) currentAiTimeEl.textContent = formatTimestamp(data.generated_at_iso || data.generated_at);
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

async function reviewPlannedTrade() {
    const qty = Number(document.getElementById('portfolio-qty')?.value || 0);
    const customPrice = Number(document.getElementById('portfolio-price')?.value || 0);
    const entryPrice = Number(customPrice || latestLivePrice || predictionData?.current_price || 0);
    const output = document.getElementById('stock-chat-output');
    if (!qty || qty <= 0 || !entryPrice || entryPrice <= 0) {
        if (output) output.textContent = 'Enter a valid buy quantity and price before review.';
        return;
    }
    if (output) output.textContent = 'Groq is reviewing your plan...';
    try {
        const res = await fetch('/api/groq-trade-review', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ticker: TICKER,
                entry_price: entryPrice,
                quantity: qty,
            }),
        });
        const data = await res.json();
        if (!res.ok || data.error) {
            if (output) output.textContent = `Review unavailable: ${data.error || res.status}`;
            return;
        }
        if (output) {
            output.innerHTML = `
                <strong>Planned Trade Review (${data.generated_at || 'now'})</strong>
                ${formatAIText(data.review || 'No review text')}`;
        }
    } catch (e) {
        if (output) output.textContent = `Review failed: ${e.message}`;
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
        const res = await fetch(`/api/news/${encodeURIComponent(TICKER)}?t=${Date.now()}`);
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
                <div class="news-body">
                    ${formatAIText(data.sentiment)}
                    <p class="muted-text">Updated: ${data.generated_at || 'now'}</p>
                </div>
            </div>`;
        } else {
            container.innerHTML = '<p class="muted-text">No news sentiment available.</p>';
        }
    } catch (e) {
        container.innerHTML = '<p class="muted-text">News loading failed.</p>';
    }
}
