/* ============================================================
   Masters AI Trading Bot — Dashboard JS (Enhanced)
   ============================================================ */

// State
let allStockData = {};
let dailyAnalysisData = null;
let currentSector = 'all';
let sectorsData = {};
let sectorTickerOrder = {};
let autoRefreshTimer = null;
let topPickFilter = 'top_buy';
let groupedTopPicks = { top_buy: [], top_sell: [], top_hold: [] };
let metricExplainCache = {};
let premarketOutlookData = [];
let highlightedPremarketTicker = null;
let metricTooltipHideTimer = null;
let useLatestStoredPredictions = false;
let currentMarketStatus = '';
let groqStatusTimer = null;
let topPicksLivePriceTimer = null;
let topPicksLivePriceBusy = false;

// ── Init ──────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    loadSectors();
    loadIndexPrices();
    loadPricesForSector('all');
    loadDailyAnalysis();
    checkGroqStatus();
    startISTClock();
    // Auto-refresh prices every 5 seconds
    autoRefreshTimer = setInterval(() => {
        if (document.visibilityState === 'visible') {
            loadPricesForSector(currentSector);
            loadIndexPrices();
        }
    }, 5000);
    // Refresh daily analysis every 60 seconds (served from server cache)
    setInterval(() => {
        if (document.visibilityState === 'visible') {
            loadDailyAnalysis();
        }
    }, 60000);
    groqStatusTimer = setInterval(() => {
        if (document.visibilityState === 'visible') {
            checkGroqStatus();
        }
    }, 12000);
    topPicksLivePriceTimer = setInterval(() => {
        if (document.visibilityState === 'visible') {
            refreshTopPicksLivePrices();
        }
    }, 1000);
    document.addEventListener('keydown', (evt) => {
        if (evt.key === 'Escape') hideMetricTooltip(true);
    });
    document.addEventListener('click', (evt) => {
        const tip = document.getElementById('metric-tooltip');
        if (!tip) return;
        if (tip.contains(evt.target)) return;
        if (evt.target.closest('.clickable-metric')) return;
        hideMetricTooltip(true);
    });
});

// ── Real-time IST Clock ───────────────────────────
function startISTClock() {
    const formatter = new Intl.DateTimeFormat('en-IN', {
        timeZone: 'Asia/Kolkata',
        day: '2-digit',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true,
    });
    function updateClock() {
        const el = document.getElementById('ist-time');
        if (el) el.textContent = formatter.format(new Date()) + ' IST';
    }
    updateClock();
    setInterval(updateClock, 1000);
}

// ── Status ────────────────────────────────────────
async function checkStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        const mktBadge = document.getElementById('market-status');
        currentMarketStatus = data.market?.status || '';
        if (data.market.status === 'market_open') {
            mktBadge.textContent = '🟢 Market Open';
            mktBadge.className = 'status-badge open';
        } else if (data.market.status === 'after_hours') {
            mktBadge.textContent = '🕓 After Hours';
            mktBadge.className = 'status-badge closed';
        } else {
            mktBadge.textContent = '🔴 ' + data.market.description;
            mktBadge.className = 'status-badge closed';
        }

        const modelBadge = document.getElementById('model-status');
        const modelDetail = document.getElementById('model-load-detail');
        if (data.models_loaded) {
            modelBadge.textContent = `✅ ${data.model_count} Models Ready`;
            modelBadge.className = 'status-badge ready';
            if (modelDetail) {
                const elapsed = data.model_load_elapsed_sec ?? 0;
                modelDetail.textContent = `Loaded in ${elapsed}s`;
            }
        } else if (data.load_error) {
            modelBadge.textContent = '❌ Model Error';
            modelBadge.className = 'status-badge closed';
            if (modelDetail) modelDetail.textContent = data.load_error;
        } else {
            const progress = data.model_load_progress || {};
            const loadedSteps = progress.loaded_steps || 0;
            const totalSteps = progress.total_steps || 0;
            const inProgress = progress.in_progress ? ` (${progress.in_progress})` : '';
            modelBadge.textContent = totalSteps > 0
                ? `⏳ Loading ${loadedSteps}/${totalSteps}${inProgress}`
                : '⏳ Loading models...';
            modelBadge.className = 'status-badge loading';
            if (modelDetail) {
                modelDetail.textContent = `Elapsed ${data.model_load_elapsed_sec ?? 0}s`;
            }
            setTimeout(checkStatus, 3000);
        }
        if (data.groq) {
            updateGroqDegradedBanner(data.groq);
        }
    } catch (e) {
        console.error('Status check failed:', e);
        setTimeout(checkStatus, 5000);
    }
}

function updateGroqDegradedBanner(groqStatus) {
    const banner = document.getElementById('groq-degraded-banner');
    const text = document.getElementById('groq-degraded-text');
    if (!banner || !text) return;
    const degraded = Boolean(groqStatus?.degraded_mode);
    if (!degraded) {
        banner.classList.add('hidden');
        return;
    }
    const reason = groqStatus?.degraded_reason || 'rate-limited';
    const until = groqStatus?.degraded_until_iso ? formatIstTimestamp(groqStatus.degraded_until_iso) : '';
    text.textContent = until
        ? `${reason}. Cached Groq content only until ${until}.`
        : `${reason}. Cached Groq content only until limits recover.`;
    banner.classList.remove('hidden');
}

async function checkGroqStatus() {
    try {
        const res = await fetch('/api/groq-status');
        if (!res.ok) return;
        const data = await res.json();
        updateGroqDegradedBanner(data);
    } catch (_) {
        // Ignore banner failures; dashboard remains functional without Groq.
    }
}

// ── Metric Tooltip (Groq Explain) ─────────────────
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
        const res = await fetch(`/api/explain-risk-term?term=${encodeURIComponent(term)}&context=${encodeURIComponent(context || 'top picks metric')}`);
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

function showAlphaFormulaTooltip(evt) {
    const alphaContext = [
        'Simplified Alpha: alpha = actual_return - benchmark_return.',
        'CAPM Alpha: alpha = R_p - [R_f + beta × (R_m - R_f)].',
        'In this dashboard, simplified alpha is primary for daily trading; CAPM alpha is shown as a risk-adjusted reference.',
        'Click the metric again to keep the popover open and scroll for full explanation.',
    ].join(' ');
    showMetricTooltip(evt, 'Alpha Formula Used', alphaContext);
}

// ── Index Prices ──────────────────────────────────
async function loadIndexPrices() {
    try {
        const res = await fetch('/api/prices?tickers=^NSEI,^NSEBANK');
        const data = await res.json();

        if (data['^NSEI']) {
            const n = data['^NSEI'];
            document.getElementById('nifty-price').textContent = formatPrice(n.price);
            const nChange = document.getElementById('nifty-change');
            nChange.textContent = `${n.change >= 0 ? '+' : ''}${n.change} (${n.change_pct}%)`;
            nChange.className = `banner-change ${n.change >= 0 ? 'up' : 'down'}`;
        }

        if (data['^NSEBANK']) {
            const bn = data['^NSEBANK'];
            document.getElementById('banknifty-price').textContent = formatPrice(bn.price);
            const bnChange = document.getElementById('banknifty-change');
            bnChange.textContent = `${bn.change >= 0 ? '+' : ''}${bn.change} (${bn.change_pct}%)`;
            bnChange.className = `banner-change ${bn.change >= 0 ? 'up' : 'down'}`;
        }
    } catch (e) {
        console.error('Index price fetch failed:', e);
    }
}

// ── Daily Analysis (Opening / Predicted / Current) ─
async function loadDailyAnalysis() {
    try {
        const res = await fetch('/api/daily-analysis');
        if (res.status === 202) {
            // Cache not ready yet — retry in 10 seconds
            setTimeout(loadDailyAnalysis, 10000);
            return;
        }
        if (!res.ok) return;
        dailyAnalysisData = await res.json();

        // Update market mood banner
        const mood = document.getElementById('market-mood');
        if (mood && dailyAnalysisData.market_summary) {
            const ms = dailyAnalysisData.market_summary;
            const ratio = ms.gainers / (ms.gainers + ms.losers || 1);
            if (ratio > 0.6) {
                mood.textContent = '🟢 Bullish';
                mood.className = 'banner-value up-color';
            } else if (ratio < 0.4) {
                mood.textContent = '🔴 Bearish';
                mood.className = 'banner-value down-color';
            } else {
                mood.textContent = '🟡 Mixed';
                mood.className = 'banner-value';
            }
        }
    } catch (e) {
        console.error('Daily analysis failed:', e);
    }
}

// ── Sectors ───────────────────────────────────────
async function loadSectors() {
    try {
        const res = await fetch('/api/sectors');
        sectorsData = await res.json();
        sectorTickerOrder = {};
        for (const [sector, payload] of Object.entries(sectorsData || {})) {
            sectorTickerOrder[sector] = (payload.tickers || []).map(t => t.symbol);
        }
        const allOrdered = [];
        for (const sec of ['large_cap', 'banking', 'mid_cap', 'high_volatility', 'commodities']) {
            if (sectorTickerOrder[sec]) allOrdered.push(...sectorTickerOrder[sec]);
        }
        sectorTickerOrder.all = [...new Set(allOrdered)];
    } catch (e) {
        console.error('Sectors load failed:', e);
    }
}

// ── Sector Selection ──────────────────────────────
function selectSector(sector) {
    currentSector = sector;

    document.querySelectorAll('.sector-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.sector === sector);
    });

    // Show stock grid, hide other sections
    document.getElementById('stock-grid').style.display = 'grid';
    document.getElementById('top-picks-section').classList.add('hidden');
    document.getElementById('expected-actual-section').classList.add('hidden');
    document.getElementById('top-analysis-section')?.classList.add('hidden');
    document.querySelector('.sector-tabs').style.display = '';
    document.querySelector('.search-container').style.display = '';

    // Update nav
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.querySelectorAll('.nav-link')[0]?.classList.add('active');

    loadPricesForSector(sector);
}

// ── Load Prices ───────────────────────────────────
async function loadPricesForSector(sector) {
    const grid = document.getElementById('stock-grid');

    if (!Object.keys(allStockData).length) {
        grid.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><p>Loading stock prices...</p></div>';
    }

    let url;
    if (sector === 'all') {
        const allTickers = [];
        for (const sec in sectorsData) {
            const tickers = sectorsData[sec].tickers.map(t => t.symbol);
            allTickers.push(...tickers);
        }
        if (allTickers.length === 0) {
            url = '/api/prices';
        } else {
            await loadPricesBatch(allTickers);
            renderStockGrid(sector);
            return;
        }
    } else {
        url = `/api/prices?sector=${sector}`;
    }

    try {
        const res = await fetch(url);
        const data = await res.json();
        Object.assign(allStockData, data);
        renderStockGrid(sector);
    } catch (e) {
        grid.innerHTML = '<div class="loading-spinner"><p>Failed to load prices. Retrying...</p></div>';
        setTimeout(() => loadPricesForSector(sector), 5000);
    }
}

async function loadPricesBatch(tickers) {
    const batchSize = 20;
    for (let i = 0; i < tickers.length; i += batchSize) {
        const batch = tickers.slice(i, i + batchSize);
        try {
            const res = await fetch(`/api/prices?tickers=${batch.join(',')}`);
            const data = await res.json();
            Object.assign(allStockData, data);
            renderStockGrid(currentSector);
        } catch (e) {
            console.error('Batch load failed:', e);
        }
    }
}

// ── Enhanced Stock Grid ───────────────────────────
function renderStockGrid(sector) {
    const grid = document.getElementById('stock-grid');
    const searchTerm = document.getElementById('search-input')?.value?.toLowerCase() || '';

    let tickers;
    if (sector === 'all') {
        tickers = (sectorTickerOrder.all || []);
        if (!tickers.length) {
            tickers = Object.keys(allStockData).filter(t => !t.startsWith('^') && t !== 'USDINR=X' && t !== 'GC=F' && t !== 'CL=F').sort();
        }
    } else if (sectorsData[sector]) {
        tickers = (sectorTickerOrder[sector] || sectorsData[sector].tickers.map(t => t.symbol));
    } else {
        tickers = Object.keys(allStockData).sort();
    }

    if (searchTerm) {
        tickers = tickers.filter(t => {
            const name = allStockData[t]?.name || t;
            return t.toLowerCase().includes(searchTerm) || name.toLowerCase().includes(searchTerm);
        });
    }

    let html = '';
    for (const ticker of tickers) {
        const data = allStockData[ticker] || {};

        const name = data.name || ticker.replace('.NS', '');
        const price = Number(data.price || 0);
        const change = Number(data.change || 0);
        const changePct = Number(data.change_pct || 0);
        const hasQuote = price > 0;
        const direction = change >= 0 ? 'up' : 'down';
        const sign = change >= 0 ? '+' : '';
        const initials = name.substring(0, 2).toUpperCase();

        // Get daily analysis data for this stock
        const analysis = dailyAnalysisData?.all_stocks?.find(s => s.ticker === ticker);
        const strategyPrice = Number(analysis?.strategy_predicted_price || analysis?.predicted_price || 0);
        const aiPrice = Number(analysis?.ai_predicted_price || 0);
        const predReturn = Number(analysis?.predicted_return || 0);
        const signal = analysis?.signal || '';
        const confidence = analysis?.confidence || 0;
        const predictionMode = analysis?.prediction_mode || 'market_open_window';
        const nextDayMode = predictionMode === 'next_day_after_close';
        const predictedForDate = analysis?.predicted_for_date || dailyAnalysisData?.predicted_for_date || '';
        const strategyPct = nextDayMode ? analysis?.close_to_strategy_pct : analysis?.open_to_predicted_pct;
        const aiPct = nextDayMode ? analysis?.close_to_ai_pct : analysis?.open_to_ai_predicted_pct;
        const strategyTime = analysis?.strategy_predicted_at || analysis?.strategy_predicted_at_open;
        const aiTime = analysis?.ai_predicted_at || analysis?.ai_predicted_at_open;
        const liveLabel = analysis?.display_price_label || 'Current Price';

        // Signal badge
        let signalBadge = '';
        if (signal) {
            const sigClass = (signal === 'BUY' || signal === 'STRONG_BUY') ? 'buy' : (signal === 'SELL' || signal === 'STRONG_SELL') ? 'sell' : 'hold';
            signalBadge = `<span class="card-signal ${sigClass}">${signal}</span>`;
        }

        // Prediction row
        let predRow = '';
        if (strategyPrice > 0 || aiPrice > 0) {
            const strategyColor = predReturn >= 0 ? 'up-color' : 'down-color';
            const aiColor = Number(aiPct || predReturn) >= 0 ? 'up-color' : 'down-color';
            const strategyLabel = nextDayMode
                ? `Strategy (Next Day ${predictedForDate})`
                : 'Strategy Prediction';
            const aiLabel = nextDayMode
                ? `AI Target (Next Day ${predictedForDate})`
                : 'AI Target';
            const modeHint = nextDayMode ? 'computed after 3:45 PM IST' : 'captured in 9:15–9:30 AM IST window';
            predRow = `
                <div class="card-prediction">
                    <div class="card-pred-left">
                        <span class="card-pred-label">${strategyLabel}</span>
                        <span class="card-pred-meta">${modeHint} • ${formatIstTimestamp(strategyTime)}</span>
                    </div>
                    <span class="card-pred-value ${strategyColor}">${formatPrice(strategyPrice)} (${formatSignedPct(strategyPct, 2)})</span>
                </div>
                <div class="card-prediction">
                    <div class="card-pred-left">
                        <span class="card-pred-label">${aiLabel}</span>
                        <span class="card-pred-meta">${modeHint} • ${formatIstTimestamp(aiTime)}</span>
                    </div>
                    <span class="card-pred-value ${aiColor}">${aiPrice > 0 ? formatPrice(aiPrice) : '—'} (${formatSignedPct(aiPct, 2)})</span>
                </div>`;
        }

        html += `
        <a href="/stock/${encodeURIComponent(ticker)}" class="stock-card" data-ticker="${ticker}">
            <div class="stock-card-top">
                <div class="stock-card-icon">${initials}</div>
                <div class="stock-card-info">
                <div class="stock-card-name">${name}</div>
                <div class="stock-card-symbol">${ticker}</div>
            </div>
                ${signalBadge}
            </div>
                <div class="stock-card-bottom">
                    <div class="stock-card-price">
                        <span class="card-live-label">${liveLabel}</span>
                        <span class="price">${formatPrice(price)}</span>
                        <span class="change ${hasQuote ? direction : ''}">
                            ${hasQuote ? `${sign}${change.toFixed(2)} (${changePct.toFixed(2)}%)` : 'Waiting for quote'}
                        </span>
                    </div>
                    ${predRow}
            </div>
        </a>`;
    }

    if (!html) {
        html = '<div class="loading-spinner"><p>No data yet — prices loading...</p></div>';
    }

    grid.innerHTML = html;
}

// ── Filter Stocks ─────────────────────────────────
function filterStocks() {
    renderStockGrid(currentSector);
}

// ── Top 10 Analysis ───────────────────────────────
async function showTopAnalysis() {
    const section = document.getElementById('top-analysis-section');
    if (!section) {
        await showTopPicks();
        return;
    }
    // Update nav
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    if (typeof event !== 'undefined' && event?.target?.classList) {
        event.target.classList.add('active');
    }

    // Show section
    document.getElementById('stock-grid').style.display = 'none';
    document.getElementById('top-picks-section').classList.add('hidden');
    document.getElementById('expected-actual-section').classList.add('hidden');
    document.querySelector('.sector-tabs').style.display = 'none';
    document.querySelector('.search-container').style.display = 'none';

    section.classList.remove('hidden');

    const grid = document.getElementById('analysis-grid');
    grid.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><p>Running deep analysis across all stocks... This uses your daily predictions.</p></div>';

    try {
        const res = await fetch('/api/daily-analysis');
        const data = await res.json();

        if (data.error) {
            grid.innerHTML = `<div class="loading-spinner"><p>⚠️ ${data.error}</p></div>`;
            return;
        }

        const top10 = data.top_10 || [];
        if (!top10.length) {
            grid.innerHTML = '<div class="loading-spinner"><p>No predictions available yet. Run daily predictions first.</p></div>';
            return;
        }

        let html = '';

        // Market summary cards
        const ms = data.market_summary;
        html += `
        <div class="analysis-summary">
            <div class="summary-card compact">
                <span class="sc-label">Total Analyzed</span>
                <span class="sc-value">${data.total_stocks}</span>
            </div>
            <div class="summary-card compact">
                <span class="sc-label">Gainers</span>
                <span class="sc-value up-color">${ms.gainers}</span>
            </div>
            <div class="summary-card compact">
                <span class="sc-label">Losers</span>
                <span class="sc-value down-color">${ms.losers}</span>
            </div>
            <div class="summary-card compact">
                <span class="sc-label">Avg Change</span>
                <span class="sc-value ${ms.avg_change_pct >= 0 ? 'up-color' : 'down-color'}">${ms.avg_change_pct >= 0 ? '+' : ''}${ms.avg_change_pct}%</span>
            </div>
        </div>`;

        // Render top 10 cards
        html += '<div class="top10-cards">';
        for (let i = 0; i < top10.length; i++) {
            const s = top10[i];
            const rank = i + 1;
            const isUp = s.predicted_return > 0;
            const nextDayMode = s.prediction_mode === 'next_day_after_close';
            const signalClass = (s.signal === 'BUY' || s.signal === 'STRONG_BUY') ? 'buy' : (s.signal === 'SELL' || s.signal === 'STRONG_SELL') ? 'sell' : 'hold';
            const predSign = s.predicted_return >= 0 ? '+' : '';
            const currSign = s.open_to_current_pct >= 0 ? '+' : '';
            const liveLabel = s.display_price_label || 'Current';
            const strategyLabel = nextDayMode ? `Strategy (Next Day ${s.predicted_for_date || ''})` : 'Strategy Predicted';
            const aiLabel = nextDayMode ? `AI (Next Day ${s.predicted_for_date || ''})` : 'AI Predicted';
            const strategyPct = nextDayMode ? s.close_to_strategy_pct : s.open_to_predicted_pct;
            const aiPct = nextDayMode ? s.close_to_ai_pct : s.open_to_ai_predicted_pct;
            const aiAvailable = Number(s.ai_predicted_price || 0) > 0;
            const strategyTime = s.strategy_predicted_at || s.strategy_predicted_at_open;
            const aiTime = s.ai_predicted_at || s.ai_predicted_at_open;

            html += `
            <div class="top10-card" onclick="window.location='/stock/${encodeURIComponent(s.ticker)}'">
                <div class="top10-rank">#${rank}</div>
                <div class="top10-header">
                    <div>
                        <h4>${s.name}</h4>
                        <span class="muted-text">${s.ticker}</span>
                    </div>
                    <span class="pick-signal ${signalClass}">${s.signal || 'N/A'}</span>
                </div>

                <div class="top10-prices">
                    <div class="top10-price-item">
                        <span class="label">Open</span>
                        <span class="value">${formatPrice(s.open_price)}</span>
                    </div>
                    <div class="top10-price-item predicted">
                        <span class="label">${strategyLabel}</span>
                        <span class="value ${isUp ? 'up-color' : 'down-color'}">${formatPrice(s.strategy_predicted_price || s.predicted_price)}</span>
                        <span class="pct ${Number(strategyPct) >= 0 ? 'up-color' : 'down-color'}">${formatSignedPct(strategyPct, 2)}</span>
                        <span class="muted-text">${formatIstTimestamp(strategyTime)}</span>
                    </div>
                    <div class="top10-price-item predicted">
                        <span class="label">${aiLabel}</span>
                        <span class="value ${Number(aiPct) >= 0 ? 'up-color' : 'down-color'}">${aiAvailable ? formatPrice(s.ai_predicted_price) : '—'}</span>
                        <span class="pct ${Number(aiPct) >= 0 ? 'up-color' : 'down-color'}">${aiAvailable ? formatSignedPct(aiPct, 2) : '—'}</span>
                        <span class="muted-text">${aiAvailable ? formatIstTimestamp(aiTime) : 'AI unavailable'}</span>
                    </div>
                    <div class="top10-price-item current">
                        <span class="label">${liveLabel}</span>
                        <span class="value">${formatPrice(s.current_price)}</span>
                        <span class="pct ${s.open_to_current_pct >= 0 ? 'up-color' : 'down-color'}">${currSign}${s.open_to_current_pct}%</span>
                    </div>
                </div>

                <div class="top10-metrics">
                    <div class="top10-metric">
                        <span class="label clickable-metric" tabindex="0" role="button"
                              onmouseenter="showMetricTooltip(event, 'Confidence', 'Top 10 model confidence')"
                              onmousemove="moveMetricTooltip(event)"
                              onmouseleave="hideMetricTooltip()"
                              onfocus="showMetricTooltip(event, 'Confidence', 'Top 10 model confidence')"
                              onblur="hideMetricTooltip()"
                              onkeydown="handleMetricLabelKey(event, 'Confidence', 'Top 10 model confidence')">Confidence</span>
                        <div class="metric-bar"><div class="metric-fill" style="width:${s.confidence}%"></div></div>
                        <span class="value">${s.confidence}%</span>
                    </div>
                    <div class="top10-metric">
                        <span class="label clickable-metric" tabindex="0" role="button"
                              onmouseenter="showMetricTooltip(event, 'Model Agreement', 'Top 10 model agreement')"
                              onmousemove="moveMetricTooltip(event)"
                              onmouseleave="hideMetricTooltip()"
                              onfocus="showMetricTooltip(event, 'Model Agreement', 'Top 10 model agreement')"
                              onblur="hideMetricTooltip()"
                              onkeydown="handleMetricLabelKey(event, 'Model Agreement', 'Top 10 model agreement')">Agreement</span>
                        <div class="metric-bar"><div class="metric-fill agreement" style="width:${s.model_agreement}%"></div></div>
                        <span class="value">${s.model_agreement}%</span>
                    </div>
                    <div class="top10-metric">
                        <span class="label">Score</span>
                        <span class="value score-badge">${s.composite_score}</span>
                    </div>
                    ${s.risk_reward ? `<div class="top10-metric"><span class="label clickable-metric" tabindex="0" role="button"
                        onmouseenter="showMetricTooltip(event, 'Risk Reward Ratio', 'Top 10 trade quality')"
                        onmousemove="moveMetricTooltip(event)"
                        onmouseleave="hideMetricTooltip()"
                        onfocus="showMetricTooltip(event, 'Risk Reward Ratio', 'Top 10 trade quality')"
                        onblur="hideMetricTooltip()"
                        onkeydown="handleMetricLabelKey(event, 'Risk Reward Ratio', 'Top 10 trade quality')">R:R</span><span class="value">${s.risk_reward}</span></div>` : ''}
                </div>
            </div>`;
        }
        html += '</div>';

        grid.innerHTML = html;
    } catch (e) {
        grid.innerHTML = `<div class="loading-spinner"><p>Error: ${e.message}</p></div>`;
    }
}

// ── Top Picks ─────────────────────────────────────
async function showTopPicks(evt = null) {
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    const trigger = evt?.target || (typeof event !== 'undefined' ? event.target : null);
    if (trigger && trigger.classList) trigger.classList.add('active');

    document.getElementById('stock-grid').style.display = 'none';
    document.getElementById('expected-actual-section').classList.add('hidden');
    document.getElementById('top-analysis-section')?.classList.add('hidden');
    document.querySelector('.sector-tabs').style.display = 'none';
    document.querySelector('.search-container').style.display = 'none';

    const section = document.getElementById('top-picks-section');
    section.classList.remove('hidden');
    updateTopPickFilterButtons();

    const grid = document.getElementById('picks-grid');
    grid.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><p>Loading Top Picks (cache-first)...</p></div>';

    let timeoutId = null;
    try {
        const controller = new AbortController();
        timeoutId = setTimeout(() => controller.abort(), 45000);
        const res = await fetch(
            '/api/top-picks?sectors=large_cap&sectors=banking&sectors=mid_cap&sectors=high_volatility&sectors=commodities&n=50&grouped=true&use_cached_analysis=true',
            { signal: controller.signal }
        );
        const data = await res.json();

        if (res.status === 202 && data.status === 'warming') {
            const waitSec = Number(data.retry_after_sec || 5);
            grid.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><p>ML picks are warming up in background... retrying shortly.</p></div>';
            setTimeout(() => showTopPicks(), Math.max(2, waitSec) * 1000);
            return;
        }

        if (data.error) {
            grid.innerHTML = `<div class="loading-spinner"><p>⚠️ ${data.error}</p></div>`;
            return;
        }
        groupedTopPicks = {
            top_buy: data.top_buy || [],
            top_sell: data.top_sell || [],
            top_hold: data.top_hold || [],
        };
        const selected = groupedTopPicks[topPickFilter] || [];
        highlightedPremarketTicker = selected.length ? selected[0].ticker : null;
        await loadPremarketOutlook();
        renderTopPicks();
        if (document.getElementById('advisor-panel') && typeof loadAdvisorOpenBuyList === 'function') {
            await loadAdvisorOpenBuyList();
        }
        if (document.getElementById('advisor-panel') && typeof refreshAdvisorSummary === 'function') {
            await refreshAdvisorSummary();
        }
        await refreshTopPicksLivePrices();
    } catch (e) {
        const msg = e?.name === 'AbortError'
            ? 'Top Picks request timed out. Please retry in a few seconds.'
            : `Error: ${e.message}`;
        grid.innerHTML = `<div class="loading-spinner"><p>${msg}</p></div>`;
    } finally {
        if (timeoutId) clearTimeout(timeoutId);
    }
}

async function refreshTopPicksAndSnapshots() {
    await showTopPicks();
}

function setTopPickFilter(filterKey) {
    topPickFilter = filterKey;
    updateTopPickFilterButtons();
    const picks = groupedTopPicks[topPickFilter] || [];
    if (!highlightedPremarketTicker || !picks.some(p => p.ticker === highlightedPremarketTicker)) {
        highlightedPremarketTicker = picks.length ? picks[0].ticker : null;
    }
    renderTopPicks();
    renderPremarketOutlookTable();
}

function togglePremarketStored(checked) {
    useLatestStoredPredictions = Boolean(checked);
    loadPremarketOutlook();
}

function updateTopPickFilterButtons() {
    const filterEl = document.getElementById('top-picks-filter');
    if (filterEl) filterEl.value = topPickFilter;
}

function renderTopPicks() {
    const grid = document.getElementById('picks-grid');
    if (!grid) return;

    const picks = groupedTopPicks[topPickFilter] || [];
    if (!picks.length) {
        grid.innerHTML = '<div class="loading-spinner"><p>No predictions available yet for this group.</p></div>';
        return;
    }

    let html = '';
    for (const pick of picks) {
        const predictedReturn = Number(pick.predicted_return);
        const hasPredictedReturn = Number.isFinite(predictedReturn);
        const isUp = hasPredictedReturn ? predictedReturn >= 0 : true;
        const signalLower = String(pick.signal || '').toLowerCase();
        const signalClass = signalLower.includes('buy') ? 'buy' : signalLower.includes('sell') ? 'sell' : 'hold';
        const returnSign = isUp ? '+' : '';
        const strategyTarget = Number(pick.target_price || pick.predicted_price || 0);
        const scoreVal = Number(pick._score || 0);
        const liqVal = Number(pick.liquidity_factor || 0);

        html += `
        <div class="pick-card" onclick="window.location='/stock/${encodeURIComponent(pick.ticker)}'">
            <div class="pick-card-header">
                <div>
                    <h4>${pick.name || pick.ticker.replace('.NS', '')}</h4>
                    <span class="muted-text">${pick.ticker}</span>
                </div>
                <span class="pick-signal ${signalClass}">${pick.signal || 'HOLD'}</span>
            </div>
            <div class="pick-details">
                <div class="pick-detail">
                    <span class="label">Current Price</span>
                    <span class="value">${formatPrice(pick.current_price)}</span>
                </div>
                <div class="pick-detail">
                    <span class="label">Predicted Return</span>
                    <span class="value ${hasPredictedReturn ? (isUp ? 'up-color' : 'down-color') : ''}">${hasPredictedReturn ? `${returnSign}${predictedReturn.toFixed(3)}%` : '—'}</span>
                </div>
                <div class="pick-detail">
                    <span class="label">Strategy Target</span>
                    <span class="value">${strategyTarget > 0 ? formatPrice(strategyTarget) : '—'}</span>
                </div>
                <div class="pick-detail">
                    <span class="label">AI Predicted</span>
                    <span class="value">${Number(pick.ai_predicted_price || 0) > 0 ? formatPrice(pick.ai_predicted_price) : '—'}</span>
                </div>
                <div class="pick-detail">
                    <span class="label clickable-metric" tabindex="0" role="button"
                          onmouseenter="showMetricTooltip(event, 'Confidence', 'Top picks confidence metric')"
                          onmousemove="moveMetricTooltip(event)"
                          onmouseleave="hideMetricTooltip()"
                          onfocus="showMetricTooltip(event, 'Confidence', 'Top picks confidence metric')"
                          onblur="hideMetricTooltip()"
                          onkeydown="handleMetricLabelKey(event, 'Confidence', 'Top picks confidence metric')">Confidence</span>
                    <span class="value">${Number(pick.confidence || 0).toFixed(0)}%</span>
                </div>
                <div class="pick-detail">
                    <span class="label clickable-metric" tabindex="0" role="button"
                          onmouseenter="showMetricTooltip(event, 'Model Agreement', 'Top picks model agreement')"
                          onmousemove="moveMetricTooltip(event)"
                          onmouseleave="hideMetricTooltip()"
                          onfocus="showMetricTooltip(event, 'Model Agreement', 'Top picks model agreement')"
                          onblur="hideMetricTooltip()"
                          onkeydown="handleMetricLabelKey(event, 'Model Agreement', 'Top picks model agreement')">Agreement</span>
                    <span class="value">${Number(pick.model_agreement || 0).toFixed(0)}%</span>
                </div>
                <div class="pick-detail">
                    <span class="label clickable-metric" tabindex="0" role="button"
                          onmouseenter="showMetricTooltip(event, 'Composite Score', 'score = |predicted_return_decimal| × (confidence/100) × (agreement/100) × liquidity_factor')"
                          onmousemove="moveMetricTooltip(event)"
                          onmouseleave="hideMetricTooltip()"
                          onfocus="showMetricTooltip(event, 'Composite Score', 'score = |predicted_return_decimal| × (confidence/100) × (agreement/100) × liquidity_factor')"
                          onblur="hideMetricTooltip()"
                          onkeydown="handleMetricLabelKey(event, 'Composite Score', 'score = |predicted_return_decimal| × (confidence/100) × (agreement/100) × liquidity_factor')">Score</span>
                    <span class="value">${Number.isFinite(scoreVal) ? scoreVal.toFixed(4) : '—'}</span>
                </div>
                <div class="pick-detail">
                    <span class="label">Liquidity</span>
                    <span class="value">${Number.isFinite(liqVal) && liqVal > 0 ? `${liqVal.toFixed(2)}x` : '—'}</span>
                </div>
            </div>
        </div>`;
    }
    grid.innerHTML = html;
}

async function refreshTopPicksLivePrices() {
    const section = document.getElementById('top-picks-section');
    if (!section || section.classList.contains('hidden')) return;
    if (topPicksLivePriceBusy) return;

    const picks = groupedTopPicks[topPickFilter] || [];
    if (!picks.length) return;

    const tickers = [...new Set(
        picks
            .slice(0, 15)
            .map((p) => String(p?.ticker || '').toUpperCase())
            .filter(Boolean)
    )];
    if (!tickers.length) return;

    topPicksLivePriceBusy = true;
    try {
        const res = await fetch(`/api/prices?tickers=${tickers.join(',')}`);
        if (!res.ok) return;
        const quotes = await res.json();
        const priceByTicker = {};
        for (const [ticker, row] of Object.entries(quotes || {})) {
            const px = Number(row?.price || 0);
            if (Number.isFinite(px) && px > 0) priceByTicker[String(ticker).toUpperCase()] = px;
        }
        if (!Object.keys(priceByTicker).length) return;

        for (const key of ['top_buy', 'top_sell', 'top_hold']) {
            groupedTopPicks[key] = (groupedTopPicks[key] || []).map((row) => {
                const ticker = String(row?.ticker || '').toUpperCase();
                const px = priceByTicker[ticker];
                return px ? { ...row, current_price: px } : row;
            });
        }
        premarketOutlookData = (premarketOutlookData || []).map((row) => {
            const ticker = String(row?.ticker || '').toUpperCase();
            const px = priceByTicker[ticker];
            return px ? { ...row, current_price: px } : row;
        });

        renderTopPicks();
        renderPremarketOutlookTable();
    } catch (_) {
        // Best-effort live price refresh; ignore transient failures.
    } finally {
        topPicksLivePriceBusy = false;
    }
}

async function loadPremarketOutlook() {
    const table = document.getElementById('premarket-table-body');
    const header = document.getElementById('premarket-captured-at');
    const toggle = document.getElementById('premarket-use-stored');
    if (!table) return;
    if (toggle) {
        const locked = currentMarketStatus === 'after_hours' || currentMarketStatus === 'weekend';
        toggle.disabled = locked;
    }

    table.innerHTML = '<tr><td colspan="7" class="muted-text">Loading premarket snapshot...</td></tr>';
    try {
        const url = `/api/premarket-outlook?use_latest_stored=${useLatestStoredPredictions ? 'true' : 'false'}`;
        const res = await fetch(url);
        const data = await res.json();
        if (!res.ok || data.error) {
            table.innerHTML = `<tr><td colspan="7" class="muted-text">${data.error || 'Premarket snapshot unavailable'}</td></tr>`;
            premarketOutlookData = [];
            if (header) header.textContent = 'Premarket snapshot unavailable';
            return;
        }
        premarketOutlookData = data.items || [];
        if (!highlightedPremarketTicker && premarketOutlookData.length) {
            highlightedPremarketTicker = premarketOutlookData[0].ticker;
        }
        if (header) {
            const openWindowTime = formatIstTimestamp(data.captured_at);
            const actualCapture = formatIstTimestamp(data.captured_at_actual);
            const snapshotType = String(data.snapshot_type || '');
            if (snapshotType === 'market_open_backfilled' && actualCapture && actualCapture !== openWindowTime) {
                header.textContent = `Market-open snapshot: ${openWindowTime} (backfilled, generated at ${actualCapture})`;
            } else {
                const modeLabel = useLatestStoredPredictions ? 'stored snapshot' : 'fresh snapshot';
                const sessionLabel = (currentMarketStatus === 'after_hours' || currentMarketStatus === 'weekend')
                    ? 'after-hours view'
                    : 'market session view';
                header.textContent = `Market-open snapshot: ${openWindowTime} (${modeLabel}; ${sessionLabel})`;
            }
        }
        renderPremarketOutlookTable();
    } catch (e) {
        table.innerHTML = `<tr><td colspan="6" class="muted-text">Premarket fetch failed: ${e.message}</td></tr>`;
        if (header) header.textContent = 'Premarket snapshot unavailable';
    }
}

function renderPremarketOutlookTable() {
    const table = document.getElementById('premarket-table-body');
    if (!table) return;
    if (!premarketOutlookData.length) {
        table.innerHTML = '<tr><td colspan="7" class="muted-text">No premarket rows available.</td></tr>';
        return;
    }

    const currentFilterSet = new Set((groupedTopPicks[topPickFilter] || []).map(p => p.ticker));
    const rows = premarketOutlookData.filter(row => currentFilterSet.size === 0 || currentFilterSet.has(row.ticker));
    const viewRows = rows.length ? rows : premarketOutlookData;

    table.innerHTML = viewRows.map(row => {
        const selected = row.ticker === highlightedPremarketTicker ? 'selected' : '';
        const aiAvailable = Number(row.ai_predicted_price || 0) > 0;
        const aligned = aiAvailable ? row.strategy_direction === row.ai_direction : null;
        const badgeClass = aligned === null ? '' : (aligned ? 'correct' : 'wrong');
        const badgeText = aligned === null ? 'AI N/A' : (aligned ? 'Aligned' : 'Divergent');
        return `
            <tr class="premarket-row ${selected}" onclick="selectPremarketTicker('${row.ticker}')">
                <td><strong>${row.name || row.ticker.replace('.NS', '')}</strong><br><span class="muted-text">${row.ticker}</span></td>
                <td>${formatPrice(row.current_price)}</td>
                <td>${Number(row.strategy_price_at_open || 0) > 0 ? formatPrice(row.strategy_price_at_open) : '—'}</td>
                <td>${Number(row.ai_predicted_price || 0) > 0 ? formatPrice(row.ai_predicted_price) : '—'}</td>
                <td>${formatIstTimestamp(row.strategy_predicted_at_open || row.captured_at)}</td>
                <td>${row.strategy_direction || 'FLAT'} / ${row.ai_direction || 'N/A'}</td>
                <td><span class="direction-badge ${badgeClass}">${badgeText}</span></td>
            </tr>
        `;
    }).join('');
}

function selectPremarketTicker(ticker) {
    highlightedPremarketTicker = ticker;
    renderPremarketOutlookTable();
}

// ── Expected vs Actual ────────────────────────────
async function showExpectedVsActual() {
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    if (typeof event !== 'undefined' && event?.target?.classList) {
        event.target.classList.add('active');
    }

    document.getElementById('stock-grid').style.display = 'none';
    document.getElementById('top-picks-section').classList.add('hidden');
    document.getElementById('top-analysis-section')?.classList.add('hidden');
    document.querySelector('.sector-tabs').style.display = 'none';
    document.querySelector('.search-container').style.display = 'none';

    const section = document.getElementById('expected-actual-section');
    section.classList.remove('hidden');

    try {
        const res = await fetch('/api/prediction-dates');
        const dates = await res.json();
        const select = document.getElementById('prediction-date');
        select.innerHTML = dates.map(d => `<option value="${d}">${d}</option>`).join('');

        if (dates.length > 0) {
            loadExpectedVsActual();
        } else {
            document.getElementById('eva-summary').innerHTML = '<div class="summary-card"><span class="sc-label">No predictions logged yet</span><span class="sc-value">—</span></div>';
        }
    } catch (e) {
        console.error('Failed to load prediction dates:', e);
    }
}

async function loadExpectedVsActual() {
    const dateStr = document.getElementById('prediction-date').value;
    if (!dateStr) return;

    const summary = document.getElementById('eva-summary');
    const table = document.getElementById('eva-table');
    summary.innerHTML = '<div class="loading-spinner"><div class="spinner"></div></div>';
    table.innerHTML = '';

    try {
        const res = await fetch(`/api/expected-vs-actual?date=${dateStr}`);
        const data = await res.json();

        if (data.error) {
            summary.innerHTML = `<div class="summary-card"><span class="sc-label">${data.error}</span></div>`;
            return;
        }

        const hitColor = data.hit_rate_pct >= 55 ? 'up-color' : (data.hit_rate_pct >= 50 ? '' : 'down-color');
        const alphaColor = data.avg_alpha_pct >= 0 ? 'up-color' : 'down-color';

        summary.innerHTML = `
            <div class="summary-card">
                <span class="sc-label">Total Predictions</span>
                <span class="sc-value">${data.total_predictions}</span>
            </div>
            <div class="summary-card">
                <span class="sc-label">Direction Hit Rate</span>
                <span class="sc-value ${hitColor}">${data.hit_rate_pct}%</span>
            </div>
            <div class="summary-card">
                <span class="sc-label clickable-metric" tabindex="0" role="button"
                      onmouseenter="showAlphaFormulaTooltip(event)"
                      onmousemove="moveMetricTooltip(event)"
                      onmouseleave="hideMetricTooltip()"
                      onfocus="showAlphaFormulaTooltip(event)"
                      onblur="hideMetricTooltip()"
                      onkeydown="handleMetricLabelKey(event, 'Alpha Formula Used', 'Simplified alpha = actual return minus benchmark return; CAPM alpha adjusts for beta and risk-free rate.')">Avg Alpha</span>
                <span class="sc-value ${alphaColor}">${data.avg_alpha_pct >= 0 ? '+' : ''}${data.avg_alpha_pct}%</span>
            </div>
            <div class="summary-card">
                <span class="sc-label">Total Alpha</span>
                <span class="sc-value ${data.total_alpha_pct >= 0 ? 'up-color' : 'down-color'}">${data.total_alpha_pct >= 0 ? '+' : ''}${data.total_alpha_pct}%</span>
            </div>
            <div class="summary-card">
                <span class="sc-label">Benchmark (Nifty)</span>
                <span class="sc-value">${data.benchmark_return_pct >= 0 ? '+' : ''}${data.benchmark_return_pct}%</span>
            </div>
        `;

        let rows = '';
        for (const r of data.results) {
            const dirClass = r.direction_comparison ? 'correct' : 'wrong';
            const dirText = r.direction_comparison ? '✓ Strategy Correct' : '✗ Strategy Wrong';
            const predColor = r.predicted_return_pct >= 0 ? 'up-color' : 'down-color';
            const actColor = r.actual_return_pct >= 0 ? 'up-color' : 'down-color';
            const alpColor = r.alpha_pct >= 0 ? 'up-color' : 'down-color';
            const capmAlpha = Number(r.alpha_capm_pct);
            const capmText = Number.isFinite(capmAlpha)
                ? `<div class="muted-text">CAPM: ${capmAlpha >= 0 ? '+' : ''}${capmAlpha.toFixed(3)}%</div>`
                : '';

            rows += `
            <tr onclick="window.location='/stock/${encodeURIComponent(r.ticker)}'" style="cursor:pointer">
                <td><strong>${r.name}</strong><br><span class="muted-text">${r.ticker}</span></td>
                <td>${r.signal}</td>
                <td class="${predColor}">${r.predicted_return_pct >= 0 ? '+' : ''}${r.predicted_return_pct}%</td>
                <td>${formatPrice(r.predicted_price)}</td>
                <td>${Number(r.strategy_price_at_open || 0) > 0 ? formatPrice(r.strategy_price_at_open) : '—'}</td>
                <td>${Number(r.ai_last_prediction || 0) > 0 ? formatPrice(r.ai_last_prediction) : '—'}</td>
                <td>${formatPrice(r.actual_price)}</td>
                <td class="${actColor}">${r.actual_return_pct >= 0 ? '+' : ''}${r.actual_return_pct}%</td>
                <td><span class="direction-badge ${dirClass}">${dirText}</span></td>
                <td class="${alpColor}">
                    <span class="clickable-metric" tabindex="0" role="button"
                          onmouseenter="showMetricTooltip(event, 'Alpha (Ticker)', 'Simplified alpha is actual_return - benchmark_return. CAPM alpha adjusts for beta and risk-free rate. Ticker: ${r.ticker}.')"
                          onmousemove="moveMetricTooltip(event)"
                          onmouseleave="hideMetricTooltip()"
                          onfocus="showMetricTooltip(event, 'Alpha (Ticker)', 'Simplified alpha is actual_return - benchmark_return. CAPM alpha adjusts for beta and risk-free rate. Ticker: ${r.ticker}.')"
                          onblur="hideMetricTooltip()"
                          onkeydown="handleMetricLabelKey(event, 'Alpha (Ticker)', 'Simplified alpha is actual_return - benchmark_return. CAPM alpha adjusts for beta and risk-free rate. Ticker: ${r.ticker}.')">${r.alpha_pct >= 0 ? '+' : ''}${r.alpha_pct}%</span>${capmText}
                </td>
            </tr>`;
        }

        table.innerHTML = `
        <table class="eva-table">
            <thead>
                <tr>
                    <th>Stock</th>
                    <th>Signal</th>
                    <th>Predicted</th>
                    <th>Pred. Price</th>
                    <th>Strategy@Open</th>
                    <th>AI Last</th>
                    <th>Actual Price</th>
                    <th>Actual Return</th>
                    <th>Strategy vs Actual</th>
                    <th><span class="clickable-metric" tabindex="0" role="button"
                              onmouseenter="showAlphaFormulaTooltip(event)"
                              onmousemove="moveMetricTooltip(event)"
                              onmouseleave="hideMetricTooltip()"
                              onfocus="showAlphaFormulaTooltip(event)"
                              onblur="hideMetricTooltip()"
                              onkeydown="handleMetricLabelKey(event, 'Alpha Formula Used', 'Simplified alpha = actual return minus benchmark return; CAPM alpha adjusts for beta and risk-free rate.')">Alpha ℹ️</span></th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>`;
    } catch (e) {
        summary.innerHTML = `<div class="summary-card"><span class="sc-label">Error: ${e.message}</span></div>`;
    }
}

// ── Helpers ───────────────────────────────────────
function formatNumber(n) {
    if (n === undefined || n === null) return '---';
    if (n >= 10000000) return (n / 10000000).toFixed(2) + ' Cr';
    if (n >= 100000) return (n / 100000).toFixed(2) + ' L';
    return n.toLocaleString('en-IN', { maximumFractionDigits: 2 });
}

function formatPrice(n) {
    const value = Number(n);
    if (!Number.isFinite(value) || value <= 0) {
        return '—';
    }
    return `₹${formatNumber(value)}`;
}

function formatIstTimestamp(value) {
    if (!value) return '—';
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return String(value);
    return d.toLocaleString('en-IN', {
        timeZone: 'Asia/Kolkata',
        day: '2-digit',
        month: 'short',
        hour: '2-digit',
        minute: '2-digit',
        hour12: true,
    }) + ' IST';
}

function formatSignedPct(value, digits = 2) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    const sign = n >= 0 ? '+' : '';
    return `${sign}${n.toFixed(digits)}%`;
}
