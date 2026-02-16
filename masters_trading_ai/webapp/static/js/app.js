/* ============================================================
   Masters AI Trading Bot — Dashboard JS (Enhanced)
   ============================================================ */

// State
let allStockData = {};
let dailyAnalysisData = null;
let currentSector = 'all';
let sectorsData = {};
let autoRefreshTimer = null;

// ── Init ──────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    loadSectors();
    loadIndexPrices();
    loadPricesForSector('all');
    loadDailyAnalysis();
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
        if (data.market.status === 'market_open') {
            mktBadge.textContent = '🟢 Market Open';
            mktBadge.className = 'status-badge open';
        } else {
            mktBadge.textContent = '🔴 ' + data.market.description;
            mktBadge.className = 'status-badge closed';
        }

        const modelBadge = document.getElementById('model-status');
        if (data.models_loaded) {
            modelBadge.textContent = `✅ ${data.model_count} Models Ready`;
            modelBadge.className = 'status-badge ready';
        } else if (data.load_error) {
            modelBadge.textContent = '❌ Model Error';
            modelBadge.className = 'status-badge closed';
        } else {
            modelBadge.textContent = '⏳ Loading models...';
            modelBadge.className = 'status-badge loading';
            setTimeout(checkStatus, 3000);
        }
    } catch (e) {
        console.error('Status check failed:', e);
        setTimeout(checkStatus, 5000);
    }
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
    document.getElementById('top-analysis-section').classList.add('hidden');
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
        tickers = Object.keys(allStockData).filter(t => !t.startsWith('^') && t !== 'USDINR=X' && t !== 'GC=F' && t !== 'CL=F');
    } else if (sectorsData[sector]) {
        tickers = sectorsData[sector].tickers.map(t => t.symbol);
    } else {
        tickers = Object.keys(allStockData);
    }

    if (searchTerm) {
        tickers = tickers.filter(t => {
            const name = allStockData[t]?.name || t;
            return t.toLowerCase().includes(searchTerm) || name.toLowerCase().includes(searchTerm);
        });
    }

    tickers.sort((a, b) => {
        const aChg = Math.abs(allStockData[a]?.change_pct || 0);
        const bChg = Math.abs(allStockData[b]?.change_pct || 0);
        return bChg - aChg;
    });

    let html = '';
    for (const ticker of tickers) {
        const data = allStockData[ticker];
        if (!data) continue;

        const name = data.name || ticker.replace('.NS', '');
        const price = data.price || 0;
        const change = data.change || 0;
        const changePct = data.change_pct || 0;
        const direction = change >= 0 ? 'up' : 'down';
        const sign = change >= 0 ? '+' : '';
        const initials = name.substring(0, 2).toUpperCase();

        // Get daily analysis data for this stock
        const analysis = dailyAnalysisData?.all_stocks?.find(s => s.ticker === ticker);
        const predPrice = analysis?.predicted_price || null;
        const predReturn = analysis?.predicted_return || null;
        const signal = analysis?.signal || '';
        const confidence = analysis?.confidence || 0;

        // Signal badge
        let signalBadge = '';
        if (signal) {
            const sigClass = (signal === 'BUY' || signal === 'STRONG_BUY') ? 'buy' : (signal === 'SELL' || signal === 'STRONG_SELL') ? 'sell' : 'hold';
            signalBadge = `<span class="card-signal ${sigClass}">${signal}</span>`;
        }

        // Prediction row
        let predRow = '';
        if (predPrice && predReturn !== null) {
            const predSign = predReturn >= 0 ? '+' : '';
            const predColor = predReturn >= 0 ? 'up-color' : 'down-color';
            predRow = `
                <div class="card-prediction">
                    <span class="card-pred-label">AI Target</span>
                    <span class="card-pred-value ${predColor}">₹${formatNumber(predPrice)} (${predSign}${predReturn.toFixed(2)}%)</span>
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
                        <span class="price">${formatPrice(price)}</span>
                        <span class="change ${direction}">${sign}${change.toFixed(2)} (${changePct.toFixed(2)}%)</span>
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
    // Update nav
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    event.target.classList.add('active');

    // Show section
    document.getElementById('stock-grid').style.display = 'none';
    document.getElementById('top-picks-section').classList.add('hidden');
    document.getElementById('expected-actual-section').classList.add('hidden');
    document.querySelector('.sector-tabs').style.display = 'none';
    document.querySelector('.search-container').style.display = 'none';

    const section = document.getElementById('top-analysis-section');
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
            const signalClass = (s.signal === 'BUY' || s.signal === 'STRONG_BUY') ? 'buy' : (s.signal === 'SELL' || s.signal === 'STRONG_SELL') ? 'sell' : 'hold';
            const predSign = s.predicted_return >= 0 ? '+' : '';
            const currSign = s.open_to_current_pct >= 0 ? '+' : '';

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
                        <span class="label">AI Predicted</span>
                        <span class="value ${isUp ? 'up-color' : 'down-color'}">${formatPrice(s.predicted_price)}</span>
                        <span class="pct ${isUp ? 'up-color' : 'down-color'}">${predSign}${s.open_to_predicted_pct}%</span>
                    </div>
                    <div class="top10-price-item current">
                        <span class="label">Current</span>
                        <span class="value">${formatPrice(s.current_price)}</span>
                        <span class="pct ${s.open_to_current_pct >= 0 ? 'up-color' : 'down-color'}">${currSign}${s.open_to_current_pct}%</span>
                    </div>
                </div>

                <div class="top10-metrics">
                    <div class="top10-metric">
                        <span class="label">Confidence</span>
                        <div class="metric-bar"><div class="metric-fill" style="width:${s.confidence}%"></div></div>
                        <span class="value">${s.confidence}%</span>
                    </div>
                    <div class="top10-metric">
                        <span class="label">Agreement</span>
                        <div class="metric-bar"><div class="metric-fill agreement" style="width:${s.model_agreement}%"></div></div>
                        <span class="value">${s.model_agreement}%</span>
                    </div>
                    <div class="top10-metric">
                        <span class="label">Score</span>
                        <span class="value score-badge">${s.composite_score}</span>
                    </div>
                    ${s.risk_reward ? `<div class="top10-metric"><span class="label">R:R</span><span class="value">${s.risk_reward}</span></div>` : ''}
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
async function showTopPicks() {
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    event.target.classList.add('active');

    document.getElementById('stock-grid').style.display = 'none';
    document.getElementById('expected-actual-section').classList.add('hidden');
    document.getElementById('top-analysis-section').classList.add('hidden');
    document.querySelector('.sector-tabs').style.display = 'none';
    document.querySelector('.search-container').style.display = 'none';

    const section = document.getElementById('top-picks-section');
    section.classList.remove('hidden');

    const grid = document.getElementById('picks-grid');
    grid.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><p>Running ML predictions across all sectors... This may take 2-5 minutes.</p></div>';

    try {
        const res = await fetch('/api/top-picks?sectors=large_cap&sectors=banking&sectors=mid_cap&n=10');
        const picks = await res.json();

        if (picks.error) {
            grid.innerHTML = `<div class="loading-spinner"><p>⚠️ ${picks.error}</p></div>`;
            return;
        }

        let html = '';
        for (const pick of picks) {
            const isUp = pick.predicted_return > 0;
            const signalClass = isUp ? 'buy' : 'sell';
            const returnSign = isUp ? '+' : '';

            html += `
            <div class="pick-card" onclick="window.location='/stock/${encodeURIComponent(pick.ticker)}'">
                <div class="pick-card-header">
                    <div>
                        <h4>${pick.name || pick.ticker.replace('.NS', '')}</h4>
                        <span class="muted-text">${pick.ticker}</span>
                    </div>
                    <span class="pick-signal ${signalClass}">${pick.signal}</span>
                </div>
                <div class="pick-details">
                    <div class="pick-detail">
                        <span class="label">Current Price</span>
                        <span class="value">${formatPrice(pick.current_price)}</span>
                    </div>
                    <div class="pick-detail">
                        <span class="label">Predicted Return</span>
                        <span class="value ${isUp ? 'up-color' : 'down-color'}">${returnSign}${pick.predicted_return?.toFixed(3)}%</span>
                    </div>
                    <div class="pick-detail">
                        <span class="label">Target</span>
                        <span class="value">${formatPrice(pick.target_price || pick.predicted_price)}</span>
                    </div>
                    <div class="pick-detail">
                        <span class="label">Confidence</span>
                        <span class="value">${pick.confidence?.toFixed(0)}%</span>
                    </div>
                    <div class="pick-detail">
                        <span class="label">Agreement</span>
                        <span class="value">${pick.model_agreement?.toFixed(0)}%</span>
                    </div>
                    <div class="pick-detail">
                        <span class="label">R:R</span>
                        <span class="value">${pick.risk_reward?.toFixed(1)}</span>
                    </div>
                </div>
            </div>`;
        }

        grid.innerHTML = html || '<div class="loading-spinner"><p>No predictions available yet.</p></div>';
    } catch (e) {
        grid.innerHTML = `<div class="loading-spinner"><p>Error: ${e.message}</p></div>`;
    }
}

// ── Expected vs Actual ────────────────────────────
async function showExpectedVsActual() {
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    event.target.classList.add('active');

    document.getElementById('stock-grid').style.display = 'none';
    document.getElementById('top-picks-section').classList.add('hidden');
    document.getElementById('top-analysis-section').classList.add('hidden');
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
                <span class="sc-label">Avg Alpha</span>
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
            const dirClass = r.direction_correct ? 'correct' : 'wrong';
            const dirText = r.direction_correct ? '✓ Correct' : '✗ Wrong';
            const predColor = r.predicted_return_pct >= 0 ? 'up-color' : 'down-color';
            const actColor = r.actual_return_pct >= 0 ? 'up-color' : 'down-color';
            const alpColor = r.alpha_pct >= 0 ? 'up-color' : 'down-color';

            rows += `
            <tr onclick="window.location='/stock/${encodeURIComponent(r.ticker)}'" style="cursor:pointer">
                <td><strong>${r.name}</strong><br><span class="muted-text">${r.ticker}</span></td>
                <td>${r.signal}</td>
                <td class="${predColor}">${r.predicted_return_pct >= 0 ? '+' : ''}${r.predicted_return_pct}%</td>
                <td>${formatPrice(r.predicted_price)}</td>
                <td>${formatPrice(r.actual_price)}</td>
                <td class="${actColor}">${r.actual_return_pct >= 0 ? '+' : ''}${r.actual_return_pct}%</td>
                <td><span class="direction-badge ${dirClass}">${dirText}</span></td>
                <td class="${alpColor}">${r.alpha_pct >= 0 ? '+' : ''}${r.alpha_pct}%</td>
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
                    <th>Actual Price</th>
                    <th>Actual Return</th>
                    <th>Direction</th>
                    <th>Alpha</th>
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
