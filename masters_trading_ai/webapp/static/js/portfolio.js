/* ============================================================
   Masters AI Trading Bot — Portfolio Page JS
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    refreshPortfolioPage();
    loadGroqTickerSuggestions();
});

let portfolioBusy = false;

async function checkStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const mktBadge = document.getElementById('market-status');
        if (!mktBadge) return;
        if (data.market?.status === 'market_open') {
            mktBadge.textContent = '🟢 Market Open';
            mktBadge.className = 'status-badge open';
        } else {
            mktBadge.textContent = '🔴 ' + (data.market?.description || 'Market Closed');
            mktBadge.className = 'status-badge closed';
        }
    } catch (_) {}
}

function formatN(n) {
    const v = Number(n);
    if (!Number.isFinite(v)) return '—';
    return v.toLocaleString('en-IN', { maximumFractionDigits: 2 });
}

function formatPrice(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n <= 0) return '—';
    return `₹${formatN(n)}`;
}

function signedClass(v) {
    return Number(v || 0) >= 0 ? 'up-color' : 'down-color';
}

function setToolbarStatus(text) {
    const el = document.getElementById('portfolio-toolbar-status');
    if (el) el.textContent = text || '';
}

async function refreshPortfolioPage() {
    if (portfolioBusy) return;
    portfolioBusy = true;
    setPortfolioBusy(true, 'Refreshing portfolio...');
    try {
        const res = await fetch('/api/portfolio/refresh?limit=300');
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
        renderSummary(data.summary || {});
        renderHoldings(data.holdings || []);
        renderTrades(data.trades || []);
        setToolbarStatus(`Refreshed at ${new Date(data.refreshed_at || Date.now()).toLocaleTimeString('en-IN')}`);
    } catch (e) {
        setToolbarStatus(`Portfolio refresh failed: ${e.message}`);
    } finally {
        setPortfolioBusy(false);
        portfolioBusy = false;
    }
}

function setPortfolioBusy(isBusy, text = '') {
    document.querySelectorAll('.portfolio-toolbar button, .portfolio-toolbar .tf-btn, .portfolio-toolbar select, .portfolio-toolbar input')
        .forEach((el) => {
            if (!(el instanceof HTMLButtonElement || el instanceof HTMLSelectElement || el instanceof HTMLInputElement)) return;
            el.disabled = Boolean(isBusy);
        });
    if (text) setToolbarStatus(text);
}

function renderSummary(data) {
    const el = document.getElementById('portfolio-summary-cards');
    el.innerHTML = `
        <div class="summary-card"><span class="summary-label position-count">Open Positions</span><span class="summary-value">${data.position_count || 0}</span></div>
        <div class="summary-card"><span class="summary-label">Trades</span><span class="summary-value">${data.trade_count || 0}</span></div>
        <div class="summary-card"><span class="summary-label">Realized P&L</span><span class="summary-value ${signedClass(data.realized_pnl)}">₹${formatN(data.realized_pnl)}</span></div>
        <div class="summary-card"><span class="summary-label">Unrealized P&L</span><span class="summary-value ${signedClass(data.unrealized_pnl)}">₹${formatN(data.unrealized_pnl)}</span></div>
        <div class="summary-card"><span class="summary-label">Total P&L</span><span class="summary-value ${signedClass(data.total_pnl)}">₹${formatN(data.total_pnl)}</span></div>
    `;
}

function renderHoldings(rows) {
    const el = document.getElementById('portfolio-holdings-table');
    if (!rows.length) {
        el.innerHTML = '<p class="muted-text">No open positions yet.</p>';
        return;
    }
    const body = rows.map(r => `
        <tr>
            <td><a href="/stock/${encodeURIComponent(r.ticker)}">${r.ticker}</a></td>
            <td>${r.quantity}</td>
            <td>${formatPrice(r.avg_buy_price || r.entry_price)}</td>
            <td>${formatPrice(r.current_price)}</td>
            <td class="${signedClass(r.unrealized_pnl)}">${Number.isFinite(Number(r.unrealized_pnl)) ? `₹${formatN(r.unrealized_pnl)}` : '—'}</td>
            <td class="${signedClass(r.unrealized_pnl_pct)}">${Number(r.unrealized_pnl_pct || 0).toFixed(2)}%</td>
        </tr>
    `).join('');
    el.innerHTML = `
        <table class="eva-table">
            <thead>
                <tr>
                    <th>Ticker</th><th>Qty</th><th>Avg Buy</th><th>Current</th><th>Unrealized P&L</th><th>Unrealized %</th>
                </tr>
            </thead>
            <tbody>${body}</tbody>
        </table>
    `;
}

function renderTrades(rows) {
    const el = document.getElementById('portfolio-trades-table');
    if (!rows.length) {
        el.innerHTML = '<p class="muted-text">No trades recorded yet.</p>';
        return;
    }
    const body = rows.map(r => {
        const dt = r.timestamp ? new Date(r.timestamp).toLocaleString() : '—';
        const qty = Number(r.quantity || 0);
        const price = Number(r.price || 0);
        const notional = qty * price;
        return `
            <tr>
                <td>${dt}</td>
                <td><a href="/stock/${encodeURIComponent(r.ticker)}">${r.ticker}</a></td>
                <td>${r.side}</td>
                <td>${qty}</td>
                <td>${formatPrice(price)}</td>
                <td>${formatPrice(notional)}</td>
                <td>
                    <button class="tf-btn" onclick="editTrade('${r.id}')">Edit</button>
                    <button class="tf-btn" onclick="deleteTrade('${r.id}')">Delete</button>
                </td>
            </tr>
        `;
    }).join('');
    el.innerHTML = `
        <table class="eva-table">
            <thead>
                <tr>
                    <th>Time</th><th>Ticker</th><th>Side</th><th>Qty</th><th>Price</th><th>Notional</th><th>Actions</th>
                </tr>
            </thead>
            <tbody>${body}</tbody>
        </table>
    `;
}

async function clearEntirePortfolio() {
    if (!confirm('Clear all trades and holdings? This cannot be undone.')) return;
    try {
        const res = await fetch('/api/portfolio', { method: 'DELETE' });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
        setToolbarStatus('Portfolio cleared.');
        await refreshPortfolioPage();
    } catch (e) {
        setToolbarStatus(`Failed to clear portfolio: ${e.message}`);
    }
}

async function cleanInvalidPortfolioRows() {
    try {
        const res = await fetch('/api/portfolio/clean', { method: 'POST' });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
        setToolbarStatus(`Cleaned invalid rows: trades=${data.removed_trade_rows || 0}, holdings=${data.removed_holding_rows || 0}`);
        await refreshPortfolioPage();
    } catch (e) {
        setToolbarStatus(`Cleanup failed: ${e.message}`);
    }
}

async function deleteTrade(id) {
    if (!id || !confirm('Delete this trade entry?')) return;
    try {
        const res = await fetch(`/api/portfolio/trade/${encodeURIComponent(id)}`, { method: 'DELETE' });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
        setToolbarStatus('Trade deleted.');
        await refreshPortfolioPage();
    } catch (e) {
        setToolbarStatus(`Delete failed: ${e.message}`);
    }
}

async function editTrade(id) {
    if (!id) return;
    const side = (prompt('Side (BUY or SELL):', 'BUY') || '').trim().toUpperCase();
    const qty = Number(prompt('Quantity:', '1') || 0);
    const price = Number(prompt('Price:', '100') || 0);
    if (!['BUY', 'SELL'].includes(side) || qty <= 0 || price <= 0) {
        setToolbarStatus('Invalid edit inputs.');
        return;
    }
    try {
        const res = await fetch(`/api/portfolio/trade/${encodeURIComponent(id)}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ side, quantity: qty, price }),
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
        setToolbarStatus('Trade updated.');
        await refreshPortfolioPage();
    } catch (e) {
        setToolbarStatus(`Edit failed: ${e.message}`);
    }
}

async function loadGroqTickerSuggestions() {
    const output = document.getElementById('groq-suggestions-output');
    const select = document.getElementById('groq-ticker-select');
    if (!output || !select) return;
    output.textContent = 'Loading Groq ticker suggestions...';
    try {
        const res = await fetch('/api/groq-ticker-suggestions?n=8');
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
        const rows = data.candidates || [];
        if (!rows.length) {
            output.textContent = 'No suggestions available right now.';
            return;
        }
        select.innerHTML = rows.map(r => `<option value="${r.ticker}">${r.ticker} (${r.signal}, ${Number(r.predicted_return || 0).toFixed(2)}%)</option>`).join('');
        output.innerHTML = `<strong>Groq Suggestion:</strong> ${data.recommendation || 'No text'}<br><span class="muted-text">Updated: ${data.generated_at || 'now'}</span>`;
    } catch (e) {
        output.textContent = `Suggestion load failed: ${e.message}`;
    }
}

async function reviewGroqTradePlan() {
    const ticker = document.getElementById('groq-ticker-select')?.value;
    const entryPrice = Number(document.getElementById('groq-entry-price')?.value || 0);
    const qty = Number(document.getElementById('groq-entry-qty')?.value || 0);
    const output = document.getElementById('groq-review-output');
    if (!ticker || entryPrice <= 0 || qty <= 0) {
        if (output) output.textContent = 'Select ticker and enter valid entry price/quantity.';
        return;
    }
    if (output) output.textContent = 'Reviewing trade plan with Groq...';
    try {
        const res = await fetch('/api/groq-trade-review', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker, entry_price: entryPrice, quantity: qty }),
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
        if (output) {
            output.innerHTML = `
                <strong>Trade Review (${data.ticker})</strong><br>
                Signal: ${data.signal}, Confidence: ${data.confidence}%, Agreement: ${data.model_agreement}%<br>
                Current: ${data.current_price ? `₹${formatN(data.current_price)}` : '—'}<br>
                ${data.review || ''}<br>
                <span class="muted-text">Generated: ${data.generated_at || 'now'}</span>
            `;
        }
    } catch (e) {
        if (output) output.textContent = `Review failed: ${e.message}`;
    }
}
