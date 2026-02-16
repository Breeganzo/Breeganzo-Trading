/* ============================================================
   Masters AI Trading Bot — Portfolio Page JS
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    refreshPortfolioPage();
});

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
    const v = Number(n || 0);
    if (!Number.isFinite(v)) return '0';
    return v.toLocaleString('en-IN', { maximumFractionDigits: 2 });
}

function signedClass(v) {
    return Number(v || 0) >= 0 ? 'up-color' : 'down-color';
}

async function refreshPortfolioPage() {
    await Promise.all([
        loadSummary(),
        loadHoldings(),
        loadTrades(),
    ]);
}

async function loadSummary() {
    const el = document.getElementById('portfolio-summary-cards');
    try {
        const res = await fetch('/api/portfolio/summary');
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        el.innerHTML = `
            <div class="summary-card"><span class="summary-label">Open Positions</span><span class="summary-value">${data.position_count || 0}</span></div>
            <div class="summary-card"><span class="summary-label">Trades</span><span class="summary-value">${data.trade_count || 0}</span></div>
            <div class="summary-card"><span class="summary-label">Realized P&L</span><span class="summary-value ${signedClass(data.realized_pnl)}">₹${formatN(data.realized_pnl)}</span></div>
            <div class="summary-card"><span class="summary-label">Unrealized P&L</span><span class="summary-value ${signedClass(data.unrealized_pnl)}">₹${formatN(data.unrealized_pnl)}</span></div>
            <div class="summary-card"><span class="summary-label">Total P&L</span><span class="summary-value ${signedClass(data.total_pnl)}">₹${formatN(data.total_pnl)}</span></div>
        `;
    } catch (e) {
        el.innerHTML = `<p class="muted-text">Summary load failed: ${e.message}</p>`;
    }
}

async function loadHoldings() {
    const el = document.getElementById('portfolio-holdings-table');
    try {
        const res = await fetch('/api/portfolio');
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        const rows = data.holdings || [];
        if (!rows.length) {
            el.innerHTML = '<p class="muted-text">No open positions yet.</p>';
            return;
        }
        const body = rows.map(r => `
            <tr>
                <td><a href="/stock/${encodeURIComponent(r.ticker)}">${r.ticker}</a></td>
                <td>${r.quantity}</td>
                <td>₹${formatN(r.avg_buy_price || r.entry_price)}</td>
                <td>${r.current_price > 0 ? `₹${formatN(r.current_price)}` : '—'}</td>
                <td class="${signedClass(r.unrealized_pnl)}">₹${formatN(r.unrealized_pnl)}</td>
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
    } catch (e) {
        el.innerHTML = `<p class="muted-text">Holdings load failed: ${e.message}</p>`;
    }
}

async function loadTrades() {
    const el = document.getElementById('portfolio-trades-table');
    try {
        const res = await fetch('/api/portfolio/trades?limit=300');
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        const rows = data.trades || [];
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
                    <td>₹${formatN(price)}</td>
                    <td>₹${formatN(notional)}</td>
                </tr>
            `;
        }).join('');
        el.innerHTML = `
            <table class="eva-table">
                <thead>
                    <tr>
                        <th>Time</th><th>Ticker</th><th>Side</th><th>Qty</th><th>Price</th><th>Notional</th>
                    </tr>
                </thead>
                <tbody>${body}</tbody>
            </table>
        `;
    } catch (e) {
        el.innerHTML = `<p class="muted-text">Trade history load failed: ${e.message}</p>`;
    }
}
