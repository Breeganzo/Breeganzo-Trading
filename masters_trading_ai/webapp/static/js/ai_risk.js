/* ============================================================
   Masters AI Trading Bot — AI Risk Page JS
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    loadAiRisk();
});

async function checkStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const badge = document.getElementById('market-status');
        if (!badge) return;
        if (data.market?.status === 'market_open') {
            badge.textContent = '🟢 Market Open';
            badge.className = 'status-badge open';
        } else {
            badge.textContent = '🔴 ' + (data.market?.description || 'Market Closed');
            badge.className = 'status-badge closed';
        }
    } catch (_) {}
}

function formatN(n) {
    const v = Number(n || 0);
    if (!Number.isFinite(v)) return '0';
    return v.toLocaleString('en-IN', { maximumFractionDigits: 2 });
}

async function loadAiRisk() {
    const summaryEl = document.getElementById('ai-risk-summary');
    const outputEl = document.getElementById('ai-risk-output');
    if (outputEl) {
        outputEl.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><p>Generating AI risk analysis...</p></div>';
    }
    try {
        const res = await fetch('/api/ai-risk-analysis');
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
        const s = data.summary || {};
        if (summaryEl) {
            summaryEl.innerHTML = `
                <div class="summary-card"><span class="summary-label">Open Positions</span><span class="summary-value">${s.position_count || 0}</span></div>
                <div class="summary-card"><span class="summary-label">Realized P&L</span><span class="summary-value ${Number(s.realized_pnl || 0) >= 0 ? 'up-color' : 'down-color'}">₹${formatN(s.realized_pnl)}</span></div>
                <div class="summary-card"><span class="summary-label">Unrealized P&L</span><span class="summary-value ${Number(s.unrealized_pnl || 0) >= 0 ? 'up-color' : 'down-color'}">₹${formatN(s.unrealized_pnl)}</span></div>
                <div class="summary-card"><span class="summary-label">Total P&L</span><span class="summary-value ${Number(s.total_pnl || 0) >= 0 ? 'up-color' : 'down-color'}">₹${formatN(s.total_pnl)}</span></div>
            `;
        }
        if (outputEl) {
            outputEl.innerHTML = `
                <div class="news-card">
                    <div class="news-body">
                        <p>${String(data.analysis || 'No analysis').replace(/\n/g, '<br>')}</p>
                        <p class="muted-text">Generated: ${data.generated_at || 'now'}</p>
                    </div>
                </div>
            `;
        }
    } catch (e) {
        if (outputEl) outputEl.innerHTML = `<p class="muted-text">AI risk analysis failed: ${e.message}</p>`;
    }
}
