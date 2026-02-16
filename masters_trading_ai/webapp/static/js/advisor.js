/* ============================================================
   Trading Desk Advisor (Simulation Only)
   ============================================================ */

let advisorOpenList = [];

function advisorFormatPrice(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n <= 0) return '—';
    return `₹${n.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
}

function advisorBudgetValue() {
    const el = document.getElementById('advisor-budget');
    const n = Number(el?.value || 0);
    if (!Number.isFinite(n) || n <= 0) return 40000;
    return Math.max(1000, n);
}

async function refreshAdvisorSummary() {
    try {
        const res = await fetch('/api/simulate/portfolio');
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Simulation summary unavailable');
        const cashEl = document.getElementById('advisor-cash');
        const openEl = document.getElementById('advisor-open-count');
        const eqEl = document.getElementById('advisor-equity');
        if (cashEl) cashEl.textContent = advisorFormatPrice(data.cash);
        if (openEl) openEl.textContent = String(data.open_positions_count || 0);
        if (eqEl) eqEl.textContent = advisorFormatPrice(data.equity_value);
    } catch (e) {
        const status = document.getElementById('advisor-status');
        if (status) status.textContent = `Simulation summary error: ${e.message}`;
    }
}

function renderAdvisorOpenList(rows) {
    const body = document.getElementById('advisor-open-buy-body');
    if (!body) return;
    if (!rows || !rows.length) {
        body.innerHTML = '<tr><td colspan="7" class="muted-text">No advisor picks available right now.</td></tr>';
        return;
    }
    body.innerHTML = rows.map((row) => `
        <tr>
            <td><strong>${row.name || row.ticker.replace('.NS', '')}</strong><br><span class="muted-text">${row.ticker}</span></td>
            <td>${advisorFormatPrice(row.strategy_price_at_open)}</td>
            <td>${Number(row.suggested_qty || 0)}</td>
            <td>${advisorFormatPrice(row.est_trade_cost)}</td>
            <td>${advisorFormatPrice(row.stop_loss_price)}</td>
            <td>${Number(row.risk_reward || 0).toFixed(2)}</td>
            <td>
                <button type="button" class="tf-btn" onclick="simulateAdvisorBuy('${row.ticker}')">Sim BUY</button>
            </td>
        </tr>
    `).join('');
}

async function loadAdvisorOpenBuyList() {
    const status = document.getElementById('advisor-status');
    const budget = advisorBudgetValue();
    if (status) status.textContent = 'Loading advisor picks...';
    try {
        const res = await fetch(`/api/advisor/open-buy-list?n=10&budget=${encodeURIComponent(budget)}`);
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Advisor list unavailable');
        advisorOpenList = data.picks || [];
        renderAdvisorOpenList(advisorOpenList);
        const warnings = (data.warnings || []).slice(0, 3);
        if (status) {
            const warnText = warnings.length ? ` | Warnings: ${warnings.join(' ; ')}` : '';
            status.textContent = `Generated ${data.count || 0} picks. Budget ₹${budget.toLocaleString('en-IN')} | Estimated total ₹${Number(data.estimated_total_cost || 0).toLocaleString('en-IN')}${warnText}`;
        }
        await refreshAdvisorSummary();
    } catch (e) {
        advisorOpenList = [];
        renderAdvisorOpenList([]);
        if (status) status.textContent = `Advisor error: ${e.message}`;
    }
}

async function simulateAdvisorBuy(ticker) {
    const row = advisorOpenList.find((r) => r.ticker === ticker);
    const status = document.getElementById('advisor-status');
    if (!row) {
        if (status) status.textContent = `No advisor row found for ${ticker}`;
        return;
    }
    if (status) status.textContent = `Submitting simulated BUY for ${ticker}...`;
    try {
        const res = await fetch('/api/simulate/trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action: 'BUY',
                ticker: row.ticker,
                quantity: Number(row.suggested_qty || 0),
                price: Number(row.strategy_price_at_open || row.current_price || 0),
                strategy_entry_price: Number(row.strategy_price_at_open || 0),
                stop_loss_price: Number(row.stop_loss_price || 0),
                target_price: Number(row.current_price || row.strategy_price_at_open || 0),
                risk_reward: Number(row.risk_reward || 1.2),
                trade_type: 'equity_delivery',
            }),
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Simulated BUY failed');
        if (status) status.textContent = `Simulated BUY: ${row.ticker} x ${row.suggested_qty} at ${advisorFormatPrice(row.strategy_price_at_open)}.`;
        await refreshAdvisorSummary();
    } catch (e) {
        if (status) status.textContent = `Simulated BUY error: ${e.message}`;
    }
}

async function runAdvisorAutoCheck() {
    const status = document.getElementById('advisor-status');
    if (status) status.textContent = 'Running auto-check for stop-loss/target triggers...';
    try {
        const res = await fetch('/api/simulate/trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'AUTO_CHECK' }),
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Auto-check failed');
        if (status) status.textContent = `Auto-check completed. Triggered ${Number(data.triggered_count || 0)} auto-sell event(s).`;
        await refreshAdvisorSummary();
    } catch (e) {
        if (status) status.textContent = `Auto-check error: ${e.message}`;
    }
}

async function resetAdvisorSimulation() {
    const status = document.getElementById('advisor-status');
    const budget = advisorBudgetValue();
    if (status) status.textContent = 'Resetting simulation portfolio...';
    try {
        const res = await fetch('/api/simulate/trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'RESET', budget }),
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Reset failed');
        if (status) status.textContent = `Simulation reset to ₹${budget.toLocaleString('en-IN')}.`;
        await refreshAdvisorSummary();
        await loadAdvisorOpenBuyList();
    } catch (e) {
        if (status) status.textContent = `Reset error: ${e.message}`;
    }
}

window.loadAdvisorOpenBuyList = loadAdvisorOpenBuyList;
window.simulateAdvisorBuy = simulateAdvisorBuy;
window.runAdvisorAutoCheck = runAdvisorAutoCheck;
window.resetAdvisorSimulation = resetAdvisorSimulation;
window.refreshAdvisorSummary = refreshAdvisorSummary;
