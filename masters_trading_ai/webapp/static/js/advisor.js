/* ============================================================
   Trading Desk Advisor (Simulation Only)
   ============================================================ */

let advisorOpenList = [];
let advisorBusy = false;
let advisorLedgerRows = [];
let advisorViewFilter = 'buy';
let advisorRealtimeTimer = null;
let advisorRealtimeBusy = false;

function advisorFormatPrice(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n <= 0) return '—';
    return `₹${n.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
}

function advisorFormatSignedPrice(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    const sign = n > 0 ? '+' : '';
    return `${sign}₹${n.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
}

function advisorFormatPriceRange(low, high) {
    const lo = Number(low);
    const hi = Number(high);
    if (!Number.isFinite(lo) || lo <= 0 || !Number.isFinite(hi) || hi <= 0) return '—';
    const min = Math.min(lo, hi);
    const max = Math.max(lo, hi);
    return `${advisorFormatPrice(min)} - ${advisorFormatPrice(max)}`;
}

function advisorFormatTimestamp(value) {
    if (!value) return '—';
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return '—';
    return d.toLocaleString('en-IN', {
        timeZone: 'Asia/Kolkata',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true,
    });
}

function advisorBudgetValue() {
    const el = document.getElementById('advisor-budget');
    const n = Number(el?.value || 0);
    if (!Number.isFinite(n) || n <= 0) return 40000;
    return Math.max(1000, n);
}

function setAdvisorBusy(isBusy) {
    advisorBusy = Boolean(isBusy);
    const panel = document.getElementById('advisor-panel');
    if (!panel) return;
    panel.querySelectorAll('button.tf-btn').forEach((btn) => {
        btn.disabled = advisorBusy;
    });
    const filterInput = document.getElementById('advisor-view-filter');
    if (filterInput) filterInput.disabled = advisorBusy;
    const budgetInput = document.getElementById('advisor-budget');
    if (budgetInput) budgetInput.disabled = advisorBusy;
}

function setAdvisorViewFilter(value) {
    const normalized = String(value || 'buy').toLowerCase();
    advisorViewFilter = ['buy', 'sell', 'hold'].includes(normalized) ? normalized : 'buy';
    const filter = document.getElementById('advisor-view-filter');
    if (filter) filter.value = advisorViewFilter;

    const openWrap = document.getElementById('advisor-open-buy-wrap');
    const ledgerWrap = document.getElementById('advisor-ledger-wrap');
    if (openWrap) openWrap.style.display = advisorViewFilter === 'buy' ? '' : 'none';
    if (ledgerWrap) ledgerWrap.style.display = advisorViewFilter === 'buy' ? 'none' : '';

    if (advisorViewFilter === 'buy') {
        renderAdvisorOpenList(advisorOpenList);
    } else {
        renderAdvisorTradeLedgerRows(advisorLedgerRows);
    }
}

function renderTransactionSummary(summary) {
    const el = document.getElementById('advisor-transactions-summary');
    if (!el) return;
    if (!summary || typeof summary !== 'object') {
        el.textContent = 'Transactions: —';
        return;
    }
    const total = Number(summary.total_transactions || 0);
    const buy = Number(summary.buy_transactions || 0);
    const sell = Number(summary.sell_transactions || 0);
    const buyFee = Number(summary.buy_transaction_cost || 0);
    const sellFee = Number(summary.sell_transaction_cost || 0);
    const totalFee = Number(summary.total_transaction_cost || 0);
    el.textContent = `Transactions today: ${total} (BUY ${buy} / SELL ${sell}) | Fees: BUY ₹${buyFee.toFixed(2)}, SELL ₹${sellFee.toFixed(2)}, TOTAL ₹${totalFee.toFixed(2)}`;
}

async function refreshAdvisorTransactions() {
    try {
        const res = await fetch('/api/simulate/transactions-summary?session=1');
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Transaction summary unavailable');
        renderTransactionSummary(data);
    } catch (_) {
        renderTransactionSummary(null);
    }
}

function renderAdvisorTradeLedgerRows(rows) {
    const body = document.getElementById('advisor-trade-ledger-body');
    if (!body) return;
    const allRows = Array.isArray(rows) ? rows : [];
    const filtered = allRows.filter((row) => {
        const status = String(row?.status || 'HOLD').toUpperCase();
        if (advisorViewFilter === 'sell') return status === 'SOLD';
        if (advisorViewFilter === 'hold') return status === 'HOLD';
        return true;
    });
    if (!filtered.length) {
        body.innerHTML = '<tr><td colspan="13" class="muted-text">No simulated transactions yet.</td></tr>';
        return;
    }
    body.innerHTML = filtered.map((row) => {
        const status = String(row.status || 'HOLD').toUpperCase();
        const qty = Number(row.quantity || 0);
        const pnl = status === 'SOLD' ? Number(row.realized_pnl || 0) : Number(row.unrealized_pnl || 0);
        const pnlStyle = pnl > 0 ? 'color: var(--green);' : (pnl < 0 ? 'color: var(--red);' : '');
        const reason = status === 'SOLD' ? (row.sell_reason || 'sold') : 'hold_open';
        return `
            <tr>
                <td><strong>${row.name || (row.ticker || '').replace('.NS', '')}</strong><br><span class="muted-text">${row.ticker || '—'}</span></td>
                <td>${status}</td>
                <td>${Number.isFinite(qty) ? qty : '—'}</td>
                <td>${advisorFormatPrice(row.buy_price)}</td>
                <td>${advisorFormatTimestamp(row.buy_timestamp)}</td>
                <td>${advisorFormatPrice(row.current_price || row.sell_price)}</td>
                <td>${advisorFormatPriceRange(row.entry_range_low, row.entry_range_high)}</td>
                <td>${advisorFormatPrice(row.stop_loss_price)}</td>
                <td>${advisorFormatPrice(row.target_price)}</td>
                <td>${advisorFormatPrice(row.sell_price)}</td>
                <td>${advisorFormatTimestamp(row.sell_timestamp)}</td>
                <td style="${pnlStyle}">${advisorFormatSignedPrice(pnl)}</td>
                <td>${reason}</td>
            </tr>
        `;
    }).join('');
}

function renderAdvisorTradeLedger(payload) {
    advisorLedgerRows = Array.isArray(payload?.rows) ? payload.rows : [];
    renderAdvisorTradeLedgerRows(advisorLedgerRows);
}

async function refreshAdvisorTradeLedger() {
    try {
        const res = await fetch('/api/simulate/trade-ledger?limit=200&session=1');
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Trade ledger unavailable');
        renderAdvisorTradeLedger(data);
    } catch (_) {
        renderAdvisorTradeLedger(null);
    }
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
        renderTransactionSummary(data.transaction_summary || null);
        await refreshAdvisorTradeLedger();
    } catch (e) {
        const status = document.getElementById('advisor-status');
        if (status) status.textContent = `Simulation summary error: ${e.message}`;
        await refreshAdvisorTransactions();
        await refreshAdvisorTradeLedger();
    }
}

function renderAdvisorOpenList(rows) {
    const body = document.getElementById('advisor-open-buy-body');
    if (!body) return;
    if (!rows || !rows.length) {
        body.innerHTML = '<tr><td colspan="11" class="muted-text">No advisor picks available right now.</td></tr>';
        return;
    }
    body.innerHTML = rows.map((row) => `
        <tr>
            <td><strong>${row.name || row.ticker.replace('.NS', '')}</strong><br><span class="muted-text">${row.ticker}</span></td>
            <td>${String(row.sector || 'other').replaceAll('_', ' ')}</td>
            <td>${advisorFormatPrice(row.strategy_price_at_open)}</td>
            <td>${advisorFormatPrice(row.current_price)}</td>
            <td>${advisorFormatPriceRange(row.entry_range_low, row.entry_range_high)}</td>
            <td>${Number(row.suggested_qty || 0)}</td>
            <td>${advisorFormatPrice(row.est_trade_cost)}</td>
            <td>${advisorFormatPrice(row.stop_loss_price)}</td>
            <td>${advisorFormatPrice(row.target_price)}</td>
            <td>${Number(row.risk_reward || 0).toFixed(2)}</td>
            <td>
                <button type="button" class="tf-btn" onclick="simulateAdvisorBuy('${row.ticker}')">Sim BUY</button>
            </td>
        </tr>
    `).join('');
}

async function loadAdvisorOpenBuyList() {
    if (advisorBusy) return;
    const status = document.getElementById('advisor-status');
    const budget = advisorBudgetValue();
    if (status) status.textContent = 'Loading advisor picks...';
    setAdvisorBusy(true);
    try {
        const res = await fetch(`/api/advisor/open-buy-list?n=10&budget=${encodeURIComponent(budget)}&allow_warming=1`);
        const data = await res.json();
        if (res.status === 202 && data.status === 'warming') {
            if (status) status.textContent = `${data.message || 'Advisor cache warming...'} Retrying in ${Number(data.retry_after_sec || 5)}s.`;
            setTimeout(() => {
                loadAdvisorOpenBuyList();
            }, Math.max(2, Number(data.retry_after_sec || 5)) * 1000);
            return;
        }
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
    } finally {
        setAdvisorBusy(false);
    }
}

async function refreshAdvisorOpenListLivePrices() {
    if (!advisorOpenList.length) return;
    const tickers = [...new Set(advisorOpenList.map((r) => String(r?.ticker || '').toUpperCase()).filter(Boolean))];
    if (!tickers.length) return;
    const res = await fetch(`/api/prices?tickers=${tickers.join(',')}`);
    if (!res.ok) return;
    const data = await res.json();
    advisorOpenList = advisorOpenList.map((row) => {
        const ticker = String(row?.ticker || '').toUpperCase();
        const px = Number(data?.[ticker]?.price || 0);
        if (!Number.isFinite(px) || px <= 0) return row;
        return { ...row, current_price: px };
    });
    if (advisorViewFilter === 'buy') {
        renderAdvisorOpenList(advisorOpenList);
    }
}

async function simulateAdvisorBuy(ticker) {
    if (advisorBusy) return;
    const row = advisorOpenList.find((r) => r.ticker === ticker);
    const status = document.getElementById('advisor-status');
    if (!row) {
        if (status) status.textContent = `No advisor row found for ${ticker}`;
        return;
    }
    if (status) status.textContent = `Submitting simulated BUY for ${ticker}...`;
    setAdvisorBusy(true);
    try {
        const res = await fetch('/api/simulate/trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action: 'BUY',
                ticker: row.ticker,
                quantity: Number(row.suggested_qty || 0),
                price: Number(row.current_price || row.strategy_price_at_open || 0),
                strategy_entry_price: Number(row.strategy_price_at_open || 0),
                entry_range_low: Number(row.entry_range_low || 0),
                entry_range_high: Number(row.entry_range_high || 0),
                stop_loss_price: Number(row.stop_loss_price || 0),
                target_price: Number(row.target_price || row.current_price || row.strategy_price_at_open || 0),
                risk_reward: Number(row.risk_reward || 1.2),
                confidence: Number(row.confidence || 0),
                atr_pct: Number(row.volatility_atr_pct || 0),
                uses_sentiment: Number(row.sentiment_weighted_score || 0) !== 0,
                trade_type: 'equity_delivery',
            }),
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Simulated BUY failed');
        if (status) {
            status.textContent = `Simulated BUY: ${row.ticker} x ${row.suggested_qty} at ${advisorFormatPrice(data?.event?.price || row.current_price || row.strategy_price_at_open)} within ${advisorFormatPriceRange(row.entry_range_low, row.entry_range_high)}.`;
        }
        await refreshAdvisorSummary();
    } catch (e) {
        if (status) status.textContent = `Simulated BUY error: ${e.message}`;
    } finally {
        setAdvisorBusy(false);
    }
}

async function runAdvisorAutoCheck() {
    if (advisorBusy) return;
    const status = document.getElementById('advisor-status');
    if (status) status.textContent = 'Running auto-check (trailing stop, auto-sell, auto-buy entry)...';
    setAdvisorBusy(true);
    try {
        const res = await fetch('/api/simulate/trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'AUTO_CHECK', auto_buy: true }),
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Auto-check failed');
        const sells = Number((data.events || []).length || 0);
        const buys = Number((data.auto_buy_events || []).length || 0);
        if (status) {
            const note = data.note ? ` ${data.note}` : '';
            status.textContent = `Auto-check completed. Auto-sell: ${sells}, auto-buy: ${buys}.${note}`;
        }
        await refreshAdvisorSummary();
    } catch (e) {
        if (status) status.textContent = `Auto-check error: ${e.message}`;
    } finally {
        setAdvisorBusy(false);
    }
}

async function resetAdvisorSimulation() {
    if (advisorBusy) return;
    const status = document.getElementById('advisor-status');
    const budget = advisorBudgetValue();
    if (status) status.textContent = 'Resetting simulation portfolio...';
    setAdvisorBusy(true);
    try {
        const res = await fetch('/api/simulate/trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action: 'RESET',
                budget,
                clear_history: true,
                clear_portfolio_sim_trades: true,
            }),
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Reset failed');
        if (status) {
            const cleaned = Number(data?.cleanup?.removed_portfolio_sim_trades || 0);
            status.textContent = `Simulation reset to ₹${budget.toLocaleString('en-IN')} (cleared ${cleaned} old simulation portfolio trades).`;
        }
        await refreshAdvisorSummary();
        setAdvisorBusy(false);
        await loadAdvisorOpenBuyList();
    } catch (e) {
        if (status) status.textContent = `Reset error: ${e.message}`;
    } finally {
        if (advisorBusy) setAdvisorBusy(false);
    }
}

async function advisorRealtimeTick() {
    if (advisorRealtimeBusy || advisorBusy) return;
    if (document.visibilityState !== 'visible') return;
    const panel = document.getElementById('advisor-panel');
    if (!panel) return;
    advisorRealtimeBusy = true;
    try {
        await refreshAdvisorOpenListLivePrices();
        await refreshAdvisorSummary();
    } catch (_) {
        // best effort polling
    } finally {
        advisorRealtimeBusy = false;
    }
}

function startAdvisorRealtimeMonitor() {
    if (advisorRealtimeTimer) return;
    advisorRealtimeTimer = setInterval(() => {
        advisorRealtimeTick();
    }, 1000);
}

window.loadAdvisorOpenBuyList = loadAdvisorOpenBuyList;
window.simulateAdvisorBuy = simulateAdvisorBuy;
window.runAdvisorAutoCheck = runAdvisorAutoCheck;
window.resetAdvisorSimulation = resetAdvisorSimulation;
window.refreshAdvisorSummary = refreshAdvisorSummary;
window.refreshAdvisorTransactions = refreshAdvisorTransactions;
window.refreshAdvisorTradeLedger = refreshAdvisorTradeLedger;
window.setAdvisorViewFilter = setAdvisorViewFilter;
window.startAdvisorRealtimeMonitor = startAdvisorRealtimeMonitor;
