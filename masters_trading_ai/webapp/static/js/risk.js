/* ============================================================
   Masters AI Trading Bot — Risk Analytics Dashboard JS
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
    loadRiskAnalytics();
    bindRiskTermHover();
});

async function loadRiskAnalytics() {
    const loading = document.getElementById('risk-loading');
    const content = document.getElementById('risk-content');

    try {
        const res = await fetch('/api/risk-analytics');
        if (res.status === 202) {
            loading.innerHTML = '<div class="spinner"></div><p>Daily analysis is being computed. Retrying in 15 seconds...</p>';
            setTimeout(loadRiskAnalytics, 15000);
            return;
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();
        if (data.error) throw new Error(data.error);

        loading.classList.add('hidden');
        content.classList.remove('hidden');

        renderRiskMetrics(data.risk_metrics);
        renderEquityCurve(data.equity_curve);
        renderSectorExposure(data.sector_exposure);
        renderMonteCarlo(data.monte_carlo);
        renderStatTests(data.statistical_tests);
        renderCorrelationMatrix(data.correlation_matrix);
        renderHoldings(data.portfolio_tickers, data.portfolio_holdings || []);

    } catch (e) {
        loading.innerHTML = `<p class="error-text">⚠️ ${e.message}</p><p class="muted-text">Add valid portfolio holdings to run risk analytics.</p>`;
    }
}

// ── Risk Metrics Grid ─────────────────────────────
function renderRiskMetrics(metrics) {
    const grid = document.getElementById('risk-metrics-grid');
    if (!metrics) { grid.innerHTML = '<p class="muted-text">No data</p>'; return; }

    const cards = [
        { label: 'Sharpe Ratio', value: metrics.sharpe_ratio?.toFixed(3), icon: '📊', color: metricColor(metrics.sharpe_ratio, 0, 1, 2) },
        { label: 'Sortino Ratio', value: metrics.sortino_ratio?.toFixed(3), icon: '📈', color: metricColor(metrics.sortino_ratio, 0, 1.5, 3) },
        { label: 'Information Ratio', value: metrics.information_ratio?.toFixed(3) || 'N/A', icon: '🎯', color: metricColor(metrics.information_ratio, 0, 0.5, 1) },
        { label: 'Max Drawdown', value: `${(metrics.max_drawdown * 100).toFixed(2)}%`, icon: '📉', color: metricColor(-metrics.max_drawdown, 0.05, 0.1, 0.2), neg: true },
        { label: 'Drawdown Duration', value: `${metrics.max_drawdown_duration_days}d`, icon: '⏱️', color: metricColor(-metrics.max_drawdown_duration_days, 5, 15, 30), neg: true },
        { label: 'Daily VaR (95%)', value: `${(metrics.daily_var_95 * 100).toFixed(3)}%`, icon: '⚠️', color: 'var(--yellow)' },
        { label: 'Daily CVaR (95%)', value: `${(metrics.daily_cvar_95 * 100).toFixed(3)}%`, icon: '🔥', color: 'var(--red)' },
        { label: 'Parametric VaR (CF)', value: `${(metrics.parametric_var_cf_95 * 100).toFixed(3)}%`, icon: '📐', color: 'var(--yellow)' },
        { label: 'Ann. Volatility', value: `${(metrics.annualized_volatility * 100).toFixed(2)}%`, icon: '🌊', color: 'var(--blue)' },
        { label: 'Skewness', value: metrics.skewness?.toFixed(4), icon: '↗️', color: metrics.skewness > 0 ? 'var(--green)' : 'var(--red)' },
        { label: 'Excess Kurtosis', value: metrics.excess_kurtosis?.toFixed(4), icon: '📊', color: metrics.excess_kurtosis > 3 ? 'var(--red)' : 'var(--blue)' },
        { label: 'Tail Ratio', value: metrics.tail_ratio?.toFixed(3), icon: '🔔', color: metrics.tail_ratio > 1 ? 'var(--green)' : 'var(--red)' },
    ];

    let html = '';
    for (const c of cards) {
        html += `
        <div class="risk-metric-card risk-term" data-risk-term="${c.label}">
            <div class="rmc-icon">${c.icon}</div>
            <div class="rmc-value" style="color:${c.color}">${c.value}</div>
            <div class="rmc-label">${c.label}</div>
        </div>`;
    }
    grid.innerHTML = html;
}

function metricColor(val, bad, ok, good) {
    if (val == null) return 'var(--text-muted)';
    if (val >= good) return 'var(--green)';
    if (val >= ok) return 'var(--yellow)';
    return 'var(--red)';
}

// ── Equity Curve Chart ────────────────────────────
function renderEquityCurve(equityData) {
    const container = document.getElementById('equity-chart');
    if (!equityData || !equityData.length) {
        container.innerHTML = '<p class="muted-text">No equity data</p>';
        return;
    }

    const chart = LightweightCharts.createChart(container, {
        layout: {
            background: { color: '#1c2128' },
            textColor: '#8b949e',
        },
        grid: {
            vertLines: { color: '#30363d' },
            horzLines: { color: '#30363d' },
        },
        width: container.clientWidth,
        height: 320,
        rightPriceScale: {
            borderColor: '#30363d',
        },
        timeScale: {
            borderColor: '#30363d',
        },
    });

    const series = chart.addAreaSeries({
        lineColor: '#5b8def',
        topColor: 'rgba(91, 141, 239, 0.3)',
        bottomColor: 'rgba(91, 141, 239, 0.02)',
        lineWidth: 2,
    });

    series.setData(equityData.map(d => ({
        time: d.date,
        value: d.value,
    })));

    chart.timeScale().fitContent();

    // Add baseline at 100000
    const baseline = chart.addLineSeries({
        color: '#484f58',
        lineWidth: 1,
        lineStyle: 2,
    });
    baseline.setData(equityData.map(d => ({ time: d.date, value: 100000 })));

    window.addEventListener('resize', () => {
        chart.applyOptions({ width: container.clientWidth });
    });
}

// ── Sector Exposure ───────────────────────────────
function renderSectorExposure(exposure) {
    const container = document.getElementById('sector-exposure');
    if (!exposure || !Object.keys(exposure).length) {
        container.innerHTML = '<p class="muted-text">No sector data</p>';
        return;
    }

    const colors = ['#00d09c', '#5b8def', '#f0b90b', '#eb5757', '#a78bfa', '#ff6b6b'];
    let html = '<div class="sector-bars">';
    let i = 0;
    for (const [sector, data] of Object.entries(exposure)) {
        const color = colors[i % colors.length];
        html += `
        <div class="sector-bar-row">
            <div class="sector-bar-label">
                <span class="sector-dot" style="background:${color}"></span>
                ${sector}
            </div>
            <div class="sector-bar-track">
                <div class="sector-bar-fill" style="width:${data.weight_pct}%; background:${color}"></div>
            </div>
            <span class="sector-bar-pct">${data.weight_pct}% (${data.count})</span>
        </div>`;
        i++;
    }
    html += '</div>';
    container.innerHTML = html;
}

// ── Monte Carlo ───────────────────────────────────
function renderMonteCarlo(mc) {
    const container = document.getElementById('monte-carlo-results');
    if (!mc || mc.error) {
        container.innerHTML = `<p class="muted-text">${mc?.error || 'No Monte Carlo data'}</p>`;
        return;
    }

    const tw = mc.terminal_wealth;
    const mdd = mc.max_drawdown;
    const sr = mc.sharpe_ratio;

    let html = `
    <div class="mc-grid">
        <div class="mc-card">
            <h4>Terminal Wealth (₹1L invested)</h4>
            <div class="mc-distributions">
                <div class="mc-row"><span class="mc-label">P5 (Worst case)</span><span class="mc-val down-color">₹${formatN(tw.p5)}</span></div>
                <div class="mc-row"><span class="mc-label">P25</span><span class="mc-val">${formatN(tw.p25)}</span></div>
                <div class="mc-row highlight"><span class="mc-label">Median (P50)</span><span class="mc-val">${formatN(tw.median)}</span></div>
                <div class="mc-row"><span class="mc-label">Mean</span><span class="mc-val">${formatN(tw.mean)}</span></div>
                <div class="mc-row"><span class="mc-label">P75</span><span class="mc-val">${formatN(tw.p75)}</span></div>
                <div class="mc-row"><span class="mc-label">P95 (Best case)</span><span class="mc-val up-color">₹${formatN(tw.p95)}</span></div>
            </div>
        </div>
        <div class="mc-card">
            <h4>Max Drawdown Distribution</h4>
            <div class="mc-distributions">
                <div class="mc-row"><span class="mc-label">Mean MDD</span><span class="mc-val down-color">${(mdd.mean * 100).toFixed(2)}%</span></div>
                <div class="mc-row"><span class="mc-label">P5 (Worst)</span><span class="mc-val down-color">${(mdd.p5_worst * 100).toFixed(2)}%</span></div>
                <div class="mc-row"><span class="mc-label">P95 (Best)</span><span class="mc-val up-color">${(mdd.p95_best * 100).toFixed(2)}%</span></div>
            </div>
            <h4 style="margin-top: 16px;">Sharpe Ratio Distribution</h4>
            <div class="mc-distributions">
                <div class="mc-row"><span class="mc-label">Mean Sharpe</span><span class="mc-val">${sr.mean.toFixed(3)}</span></div>
                <div class="mc-row"><span class="mc-label">P5</span><span class="mc-val">${sr.p5.toFixed(3)}</span></div>
                <div class="mc-row"><span class="mc-label">P95</span><span class="mc-val">${sr.p95.toFixed(3)}</span></div>
            </div>
        </div>
        <div class="mc-card">
            <h4>Probability Analysis</h4>
            <div class="mc-distributions">
                <div class="mc-row highlight">
                    <span class="mc-label risk-term" data-risk-term="Probability of Profit">Prob(Profit)</span>
                    <span class="mc-val ${mc.probability_of_profit > 0.5 ? 'up-color' : 'down-color'}">${(mc.probability_of_profit * 100).toFixed(1)}%</span>
                </div>
                <div class="mc-row">
                    <span class="mc-label risk-term" data-risk-term="Probability of Loss greater than 10%">Prob(Loss > 10%)</span>
                    <span class="mc-val down-color">${(mc.probability_of_loss_gt_10pct * 100).toFixed(1)}%</span>
                </div>
                <div class="mc-row">
                    <span class="mc-label">Simulations</span>
                    <span class="mc-val">${mc.n_simulations}</span>
                </div>
                <div class="mc-row">
                    <span class="mc-label">Horizon</span>
                    <span class="mc-val">${mc.n_days} days</span>
                </div>
            </div>
        </div>
    </div>`;

    container.innerHTML = html;
}

// ── Statistical Tests ─────────────────────────────
function renderStatTests(tests) {
    const container = document.getElementById('stat-tests');
    if (!tests) { container.innerHTML = '<p class="muted-text">No test results</p>'; return; }

    let html = '<div class="stat-tests-grid">';

    // Jarque-Bera
    if (tests.jarque_bera) {
        const jb = tests.jarque_bera;
        const passed = jb.is_normal;
        html += `
        <div class="stat-test-card">
            <div class="test-header">
                <span class="test-icon">${passed ? '✅' : '⚠️'}</span>
                <h4>Jarque-Bera Normality Test</h4>
            </div>
            <p class="test-hypothesis">H₀: Returns are normally distributed</p>
            <div class="test-results">
                <div class="test-row"><span>Statistic</span><span>${jb.statistic}</span></div>
                <div class="test-row"><span>p-value</span><span>${jb.p_value}</span></div>
                <div class="test-row"><span>Result</span>
                    <span class="${passed ? 'up-color' : 'down-color'}">
                        ${passed ? 'Cannot reject H₀ (returns appear normal)' : 'Reject H₀ (returns are non-normal)'}
                    </span>
                </div>
            </div>
            <p class="test-implication ${passed ? '' : 'down-color'}">
                ${passed
                    ? 'Parametric VaR and standard Sharpe ratio are valid.'
                    : 'Use Cornish-Fisher VaR and non-parametric methods. Standard Sharpe may be misleading.'}
            </p>
        </div>`;
    }

    html += '</div>';
    container.innerHTML = html;
}

// ── Correlation Matrix ────────────────────────────
function renderCorrelationMatrix(corr) {
    const container = document.getElementById('correlation-matrix');
    if (!corr || !Object.keys(corr).length) {
        container.innerHTML = '<p class="muted-text">No correlation data</p>';
        return;
    }

    const tickers = Object.keys(corr);
    const shortName = t => t.replace('.NS', '').substring(0, 8);

    let html = '<div class="corr-table-wrapper"><table class="corr-table"><thead><tr><th></th>';
    for (const t of tickers) {
        html += `<th title="${t}">${shortName(t)}</th>`;
    }
    html += '</tr></thead><tbody>';

    for (const row of tickers) {
        html += `<tr><td class="corr-row-label" title="${row}">${shortName(row)}</td>`;
        for (const col of tickers) {
            const val = corr[row]?.[col] ?? 0;
            const bg = corrColor(val);
            const display = row === col ? '1.00' : val.toFixed(2);
            html += `<td class="corr-cell" style="background:${bg}" title="${row} × ${col}: ${val.toFixed(3)}">${display}</td>`;
        }
        html += '</tr>';
    }
    html += '</tbody></table></div>';
    container.innerHTML = html;
}

function corrColor(val) {
    // -1 (red) → 0 (gray) → 1 (green)
    if (val >= 0.8) return 'rgba(0, 208, 156, 0.5)';
    if (val >= 0.5) return 'rgba(0, 208, 156, 0.25)';
    if (val >= 0.2) return 'rgba(0, 208, 156, 0.1)';
    if (val >= -0.2) return 'rgba(139, 148, 158, 0.1)';
    if (val >= -0.5) return 'rgba(235, 87, 87, 0.15)';
    return 'rgba(235, 87, 87, 0.3)';
}

// ── Holdings List ─────────────────────────────────
function renderHoldings(tickers, holdings = []) {
    const container = document.getElementById('portfolio-holdings');
    if (!tickers || !tickers.length) {
        container.innerHTML = '<p class="muted-text">No holdings data</p>';
        return;
    }

    let html = '<div class="holdings-chips">';
    for (const t of tickers) {
        const row = holdings.find(h => h.ticker === t);
        const name = (row?.name || t.replace('.NS', ''));
        const suffix = row ? ` • Qty ${row.quantity} @ ₹${Number(row.entry_price).toFixed(2)}` : '';
        html += `<a href="/stock/${encodeURIComponent(t)}" class="holding-chip">${name}${suffix}</a>`;
    }
    html += '</div>';
    container.innerHTML = html;
}

function bindRiskTermHover() {
    document.addEventListener('mouseover', async (event) => {
        const el = event.target.closest('.risk-term');
        if (!el) return;
        const term = el.dataset.riskTerm || el.textContent?.trim();
        if (!term) return;
        const tooltip = document.getElementById('risk-tooltip');
        const titleEl = document.getElementById('risk-tooltip-title');
        const bodyEl = document.getElementById('risk-tooltip-body');
        if (!tooltip || !titleEl || !bodyEl) return;

        const rect = el.getBoundingClientRect();
        tooltip.style.top = `${window.scrollY + rect.bottom + 10}px`;
        tooltip.style.left = `${Math.min(window.scrollX + rect.left, window.scrollX + window.innerWidth - 380)}px`;
        tooltip.classList.remove('hidden');
        titleEl.textContent = term;
        bodyEl.textContent = 'Loading explanation...';

        try {
            const res = await fetch(`/api/explain-risk-term?term=${encodeURIComponent(term)}&context=${encodeURIComponent('portfolio risk analytics')}`);
            const data = await res.json();
            bodyEl.textContent = data.explanation || data.error || 'No explanation available';
        } catch (e) {
            bodyEl.textContent = `Explanation unavailable: ${e.message}`;
        }
    });

    document.addEventListener('mouseout', (event) => {
        if (!event.target.closest('.risk-term')) return;
        const tooltip = document.getElementById('risk-tooltip');
        if (tooltip) tooltip.classList.add('hidden');
    });
}

// ── Utility ───────────────────────────────────────
function formatN(n) {
    if (n == null) return '--';
    return Number(n).toLocaleString('en-IN', { maximumFractionDigits: 0 });
}
