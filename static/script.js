/* ============================================
   FraudShield — Frontend JavaScript
   PaySim + Email Phishing Dashboard
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {

    // ─── DOM REFS ───
    const navItems       = document.querySelectorAll('.nav-item[data-section]');
    const sections       = document.querySelectorAll('.section');
    const pageTitle      = document.getElementById('pageTitle');
    const liveTime       = document.getElementById('liveTime');
    const menuToggle     = document.getElementById('menuToggle');
    const sidebar        = document.getElementById('sidebar');
    const toast          = document.getElementById('toast');

    const statTotal      = document.getElementById('statTotal');
    const statFraud      = document.getElementById('statFraud');
    const statSafe       = document.getElementById('statSafe');
    const statAccuracy   = document.getElementById('statAccuracy');
    const statModelType  = document.getElementById('statModelType');
    const sidebarStatus  = document.getElementById('sidebarModelStatus');

    const metricPrecision = document.getElementById('metricPrecision');
    const metricRecall    = document.getElementById('metricRecall');
    const metricF1        = document.getElementById('metricF1');
    const metricAUC       = document.getElementById('metricAUC');
    const barPrecision    = document.getElementById('barPrecision');
    const barRecall       = document.getElementById('barRecall');
    const barF1           = document.getElementById('barF1');
    const barAUC          = document.getElementById('barAUC');

    const groupDiv    = document.getElementById('groupImportances');
    const featureDiv  = document.getElementById('featureImportances');
    const cmDiv       = document.getElementById('confusionMatrix');
    const dsSource    = document.getElementById('dsSource');
    const dsTotalRows = document.getElementById('dsTotalRows');
    const dsFraudCases = document.getElementById('dsFraudCases');
    const dsBalanced   = document.getElementById('dsBalanced');

    const analyzerForm = document.getElementById('analyzerForm');
    const analyzeBtn   = document.getElementById('analyzeBtn');
    const gaugeCanvas  = document.getElementById('riskGauge');
    const gaugeScore   = document.getElementById('gaugeScore');
    const gaugeSev     = document.getElementById('gaugeSeverity');
    const gaugeRec     = document.getElementById('gaugeRecommendation');
    const resultMethod = document.getElementById('resultMethod');
    const reasonsSec   = document.getElementById('reasonsSection');
    const reasonsList  = document.getElementById('reasonsList');
    const actualBanner = document.getElementById('actualLabelBanner');

    const loadRandomBtn = document.getElementById('loadRandomBtn');
    const loadFraudBtn  = document.getElementById('loadFraudBtn');
    const loadSafeBtn   = document.getElementById('loadSafeBtn');

    const historyBody  = document.getElementById('historyTableBody');
    const refreshBtn   = document.getElementById('refreshHistoryBtn');
    const datasetBody  = document.getElementById('datasetTableBody');
    const refreshDsBtn = document.getElementById('refreshDatasetBtn');
    const dsTotal      = document.getElementById('dsTotal');
    const dsFraudCount = document.getElementById('dsFraudCount');
    const dsLegitCount = document.getElementById('dsLegitCount');
    const dsFraudPct   = document.getElementById('dsFraudPct');
    const clearBtn     = document.getElementById('navClearLogs');

    // Phishing
    const tabText       = document.getElementById('tabText');
    const tabImage      = document.getElementById('tabImage');
    const tabContentText  = document.getElementById('tabContentText');
    const tabContentImage = document.getElementById('tabContentImage');
    const emailTextInput  = document.getElementById('emailTextInput');
    const emailImageInput = document.getElementById('emailImageInput');
    const uploadZone      = document.getElementById('uploadZone');
    const uploadPreview   = document.getElementById('uploadPreview');
    const previewImg      = document.getElementById('previewImg');
    const clearImageBtn   = document.getElementById('clearImageBtn');
    const analyzeEmailBtn = document.getElementById('analyzeEmailBtn');
    const factorScores       = document.getElementById('factorScores');
    const factorScoresGrid   = document.getElementById('factorScoresGrid');
    const phishingScoreRing   = document.getElementById('phishingScoreRing');
    const phishingScoreValue  = document.getElementById('phishingScoreValue');
    const phishingVerdict     = document.getElementById('phishingVerdict');
    const phishingSummary     = document.getElementById('phishingSummary');
    const phishingIndicators  = document.getElementById('phishingIndicators');
    const phishingIndicatorsList = document.getElementById('phishingIndicatorsList');

    let uploadedImageBase64 = '';


    // ═══ NAVIGATION ═══
    const sectionTitles = { dashboard:'Dashboard', analyzer:'Analyzer', phishing:'Email Phishing', history:'History', dataset:'Dataset' };
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const target = item.dataset.section;
            sections.forEach(s => s.classList.add('hidden'));
            document.getElementById(target)?.classList.remove('hidden');
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            pageTitle.textContent = sectionTitles[target] || 'Dashboard';
            sidebar.classList.remove('open');
            if (target === 'dashboard') loadStats();
            if (target === 'history') loadHistory();
            if (target === 'dataset') loadDataset();
        });
    });
    menuToggle?.addEventListener('click', () => sidebar.classList.toggle('open'));


    // ═══ LIVE TIME ═══
    function updateTime() {
        liveTime.textContent = new Date().toLocaleString('en-US', {
            weekday:'short', hour:'2-digit', minute:'2-digit', second:'2-digit'
        });
    }
    updateTime(); setInterval(updateTime, 1000);


    // ═══ UTILITIES ═══
    function showToast(msg, type = 'success') {
        toast.textContent = msg;
        toast.className = 'toast show ' + type;
        setTimeout(() => { toast.className = 'toast'; }, 4000);
    }

    function animateCounter(el, target) {
        if (!el) return;
        const start = parseInt(el.textContent) || 0;
        const diff = target - start;
        if (diff === 0) { el.textContent = target.toLocaleString(); return; }
        let step = 0; const steps = 35;
        const timer = setInterval(() => {
            step++;
            el.textContent = Math.round(start + (diff * step / steps)).toLocaleString();
            if (step >= steps) { el.textContent = target.toLocaleString(); clearInterval(timer); }
        }, 22);
    }


    // ═══ GAUGE ═══
    function drawGauge(canvas, score) {
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        const cx = w/2, cy = h/2, r = 92;
        ctx.clearRect(0, 0, w, h);

        ctx.beginPath();
        ctx.arc(cx, cy, r, 0.75*Math.PI, 2.25*Math.PI, false);
        ctx.lineWidth = 15; ctx.strokeStyle = 'rgba(255,255,255,0.04)';
        ctx.lineCap = 'round'; ctx.stroke();

        const endAngle = 0.75*Math.PI + (score * 1.5*Math.PI);
        let color;
        if (score < 0.3) color = '#46D369';
        else if (score < 0.55) color = '#E89806';
        else if (score < 0.80) color = '#E89806';
        else color = '#E50914';

        const grad = ctx.createLinearGradient(0, 0, w, h);
        grad.addColorStop(0, '#E50914');
        grad.addColorStop(1, color);

        ctx.beginPath();
        ctx.arc(cx, cy, r, 0.75*Math.PI, endAngle, false);
        ctx.lineWidth = 15; ctx.strokeStyle = grad; ctx.lineCap = 'round'; ctx.stroke();

        ctx.save();
        ctx.shadowBlur = 20; ctx.shadowColor = color;
        ctx.beginPath();
        ctx.arc(cx, cy, r, Math.max(0.75*Math.PI, endAngle - 0.12), endAngle, false);
        ctx.lineWidth = 15; ctx.strokeStyle = color; ctx.lineCap = 'round'; ctx.stroke();
        ctx.restore();
    }
    drawGauge(gaugeCanvas, 0);


    // ═══ LOAD STATS ═══
    async function loadStats() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();

            animateCounter(statTotal, data.total_predictions || 0);
            animateCounter(statFraud, data.fraud_detected || 0);
            animateCounter(statSafe, data.safe_transactions || 0);

            dsSource.textContent = data.dataset_source || data.dataset || 'N/A';
            dsTotalRows.textContent = (data.total_dataset_rows || 0).toLocaleString();
            dsFraudCases.textContent = (data.total_dataset_fraud || 0).toLocaleString();
            dsBalanced.textContent = (data.total_balanced || 0).toLocaleString();

            if (data.model_loaded) {
                statAccuracy.textContent = (data.accuracy * 100).toFixed(1) + '%';
                statModelType.textContent = 'RandomForest • Online';
                sidebarStatus.textContent = 'ML Model active';
                sidebarStatus.previousElementSibling.className = 'dot green';

                setMetric(metricPrecision, barPrecision, data.precision);
                setMetric(metricRecall, barRecall, data.recall);
                setMetric(metricF1, barF1, data.f1_score);
                setMetric(metricAUC, barAUC, data.roc_auc);

                if (data.confusion_matrix && Object.keys(data.confusion_matrix).length) {
                    const cm = data.confusion_matrix;
                    cmDiv.innerHTML = `
                        <div class="cm-cell tn"><div class="cm-label">True Negative</div><div class="cm-value">${(cm.true_negative||0).toLocaleString()}</div></div>
                        <div class="cm-cell fp"><div class="cm-label">False Positive</div><div class="cm-value">${(cm.false_positive||0).toLocaleString()}</div></div>
                        <div class="cm-cell fn"><div class="cm-label">False Negative</div><div class="cm-value">${(cm.false_negative||0).toLocaleString()}</div></div>
                        <div class="cm-cell tp"><div class="cm-label">True Positive</div><div class="cm-value">${(cm.true_positive||0).toLocaleString()}</div></div>
                    `;
                }
                if (data.group_importances) renderGroupImportances(data.group_importances);
                if (data.feature_importances) renderFeatureImportances(data.feature_importances, data.feature_groups || {});
            } else {
                statAccuracy.textContent = '—';
                statModelType.textContent = 'No model loaded';
                sidebarStatus.textContent = 'Rule-based mode';
                sidebarStatus.previousElementSibling.className = 'dot yellow';
            }
        } catch (err) { console.error('Stats error:', err); }
    }

    function setMetric(valueEl, barEl, val) {
        if (!valueEl || !barEl) return;
        valueEl.textContent = (val * 100).toFixed(1) + '%';
        setTimeout(() => { barEl.style.width = (val * 100) + '%'; }, 200);
    }

    const groupIcons = { Transaction:'💳', Sender:'🏦', Receiver:'👤', Derived:'📊' };

    function renderGroupImportances(groups) {
        const sorted = Object.entries(groups).sort((a,b) => b[1]-a[1]);
        const max = sorted[0][1];
        groupDiv.innerHTML = sorted.map(([name, val]) => {
            const cls = name.toLowerCase();
            const icon = groupIcons[name] || '🔹';
            const pct = (val/max*100).toFixed(1);
            return `<div class="group-bar-item">
                <div class="group-bar-header"><span class="group-bar-name">${icon} ${name}</span><span class="group-bar-value">${(val*100).toFixed(1)}%</span></div>
                <div class="group-bar-track"><div class="group-bar-fill ${cls}" data-width="${pct}"></div></div>
            </div>`;
        }).join('');
        setTimeout(() => {
            groupDiv.querySelectorAll('.group-bar-fill').forEach(bar => { bar.style.width = bar.dataset.width + '%'; });
        }, 300);
    }

    function renderFeatureImportances(importances, groups) {
        const sorted = Object.entries(importances).sort((a,b) => b[1]-a[1]).slice(0, 14);
        const max = sorted[0][1];
        featureDiv.innerHTML = sorted.map(([feat, val]) => {
            const pct = (val/max*100).toFixed(1);
            return `<div class="feature-bar-item">
                <div class="feature-bar-name">${feat}</div>
                <div class="feature-bar-track"><div class="feature-bar-fill" data-width="${pct}"></div></div>
                <div class="feature-bar-value">${(val*100).toFixed(1)}%</div>
            </div>`;
        }).join('');
        setTimeout(() => {
            featureDiv.querySelectorAll('.feature-bar-fill').forEach(bar => { bar.style.width = bar.dataset.width + '%'; });
        }, 400);
    }

    loadStats();


    // ═══ QUICK LOAD ═══
    async function loadRandomTransaction(fraudFilter) {
        try {
            let url = '/api/random_transaction';
            if (fraudFilter !== undefined) url += '?fraud=' + fraudFilter;
            const res = await fetch(url);
            const data = await res.json();
            if (data.success) {
                const t = data.transaction;
                document.getElementById('inputType').value = t.type || 'TRANSFER';
                document.getElementById('inputAmount').value = t.amount || 0;
                document.getElementById('inputStep').value = t.step || 1;
                document.getElementById('inputOldBalOrg').value = t.oldbalanceOrg || 0;
                document.getElementById('inputNewBalOrg').value = t.newbalanceOrig || 0;
                document.getElementById('inputOldBalDest').value = t.oldbalanceDest || 0;
                document.getElementById('inputNewBalDest').value = t.newbalanceDest || 0;

                if (t.actual_label !== undefined) {
                    actualBanner.classList.remove('hidden', 'fraud', 'legit');
                    if (t.actual_label === 1) {
                        actualBanner.textContent = '🚨 Actual: FRAUD — from dataset';
                        actualBanner.classList.add('fraud');
                    } else {
                        actualBanner.textContent = '✅ Actual: LEGIT — from dataset';
                        actualBanner.classList.add('legit');
                    }
                }
                showToast('📥 Loaded real transaction', 'success');
            }
        } catch (err) { showToast('❌ Failed: ' + err.message, 'error'); }
    }

    loadRandomBtn?.addEventListener('click', () => loadRandomTransaction('random'));
    loadFraudBtn?.addEventListener('click', () => loadRandomTransaction('1'));
    loadSafeBtn?.addEventListener('click', () => loadRandomTransaction('0'));


    // ═══ ANALYZER ═══
    analyzerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        analyzeBtn.textContent = '⏳ Analyzing...';
        analyzeBtn.classList.add('loading');

        const payload = {
            type: document.getElementById('inputType').value,
            amount: parseFloat(document.getElementById('inputAmount').value) || 0,
            step: parseInt(document.getElementById('inputStep').value) || 1,
            oldbalanceOrg: parseFloat(document.getElementById('inputOldBalOrg').value) || 0,
            newbalanceOrig: parseFloat(document.getElementById('inputNewBalOrg').value) || 0,
            oldbalanceDest: parseFloat(document.getElementById('inputOldBalDest').value) || 0,
            newbalanceDest: parseFloat(document.getElementById('inputNewBalDest').value) || 0,
        };

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();

            if (data.success) {
                const prob = data.fraud_probability;
                drawGauge(gaugeCanvas, prob);
                gaugeScore.textContent = (prob * 100).toFixed(1) + '%';

                const sev = data.severity.toLowerCase();
                gaugeSev.textContent = data.severity;
                gaugeSev.className = 'gauge-severity ' + sev;
                gaugeRec.textContent = data.recommendation;
                resultMethod.textContent = data.method === 'ml_model'
                    ? '🤖 RandomForest (PaySim balanced)' : '📏 Rule-based fallback';

                if (data.reasons && data.reasons.length > 0) {
                    reasonsSec.classList.remove('hidden');
                    reasonsList.innerHTML = data.reasons.map((r, i) => `
                        <div class="reason-item ${r.severity}" style="animation-delay:${i*0.08}s">
                            <div class="reason-icon">${r.icon}</div>
                            <div class="reason-content">
                                <div class="reason-type">${r.type}</div>
                                <div class="reason-text">${r.text}</div>
                            </div>
                        </div>
                    `).join('');
                } else { reasonsSec.classList.add('hidden'); }

                showToast(data.is_fraud ? `🚨 FRAUD — ${(prob*100).toFixed(1)}%` : `✅ Safe — ${(prob*100).toFixed(1)}%`, data.is_fraud ? 'error' : 'success');
                loadStats();
            } else { showToast('❌ ' + (data.error || 'Failed'), 'error'); }
        } catch (err) { showToast('❌ ' + err.message, 'error'); }

        analyzeBtn.textContent = '🔍 Analyze Transaction';
        analyzeBtn.classList.remove('loading');
    });


    // ═══ EMAIL PHISHING ═══
    tabText?.addEventListener('click', () => {
        tabText.classList.add('active'); tabImage.classList.remove('active');
        tabContentText.classList.remove('hidden'); tabContentImage.classList.add('hidden');
    });
    tabImage?.addEventListener('click', () => {
        tabImage.classList.add('active'); tabText.classList.remove('active');
        tabContentImage.classList.remove('hidden'); tabContentText.classList.add('hidden');
    });

    uploadZone?.addEventListener('click', () => emailImageInput.click());
    uploadZone?.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.style.borderColor = '#E50914'; });
    uploadZone?.addEventListener('dragleave', () => { uploadZone.style.borderColor = ''; });
    uploadZone?.addEventListener('drop', (e) => {
        e.preventDefault(); uploadZone.style.borderColor = '';
        if (e.dataTransfer.files.length) handleImageFile(e.dataTransfer.files[0]);
    });

    emailImageInput?.addEventListener('change', () => {
        if (emailImageInput.files.length) handleImageFile(emailImageInput.files[0]);
    });

    function handleImageFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImageBase64 = e.target.result;
            previewImg.src = uploadedImageBase64;
            uploadZone.classList.add('hidden');
            uploadPreview.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    clearImageBtn?.addEventListener('click', () => {
        uploadedImageBase64 = '';
        emailImageInput.value = '';
        uploadZone.classList.remove('hidden');
        uploadPreview.classList.add('hidden');
    });

    analyzeEmailBtn?.addEventListener('click', async () => {
        const text = emailTextInput.value.trim();
        const image = uploadedImageBase64;

        if (!text && !image) { showToast('📧 Paste email text or upload image', 'error'); return; }

        analyzeEmailBtn.textContent = '⏳ Analyzing with AI...';
        analyzeEmailBtn.classList.add('loading');

        try {
            const payload = {};
            if (image) payload.image = image;
            else payload.text = text;

            const res = await fetch('/api/analyze_email', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();

            if (data.success) {
                const score = data.score || 0;
                phishingScoreValue.textContent = score;

                // Color the ring based on score
                let ringColor;
                if (score <= 30) ringColor = '#46D369';
                else if (score <= 60) ringColor = '#E89806';
                else ringColor = '#E50914';
                phishingScoreRing.style.borderColor = ringColor;
                phishingScoreRing.style.boxShadow = `0 0 30px ${ringColor}33, inset 0 0 30px ${ringColor}11`;

                const verdict = (data.verdict || 'SUSPICIOUS').toUpperCase();
                phishingVerdict.textContent = verdict;
                phishingVerdict.className = 'phishing-verdict ' + verdict.toLowerCase();

                phishingSummary.textContent = data.summary || '';

                // Render factor-by-factor scores
                if (data.factor_scores && Object.keys(data.factor_scores).length > 0) {
                    factorScores.classList.remove('hidden');
                    const factorIcons = {
                        urgency:'⏰', sender_legitimacy:'📤', link_safety:'🔗',
                        grammar_quality:'📝', impersonation:'🎭', data_request:'💳',
                        attachment_risk:'📎', emotional_manipulation:'😰'
                    };
                    const factorLabels = {
                        urgency:'Urgency', sender_legitimacy:'Sender Legitimacy',
                        link_safety:'Link Safety', grammar_quality:'Grammar Quality',
                        impersonation:'Impersonation', data_request:'Data Request',
                        attachment_risk:'Attachment Risk', emotional_manipulation:'Emotional Manipulation'
                    };
                    factorScoresGrid.innerHTML = Object.entries(data.factor_scores).map(([key, val]) => {
                        const s = val.score || 0;
                        const cls = s <= 30 ? 'safe' : s <= 60 ? 'warning' : 'danger';
                        const icon = factorIcons[key] || '📊';
                        const label = factorLabels[key] || key.replace(/_/g, ' ');
                        return `<div class="factor-score-card">
                            <div class="factor-score-header">
                                <span class="factor-score-name">${icon} ${label}</span>
                                <span class="factor-score-value ${cls}">${s}/100</span>
                            </div>
                            <div class="factor-score-bar"><div class="factor-score-bar-fill ${cls}" style="width:${s}%"></div></div>
                            <div class="factor-score-reason">${val.reason || ''}</div>
                        </div>`;
                    }).join('');
                } else { factorScores.classList.add('hidden'); }

                if (data.indicators && data.indicators.length > 0) {
                    phishingIndicators.classList.remove('hidden');
                    const catIcons = { urgency:'⏰', sender:'📤', link:'🔗', grammar:'📝', impersonation:'🎭', attachment:'📎', request:'💳', other:'🔍' };
                    phishingIndicatorsList.innerHTML = data.indicators.map((ind, i) => `
                        <div class="reason-item ${ind.severity}" style="animation-delay:${i*0.08}s">
                            <div class="reason-icon">${catIcons[ind.category] || '🔍'}</div>
                            <div class="reason-content">
                                <div class="reason-type">${ind.category || 'indicator'} ${ind.score !== undefined ? '<span style="float:right;font-weight:900;color:var(--primary-light)">' + ind.score + '/100</span>' : ''}</div>
                                <div class="reason-text">${ind.text}</div>
                            </div>
                        </div>
                    `).join('');
                } else { phishingIndicators.classList.add('hidden'); }

                showToast(score > 60 ? `🚨 PHISHING — Score ${score}/100` : `✅ Score ${score}/100`, score > 60 ? 'error' : 'success');
            } else {
                showToast('❌ ' + (data.error || 'Analysis failed'), 'error');
            }
        } catch (err) { showToast('❌ ' + err.message, 'error'); }

        analyzeEmailBtn.textContent = '🔍 Analyze for Phishing';
        analyzeEmailBtn.classList.remove('loading');
    });


    // ═══ HISTORY ═══
    async function loadHistory() {
        try {
            const res = await fetch('/api/recent');
            const logs = await res.json();
            if (!logs.length) {
                historyBody.innerHTML = `<tr><td colspan="8"><div class="empty-state"><div class="empty-state-icon">📋</div><div class="empty-state-text">No predictions yet</div></div></td></tr>`;
                return;
            }
            historyBody.innerHTML = logs.map(log => {
                const sev = (log.severity || 'low').toLowerCase();
                return `<tr>
                    <td>${log.id}</td>
                    <td>${log.type || '—'}</td>
                    <td>$${(log.amount||0).toLocaleString()}</td>
                    <td>${((log.fraud_probability||0)*100).toFixed(1)}%</td>
                    <td><span class="badge ${sev}">${log.severity}</span></td>
                    <td><span class="badge ${log.is_fraud?'fraud':'safe'}">${log.is_fraud?'Fraud':'Safe'}</span></td>
                    <td>${log.reasons_count||0} flags</td>
                    <td>${log.timestamp||''}</td>
                </tr>`;
            }).join('');
        } catch (err) { console.error('History error:', err); }
    }
    refreshBtn?.addEventListener('click', loadHistory);


    // ═══ DATASET ═══
    async function loadDataset() {
        try {
            const res = await fetch('/api/dataset');
            const data = await res.json();
            if (!data.success) {
                datasetBody.innerHTML = `<tr><td colspan="9"><div class="empty-state"><div class="empty-state-icon">❌</div><div class="empty-state-text">${data.error}</div></div></td></tr>`;
                return;
            }
            animateCounter(dsTotal, data.total_rows);
            animateCounter(dsFraudCount, data.total_fraud);
            animateCounter(dsLegitCount, data.total_legit);
            if (dsFraudPct) dsFraudPct.textContent = data.fraud_pct + '% fraud';

            datasetBody.innerHTML = data.sample.map((r, i) => {
                const cls = r.isFraud ? 'fraud' : 'safe';
                const txt = r.isFraud ? 'Fraud' : 'Legit';
                return `<tr>
                    <td>${i+1}</td>
                    <td>${r.step}</td>
                    <td>${r.type}</td>
                    <td>$${r.amount.toLocaleString()}</td>
                    <td>$${r.oldbalanceOrg.toLocaleString()}</td>
                    <td>$${r.newbalanceOrig.toLocaleString()}</td>
                    <td>$${r.oldbalanceDest.toLocaleString()}</td>
                    <td>$${r.newbalanceDest.toLocaleString()}</td>
                    <td><span class="badge ${cls}">${txt}</span></td>
                </tr>`;
            }).join('');
        } catch (err) { console.error('Dataset error:', err); }
    }
    refreshDsBtn?.addEventListener('click', loadDataset);


    // ═══ CLEAR LOGS ═══
    clearBtn?.addEventListener('click', async (e) => {
        e.preventDefault();
        if (!confirm('Clear all prediction logs?')) return;
        try {
            await fetch('/api/clear', { method: 'POST' });
            showToast('🗑️ Logs cleared', 'success');
            loadStats(); loadHistory();
        } catch (err) { showToast('❌ Failed', 'error'); }
    });

});
