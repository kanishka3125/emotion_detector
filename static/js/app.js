document.addEventListener('DOMContentLoaded', () => {

    // ── State ────────────────────────────────────────────────
    let sessionId        = null;
    let stream           = null;
    let isAnalyzing      = false;
    let chartInstance    = null;
    let summaryChart     = null;
    let sessionStartTime = null;
    let durationInterval = null;
    let framesAnalyzed   = 0;
    let emotionCounts    = {};   // { emotion: count }
    let timelineData     = [];   // [{ time, emotion, confidence }]
    let lastBaseline     = null; // stored after calibration for delta display
    let isCalibrated     = false;

    // ── DOM ──────────────────────────────────────────────────
    const screens = {
        landing:   document.getElementById('screen-landing'),
        dashboard: document.getElementById('screen-dashboard'),
        summary:   document.getElementById('screen-summary'),
    };

    const video    = document.getElementById('webcam');
    const canvas   = document.getElementById('canvas');
    const ctx      = canvas.getContext('2d', { willReadFrequently: true });

    // Header indicators
    const indCamera     = document.getElementById('ind-camera');
    const indModel      = document.getElementById('ind-model');
    const indCalibrated = document.getElementById('ind-calibrated');
    const indAnalyzing  = document.getElementById('ind-analyzing');

    // Calibration
    const calibrationOverlay    = document.getElementById('calibration-overlay');
    const calibrationPercentText= document.getElementById('calibration-percent-text');
    const calibBarFill          = document.getElementById('calib-bar-fill');

    // Camera extras
    const recIndicator = document.getElementById('rec-indicator');

    // Adaptive result panel
    const resultPanel       = document.getElementById('result-panel');
    const resultEmotionName = document.getElementById('result-emotion-name');
    const resultRawConf     = document.getElementById('result-raw-conf');
    const resultDelta       = document.getElementById('result-delta');
    const resultDeltaWrap   = document.getElementById('result-delta-wrap');
    const explainToggle     = document.getElementById('explain-toggle');
    const explainDetail     = document.getElementById('explain-detail');

    // Session metrics
    const sessionMetrics = document.getElementById('session-metrics');
    const metricDuration = document.getElementById('metric-duration');
    const metricFrames   = document.getElementById('metric-frames');
    const metricDominant = document.getElementById('metric-dominant');
    const metricCurrent  = document.getElementById('metric-current');

    // Controls
    const btnStart       = document.getElementById('btn-start');
    const btnRecalibrate = document.getElementById('btn-recalibrate');
    const btnEnd         = document.getElementById('btn-end');
    const btnRestart     = document.getElementById('btn-restart');
    const btnDownload    = document.getElementById('btn-download');

    // Probability bars
    const barsContainer = document.getElementById('bars-container');

    // Summary
    const summaryDuration = document.getElementById('summary-duration');
    const summaryFrames   = document.getElementById('summary-frames');
    const summaryDominant = document.getElementById('summary-dominant');
    const summaryDistBars = document.getElementById('summary-dist-bars');


    // ── Helpers ───────────────────────────────────────────────

    const showScreen = (name) => {
        Object.values(screens).forEach(s => {
            s.classList.remove('active');
            s.classList.add('hidden');
        });
        screens[name].classList.remove('hidden');
        screens[name].classList.add('active');
        // switch body theme: dark on landing, light on dashboard/summary
        document.body.classList.toggle('in-session', name !== 'landing');
    };

    const setIndicator = (el, state) => {
        // state: 'active' | 'warning' | 'error' | '' (off)
        el.className = 'indicator' + (state ? ` ${state}` : '');
    };

    const formatDuration = (secs) => {
        const m = Math.floor(secs / 60);
        const s = secs % 60;
        return `${m}:${String(s).padStart(2, '0')}`;
    };

    const getDominantEmotion = () => {
        if (!Object.keys(emotionCounts).length) return '—';
        return Object.entries(emotionCounts).sort((a, b) => b[1] - a[1])[0][0];
    };

    // ── Camera ─────────────────────────────────────────────────

    const startCamera = async () => {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: false
        });
        video.srcObject = stream;
        return new Promise(resolve => {
            video.onloadedmetadata = () => {
                canvas.width  = video.videoWidth;
                canvas.height = video.videoHeight;
                resolve();
            };
        });
    };

    const stopCamera = () => {
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            video.srcObject = null;
            stream = null;
        }
    };

    // ── Session ────────────────────────────────────────────────

    const initSession = async () => {
        const response = await fetch('/api/start_session', { method: 'POST' });
        const data = await response.json();
        if (!data.session_id) throw new Error('No session ID');
        sessionId = data.session_id;

        // Reset state
        framesAnalyzed   = 0;
        emotionCounts    = {};
        timelineData     = [];
        lastBaseline     = null;
        isCalibrated     = false;
        sessionStartTime = Date.now();

        // UI
        calibrationOverlay.classList.add('active');
        recIndicator.classList.add('hidden');
        resultPanel.classList.add('hidden');
        sessionMetrics.classList.add('hidden');

        setIndicator(indCalibrated, 'warning');
        setIndicator(indAnalyzing, '');

        initChart();

        // Start duration timer
        if (durationInterval) clearInterval(durationInterval);
        durationInterval = setInterval(() => {
            if (!sessionStartTime) return;
            const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
            metricDuration.textContent = formatDuration(elapsed);
        }, 1000);

        // Enable controls
        btnRecalibrate.disabled = true;  // only after calibration
        btnEnd.disabled = false;

        isAnalyzing = true;
        processFrameLoop();
    };

    // ── Frame Loop ────────────────────────────────────────────

    const processFrameLoop = async () => {
        if (!isAnalyzing || !sessionId) return;
        await processFrame();
        setTimeout(processFrameLoop, 10);
    };

    const processFrame = async () => {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        try {
            const res = await fetch('/api/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, image: imageData })
            });
            if (!res.ok) return;
            const data = await res.json();
            updateUI(data);
        } catch (e) {
            // network error — ignore silently
        }
    };

    // ── UI Update ─────────────────────────────────────────────

    const updateUI = (data) => {
        if (data.status === 'calibrating') {
            const pct = data.progress || 0;
            calibrationPercentText.textContent = `Calibrating… ${pct}%`;
            calibBarFill.style.width = `${pct}%`;
            setIndicator(indCalibrated, 'warning');
            return;
        }

        if (data.status === 'active') {
            // First active frame
            if (!isCalibrated) {
                isCalibrated = true;
                calibrationOverlay.classList.remove('active');
                recIndicator.classList.remove('hidden');
                resultPanel.classList.remove('hidden');
                sessionMetrics.classList.remove('hidden');

                setIndicator(indCalibrated, 'active');
                setIndicator(indAnalyzing, 'active');

                btnRecalibrate.disabled = false;
            }

            framesAnalyzed++;
            const emotion    = (data.emotion || 'neutral').toLowerCase();
            const confidence = data.confidence || 0;
            const probs      = data.probabilities || {};

            // Track counts
            emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;

            // Timeline record
            const now = new Date();
            timelineData.push({
                time: now.toLocaleTimeString(),
                emotion,
                confidence
            });

            // Adaptive result panel
            updateResultPanel(emotion, confidence, probs);

            // Session metrics
            metricFrames.textContent  = framesAnalyzed;
            metricDominant.textContent = getDominantEmotion();
            metricCurrent.textContent = emotion;

            // Probability bars
            renderBars(probs, emotion);

            // Timeline chart
            updateChart(now.toLocaleTimeString(), emotion, confidence);
        }
    };

    // ── Adaptive Result Panel ─────────────────────────────────

    const EMOTION_BASELINE_FALLBACK = {
        // used to compute delta when lastBaseline is available
    };

    const updateResultPanel = (emotion, confidence, probs) => {
        // Remove old emotion classes
        resultPanel.className = 'result-panel';
        resultPanel.classList.add(`emotion-${emotion}`);

        resultEmotionName.textContent = emotion;
        resultRawConf.textContent = `${confidence.toFixed(1)}%`;

        // Delta vs baseline — baseline isn't exposed from API, so we show
        // the difference between this emotion's raw prob and the raw average
        // as a useful proxy the user can understand
        const rawForEmotion = (probs[emotion] || 0);
        const allVals = Object.values(probs);
        if (allVals.length > 0) {
            const avg = allVals.reduce((a, b) => a + b, 0) / allVals.length;
            const delta = rawForEmotion - avg;
            const sign = delta >= 0 ? '+' : '';
            resultDelta.textContent = `${sign}${delta.toFixed(1)}% vs avg`;
            resultDelta.className = 'meta-value ' + (delta >= 0 ? 'positive' : 'negative');
            resultDeltaWrap.style.display = '';
        } else {
            resultDeltaWrap.style.display = 'none';
        }
    };

    // ── Explain Toggle ────────────────────────────────────────

    explainToggle.addEventListener('click', (e) => {
        e.preventDefault();
        const hidden = explainDetail.classList.toggle('hidden');
        explainToggle.textContent = hidden ? 'Why might this differ?' : 'Hide explanation';
    });

    // ── Probability Bars ──────────────────────────────────────

    const EMOTION_ORDER = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];

    const renderBars = (probs, adaptiveWinner) => {
        barsContainer.innerHTML = '';

        // Find the raw max for annotation
        let rawMax = '';
        let rawMaxVal = -1;
        EMOTION_ORDER.forEach(e => {
            if ((probs[e] || 0) > rawMaxVal) { rawMaxVal = probs[e] || 0; rawMax = e; }
        });

        EMOTION_ORDER.forEach(emotion => {
            const val = probs[emotion] || 0;

            const row = document.createElement('div');
            row.className = 'emotion-bar-row';

            const header = document.createElement('div');
            header.className = 'emotion-bar-header';

            const nameSpan = document.createElement('span');
            nameSpan.className = 'bar-name';
            nameSpan.textContent = emotion;
            // Mark if this is the adaptive winner (may differ from raw max)
            if (emotion === adaptiveWinner) {
                nameSpan.classList.add('is-adaptive-winner');
                nameSpan.title = 'Current adaptive result';
            }

            const pctSpan = document.createElement('span');
            pctSpan.className = 'bar-pct';
            pctSpan.textContent = `${val.toFixed(1)}%`;

            header.appendChild(nameSpan);
            header.appendChild(pctSpan);

            const bg   = document.createElement('div');
            bg.className = 'emotion-bar-bg';

            const fill = document.createElement('div');
            fill.className = 'emotion-bar-fill';
            if (emotion === rawMax && emotion !== adaptiveWinner) {
                fill.classList.add('top-raw');  // grey — highest raw but not adaptive winner
            }
            fill.style.width = `${Math.min(100, Math.max(0, val))}%`;

            bg.appendChild(fill);
            row.appendChild(header);
            row.appendChild(bg);
            barsContainer.appendChild(row);
        });
    };

    // ── Timeline Chart ────────────────────────────────────────

    const EMOTION_COLORS = {
        happy:    '#10b981',
        surprise: '#f97316',
        neutral:  '#a8a29e',
        sad:      '#f59e0b',
        angry:    '#ef4444',
        fear:     '#fb923c',
        disgust:  '#e11d48',
    };

    const emotionToY = (em) => {
        const order = ['disgust', 'angry', 'fear', 'sad', 'neutral', 'surprise', 'happy'];
        const i = order.indexOf(em.toLowerCase());
        return i >= 0 ? i : 4;
    };

    const initChart = () => {
        const el = document.getElementById('timeline-chart');
        if (!el) return;
        if (chartInstance) chartInstance.destroy();

        chartInstance = new Chart(el, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Adaptive Emotion',
                    data: [],
                    pointBackgroundColor: [],
                    pointRadius: 5,
                    pointHoverRadius: 6,
                    showLine: true,
                    borderColor: 'rgba(249,115,22,0.3)',
                    borderWidth: 1.5,
                    tension: 0,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    y: {
                        min: -0.5,
                        max: 6.5,
                        ticks: {
                            stepSize: 1,
                            callback: (v) => {
                                const order = ['disgust','angry','fear','sad','neutral','surprise','happy'];
                                return order[Math.round(v)] || '';
                            },
                            font: { size: 10 }
                        },
                        grid: { color: 'rgba(0,0,0,0.04)' }
                    },
                    x: { display: false }
                },
                plugins: { legend: { display: false }, tooltip: {
                    callbacks: {
                        label: (item) => {
                            const order = ['disgust','angry','fear','sad','neutral','surprise','happy'];
                            return order[Math.round(item.parsed.y)] || '';
                        }
                    }
                }}
            }
        });
    };

    const updateChart = (timeStr, emotion, confidence) => {
        if (!chartInstance) return;
        const ds = chartInstance.data.datasets[0];
        const y  = emotionToY(emotion);
        ds.data.push({ x: timeStr, y });
        ds.pointBackgroundColor.push(EMOTION_COLORS[emotion] || '#f97316');

        // Keep last 40 points
        if (ds.data.length > 40) {
            ds.data.shift();
            ds.pointBackgroundColor.shift();
        }
        chartInstance.update();
    };

    // ── End Session ───────────────────────────────────────────

    const endSession = async () => {
        isAnalyzing = false;
        stopCamera();

        if (durationInterval) {
            clearInterval(durationInterval);
            durationInterval = null;
        }

        setIndicator(indCamera, '');
        setIndicator(indAnalyzing, '');

        try {
            const res  = await fetch(`/api/end_session/${sessionId}`);
            const data = await res.json();

            const elapsed = sessionStartTime
                ? Math.floor((Date.now() - sessionStartTime) / 1000)
                : 0;

            // Populate summary
            summaryDuration.textContent = formatDuration(elapsed);
            summaryFrames.textContent   = data.frames_analyzed || 0;
            summaryDominant.textContent = data.dominant_emotion || '—';

            // Distribution bars
            renderDistBars(data.emotion_distribution || {});

            // Replay timeline chart
            buildSummaryChart();

            // Download link
            btnDownload.href = `/api/download_report/${sessionId}`;

            showScreen('summary');
        } catch (e) {
            alert('Error ending session. Please try again.');
        }
    };

    const renderDistBars = (dist) => {
        summaryDistBars.innerHTML = '';
        const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
        const sorted = Object.entries(dist).sort((a, b) => b[1] - a[1]);
        sorted.forEach(([emotion, count]) => {
            const pct = (count / total) * 100;
            const row = document.createElement('div');
            row.className = 'emotion-bar-row';
            row.innerHTML = `
                <div class="emotion-bar-header">
                    <span class="bar-name" style="text-transform:capitalize">${emotion}</span>
                    <span class="bar-pct">${pct.toFixed(1)}%</span>
                </div>
                <div class="emotion-bar-bg">
                    <div class="emotion-bar-fill" style="width:${pct}%;background:${EMOTION_COLORS[emotion] || '#6366f1'}"></div>
                </div>`;
            summaryDistBars.appendChild(row);
        });
    };

    const buildSummaryChart = () => {
        const el = document.getElementById('summary-chart');
        if (!el) return;
        if (summaryChart) summaryChart.destroy();

        // Thin out to last 60 points for readability
        const slice = timelineData.slice(-60);
        summaryChart = new Chart(el, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Emotion Timeline',
                    data: slice.map((d, i) => ({ x: i, y: emotionToY(d.emotion) })),
                    pointBackgroundColor: slice.map(d => EMOTION_COLORS[d.emotion] || '#f97316'),
                    pointRadius: 4,
                    showLine: true,
                    borderColor: 'rgba(249,115,22,0.25)',
                    borderWidth: 1.5,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    y: {
                        min: -0.5, max: 6.5,
                        ticks: {
                            stepSize: 1,
                            callback: (v) => {
                                const order = ['disgust','angry','fear','sad','neutral','surprise','happy'];
                                return order[Math.round(v)] || '';
                            },
                            font: { size: 10 }
                        },
                        grid: { color: 'rgba(0,0,0,0.04)' }
                    },
                    x: { display: false }
                },
                plugins: { legend: { display: false } }
            }
        });
    };

    // ── Recalibrate ────────────────────────────────────────────

    const doRecalibrate = async () => {
        if (!sessionId) return;
        isAnalyzing = false;
        isCalibrated = false;

        // Reset server session
        await fetch(`/api/end_session/${sessionId}`).catch(() => {});
        const res  = await fetch('/api/start_session', { method: 'POST' });
        const data = await res.json();
        if (!data.session_id) return;
        sessionId = data.session_id;

        // Reset client state
        framesAnalyzed = 0;
        emotionCounts  = {};
        timelineData   = [];
        sessionStartTime = Date.now();

        calibrationOverlay.classList.add('active');
        recIndicator.classList.add('hidden');
        resultPanel.classList.add('hidden');
        sessionMetrics.classList.add('hidden');
        setIndicator(indCalibrated, 'warning');
        setIndicator(indAnalyzing, '');
        barsContainer.innerHTML = '<div class="placeholder-bars"><div class="placeholder-text">Waiting for analysis…</div></div>';

        initChart();

        btnRecalibrate.disabled = true;

        isAnalyzing = true;
        processFrameLoop();
    };

    // ── Event Listeners ────────────────────────────────────────

    btnStart.addEventListener('click', async () => {
        btnStart.disabled = true;
        btnStart.textContent = 'Accessing Camera…';
        try {
            await startCamera();
            setIndicator(indCamera, 'active');
            setIndicator(indModel, 'active');
            showScreen('dashboard');
            await initSession();
        } catch (e) {
            btnStart.disabled = false;
            btnStart.textContent = 'Start Analysis';
            setIndicator(indCamera, 'error');
            alert('Could not access camera. Please grant permissions and reload.');
        }
    });

    btnRecalibrate.addEventListener('click', () => {
        if (confirm('Recalibrate? This resets your baseline. Current session data will be cleared.')) {
            doRecalibrate();
        }
    });

    btnEnd.addEventListener('click', () => {
        if (confirm('End session and view results?')) {
            btnEnd.disabled = true;
            endSession();
        }
    });

    btnRestart.addEventListener('click', () => {
        window.location.reload();
    });
});
