// Veritas AI — Client Application Logic
document.addEventListener("DOMContentLoaded", () => {
    // 1. Tab Navigation
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));
            btn.classList.add("active");
            const targetPane = document.getElementById(target);
            if (targetPane) targetPane.classList.add("active");

            if (target === "tab-metrics") {
                loadMetrics();
            }
        });
    });

    // 2. Character & Word Counter
    const articleInput = document.getElementById("article-input");
    const charCounter = document.getElementById("char-counter");
    const btnClear = document.getElementById("btn-clear");

    function updateCounter() {
        const text = articleInput.value.trim();
        const words = text ? text.split(/\s+/).length : 0;
        const chars = articleInput.value.length;
        charCounter.textContent = `${words} words | ${chars} characters`;
    }

    articleInput.addEventListener("input", updateCounter);

    btnClear.addEventListener("click", () => {
        articleInput.value = "";
        updateCounter();
        resetResults();
    });

    // 3. Quick Test Samples
    let cachedSamples = [];
    async function fetchSamples() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();
            if (data.status === "success") {
                cachedSamples = data.samples;
            }
        } catch (e) {
            console.error("Failed to load sample articles:", e);
        }
    }
    fetchSamples();

    const sampleBtns = document.querySelectorAll(".btn-sample");
    sampleBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const idx = parseInt(btn.dataset.sample, 10);
            if (cachedSamples[idx]) {
                articleInput.value = cachedSamples[idx].title + "\n\n" + cachedSamples[idx].text;
                updateCounter();
                triggerAnalysis();
            }
        });
    });

    // 4. Live News Analysis
    const btnAnalyze = document.getElementById("btn-analyze");
    const resultsCard = document.getElementById("results-card");
    const emptyState = document.getElementById("empty-state");
    const resultDetails = document.getElementById("result-details");

    btnAnalyze.addEventListener("click", triggerAnalysis);

    async function triggerAnalysis() {
        const text = articleInput.value.trim();
        if (!text) {
            alert("Please paste or type a news article or headline first.");
            articleInput.focus();
            return;
        }

        btnAnalyze.disabled = true;
        const btnText = btnAnalyze.querySelector(".btn-text");
        const spinner = btnAnalyze.querySelector(".loader-spinner");
        btnText.textContent = "Analyzing NLP Patterns...";
        if (spinner) spinner.style.display = "inline-block";

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            });

            const resData = await response.json();
            if (resData.status === "success") {
                displayResult(resData.data);
            } else {
                alert("Error during prediction: " + (resData.message || resData.error));
            }
        } catch (err) {
            alert("Network or server error: " + err.message);
        } finally {
            btnAnalyze.disabled = false;
            btnText.textContent = "🔍 Analyze News";
            if (spinner) spinner.style.display = "none";
        }
    }

    function resetResults() {
        emptyState.style.display = "block";
        resultDetails.style.display = "none";
    }

    function displayResult(data) {
        emptyState.style.display = "none";
        resultDetails.style.display = "block";

        const isReal = data.prediction === "REAL";
        const banner = document.getElementById("verdict-banner");
        const verdictLabel = document.getElementById("verdict-label");
        const verdictSub = document.getElementById("verdict-sub");
        const verdictIcon = document.getElementById("verdict-icon");
        const riskBadge = document.getElementById("risk-badge");

        // Banner setup
        banner.className = `verdict-banner ${isReal ? "verdict-real" : "verdict-fake"}`;
        verdictIcon.textContent = isReal ? "🛡️" : "⚠️";
        verdictLabel.textContent = isReal ? "CREDIBLE / RELIABLE NEWS" : "DECEPTIVE / FAKE NEWS";
        verdictSub.textContent = isReal
            ? "High linguistic and contextual alignment with verified reporting."
            : "High concentration of sensationalism, deceptive markers, or unverified claims.";

        // Risk badge
        riskBadge.textContent = data.risk_level;
        riskBadge.className = `risk-badge ${
            data.risk_level.includes("LOW") ? "risk-low" :
            data.risk_level.includes("MODERATE") ? "risk-mod" : "risk-high"
        }`;

        // Confidence meter
        const confVal = document.getElementById("confidence-val");
        const meterFill = document.getElementById("meter-bar-fill");
        confVal.textContent = `${data.confidence_percent}%`;
        meterFill.className = `meter-bar-fill ${isReal ? "fill-real" : "fill-fake"}`;
        setTimeout(() => {
            meterFill.style.width = `${data.confidence_percent}%`;
        }, 50);

        // Probability tags
        document.getElementById("prob-real-tag").textContent = `Credible: ${(data.probability_real * 100).toFixed(1)}%`;
        document.getElementById("prob-fake-tag").textContent = `Deceptive: ${(data.probability_fake * 100).toFixed(1)}%`;

        // Stylometrics
        const sty = data.stylometrics;
        document.getElementById("sty-sensational").textContent = `${Math.round(sty.sensational_score * 100)}%`;
        document.getElementById("sty-credibility").textContent = `${Math.round(sty.credibility_score * 100)}%`;
        document.getElementById("sty-caps").textContent = `${(sty.uppercase_ratio * 100).toFixed(1)}%`;
        document.getElementById("sty-punctuation").textContent = `${sty.exclamation_count + sty.question_count}`;

        // Explainability pills
        const deceptiveContainer = document.getElementById("deceptive-pills");
        const credibleContainer = document.getElementById("credible-pills");
        deceptiveContainer.innerHTML = "";
        credibleContainer.innerHTML = "";

        const deceptiveCues = data.explanation.top_deceptive_cues || [];
        const credibleCues = data.explanation.top_credible_cues || [];
        const sensWords = data.explanation.sensational_keywords_found || [];
        const credWords = data.explanation.credibility_keywords_found || [];

        if (deceptiveCues.length === 0 && sensWords.length === 0) {
            deceptiveContainer.innerHTML = `<span class="pill-none">No deceptive triggers detected</span>`;
        } else {
            sensWords.forEach(word => {
                deceptiveContainer.innerHTML += `<span class="pill-cue pill-deceptive">⚡ ${escapeHtml(word)}</span>`;
            });
            deceptiveCues.forEach(cue => {
                if (!sensWords.includes(cue.word)) {
                    deceptiveContainer.innerHTML += `<span class="pill-cue pill-deceptive">"${escapeHtml(cue.word)}" (+${cue.weight.toFixed(1)})</span>`;
                }
            });
        }

        if (credibleCues.length === 0 && credWords.length === 0) {
            credibleContainer.innerHTML = `<span class="pill-none">No distinctive credibility markers</span>`;
        } else {
            credWords.forEach(word => {
                credibleContainer.innerHTML += `<span class="pill-cue pill-credible">🏛️ ${escapeHtml(word)}</span>`;
            });
            credibleCues.forEach(cue => {
                if (!credWords.includes(cue.word)) {
                    credibleContainer.innerHTML += `<span class="pill-cue pill-credible">"${escapeHtml(cue.word)}" (+${cue.weight.toFixed(1)})</span>`;
                }
            });
        }
    }

    // 5. Batch CSV Handling
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const btnBrowse = document.getElementById("btn-browse");
    const batchStatus = document.getElementById("batch-status");
    const batchResultsContainer = document.getElementById("batch-results-container");
    const batchTbody = document.getElementById("batch-tbody");
    const batchCountTitle = document.getElementById("batch-count-title");
    const btnExportCsv = document.getElementById("btn-export-csv");

    let currentBatchData = [];

    btnBrowse.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("click", (e) => {
        if (e.target !== btnBrowse && !e.target.closest("a")) {
            fileInput.click();
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleCsvUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleCsvUpload(fileInput.files[0]);
        }
    });

    async function handleCsvUpload(file) {
        if (!file.name.endsWith(".csv")) {
            alert("Please upload a .csv file.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        batchStatus.style.display = "flex";
        batchResultsContainer.style.display = "none";

        try {
            const res = await fetch("/api/batch-predict", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (data.status === "success") {
                currentBatchData = data.results;
                renderBatchResults(currentBatchData);
                batchCountTitle.textContent = `Batch Results (${data.total_processed} items processed)`;
                batchResultsContainer.style.display = "block";
            } else {
                alert("Error processing CSV: " + (data.message || data.error));
            }
        } catch (err) {
            alert("Upload error: " + err.message);
        } finally {
            batchStatus.style.display = "none";
        }
    }

    function renderBatchResults(items) {
        batchTbody.innerHTML = "";
        items.forEach((item, idx) => {
            const isReal = item.prediction === "REAL";
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${item.id || (idx + 1)}</td>
                <td title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</td>
                <td>
                    <span class="badge ${isReal ? "badge-success" : "badge-outline"}" style="${!isReal ? 'color: var(--color-rose); border-color: rgba(244,63,94,0.3); background: rgba(244,63,94,0.1);' : ''}">
                        ${isReal ? "✅ REAL" : "🚨 FAKE"}
                    </span>
                </td>
                <td><strong>${item.confidence}%</strong></td>
                <td><span class="risk-badge ${item.risk_level.includes('LOW') ? 'risk-low' : item.risk_level.includes('MODERATE') ? 'risk-mod' : 'risk-high'}">${item.risk_level}</span></td>
                <td>${Math.round((item.sensational_score || 0) * 100)}%</td>
            `;
            batchTbody.appendChild(row);
        });
    }

    // Filter batch table
    const filterBtns = document.querySelectorAll(".filter-btn");
    filterBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            filterBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            const filter = btn.dataset.filter;
            if (filter === "all") {
                renderBatchResults(currentBatchData);
            } else {
                renderBatchResults(currentBatchData.filter(i => i.prediction === filter));
            }
        });
    });

    // Export CSV
    btnExportCsv.addEventListener("click", () => {
        if (!currentBatchData.length) return;
        const headers = ["ID", "Excerpt", "Prediction", "Confidence", "Risk Level", "Sensationalism Score"];
        const rows = currentBatchData.map(item => [
            item.id,
            `"${item.text.replace(/"/g, '""')}"`,
            item.prediction,
            item.confidence + "%",
            item.risk_level,
            item.sensational_score
        ]);
        const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "fake_news_predictions_export.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // 6. Metrics & Performance Tab
    let metricsLoaded = false;
    async function loadMetrics() {
        if (metricsLoaded) return;
        try {
            const res = await fetch("/api/metrics");
            const data = await res.json();
            if (data && data.models_benchmark) {
                document.getElementById("metric-best-model").textContent = data.best_model || "Passive-Aggressive";
                const stats = data.dataset_stats || {};
                document.getElementById("metric-samples").textContent = stats.total_samples || "700";

                const bestBenchmark = data.models_benchmark[data.best_model] || Object.values(data.models_benchmark)[0];
                if (bestBenchmark) {
                    document.getElementById("metric-accuracy").textContent = `${(bestBenchmark.test_accuracy * 100).toFixed(1)}%`;
                    document.getElementById("metric-f1").textContent = bestBenchmark.test_f1.toFixed(4);

                    const cm = bestBenchmark.confusion_matrix;
                    if (cm && cm.length === 2) {
                        document.getElementById("cm-tn").textContent = cm[0][0];
                        document.getElementById("cm-fp").textContent = cm[0][1];
                        document.getElementById("cm-fn").textContent = cm[1][0];
                        document.getElementById("cm-tp").textContent = cm[1][1];
                    }
                }

                // Populate comparison table
                const tbody = document.getElementById("benchmark-tbody");
                tbody.innerHTML = "";
                for (const [name, m] of Object.entries(data.models_benchmark)) {
                    const row = document.createElement("tr");
                    const isBest = name === data.best_model;
                    row.innerHTML = `
                        <td><strong>${escapeHtml(name)}</strong> ${isBest ? '<span class="badge badge-success" style="font-size:10px; padding:2px 8px; margin-left:6px;">BEST</span>' : ''}</td>
                        <td>${m.cv_f1_mean.toFixed(4)} &plusmn; ${m.cv_f1_std.toFixed(4)}</td>
                        <td><strong>${(m.test_accuracy * 100).toFixed(1)}%</strong></td>
                        <td>${(m.test_precision * 100).toFixed(1)}%</td>
                        <td>${(m.test_recall * 100).toFixed(1)}%</td>
                        <td><strong>${m.test_f1.toFixed(4)}</strong></td>
                        <td>${m.test_roc_auc.toFixed(4)}</td>
                    `;
                    tbody.appendChild(row);
                }
                metricsLoaded = true;
            }
        } catch (e) {
            console.error("Failed to load metrics:", e);
        }
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }
});
