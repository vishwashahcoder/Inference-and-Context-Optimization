document.addEventListener("DOMContentLoaded", () => {
    // --- Navigation Tabs ---
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            tabButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const target = btn.getAttribute("data-tab");
            const targetElem = document.getElementById(target);
            if (targetElem) {
                targetElem.classList.add("active");
            }
        });
    });

    // --- Document Upload & RAG Elements ---
    const dropZone = document.getElementById("dropZone");
    const browseBtn = document.getElementById("browseBtn");
    const fileInput = document.getElementById("fileInput");
    const docList = document.getElementById("docList");
    const docCount = document.getElementById("docCount");
    const totalTokensIndexed = document.getElementById("totalTokensIndexed");

    const docRagPrompt = document.getElementById("docRagPrompt");
    const docRagBudget = document.getElementById("docRagBudget");
    const chkDocForceNoCache = document.getElementById("chkDocForceNoCache");
    const btnRunDocRag = document.getElementById("btnRunDocRag");

    const docRagResults = document.getElementById("docRagResults");
    const docLoadingSpinner = document.getElementById("docLoadingSpinner");
    const docDeltaBox = document.getElementById("docDeltaBox");

    const docDeltaSpeedup = document.getElementById("docDeltaSpeedup");
    const docDeltaTokens = document.getElementById("docDeltaTokens");
    const docDeltaCost = document.getElementById("docDeltaCost");

    const docBaseModel = document.getElementById("docBaseModel");
    const docBaseLatency = document.getElementById("docBaseLatency");
    const docBaseTokens = document.getElementById("docBaseTokens");
    const docBaseCost = document.getElementById("docBaseCost");
    const docBaseResponse = document.getElementById("docBaseResponse");

    const docOptModel = document.getElementById("docOptModel");
    const docOptLatency = document.getElementById("docOptLatency");
    const docOptTokens = document.getElementById("docOptTokens");
    const docOptCacheStatus = document.getElementById("docOptCacheStatus");
    const docOptResponse = document.getElementById("docOptResponse");

    // --- Playground Elements ---
    const promptInput = document.getElementById("promptInput");
    const contextInput = document.getElementById("contextInput");
    const budgetInput = document.getElementById("budgetInput");
    const chkForceNoCache = document.getElementById("chkForceNoCache");
    const btnRunBenchmark = document.getElementById("btnRunBenchmark");
    const btnClearCache = document.getElementById("btnClearCache");

    const btnPresetFAQ = document.getElementById("btnPresetFAQ");
    const btnPresetRAG = document.getElementById("btnPresetRAG");
    const btnPresetRouting = document.getElementById("btnPresetRouting");

    const loadingSpinner = document.getElementById("loadingSpinner");
    const deltaResults = document.getElementById("deltaResults");

    // KPI Banner
    const kpiCacheHitRate = document.getElementById("kpiCacheHitRate");
    const kpiCacheHitsCount = document.getElementById("kpiCacheHitsCount");
    const kpiLatencySpeedup = document.getElementById("kpiLatencySpeedup");
    const kpiP50P95 = document.getElementById("kpiP50P95");
    const kpiTokensSaved = document.getElementById("kpiTokensSaved");
    const kpiCostSaved = document.getElementById("kpiCostSaved");

    // Delta Badges
    const deltaSpeedup = document.getElementById("deltaSpeedup");
    const deltaTokensPct = document.getElementById("deltaTokensPct");
    const deltaCost = document.getElementById("deltaCost");

    // Side-by-Side Playground Output
    const baseModel = document.getElementById("baseModel");
    const baseLatency = document.getElementById("baseLatency");
    const baseTokens = document.getElementById("baseTokens");
    const baseCost = document.getElementById("baseCost");
    const baseResponse = document.getElementById("baseResponse");

    const optModel = document.getElementById("optModel");
    const optLatency = document.getElementById("optLatency");
    const optTokens = document.getElementById("optTokens");
    const optCacheStatus = document.getElementById("optCacheStatus");
    const optResponse = document.getElementById("optResponse");

    // Context Inspector Views
    const rawTokensCount = document.getElementById("rawTokensCount");
    const viewRawContext = document.getElementById("viewRawContext");
    const optTokensCount = document.getElementById("optTokensCount");
    const optReductionPct = document.getElementById("optReductionPct");
    const viewOptContext = document.getElementById("viewOptContext");

    // Observability Tab Elements
    const valP50 = document.getElementById("valP50");
    const valP95 = document.getElementById("valP95");
    const valP99 = document.getElementById("valP99");
    const valAvgBase = document.getElementById("valAvgBase");
    const valAvgOpt = document.getElementById("valAvgOpt");
    const tierDistributionList = document.getElementById("tierDistributionList");

    // --- Drag and Drop File Upload Handlers ---
    if (browseBtn && fileInput) {
        browseBtn.addEventListener("click", () => fileInput.click());
    }

    if (dropZone) {
        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        });

        dropZone.addEventListener("dragleave", () => {
            dropZone.classList.remove("dragover");
        });

        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                uploadFiles(e.dataTransfer.files);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                uploadFiles(e.target.files);
            }
        });
    }

    async function uploadFiles(files) {
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const formData = new FormData();
            formData.append("file", file);

            try {
                const resp = await fetch("/v1/documents/upload", {
                    method: "POST",
                    body: formData
                });
                if (!resp.ok) {
                    throw new Error(`Upload failed: ${resp.statusText}`);
                }
            } catch (err) {
                alert(`Error uploading file ${file.name}: ${err.message}`);
            }
        }
        fetchDocumentList();
    }

    async function fetchDocumentList() {
        try {
            const resp = await fetch("/v1/documents");
            if (!resp.ok) return;
            const docs = await resp.json();
            renderDocumentList(docs);
        } catch (err) {
            console.error("Error fetching documents:", err);
        }
    }

    function renderDocumentList(docs) {
        if (!docList || !docCount || !totalTokensIndexed) return;

        docCount.textContent = docs.length;
        let sumTokens = 0;
        docs.forEach(d => sumTokens += d.total_tokens);
        totalTokensIndexed.textContent = `${sumTokens.toLocaleString()} total tokens`;

        if (docs.length === 0) {
            docList.innerHTML = `<p class="txt-muted text-center py-4" style="font-size: 13px; text-align: center;">No documents uploaded yet. Upload a PDF/TXT document to get started!</p>`;
            return;
        }

        docList.innerHTML = "";
        docs.forEach(doc => {
            const item = document.createElement("div");
            item.className = "doc-item";
            item.innerHTML = `
                <div class="doc-info-meta">
                    <span class="doc-type-badge">${doc.file_type}</span>
                    <div>
                        <div class="doc-name">${doc.filename}</div>
                        <div class="doc-stats">${doc.total_chunks} chunks | ${doc.total_tokens.toLocaleString()} tokens | ${doc.char_count.toLocaleString()} chars</div>
                    </div>
                </div>
                <button class="doc-delete-btn" title="Delete document" data-id="${doc.doc_id}">🗑️</button>
            `;
            docList.appendChild(item);
        });

        // Add Delete Handlers
        document.querySelectorAll(".doc-delete-btn").forEach(btn => {
            btn.addEventListener("click", async (e) => {
                const docId = e.currentTarget.getAttribute("data-id");
                if (confirm("Are you sure you want to delete this document from the vector store?")) {
                    await fetch(`/v1/documents/${docId}`, { method: "DELETE" });
                    fetchDocumentList();
                }
            });
        });
    }

    // --- RAG Document Chat Runner ---
    if (btnRunDocRag) {
        btnRunDocRag.addEventListener("click", async () => {
            const prompt = docRagPrompt.value.trim();
            const budget = parseInt(docRagBudget.value) || 1500;
            const forceNoCache = chkDocForceNoCache.checked;

            if (!prompt) {
                alert("Please enter a question about your uploaded documents.");
                return;
            }

            if (docRagResults) docRagResults.classList.remove("hidden");
            if (docLoadingSpinner) docLoadingSpinner.classList.remove("hidden");
            if (docDeltaBox) docDeltaBox.classList.add("hidden");

            try {
                const resp = await fetch("/v1/chat/document-rag", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        prompt: prompt,
                        token_budget: budget,
                        force_no_cache: forceNoCache
                    })
                });

                if (!resp.ok) {
                    throw new Error(`HTTP error ${resp.status}`);
                }

                const data = await resp.json();
                renderDocRagResults(data);
                fetchMetrics();
            } catch (err) {
                alert("RAG query failed: " + err.message);
            } finally {
                if (docLoadingSpinner) docLoadingSpinner.classList.add("hidden");
                if (docDeltaBox) docDeltaBox.classList.remove("hidden");
            }
        });
    }

    function renderDocRagResults(data) {
        if (docDeltaSpeedup) docDeltaSpeedup.textContent = data.delta.speedup_factor || "1.0x";
        if (docDeltaTokens) docDeltaTokens.textContent = `${data.delta.token_savings_percent}% (${data.delta.tokens_saved} tokens saved)`;
        if (docDeltaCost) docDeltaCost.textContent = `$${data.delta.cost_saved_usd} (${data.delta.cost_savings_percent}% saved)`;

        if (docBaseModel) docBaseModel.textContent = data.baseline.model_used;
        if (docBaseLatency) docBaseLatency.textContent = `${data.baseline.latency_ms} ms`;
        if (docBaseTokens) docBaseTokens.textContent = data.baseline.prompt_tokens;
        if (docBaseCost) docBaseCost.textContent = `$${data.baseline.estimated_cost_usd}`;
        if (docBaseResponse) docBaseResponse.textContent = data.baseline.response_text;

        if (docOptModel) docOptModel.textContent = data.optimized.model_used;
        if (docOptLatency) docOptLatency.textContent = `${data.optimized.latency_ms} ms`;
        if (docOptTokens) docOptTokens.textContent = data.optimized.prompt_tokens;
        if (docOptCacheStatus) {
            docOptCacheStatus.textContent = data.optimized.cache_hit 
                ? `HIT (${data.optimized.cache_type.toUpperCase()})` 
                : `MISS (Hybrid RAG Search: ${data.delta.retrieved_chunks_count || 1} chunks)`;
        }
        if (docOptResponse) docOptResponse.textContent = data.optimized.response_text;
    }

    // --- Playground Handlers ---
    if (btnPresetFAQ) {
        btnPresetFAQ.addEventListener("click", () => {
            promptInput.value = "What is the refund policy for software subscription licenses?";
            contextInput.value = "FAQ Document:\nCustomers are entitled to a full 100% money-back guarantee refund within 30 days of purchase for any software subscription. Requests submitted after 30 days are handled on a case-by-case basis by customer support.";
            budgetInput.value = "1000";
        });
    }

    if (btnPresetRAG) {
        btnPresetRAG.addEventListener("click", () => {
            promptInput.value = "How does continuous batching and KV caching optimize LLM throughput and latency?";
            contextInput.value = `INFERENCE OPTIMIZATION PLATFORM TECHNICAL BRIEF:

1. CONTINUOUS BATCHING:
Standard LLM inference processes requests in static batches. Continuous batching dynamically inserts new requests into running batch iterations as soon as existing requests finish. This increases GPU memory utilization and throughput by up to 4x.

2. KV CACHING:
Transformer models recompute key-value vectors for all previous tokens at every decoding step. Key-Value (KV) Caching stores intermediate attention states in GPU RAM so only the new token's KV vector needs to be computed.

3. UNRELATED FLUFF & HISTORY:
Ancient computing history dates back to mechanical analytical engines in the 19th century operating with vacuum tubes.`;
            budgetInput.value = "400";
        });
    }

    if (btnPresetRouting) {
        btnPresetRouting.addEventListener("click", () => {
            promptInput.value = "Hi, define what an API Gateway is in simple terms.";
            contextInput.value = "";
            budgetInput.value = "1000";
        });
    }

    // --- Text Playground Runner ---
    async function runBenchmark() {
        const prompt = promptInput.value.trim();
        const context = contextInput.value.trim();
        const budget = parseInt(budgetInput.value) || 1500;
        const forceNoCache = chkForceNoCache.checked;

        if (!prompt) {
            alert("Please enter a question or prompt.");
            return;
        }

        loadingSpinner.classList.remove("hidden");
        deltaResults.classList.add("hidden");

        try {
            const resp = await fetch("/v1/benchmark", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    prompt: prompt,
                    context_documents: context ? [context] : [],
                    token_budget: budget,
                    force_no_cache: forceNoCache
                })
            });

            if (!resp.ok) {
                throw new Error(`HTTP error ${resp.status}`);
            }

            const data = await resp.json();
            renderBenchmarkResults(data, context);
            fetchMetrics();
        } catch (err) {
            alert("Failed to run benchmark: " + err.message);
        } finally {
            loadingSpinner.classList.add("hidden");
            deltaResults.classList.remove("hidden");
        }
    }

    function renderBenchmarkResults(data, rawContextText) {
        deltaSpeedup.textContent = data.delta.speedup_factor || "1.0x";
        deltaTokensPct.textContent = `${data.delta.token_savings_percent}% (${data.delta.tokens_saved} tokens)`;
        deltaCost.textContent = `$${data.delta.cost_saved_usd} (${data.delta.cost_savings_percent}% saved)`;

        baseModel.textContent = data.baseline.model_used;
        baseLatency.textContent = `${data.baseline.latency_ms} ms`;
        baseTokens.textContent = data.baseline.prompt_tokens;
        baseCost.textContent = `$${data.baseline.estimated_cost_usd}`;
        baseResponse.textContent = data.baseline.response_text;

        optModel.textContent = data.optimized.model_used;
        optLatency.textContent = `${data.optimized.latency_ms} ms`;
        optTokens.textContent = data.optimized.prompt_tokens;
        optCacheStatus.textContent = data.optimized.cache_hit 
            ? `HIT (${data.optimized.cache_type.toUpperCase()})` 
            : "MISS (Live Inference)";
        optResponse.textContent = data.optimized.response_text;

        rawTokensCount.textContent = data.baseline.prompt_tokens;
        viewRawContext.textContent = rawContextText || data.prompt;
        optTokensCount.textContent = data.optimized.prompt_tokens;
        optReductionPct.textContent = `${data.delta.token_savings_percent}%`;
        viewOptContext.textContent = data.optimized.context_length_chars > 0 
            ? `[Optimized Context Chunks within Token Budget]\nTokens: ${data.optimized.prompt_tokens}\n\n${data.optimized.response_text}`
            : data.prompt;
    }

    // --- Fetch Metrics ---
    async function fetchMetrics() {
        try {
            const resp = await fetch("/v1/metrics");
            if (!resp.ok) return;
            const m = await resp.json();

            if (kpiCacheHitRate) kpiCacheHitRate.textContent = `${m.cache_hit_rate_percent}%`;
            if (kpiCacheHitsCount) kpiCacheHitsCount.textContent = `${m.cache_hits_exact} Exact / ${m.cache_hits_semantic} Semantic`;

            const speedup = m.avg_latency_optimized_ms > 0 
                ? (m.avg_latency_baseline_ms / m.avg_latency_optimized_ms).toFixed(1) 
                : "1.0";
            if (kpiLatencySpeedup) kpiLatencySpeedup.textContent = `${speedup}x`;
            if (kpiP50P95) kpiP50P95.textContent = `P50: ${m.p50_latency_ms}ms | P95: ${m.p95_latency_ms}ms`;

            if (kpiTokensSaved) kpiTokensSaved.textContent = m.total_prompt_tokens_saved.toLocaleString();
            if (kpiCostSaved) kpiCostSaved.textContent = `$${m.total_cost_saved_usd.toFixed(4)}`;

            if (valP50) valP50.textContent = `${m.p50_latency_ms} ms`;
            if (valP95) valP95.textContent = `${m.p95_latency_ms} ms`;
            if (valP99) valP99.textContent = `${m.p99_latency_ms} ms`;
            if (valAvgBase) valAvgBase.textContent = `${m.avg_latency_baseline_ms} ms`;
            if (valAvgOpt) valAvgOpt.textContent = `${m.avg_latency_optimized_ms} ms`;

            renderTierDistribution(m.model_tier_distribution);
        } catch (err) {
            console.error("Error fetching metrics:", err);
        }
    }

    function renderTierDistribution(dist) {
        if (!dist || !tierDistributionList) return;
        tierDistributionList.innerHTML = "";
        for (const [tier, count] of Object.entries(dist)) {
            const item = document.createElement("div");
            item.className = "metric-row";
            item.style.padding = "8px 0";
            item.style.borderBottom = "1px solid var(--border-color)";
            item.innerHTML = `<span>${tier}:</span> <strong>${count} requests</strong>`;
            tierDistributionList.appendChild(item);
        }
    }

    if (btnClearCache) {
        btnClearCache.addEventListener("click", async () => {
            try {
                await fetch("/v1/cache/clear", { method: "POST" });
                alert("Cache cleared successfully.");
                fetchMetrics();
            } catch (err) {
                alert("Error clearing cache.");
            }
        });
    }

    if (btnRunBenchmark) {
        btnRunBenchmark.addEventListener("click", runBenchmark);
    }

    // Initializations
    fetchDocumentList();
    fetchMetrics();
});
