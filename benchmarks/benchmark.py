import asyncio
import time
import httpx

BENCHMARK_PROMPTS = [
    {
        "name": "Repeated FAQ Query (Cache Hit Test)",
        "prompt": "What is the refund policy for software subscription licenses?",
        "context": ["Customers receive a 100% full refund within 30 days of purchase."],
        "budget": 1000
    },
    {
        "name": "Long Document RAG QA (Context Trimming Test)",
        "prompt": "How does continuous batching and KV caching improve LLM latency?",
        "context": [
            "Continuous batching inserts new requests into running iterations without waiting for long sequences to finish.",
            "KV caching stores attention states in GPU memory to avoid redundant recomputations.",
            "Ancient history of silicon transistors and mechanical engines dating back to 1800s." * 10
        ],
        "budget": 300
    },
    {
        "name": "Simple Greeting / Classification (Model Routing Test)",
        "prompt": "Hello, please define API Gateway in simple terms.",
        "context": [],
        "budget": 1000
    }
]

async def run_cli_benchmark():
    url = "http://localhost:8000/v1/benchmark"
    print("=" * 80)
    print("[BENCHMARK] PRODUCTION-GRADE AI INFERENCE & CONTEXT OPTIMIZATION PLATFORM BENCHMARK")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Check if server is running
        try:
            health = await client.get("http://localhost:8000/v1/metrics")
            if health.status_code != 200:
                print("[ERROR] Server is not healthy. Please start FastAPI server first.")
                return
        except Exception:
            print("[WARN] Server not detected at http://localhost:8000. Start FastAPI server with 'python -m uvicorn app.main:app --reload'")
            return

        for idx, item in enumerate(BENCHMARK_PROMPTS, 1):
            print(f"\n--- Scenario {idx}: {item['name']} ---")
            payload = {
                "prompt": item["prompt"],
                "context_documents": item["context"],
                "token_budget": item["budget"],
                "force_no_cache": False
            }
            res = await client.post(url, json=payload)
            data = res.json()

            base = data["baseline"]
            opt = data["optimized"]
            delta = data["delta"]

            print(f"  [BASELINE] Model:  {base['model_used']} | Latency: {base['latency_ms']} ms | Tokens: {base['prompt_tokens']} | Cost: ${base['estimated_cost_usd']}")
            print(f"  [OPTIMIZED] Model: {opt['model_used']} | Latency: {opt['latency_ms']} ms | Tokens: {opt['prompt_tokens']} | Cost: ${opt['estimated_cost_usd']}")
            print(f"  [DELTA] Speedup:   {delta['speedup_factor']} | Token Savings: {delta['token_savings_percent']}% | Cost Saved: ${delta['cost_saved_usd']}")

        # Fetch Summary Metrics
        metrics_res = await client.get("http://localhost:8000/v1/metrics")
        m = metrics_res.json()

        print("\n" + "=" * 80)
        print("[METRICS] AGGREGATED PLATFORM METRICS SUMMARY")
        print("=" * 80)
        print(f"  Total Benchmark Requests Processed: {m['total_requests']}")
        print(f"  Cache Hit Rate:                   {m['cache_hit_rate_percent']}% ({m['cache_hits_exact']} Exact, {m['cache_hits_semantic']} Semantic)")
        print(f"  Latency P50 / P95 / P99:          P50: {m['p50_latency_ms']} ms | P95: {m['p95_latency_ms']} ms | P99: {m['p99_latency_ms']} ms")
        print(f"  Total Prompt Tokens Saved:        {m['total_prompt_tokens_saved']}")
        print(f"  Total Estimated Cost Saved:       ${m['total_cost_saved_usd']}")
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_cli_benchmark())
