import numpy as np
from typing import List, Dict, Any
from app.models.schemas import MetricsSummary

class ObservabilityCollector:
    """
    Metrics and Observability Collector tracking P50/P95/P99 latency,
    cache hit rates, token savings, cost savings, and model distribution.
    """

    def __init__(self):
        self.total_requests: int = 0
        self.cache_hits_exact: int = 0
        self.cache_hits_semantic: int = 0
        
        self.latencies_baseline_ms: List[float] = []
        self.latencies_optimized_ms: List[float] = []
        
        self.total_prompt_tokens_saved: int = 0
        self.total_cost_saved_usd: float = 0.0
        
        self.model_tier_distribution: Dict[str, int] = {
            "Tier 1 (Small)": 0,
            "Tier 2 (Medium)": 0,
            "Tier 3 (Large)": 0,
            "Exact Cache Hit": 0,
            "Semantic Cache Hit": 0
        }

    def record_request(
        self, 
        baseline_latency_ms: float, 
        optimized_latency_ms: float, 
        tokens_saved: int, 
        cost_saved_usd: float, 
        cache_hit: bool, 
        cache_type: str, 
        tier_name: str
    ):
        self.total_requests += 1
        
        if cache_hit:
            if cache_type == "exact":
                self.cache_hits_exact += 1
                self.model_tier_distribution["Exact Cache Hit"] += 1
            else:
                self.cache_hits_semantic += 1
                self.model_tier_distribution["Semantic Cache Hit"] += 1
        else:
            if "Tier 1" in tier_name:
                self.model_tier_distribution["Tier 1 (Small)"] += 1
            elif "Tier 3" in tier_name:
                self.model_tier_distribution["Tier 3 (Large)"] += 1
            else:
                self.model_tier_distribution["Tier 2 (Medium)"] += 1

        self.latencies_baseline_ms.append(baseline_latency_ms)
        self.latencies_optimized_ms.append(optimized_latency_ms)
        self.total_prompt_tokens_saved += max(0, tokens_saved)
        self.total_cost_saved_usd += max(0.0, cost_saved_usd)

    def get_summary(self) -> MetricsSummary:
        total = self.total_requests
        if total == 0:
            return MetricsSummary(
                total_requests=0,
                cache_hits_exact=0,
                cache_hits_semantic=0,
                cache_hit_rate_percent=0.0,
                avg_latency_baseline_ms=0.0,
                avg_latency_optimized_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                total_prompt_tokens_saved=0,
                total_cost_saved_usd=0.0,
                model_tier_distribution=self.model_tier_distribution
            )

        cache_hits_total = self.cache_hits_exact + self.cache_hits_semantic
        hit_rate_pct = round((cache_hits_total / total) * 100.0, 2)

        arr_opt = np.array(self.latencies_optimized_ms)
        arr_base = np.array(self.latencies_baseline_ms)

        avg_base = round(float(np.mean(arr_base)), 2)
        avg_opt = round(float(np.mean(arr_opt)), 2)

        p50 = round(float(np.percentile(arr_opt, 50)), 2)
        p95 = round(float(np.percentile(arr_opt, 95)), 2)
        p99 = round(float(np.percentile(arr_opt, 99)), 2)

        return MetricsSummary(
            total_requests=total,
            cache_hits_exact=self.cache_hits_exact,
            cache_hits_semantic=self.cache_hits_semantic,
            cache_hit_rate_percent=hit_rate_pct,
            avg_latency_baseline_ms=avg_base,
            avg_latency_optimized_ms=avg_opt,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            total_prompt_tokens_saved=self.total_prompt_tokens_saved,
            total_cost_saved_usd=round(self.total_cost_saved_usd, 6),
            model_tier_distribution=self.model_tier_distribution
        )

# Global Metrics Collector Instance
metrics_collector = ObservabilityCollector()
