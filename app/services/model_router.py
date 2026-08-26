from typing import Dict, Any
from app.core.config import settings
from app.services.analyzer import RequestAnalyzer

class ModelRouter:
    """
    Intelligent Model Router that maps analyzed requests to the optimal model tier:
    - Tier 1 (Small / Ultra-fast / Cheap)
    - Tier 2 (Medium / Balanced)
    - Tier 3 (Large / Complex Reasoning)
    """

    @classmethod
    def route(cls, prompt: str, context_len: int = 0, requested_model: str = "auto") -> Dict[str, Any]:
        analysis = RequestAnalyzer.analyze_complexity(prompt, context_len)
        complexity = analysis["complexity"]

        # Document context queries require high-capacity reasoning models (Tier 2 / Tier 3)
        if context_len > 300 and complexity == "SIMPLE":
            complexity = "MEDIUM"

        if requested_model != "auto" and requested_model:
            model_name = requested_model
            tier_name = "Custom / User-Selected"
            input_rate = settings.COST_TIER2_INPUT
            output_rate = settings.COST_TIER2_OUTPUT
        elif complexity == "SIMPLE":
            model_name = settings.MODEL_TIER_1_SMALL
            tier_name = "Tier 1 (Small / Ultra-Fast)"
            input_rate = settings.COST_TIER1_INPUT
            output_rate = settings.COST_TIER1_OUTPUT
        elif complexity == "COMPLEX":
            model_name = settings.MODEL_TIER_3_LARGE
            tier_name = "Tier 3 (Large / Complex Reasoning)"
            input_rate = settings.COST_TIER3_INPUT
            output_rate = settings.COST_TIER3_OUTPUT
        else:
            model_name = settings.MODEL_TIER_2_MEDIUM
            tier_name = "Tier 2 (Medium / Balanced)"
            input_rate = settings.COST_TIER2_INPUT
            output_rate = settings.COST_TIER2_OUTPUT

        return {
            "tier_name": tier_name,
            "model_name": model_name,
            "complexity": complexity,
            "reason": analysis["reason"],
            "input_cost_per_1k": input_rate,
            "output_cost_per_1k": output_rate
        }

    @classmethod
    def calculate_cost(
        cls, 
        prompt_tokens: int, 
        completion_tokens: int, 
        tier_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculates actual cost vs direct unoptimized baseline cost."""
        input_rate = tier_info.get("input_cost_per_1k", settings.COST_TIER2_INPUT)
        output_rate = tier_info.get("output_cost_per_1k", settings.COST_TIER2_OUTPUT)

        actual_cost = (prompt_tokens / 1000.0 * input_rate) + (completion_tokens / 1000.0 * output_rate)
        
        # Baseline assumes full uncompressed prompt tokens to Direct Large Model
        baseline_prompt_tokens = max(prompt_tokens, int(prompt_tokens * 3.5))
        baseline_cost = (baseline_prompt_tokens / 1000.0 * settings.COST_BASELINE_LARGE_INPUT) + \
                        (completion_tokens / 1000.0 * settings.COST_BASELINE_LARGE_OUTPUT)

        cost_saved = max(0.0, baseline_cost - actual_cost)
        
        return {
            "actual_cost_usd": round(actual_cost, 6),
            "baseline_cost_usd": round(baseline_cost, 6),
            "cost_saved_usd": round(cost_saved, 6)
        }
