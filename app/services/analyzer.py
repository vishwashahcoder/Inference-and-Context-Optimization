import re
from typing import Dict, Any, List

class RequestAnalyzer:
    """
    Analyzes prompts and context to estimate token counts, intent,
    and task complexity (Simple, Medium, Complex).
    """

    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        # Heuristic estimation: ~4 chars per token or ~0.75 words per token
        words = len(text.split())
        chars = len(text)
        return max(1, int(max(words * 1.3, chars / 3.8)))

    @classmethod
    def analyze_complexity(cls, prompt: str, context_len: int = 0) -> Dict[str, Any]:
        prompt_lower = prompt.lower()
        token_count = cls.estimate_tokens(prompt)

        # Keyword heuristics for simple vs complex tasks
        complex_indicators = [
            "step by step", "explain in detail", "architect", "refactor",
            "proof", "algorithm", "trade-offs", "compare and contrast",
            "mathematical", "debug", "analyze deep", "multi-threading"
        ]
        
        simple_indicators = [
            "hi", "hello", "hey", "what is", "define", "capital of", 
            "translate", "synonym", "yes or no", "true or false"
        ]

        complex_score = sum(1 for kw in complex_indicators if kw in prompt_lower)
        simple_score = sum(1 for kw in simple_indicators if kw in prompt_lower)

        # Determine complexity tier
        if token_count < 25 and simple_score > 0 and context_len < 100:
            complexity = "SIMPLE"
            recommended_tier = "Tier 1 (Small)"
            reason = "Short prompt with simple query indicator."
        elif complex_score >= 2 or token_count > 300 or context_len > 3000:
            complexity = "COMPLEX"
            recommended_tier = "Tier 3 (Large)"
            reason = "High token count or complex multi-step reasoning requested."
        elif token_count < 80 and complex_score == 0:
            complexity = "SIMPLE"
            recommended_tier = "Tier 1 (Small)"
            reason = "Low token prompt without complex indicators."
        else:
            complexity = "MEDIUM"
            recommended_tier = "Tier 2 (Medium)"
            reason = "Standard task requiring balanced model."

        return {
            "token_count": token_count,
            "complexity": complexity,
            "recommended_tier": recommended_tier,
            "reason": reason,
            "complex_score": complex_score,
            "simple_score": simple_score
        }
