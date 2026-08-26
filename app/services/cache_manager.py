import re
import hashlib
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from app.core.config import settings

class CacheManager:
    """
    Dual-layer Caching Manager:
    - Layer 1: Exact SHA-256 Hash Matching
    - Layer 2: Vector Embedding Cosine Similarity Semantic Cache
    """

    def __init__(self, similarity_threshold: float = settings.CACHE_SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        # Exact cache: hash_key -> entry dict
        self.exact_cache: Dict[str, Dict[str, Any]] = {}
        # Semantic cache list: [(vector, prompt_text, response_data)]
        self.semantic_cache: List[Tuple[np.ndarray, str, Dict[str, Any]]] = []

    @staticmethod
    def _compute_exact_hash(prompt: str, system_prompt: str = "", context_text: str = "") -> str:
        raw = f"{system_prompt.strip()}|||{prompt.strip()}|||{context_text.strip()}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _compute_vector(text: str) -> np.ndarray:
        """Fast normalized vector embedding using normalized word & character trigram hashing."""
        vec = np.zeros(256, dtype=np.float32)
        # Normalize punctuation and possessives like France's -> France
        cleaned = re.sub(r"'s\b", "", text.lower())
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        words = [w.strip() for w in cleaned.split() if len(w.strip()) > 1]
        
        for word in words:
            # Word feature
            idx = (hash(word) & 0x7FFFFFFF) % 256
            vec[idx] += 3.0
            # Character trigram features
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    trigram = word[i:i+3]
                    t_idx = (hash(trigram) & 0x7FFFFFFF) % 256
                    vec[t_idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)

    def get(self, prompt: str, system_prompt: str = "", context_text: str = "") -> Tuple[Optional[Dict[str, Any]], str, float]:
        """
        Looks up cache. Returns (cached_entry, hit_type, similarity_score).
        hit_type: "exact", "semantic", or "miss"
        """
        # 1. Exact Match Check
        hash_key = self._compute_exact_hash(prompt, system_prompt, context_text)
        if hash_key in self.exact_cache:
            entry = self.exact_cache[hash_key]
            # Ignore old mock entries if live key is active
            if "Response to query:" not in entry.get("response_text", "") and "successfully processed by" not in entry.get("response_text", ""):
                entry["hit_count"] = entry.get("hit_count", 0) + 1
                return entry, "exact", 1.0

        # 2. Semantic Similarity Check (combined prompt + context text)
        query_str = f"{prompt} {context_text[:200]}" if context_text else prompt
        query_vec = self._compute_vector(query_str)
        best_score = 0.0
        best_entry = None

        for cached_vec, cached_prompt, entry in self.semantic_cache:
            if "Response to query:" in entry.get("response_text", "") or "successfully processed by" in entry.get("response_text", ""):
                continue
            sim = float(np.dot(query_vec, cached_vec))
            if sim > best_score:
                best_score = sim
                best_entry = entry

        if best_entry and best_score >= self.similarity_threshold:
            best_entry["hit_count"] = best_entry.get("hit_count", 0) + 1
            return best_entry, "semantic", round(best_score, 4)

        return None, "miss", 0.0

    def set(
        self, 
        prompt: str, 
        response_text: str, 
        system_prompt: str = "", 
        context_text: str = "",
        model_used: str = "Tier 1",
        tokens_used: int = 0,
        estimated_latency_ms: float = 800.0
    ):
        """Stores result in both exact hash cache and vector semantic cache."""
        hash_key = self._compute_exact_hash(prompt, system_prompt, context_text)
        entry = {
            "prompt": prompt,
            "response_text": response_text,
            "model_used": model_used,
            "tokens_used": tokens_used,
            "estimated_latency_ms": estimated_latency_ms,
            "created_at": time.time(),
            "hit_count": 0
        }
        self.exact_cache[hash_key] = entry

        # Save to semantic vector cache
        query_str = f"{prompt} {context_text[:200]}" if context_text else prompt
        vec = self._compute_vector(query_str)
        self.semantic_cache.append((vec, query_str, entry))

        # Enforce max size eviction if cache exceeds capacity
        if len(self.semantic_cache) > settings.MAX_CACHE_SIZE:
            self.semantic_cache.pop(0)

        return entry

    def clear(self):
        self.exact_cache.clear()
        self.semantic_cache.clear()

# Global singleton cache instance
global_cache = CacheManager()
