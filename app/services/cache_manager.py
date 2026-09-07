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

        # 2. Semantic Similarity Check with Secondary Intent & Critical Entity Verification
        query_str = f"{prompt} {context_text[:200]}" if context_text else prompt
        query_vec = self._compute_vector(query_str)
        best_score = 0.0
        best_entry = None
        best_cached_prompt = ""

        for cached_vec, cached_prompt, entry in self.semantic_cache:
            if "Response to query:" in entry.get("response_text", "") or "successfully processed by" in entry.get("response_text", ""):
                continue
            sim = float(np.dot(query_vec, cached_vec))
            if sim > best_score:
                best_score = sim
                best_entry = entry
                best_cached_prompt = cached_prompt

        # Dual validation: 1. Cosine similarity >= 0.88 AND 2. Secondary Intent/Entity Match
        if best_entry and best_score >= self.similarity_threshold:
            if self._verify_intent_match(prompt, best_cached_prompt):
                best_entry["hit_count"] = best_entry.get("hit_count", 0) + 1
                return best_entry, "semantic", round(best_score, 4)
            else:
                # Rejected by secondary intent verification (e.g. RTO vs RPO, Section 5 vs Section 4)
                pass

        return None, "miss", 0.0

    @staticmethod
    def _extract_critical_entities(text: str) -> Dict[str, Any]:
        """
        Extracts acronyms, section/step numbers, and domain topic nouns to guarantee
        that two semantically close queries are truly asking for the same information target.
        """
        t_low = text.lower()
        # 1. Acronyms & uppercase codes (e.g., RTO, RPO, AWS, BCDR, CTO, LAN, WAN, SOP)
        acronyms = set(re.findall(r'\b[A-Z]{2,6}\b', text))
        # 2. Section / step / annexure numbers (e.g. '5', '6.1', 'step 3')
        numbers = set(re.findall(r'\b(?:section|step|annexure|part)?\s*(\d+(?:\.\d+)?)\b', t_low))
        # 3. Specific domain topic nouns
        domain_topics = set()
        for kw in ["annexure", "procedure", "abbreviation", "responsibility", "definition", 
                   "scope", "purpose", "incident", "recovery", "objective", "policy", "form", "logbook"]:
            if kw in t_low:
                domain_topics.add(kw)
        return {
            "acronyms": acronyms,
            "numbers": numbers,
            "domain_topics": domain_topics
        }

    @classmethod
    def _verify_intent_match(cls, query: str, cached_prompt: str) -> bool:
        """
        Secondary intent validation layer.
        Rejects semantic cache hits if critical acronyms, section numbers, or domain topics mismatch!
        """
        q_ent = cls._extract_critical_entities(query)
        c_ent = cls._extract_critical_entities(cached_prompt)

        # 1. Strict Acronym check: If both queries specify acronyms, they MUST have at least one overlap!
        if q_ent["acronyms"] and c_ent["acronyms"]:
            if not (q_ent["acronyms"] & c_ent["acronyms"]):
                return False

        # 2. Strict Section/Number check: If queries specify section/step numbers, they must match!
        if q_ent["numbers"] and c_ent["numbers"]:
            if not (q_ent["numbers"] & c_ent["numbers"]):
                return False

        # 3. Core Domain Topic check: If both have recognized domain topics, they must overlap!
        if q_ent["domain_topics"] and c_ent["domain_topics"]:
            if not (q_ent["domain_topics"] & c_ent["domain_topics"]):
                return False

        return True

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
