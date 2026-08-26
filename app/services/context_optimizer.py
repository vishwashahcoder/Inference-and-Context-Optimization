import re
import numpy as np
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from app.services.analyzer import RequestAnalyzer

class ContextOptimizer:
    """
    Optimizes context using Sparse (BM25) + Dense similarity hybrid search,
    deduplication, reranking, and token budget enforcement.
    """

    @staticmethod
    def _simple_vector_embedding(text: str) -> np.ndarray:
        """Lightweight zero-dependency CPU vector embedding generator (TF-IDF / character n-gram hash)."""
        vec = np.zeros(128, dtype=np.float32)
        words = re.findall(r'\w+', text.lower())
        for word in words:
            idx = hash(word) % 128
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)

    @classmethod
    def optimize(
        cls, 
        query: str, 
        documents: List[str], 
        token_budget: int = 1500
    ) -> Dict[str, Any]:
        """
        Takes raw context documents or long text, breaks into chunks,
        reranks using hybrid BM25 + Vector scoring, deduplicates,
        and enforces the token budget.
        """
        if not documents:
            return {
                "optimized_text": "",
                "original_tokens": 0,
                "optimized_tokens": 0,
                "token_savings_percent": 0.0,
                "chunks_selected": 0,
                "total_chunks": 0
            }

        # 1. Break documents into paragraph chunks
        raw_chunks: List[str] = []
        for doc in documents:
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\n(?=[A-Z0-9])', doc) if p.strip()]
            raw_chunks.extend(paragraphs)

        if not raw_chunks:
            return {
                "optimized_text": "",
                "original_tokens": 0,
                "optimized_tokens": 0,
                "token_savings_percent": 0.0,
                "chunks_selected": 0,
                "total_chunks": 0
            }

        original_combined = "\n\n".join(raw_chunks)
        original_tokens = RequestAnalyzer.estimate_tokens(original_combined)

        # If total context is already within token budget, return intact
        if original_tokens <= token_budget:
            return {
                "optimized_text": original_combined,
                "original_tokens": original_tokens,
                "optimized_tokens": original_tokens,
                "token_savings_percent": 0.0,
                "chunks_selected": len(raw_chunks),
                "total_chunks": len(raw_chunks)
            }

        # 2. Sparse Search: BM25 Scoring
        tokenized_corpus = [re.findall(r'\w+', chunk.lower()) for chunk in raw_chunks]
        tokenized_query = re.findall(r'\w+', query.lower())
        
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_query)
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        norm_bm25 = bm25_scores / max_bm25

        # 3. Dense Search: Vector Embedding Cosine Similarity
        query_vec = cls._simple_vector_embedding(query)
        vector_scores = []
        for chunk in raw_chunks:
            chunk_vec = cls._simple_vector_embedding(chunk)
            sim = float(np.dot(query_vec, chunk_vec))
            vector_scores.append(sim)
        vector_scores = np.array(vector_scores)

        # 4. Hybrid Reciprocal Scoring (50% BM25 + 50% Vector Similarity)
        hybrid_scores = (0.5 * norm_bm25) + (0.5 * vector_scores)

        # 5. Sort Chunks by Hybrid Relevance Score
        scored_chunks: List[Tuple[float, str]] = sorted(
            zip(hybrid_scores, raw_chunks), 
            key=lambda x: x[0], 
            reverse=True
        )

        # 6. Deduplication & Token Budget Selection
        selected_chunks: List[str] = []
        accumulated_tokens = 0
        seen_fingerprints = set()

        for score, chunk in scored_chunks:
            # Simple fingerprint to skip near-duplicate chunks
            fingerprint = " ".join(re.findall(r'\w+', chunk.lower())[:10])
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)

            chunk_tokens = RequestAnalyzer.estimate_tokens(chunk)
            if accumulated_tokens + chunk_tokens <= token_budget:
                selected_chunks.append(chunk)
                accumulated_tokens += chunk_tokens
            elif accumulated_tokens == 0:
                # If chunk alone exceeds budget, truncate it to fit budget
                words = chunk.split()
                truncated_word_count = int(token_budget * 0.75)
                truncated_chunk = " ".join(words[:truncated_word_count]) + "..."
                selected_chunks.append(truncated_chunk)
                accumulated_tokens = token_budget
                break

        optimized_text = "\n\n".join(selected_chunks)
        optimized_tokens = RequestAnalyzer.estimate_tokens(optimized_text)
        token_savings_pct = round(
            max(0.0, (original_tokens - optimized_tokens) / (original_tokens + 1e-9) * 100), 2
        )

        return {
            "optimized_text": optimized_text,
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "token_savings_percent": token_savings_pct,
            "chunks_selected": len(selected_chunks),
            "total_chunks": len(raw_chunks)
        }
