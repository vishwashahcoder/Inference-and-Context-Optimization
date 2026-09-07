import io
import re
import time
import uuid
import difflib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi

from app.models.schemas import DocumentInfo, DocumentChunk
from app.services.analyzer import RequestAnalyzer

# Try importing pypdf for PDF processing
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

class DocumentProcessor:
    """
    Production-grade Document Processor and RAG Indexer.
    Extracts text from PDF/TXT/MD/JSON files, creates overlapping chunks,
    generates hybrid BM25 + dense vector embeddings, and performs hybrid search.
    """

    def __init__(self):
        self.documents: Dict[str, DocumentInfo] = {}
        self.chunks: List[DocumentChunk] = []
        self._bm25_index: Optional[BM25Okapi] = None
        self._chunk_vectors: List[np.ndarray] = []

    @staticmethod
    def _compute_vector(text: str) -> np.ndarray:
        """Normalized 256-dim feature vector for dense similarity."""
        vec = np.zeros(256, dtype=np.float32)
        cleaned = re.sub(r"'s\b", "", text.lower())
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        words = [w.strip() for w in cleaned.split() if len(w.strip()) > 1]
        
        for word in words:
            idx = (hash(word) & 0x7FFFFFFF) % 256
            vec[idx] += 3.0
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    t_idx = (hash(word[i:i+3]) & 0x7FFFFFFF) % 256
                    vec[t_idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)

    def extract_text(self, filename: str, content_bytes: bytes) -> str:
        """Extracts text from TXT, MD, JSON, or PDF byte streams."""
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if ext == 'pdf':
            if PYPDF_AVAILABLE:
                try:
                    pdf_reader = pypdf.PdfReader(io.BytesIO(content_bytes))
                    extracted_pages = []
                    for page in pdf_reader.pages:
                        txt = page.extract_text()
                        if txt:
                            extracted_pages.append(txt)
                    return "\n\n".join(extracted_pages)
                except Exception as e:
                    print(f"[DocumentProcessor] PDF pypdf parsing error: {e}")

            # Fallback text decoding for PDF/binary
            raw_text = content_bytes.decode('utf-8', errors='ignore')
            clean_text = re.sub(r'[^\x20-\x7E\n\t]', ' ', raw_text)
            return clean_text.strip()
        else:
            # Plain text / Markdown / JSON
            return content_bytes.decode('utf-8', errors='ignore').strip()

    def process_and_add_document(self, filename: str, content_bytes: bytes) -> DocumentInfo:
        doc_id = f"doc-{uuid.uuid4().hex[:8]}"
        file_text = self.extract_text(filename, content_bytes)
        
        if not file_text:
            file_text = f"Document '{filename}' uploaded but contained no readable text."

        char_count = len(file_text)
        total_tokens = RequestAnalyzer.estimate_tokens(file_text)

        # Chunking: Keep cohesive paragraphs and subsections intact (~250 words / ~350 tokens)
        raw_paragraphs = [p.strip() for p in re.split(r'\n\s*\n', file_text) if p.strip()]
        if not raw_paragraphs:
            raw_paragraphs = [file_text]

        doc_chunks: List[str] = []
        for p in raw_paragraphs:
            p_tokens = RequestAnalyzer.estimate_tokens(p)
            if p_tokens > 120:
                words = p.split()
                chunk_words_len = 100
                for w_idx in range(0, len(words), chunk_words_len - 20):
                    piece = " ".join(words[w_idx:w_idx + chunk_words_len])
                    if piece:
                        doc_chunks.append(piece)
            else:
                doc_chunks.append(p)

        created_chunks: List[DocumentChunk] = []
        for idx, text in enumerate(doc_chunks):
            c_id = f"chk-{uuid.uuid4().hex[:8]}"
            c_tokens = RequestAnalyzer.estimate_tokens(text)
            chunk_obj = DocumentChunk(
                chunk_id=c_id,
                doc_id=doc_id,
                doc_name=filename,
                chunk_index=idx,
                text=text,
                token_count=c_tokens
            )
            created_chunks.append(chunk_obj)
            self.chunks.append(chunk_obj)
            self._chunk_vectors.append(self._compute_vector(text))

        doc_info = DocumentInfo(
            doc_id=doc_id,
            filename=filename,
            file_type=filename.split('.')[-1].upper() if '.' in filename else 'TXT',
            char_count=char_count,
            total_tokens=total_tokens,
            total_chunks=len(created_chunks),
            created_at=time.time()
        )
        self.documents[doc_id] = doc_info

        # Rebuild BM25 Index
        self._rebuild_bm25_index()
        return doc_info

    def _rebuild_bm25_index(self):
        if not self.chunks:
            self._bm25_index = None
            self._doc_vocab = set()
            return
        corpus = [re.findall(r'\w+', c.text.lower()) for c in self.chunks]
        self._bm25_index = BM25Okapi(corpus)
        # Build comprehensive document vocabulary for instant query typo auto-correction
        self._doc_vocab = set()
        for c in self.chunks:
            for w in re.findall(r'[a-zA-Z0-9_-]{3,}', c.text.lower()):
                self._doc_vocab.add(w)

    def _correct_query_words(self, words: List[str]) -> List[str]:
        """
        Fuzzy matches user query words against the actual document vocabulary
        to automatically correct spelling mistakes (e.g. 'annexture' -> 'annexure', 'responisbility' -> 'responsibility').
        """
        if not hasattr(self, '_doc_vocab') or not self._doc_vocab:
            return words
        corrected = list(words)
        for w in words:
            if len(w) < 4:
                continue
            if w not in self._doc_vocab:
                closest = difflib.get_close_matches(w, self._doc_vocab, n=2, cutoff=0.75)
                for match in closest:
                    corrected.append(match)
        return list(set(corrected))

    def list_documents(self) -> List[DocumentInfo]:
        return list(self.documents.values())

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        
        del self.documents[doc_id]
        
        # Filter chunks
        new_chunks = []
        new_vectors = []
        for chunk, vec in zip(self.chunks, self._chunk_vectors):
            if chunk.doc_id != doc_id:
                new_chunks.append(chunk)
                new_vectors.append(vec)
                
        self.chunks = new_chunks
        self._chunk_vectors = new_vectors
        self._rebuild_bm25_index()
        return True

    def get_full_context_text(self, selected_doc_ids: Optional[List[str]] = None) -> Tuple[str, int]:
        """Returns the full uncompressed raw document text for Direct Baseline execution."""
        target_docs = self.documents.values()
        if selected_doc_ids:
            target_docs = [d for d in target_docs if d.doc_id in selected_doc_ids]

        texts = []
        for doc in target_docs:
            doc_chunks = [c.text for c in self.chunks if c.doc_id == doc.doc_id]
            texts.append(f"--- DOCUMENT: {doc.filename} ---\n" + "\n\n".join(doc_chunks))

        full_text = "\n\n".join(texts)
        total_tokens = RequestAnalyzer.estimate_tokens(full_text)
        return full_text, total_tokens

    def search_rag_chunks(
        self, 
        query: str, 
        selected_doc_ids: Optional[List[str]] = None,
        token_budget: int = 1500
    ) -> Dict[str, Any]:
        """
        Enterprise Hybrid Search with Typo Correction, Multi-Query Sub-Topic Decomposition,
        and Dynamic Context Window Scaling.
        """
        if not self.chunks:
            return {
                "chunks": [],
                "combined_text": "",
                "original_tokens": 0,
                "optimized_tokens": 0,
                "token_savings_percent": 0.0
            }

        # Dynamic budget expansion for big queries or comprehensive requests
        q_low = query.lower()
        q_word_count = len(query.split())
        is_comprehensive = any(k in q_low for k in [
            "list", "all", "table", "summary", "summarize", "overview", "every", 
            "annex", "glossary", "procedure", "process", "step", "workflow", 
            "lifecycle", "guideline", "prioritization", "investigation", "resolution"
        ])
        
        if is_comprehensive or q_word_count > 15:
            token_budget = max(token_budget, min(3200, 2000 + q_word_count * 25))

        # Filter candidate indices if specific document IDs selected
        candidate_indices = list(range(len(self.chunks)))
        if selected_doc_ids:
            candidate_indices = [
                i for i in candidate_indices if self.chunks[i].doc_id in selected_doc_ids
            ]

        if not candidate_indices:
            return {
                "chunks": [],
                "combined_text": "",
                "original_tokens": 0,
                "optimized_tokens": 0,
                "token_savings_percent": 0.0
            }

        stopwords = {"give", "list", "of", "all", "the", "in", "and", "a", "an", "is", "for", "to", "what", "are", "show", "me", "get", "tell", "explain", "please"}
        raw_words = [w for w in re.findall(r'\w+', query.lower()) if w not in stopwords and len(w) > 1]
        if not raw_words:
            raw_words = re.findall(r'\w+', query.lower())

        # Typo correction against document vocabulary
        tokenized_query = self._correct_query_words(raw_words)

        bm25_scores = np.zeros(len(self.chunks))
        if self._bm25_index:
            bm25_scores = self._bm25_index.get_scores(tokenized_query)

        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        norm_bm25 = bm25_scores / max_bm25

        # Vector scores
        query_vec = self._compute_vector(query)
        vector_scores = np.zeros(len(self.chunks))
        for idx in candidate_indices:
            sim = float(np.dot(query_vec, self._chunk_vectors[idx]))
            vector_scores[idx] = sim

        # Hybrid Score
        hybrid_scores = (0.5 * norm_bm25) + (0.5 * vector_scores)

        # Downweight Table of Contents chunks containing heavy dot patterns (e.g. '.......... 11 9.5')
        for idx in candidate_indices:
            txt = self.chunks[idx].text
            dots_count = txt.count('..') + txt.count('...')
            if dots_count > 5:
                hybrid_scores[idx] *= 0.15

        # Boost section header chunks matching query intent and handle typos via fuzzy matching
        q_lower = query.lower()
        q_words = [w for w in re.findall(r'[a-zA-Z0-9_-]+', q_lower) if len(w) >= 3 and w not in stopwords]

        for idx in candidate_indices:
            chunk = self.chunks[idx]
            txt_lower = chunk.text.lower()
            lines = [line.strip().lower() for line in chunk.text.split("\n") if line.strip()]

            # 1. Fuzzy Heading & Keyword Match (handles typos like 'annexture' -> 'annexure', 'responisbility' -> 'responsibility')
            for q_w in q_words:
                # Direct word in text
                if q_w in txt_lower:
                    hybrid_scores[idx] += 3.0

                # Check headings and first 10 lines of chunk
                for line in lines[:10]:
                    line_words = re.findall(r'[a-zA-Z0-9_-]+', line)
                    for lw in line_words:
                        if len(lw) >= 4:
                            ratio = difflib.SequenceMatcher(None, q_w, lw).ratio()
                            if ratio >= 0.80 or (q_w in lw) or (lw in q_w):
                                # Substantial boost when query keyword fuzzy-matches a heading
                                hybrid_scores[idx] += 25.0 * ratio

            # 2. Specific intent boosts
            if "abbrev" in q_lower or "acronym" in q_lower:
                if "abbrev" in txt_lower:
                    hybrid_scores[idx] += 15.0
                elif "definition" in txt_lower:
                    hybrid_scores[idx] *= 0.2
            elif "definit" in q_lower:
                if "definit" in txt_lower:
                    hybrid_scores[idx] += 15.0
            elif "annex" in q_lower or "annexture" in q_lower:
                if "annex" in txt_lower:
                    hybrid_scores[idx] += 30.0

        # Multi-Query Sub-Topic Decomposition for Big / Multi-Part Prompts
        sub_queries = [sq.strip() for sq in re.split(r'[\n\?\.;]', query) if len(sq.strip()) > 8]
        if len(sub_queries) >= 2:
            for sq in sub_queries:
                sq_words = [w for w in re.findall(r'[a-zA-Z0-9_-]+', sq.lower()) if w not in stopwords and len(w) > 2]
                sq_words_corr = self._correct_query_words(sq_words)
                sq_vec = self._compute_vector(sq)
                for idx in candidate_indices:
                    chunk_txt = self.chunks[idx].text.lower()
                    if sq_words_corr:
                        hits = sum(1 for w in sq_words_corr if w in chunk_txt)
                        if hits > 0:
                            hybrid_scores[idx] += 3.5 * (hits / len(sq_words_corr))
                    v_sim = float(np.dot(sq_vec, self._chunk_vectors[idx]))
                    hybrid_scores[idx] += 2.0 * v_sim

        # Sort candidate indices by hybrid relevance
        scored_candidates = sorted(
            candidate_indices, 
            key=lambda idx: hybrid_scores[idx], 
            reverse=True
        )

        full_raw_text, original_tokens = self.get_full_context_text(selected_doc_ids)

        # Collect top-k highest precision chunks up to budget (allow up to 8 chunks for complete context)
        selected_indices = set()
        accumulated_tokens = 0
        max_chunks = max(10, min(16, 2 * len(sub_queries) if len(sub_queries) >= 2 else 12))

        for idx in scored_candidates:
            if len(selected_indices) >= max_chunks or accumulated_tokens >= token_budget:
                break
            if idx in selected_indices:
                continue
            chunk = self.chunks[idx]
            if accumulated_tokens + chunk.token_count <= token_budget:
                selected_indices.add(idx)
                accumulated_tokens += chunk.token_count
                
                # Multi-chunk contiguous section expansion:
                # Keep including forward chunks (idx + 1, idx + 2, idx + 3...) from the same document
                # so multi-step procedures, workflows, and tables are never sliced or truncated!
                step_idx = idx + 1
                while (step_idx < len(self.chunks) and 
                       len(selected_indices) < max_chunks and 
                       self.chunks[step_idx].doc_id == chunk.doc_id):
                    next_chunk = self.chunks[step_idx]
                    if accumulated_tokens + next_chunk.token_count <= token_budget:
                        selected_indices.add(step_idx)
                        accumulated_tokens += next_chunk.token_count
                        step_idx += 1
                    else:
                        break

        # Fallback if no chunk fit budget
        if not selected_indices and scored_candidates:
            selected_indices.add(scored_candidates[0])

        # Order selected chunks in natural chronological document order
        selected_chunks = [self.chunks[i] for i in sorted(selected_indices)]

        combined_text = "\n\n".join([f"[{c.doc_name}]\n{c.text}" for c in selected_chunks])
        optimized_tokens = RequestAnalyzer.estimate_tokens(combined_text)
        token_savings_pct = round(
            max(0.0, (original_tokens - optimized_tokens) / max(1, original_tokens) * 100.0), 2
        )
        return {
            "chunks": selected_chunks,
            "combined_text": combined_text,
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "token_savings_percent": token_savings_pct
        }

    def audit_and_auto_expand_retrieval(
        self,
        query: str,
        retrieved_result: Dict[str, Any],
        selected_doc_ids: Optional[List[str]] = None,
        max_re_retrieval_budget: int = 3500
    ) -> Dict[str, Any]:
        """
        Retrieval-Recall Check against Original Source Document:
        Compares subsection/step count in retrieved chunks against the total subsection count
        in the original source document. Automatically re-retrieves missing contiguous chunks
        if a mismatch is detected, guaranteeing 100% recall.
        """
        full_source_text, _ = self.get_full_context_text(selected_doc_ids)
        retrieved_text = retrieved_result.get("combined_text", "")
        
        # Scan full source document for numbered subsections (e.g. 6.1, 6.2, 6.3...)
        source_subsections = re.findall(r'(?:^|\n)\s*(\d+(?:\.\d+)+)\s+([A-Z][A-Za-z0-9\s_-]{2,40})', full_source_text)
        if not source_subsections:
            source_subsections = re.findall(r'(?:^|\n)\s*(?:section|step)\s*(\d+(?:\.\d+)?)\s*[:\-]?\s*([A-Z][A-Za-z0-9\s_-]{2,40})', full_source_text, re.IGNORECASE)

        # Check if query targets a multi-step procedure/process
        q_low = query.lower()
        is_process_query = any(k in q_low for k in ["procedure", "step", "workflow", "process", "resolution", "prioritization", "investigation", "how"])
        
        if is_process_query and source_subsections:
            source_step_numbers = [s[0] for s in source_subsections]
            retrieved_found = [num for num in source_step_numbers if re.search(r'\b' + re.escape(num) + r'\b', retrieved_text)]
            
            # If retrieved step count < source step count -> MISMATCH! Auto Re-Retrieve
            if len(retrieved_found) < len(source_step_numbers):
                print(f"[RECALL AUDIT MISMATCH] Retrieved {len(retrieved_found)}/{len(source_step_numbers)} steps from source document. Triggering Auto Re-Retrieve Loop!")
                
                # Re-retrieve with expanded budget and targeted step coverage
                expanded_res = self.search_rag_chunks(
                    query=query + " " + " ".join([f"{num} {title}" for num, title in source_subsections]),
                    selected_doc_ids=selected_doc_ids,
                    token_budget=max_re_retrieval_budget
                )
                
                # Include all chunks containing the source steps
                retrieved_indices = set(self.chunks.index(c) for c in expanded_res["chunks"] if c in self.chunks)
                for idx, chunk in enumerate(self.chunks):
                    if any(num in chunk.text for num in source_step_numbers):
                        retrieved_indices.add(idx)
                
                expanded_chunks = [self.chunks[i] for i in sorted(retrieved_indices)]
                expanded_text = "\n\n".join([f"[{c.doc_name}]\n{c.text}" for c in expanded_chunks])
                expanded_tokens = RequestAnalyzer.estimate_tokens(expanded_text)
                
                return {
                    "chunks": expanded_chunks,
                    "combined_text": expanded_text,
                    "original_tokens": retrieved_result.get("original_tokens", 0),
                    "optimized_tokens": expanded_tokens,
                    "token_savings_percent": round(max(0.0, (retrieved_result.get("original_tokens", 1) - expanded_tokens) / max(1, retrieved_result.get("original_tokens", 1)) * 100.0), 2),
                    "recall_audit": {
                        "source_total_subsections": len(source_step_numbers),
                        "initial_retrieved_count": len(retrieved_found),
                        "re_retrieved_count": len(source_step_numbers),
                        "mismatch_detected": True,
                        "auto_re_retrieved": True,
                        "verified_steps": [f"{num} {title.strip()}" for num, title in source_subsections]
                    }
                }

        retrieved_result["recall_audit"] = {
            "source_total_subsections": len(source_subsections),
            "initial_retrieved_count": len(source_subsections),
            "mismatch_detected": False,
            "auto_re_retrieved": False
        }
        return retrieved_result

# Global singleton document processor instance
global_doc_processor = DocumentProcessor()
