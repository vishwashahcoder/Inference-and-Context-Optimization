from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# OpenAI Compatible Schemas
class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "auto"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    context_documents: Optional[List[str]] = None  # Optional raw context docs for RAG optimization
    token_budget: Optional[int] = 1500             # Max context tokens after optimization

class ChatChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str

class ChatChoice(BaseModel):
    index: int = 0
    message: ChatChoiceMessage
    finish_reason: str = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    original_prompt_tokens: int
    tokens_saved: int
    token_savings_percent: float
    estimated_cost_usd: float
    baseline_estimated_cost_usd: float
    cost_savings_usd: float

class OptimizationDetails(BaseModel):
    cache_hit: bool
    cache_type: Optional[str] = None  # "exact", "semantic", or None
    similarity_score: Optional[float] = None
    routed_tier: str                  # "Tier 1 (Small)", "Tier 2 (Medium)", "Tier 3 (Large)"
    routed_model: str
    context_original_length: int
    context_optimized_length: int
    context_compression_ratio: float
    latency_ms: float
    baseline_estimated_latency_ms: float
    latency_saved_ms: float
    latency_savings_percent: float

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo
    optimization: OptimizationDetails

# Direct Benchmark & Playground Schemas
class BenchmarkRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = "You are a helpful assistant."
    context_documents: Optional[List[str]] = []
    token_budget: Optional[int] = 1500
    force_no_cache: Optional[bool] = False

class SingleRunResult(BaseModel):
    mode: str  # "Baseline (Direct Large Model)" vs "Optimized Gateway"
    model_used: str
    response_text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    cache_hit: bool
    cache_type: Optional[str] = None
    context_length_chars: int

class BenchmarkResponse(BaseModel):
    benchmark_id: str
    prompt: str
    context_provided: bool
    baseline: SingleRunResult
    optimized: SingleRunResult
    delta: Dict[str, Any]  # Latency saved, Token savings %, Cost saved $, Speedup factor

class MetricsSummary(BaseModel):
    total_requests: int
    cache_hits_exact: int
    cache_hits_semantic: int
    cache_hit_rate_percent: float
    avg_latency_baseline_ms: float
    avg_latency_optimized_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_prompt_tokens_saved: int
    total_cost_saved_usd: float
    model_tier_distribution: Dict[str, int]

# Document RAG Schemas
class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    chunk_index: int
    text: str
    token_count: int

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    char_count: int
    total_tokens: int
    total_chunks: int
    created_at: float

class DocumentUploadResponse(BaseModel):
    status: str = "success"
    document: DocumentInfo

class DocumentChatRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = "You are a helpful AI assistant answering questions based on uploaded documents."
    selected_doc_ids: Optional[List[str]] = None  # None = search across all docs
    token_budget: Optional[int] = 1500
    force_no_cache: Optional[bool] = False

