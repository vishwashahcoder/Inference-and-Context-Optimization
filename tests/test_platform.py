import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.analyzer import RequestAnalyzer
from app.services.context_optimizer import ContextOptimizer
from app.services.cache_manager import CacheManager
from app.services.model_router import ModelRouter
from app.services.document_processor import DocumentProcessor

client = TestClient(app)

def test_request_analyzer_simple_vs_complex():
    simple_analysis = RequestAnalyzer.analyze_complexity("Hi, what is Paris?")
    assert simple_analysis["complexity"] == "SIMPLE"
    assert simple_analysis["recommended_tier"] == "Tier 1 (Small)"

    complex_analysis = RequestAnalyzer.analyze_complexity(
        "Please explain in detail step by step how to architect a multi-threaded parallel GPU inference engine with lock-free queues."
    )
    assert complex_analysis["complexity"] == "COMPLEX"
    assert complex_analysis["recommended_tier"] == "Tier 3 (Large)"

def test_context_optimizer_budget_compression():
    doc1 = "Continuous batching improves GPU memory utilization by dynamically inserting requests into running iteration batches."
    doc2 = "KV Caching stores key-value vectors in memory to eliminate recomputing attention states for previous tokens."
    doc3 = "Unrelated fluff about ancient history and vacuum tubes operating before transistors were invented." * 5

    res = ContextOptimizer.optimize(
        query="How does continuous batching and KV caching work?",
        documents=[doc1, doc2, doc3],
        token_budget=100
    )

    assert res["optimized_tokens"] <= 120
    assert res["token_savings_percent"] > 0
    assert "Continuous batching" in res["optimized_text"] or "KV Caching" in res["optimized_text"]

def test_cache_manager_exact_and_semantic_hits():
    cache = CacheManager(similarity_threshold=0.80)
    prompt = "What is the capital city of France?"
    sys_prompt = "You are a helpful assistant."

    # Store entry
    cache.set(
        prompt=prompt,
        system_prompt=sys_prompt,
        response_text="The capital of France is Paris.",
        model_used="TestModel",
        tokens_used=15,
        estimated_latency_ms=100.0
    )

    # 1. Test Exact Hit
    entry_exact, hit_exact, score_exact = cache.get(prompt, sys_prompt)
    assert hit_exact == "exact"
    assert entry_exact["response_text"] == "The capital of France is Paris."

    # 2. Test Semantic Hit (similar phrasing)
    entry_sem, hit_sem, score_sem = cache.get("What is France's capital city?", sys_prompt)
    assert hit_sem == "semantic"
    assert score_sem >= 0.80

def test_model_router_tier_selection():
    route_simple = ModelRouter.route("Hi")
    assert route_simple["complexity"] == "SIMPLE"

    route_complex = ModelRouter.route("Explain in detail step by step trade-offs of microservices vs monoliths.")
    assert route_complex["complexity"] == "COMPLEX"

def test_document_processor_chunking_and_rag_search():
    dp = DocumentProcessor()
    text = "AI Context Gateway optimizes prompt length. " + "Unrelated text about ancient history. " * 20
    doc_info = dp.process_and_add_document("tech_brief.txt", text.encode("utf-8"))
    
    assert doc_info.filename == "tech_brief.txt"
    assert doc_info.total_chunks > 0
    
    rag_res = dp.search_rag_chunks("What does AI Context Gateway optimize?", token_budget=100)
    assert len(rag_res["chunks"]) > 0
    assert rag_res["optimized_tokens"] <= 200

def test_api_chat_completions_endpoint():
    payload = {
        "messages": [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Explain AI optimization"}
        ],
        "token_budget": 500
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert "usage" in data
    assert "optimization" in data

def test_api_benchmark_endpoint():
    payload = {
        "prompt": "What is continuous batching?",
        "context_documents": ["Continuous batching maximizes GPU throughput during LLM serving."],
        "token_budget": 300,
        "force_no_cache": True
    }
    response = client.post("/v1/benchmark", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "baseline" in data
    assert "optimized" in data
    assert "delta" in data
    assert "speedup_factor" in data["delta"]

def test_api_document_upload_and_chat_rag():
    # 1. Upload test file
    files = {"file": ("test_policy.txt", b"Company refund policy grants 100% money back within 30 days of purchase.", "text/plain")}
    up_res = client.post("/v1/documents/upload", files=files)
    assert up_res.status_code == 200
    up_data = up_res.json()
    assert up_data["status"] == "success"
    doc_id = up_data["document"]["doc_id"]

    # 2. List documents
    list_res = client.get("/v1/documents")
    assert list_res.status_code == 200
    assert any(d["doc_id"] == doc_id for d in list_res.json())

    # 3. Document RAG Chat
    rag_req = {
        "prompt": "What is the refund policy?",
        "token_budget": 500,
        "force_no_cache": True
    }
    chat_res = client.post("/v1/chat/document-rag", json=rag_req)
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert "baseline" in chat_data
    assert "optimized" in chat_data
    assert "delta" in chat_data

    # 4. Clean up / Delete document
    del_res = client.delete(f"/v1/documents/{doc_id}")
    assert del_res.status_code == 200
