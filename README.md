# Production-Grade AI Inference & Context Optimization Platform

An intelligent LLM infrastructure platform that automatically optimizes **context, semantic caching, intelligent model routing, and inference serving** to deliver high-quality AI responses with **lower latency, lower cost, and higher throughput** in production.

---

## Architecture Overview

```text
User Request
     ↓
Request Analyzer
     ↓
Context Optimizer
     ├── Hybrid Retrieval (Vector + Keyword)
     ├── Cross-Encoder Reranking
     ├── Context Compression & Filtering
     └── Token Budget Management
     ↓
Semantic Cache (Vector similarity lookup)
     ↓
Intelligent Model Router (Complexity-based Tier 1 / 2 / 3 routing)
     ↓
Inference Engine (vLLM / Groq / Gemini / HF / Mock)
     ↓
Reliability Layer (Retries, Circuit Breaker, Fallbacks)
     ↓
Response + Observability (Metrics, Latency, Token tracking)
```

---

## Key Features

- **Context Optimization**: Selective retrieval, token budgeting, and compression to avoid sending unnecessary context to the LLM.
- **Semantic Caching**: High-performance semantic cache with cosine similarity thresholds to eliminate redundant queries.
- **Dynamic Model Routing**: Intelligently routes simple queries to faster/cheaper models and complex tasks to frontier models.
- **Reliability & Resilience**: Circuit breakers, rate limiters, fallback chains, and retries.
- **Benchmarking Suite**: Built-in benchmark tools to evaluate latency, throughput, token reduction, and cost savings.

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- Virtual environment

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd "Inference & Context Optimization"

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy the example configuration and fill in your API keys:
```bash
cp .env.example .env
```

### 4. Running the Platform
```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Running Tests & Benchmarks
```bash
# Run test suite
pytest

# Run benchmarks
pytest benchmarks/
```

---

## License
MIT License
