import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Production-Grade AI Inference & Context Optimization Platform"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/v1"

    # API Keys (Free tier APIs supported: Groq, Gemini, HuggingFace)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    HF_API_KEY: str = os.getenv("HF_API_KEY", "")

    # Provider Mode: "auto", "groq", "gemini", "hf", "mock"
    DEFAULT_PROVIDER: str = "auto"

    # Semantic Cache Settings
    CACHE_SIMILARITY_THRESHOLD: float = 0.88  # Cosine similarity score >= 0.88 = semantic cache hit
    MAX_CACHE_SIZE: int = 1000

    # Token Budget Defaults
    DEFAULT_TOKEN_BUDGET: int = 1500  # Max context tokens allowed after compression
    MIN_CHUNK_SCORE: float = 0.15

    # Pricing Models (Cost per 1,000 tokens in USD)
    COST_BASELINE_LARGE_INPUT: float = 0.0050   # Baseline direct large model input cost ($0.005 / 1k)
    COST_BASELINE_LARGE_OUTPUT: float = 0.0150  # Baseline direct large model output cost ($0.015 / 1k)

    COST_TIER1_INPUT: float = 0.0001   # Tier 1 (Small fast) input cost
    COST_TIER1_OUTPUT: float = 0.0003  # Tier 1 output cost

    COST_TIER2_INPUT: float = 0.0006   # Tier 2 (Medium) input cost
    COST_TIER2_OUTPUT: float = 0.0018  # Tier 2 output cost

    COST_TIER3_INPUT: float = 0.0030   # Tier 3 (Large complex) input cost
    COST_TIER3_OUTPUT: float = 0.0090  # Tier 3 output cost

    # Model Router Tier Mappings (Active Groq Models)
    MODEL_TIER_1_SMALL: str = "openai/gpt-oss-120b"
    MODEL_TIER_2_MEDIUM: str = "openai/gpt-oss-120b"
    MODEL_TIER_3_LARGE: str = "openai/gpt-oss-120b"

    # Optional LangSmith Tracing Fields
    LANGCHAIN_TRACING_V2: Optional[str] = "true"
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: Optional[str] = "inference"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

settings = Settings()
