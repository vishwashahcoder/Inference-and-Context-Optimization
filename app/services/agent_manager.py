import os
import time
from typing import Dict, Any, List, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from app.core.config import settings
from app.services.document_processor import global_doc_processor
from app.services.cache_manager import global_cache
from app.services.analyzer import RequestAnalyzer

# Define Agent Tools
@tool
def search_document_knowledge_base(query: str) -> str:
    """
    Search the uploaded document knowledge base for relevant facts, definitions,
    abbreviations, specifications, or section contents using hybrid BM25 + Vector retrieval.
    """
    res = global_doc_processor.search_rag_chunks(query=query, token_budget=1800)
    if not res["chunks"]:
        return "No relevant document chunks found in the uploaded knowledge base."
    return res["combined_text"]

@tool
def check_semantic_cache(query: str) -> str:
    """
    Check if an exact or semantically equivalent question has already been answered
    and cached in memory to achieve sub-5ms response time at zero token cost.
    """
    cached_entry, hit_type, sim_score = global_cache.get(query)
    if cached_entry:
        return f"CACHE_HIT ({hit_type}, similarity: {sim_score}): {cached_entry.get('response_text', '')}"
    return "CACHE_MISS: No cached answer found for this query."

tools = [search_document_knowledge_base, check_semantic_cache]

class LangChainAgentManager:
    """
    Production-Grade LangChain Tool-Calling Agent with Native LangSmith Tracing,
    automatic token calculation, and multi-step reasoning.
    """
    _executor: Optional[AgentExecutor] = None

    @classmethod
    def get_llm(cls):
        azure_key = (settings.AZURE_OPENAI_API_KEY or "").strip()
        azure_endpoint = (settings.AZURE_OPENAI_ENDPOINT or "").strip()
        deployment = settings.LLM_MODEL or "gpt-4o"
        api_version = settings.API_VERSION or "2024-12-01-preview"

        if azure_key and azure_endpoint:
            return AzureChatOpenAI(
                azure_deployment=deployment,
                azure_endpoint=azure_endpoint,
                api_key=azure_key,
                api_version=api_version,
                temperature=0.2,
                timeout=30.0
            )
        else:
            from langchain_community.chat_models import ChatOpenAI
            return ChatOpenAI(
                model_name="openai/gpt-oss-120b",
                openai_api_base="https://api.groq.com/openai/v1",
                openai_api_key=settings.GROQ_API_KEY,
                temperature=0.2
            )

    @classmethod
    def get_agent_executor(cls) -> AgentExecutor:
        llm = cls.get_llm()
        
        system_prompt = (
            "You are an expert AI Inference & Context Optimization Agent.\n"
            "Your objective is to provide 100% accurate, factual, and hallucination-free answers.\n"
            "GUIDELINES:\n"
            "1. When asked questions about uploaded documents or technical terms, ALWAYS use the "
            "`search_document_knowledge_base` tool to retrieve verified facts from the document.\n"
            "2. For abbreviations or acronyms, output each abbreviation and its full definition as written in the text.\n"
            "3. Do not assume or invent facts outside the retrieved context."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )

    @classmethod
    async def run(cls, user_input: str) -> Dict[str, Any]:
        """
        Executes the LangChain Agent.
        Because LANGCHAIN_TRACING_V2=true is set, this invocation automatically
        streams full execution runs, tool steps, and token counts directly to LangSmith!
        """
        start_time = time.time()
        executor = cls.get_agent_executor()
        
        # Invoke agent
        result = await executor.ainvoke({"input": user_input})
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        output_text = result.get("output", "")
        steps = []
        for action, observation in result.get("intermediate_steps", []):
            steps.append({
                "tool": getattr(action, "tool", str(action)),
                "tool_input": getattr(action, "tool_input", ""),
                "tool_output": str(observation)[:300] + ("..." if len(str(observation)) > 300 else "")
            })

        prompt_tokens = RequestAnalyzer.estimate_tokens(user_input)
        completion_tokens = RequestAnalyzer.estimate_tokens(output_text)

        return {
            "query": user_input,
            "answer": output_text,
            "latency_ms": latency_ms,
            "intermediate_steps": steps,
            "estimated_prompt_tokens": prompt_tokens,
            "estimated_completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "langsmith_project": os.getenv("LANGCHAIN_PROJECT", "inference"),
            "langsmith_traced": os.getenv("LANGCHAIN_TRACING_V2") == "true"
        }
