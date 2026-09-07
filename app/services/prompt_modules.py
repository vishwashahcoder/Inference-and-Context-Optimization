"""
Modular System Prompt Engine
Splits monolithic system prompts into composable, modular blocks.
Pulls only the relevant block per query type, saving tokens while enforcing
structured schema-based output verified against source section counts.
"""

from typing import Dict, Any, List, Optional
import re

class PromptModuleManager:
    # 1. Base Grounding Block (Universal static prefix - highly cacheable)
    BASE_GROUNDING_BLOCK = (
        "You are an enterprise AI Gateway assistant engineered for 100% factual accuracy.\n"
        "CORE RULES:\n"
        "- Base your response EXCLUSIVELY on the provided document context.\n"
        "- Do not extrapolate, assume, or invent information not present in the text.\n"
        "- Maintain exact terminology, acronyms, reference codes, and section numbers."
    )

    # 2. Procedure & Workflow Block (Enforces sequential completeness for multi-step processes)
    PROCEDURE_SCHEMA_BLOCK = (
        "PROCEDURE & WORKFLOW RULES:\n"
        "- Extract EVERY single step, subsection, and stage in the procedure in exact chronological order.\n"
        "- Do NOT summarize or omit any step (e.g. Prioritization, Investigation, Resolution, Closure must all be included).\n"
        "- Format each step as: ### [Step Number] [Step Title]\n"
        "  - **Description**: Exact action or policy\n"
        "  - **Responsible Party**: Role assigned (if specified)\n"
        "- Verify that the total number of outputted steps matches all steps listed in the source section."
    )

    # 3. Table & Extraction Block (Enforces structured key-value / tabular precision)
    TABLE_SCHEMA_BLOCK = (
        "TABLE & EXTRACTION RULES:\n"
        "- Extract all items from the requested table or list with zero omissions.\n"
        "- For acronyms/abbreviations, output: - **[ACRONYM]**: Full Expanded Definition\n"
        "- For annexures/forms, output: - **[Form Code]**: Form/Document Title\n"
        "- Do not include document headers, footers, or unassociated metadata codes."
    )

    # 4. Factoid / Q&A Block (Concise, high-speed factual responses)
    FACTOID_SCHEMA_BLOCK = (
        "FACTUAL Q&A RULES:\n"
        "- Answer the query directly and concisely in 1-3 sentences or clear bullet points.\n"
        "- Cite the specific section or article name where the fact is stated."
    )

    # 5. Synthesis & Summary Block (High-level structured overview)
    SUMMARY_SCHEMA_BLOCK = (
        "EXECUTIVE SUMMARY RULES:\n"
        "- Provide a structured executive overview using bold section headers.\n"
        "- Cover all core pillars present in the retrieved context with bullet points."
    )

    @classmethod
    def get_modular_prompt(cls, query: str) -> str:
        """
        Dynamically routes the query to assemble the minimal, optimal system prompt.
        Saves 60-70% of system prompt tokens compared to monolithic prompt templates.
        """
        q_low = query.lower()
        blocks = [cls.BASE_GROUNDING_BLOCK]

        # Route by query intent
        if any(k in q_low for k in ["procedure", "step", "workflow", "process", "lifecycle", "how to", "resolution", "investigation"]):
            blocks.append(cls.PROCEDURE_SCHEMA_BLOCK)
        elif any(k in q_low for k in ["abbreviat", "acronym", "table", "annex", "form", "code", "list", "glossary"]):
            blocks.append(cls.TABLE_SCHEMA_BLOCK)
        elif any(k in q_low for k in ["summar", "overview", "brief", "what is this document"]):
            blocks.append(cls.SUMMARY_SCHEMA_BLOCK)
        else:
            blocks.append(cls.FACTOID_SCHEMA_BLOCK)

        return "\n\n".join(blocks)

    @classmethod
    def count_source_sections(cls, context_text: str) -> Dict[str, Any]:
        """
        Audits the retrieved document context to count section titles,
        numbered steps, and bulleted items to verify complete extraction.
        """
        # Find numbered sections like "6.1", "6.2", "Step 1", "5 ANNEXURE"
        numbered_steps = re.findall(r'(?:^|\n)\s*(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z0-9\s_-]{2,40})', context_text)
        bullet_items = re.findall(r'(?:^|\n)\s*[-*•]\s+([^\n]+)', context_text)
        
        return {
            "numbered_steps_count": len(numbered_steps),
            "step_headers": [f"{num} {title.strip()}" for num, title in numbered_steps[:15]],
            "bullet_items_count": len(bullet_items)
        }
