"""Document summarization service with safe context budgeting and provider routing."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.document_registry import registry
from app.services.mock_llm_service import is_noise_line, clean_statement
from app.services.openai_service import openai_service

logger = logging.getLogger(__name__)

# Maximum safe character budget for LLM summarization context
MAX_SUMMARY_CONTEXT_CHARS = 12000
MAX_PAGES_FULL_SCAN = 6


def prepare_budgeted_document_context(doc: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Prepare a token-budgeted representation of document pages to prevent oversized requests.
    
    Args:
        doc: Document dictionary from document registry containing 'pages'.
        
    Returns:
        Tuple of (budgeted_context_text, sampled_pages_metadata)
    """
    pages = doc.get("pages", [])
    if not pages:
        return ("", [])

    total_pages = len(pages)
    sampled_pages: List[Dict[str, Any]] = []

    # If document is short, use all pages directly
    if total_pages <= MAX_PAGES_FULL_SCAN:
        selected_pages = pages
    else:
        # Long document: sample front (pages 1, 2), middle, and end (last page)
        selected_indices = {0, 1, total_pages - 1}
        # Add 1-2 intermediate pages evenly
        step = max(2, total_pages // 3)
        for idx in range(2, total_pages - 1, step):
            selected_indices.add(idx)
        selected_pages = [pages[i] for i in sorted(selected_indices) if i < total_pages]

    context_blocks = []
    current_chars = 0

    for p in selected_pages:
        page_num = p.get("page", 1)
        raw_text = p.get("text", "")
        clean_lines = [line.strip() for line in raw_text.splitlines() if line.strip() and not is_noise_line(line)]
        page_content = "\n".join(clean_lines)

        # Truncate if exceeds remaining budget
        remaining_budget = MAX_SUMMARY_CONTEXT_CHARS - current_chars
        if remaining_budget <= 200:
            break

        if len(page_content) > remaining_budget:
            page_content = page_content[:remaining_budget] + "... [page content truncated for budget]"

        context_blocks.append(f"--- PAGE {page_num} ---\n{page_content}\n")
        current_chars += len(page_content)
        sampled_pages.append({"page": page_num, "char_count": len(page_content)})

    return ("\n".join(context_blocks), sampled_pages)


class DocumentSummaryService:
    """Orchestrates structured document summarization via Mock or OpenAI providers."""

    def summarize_document(self, document_id: str) -> Dict[str, Any]:
        """Generate a structured summary for a document registered in memory.
        
        Args:
            document_id: UUID of uploaded document.
            
        Returns:
            Structured dictionary matching DocumentSummaryResponse.
        """
        doc = registry.get_document(document_id)
        if not doc:
            raise KeyError(f"Document '{document_id}' not found in in-memory session.")

        budgeted_text, sampled_pages = prepare_budgeted_document_context(doc)
        if not budgeted_text.strip():
            return {
                "document_id": document_id,
                "filename": doc["filename"],
                "total_pages": doc["total_pages"],
                "overview": "The document contains no extractable text.",
                "objectives": [],
                "key_points": [],
                "findings": [],
                "conclusion": "No conclusion could be derived from empty text.",
                "llm_provider": settings.LLM_PROVIDER,
                "is_simulated": settings.is_mock_mode,
                "model_used": "mock-synthesizer" if settings.is_mock_mode else settings.OPENAI_MODEL,
                "sampled_pages_count": len(sampled_pages),
            }

        if settings.is_mock_mode:
            return self._summarize_mock(doc, budgeted_text, sampled_pages)
        else:
            return self._summarize_openai(doc, budgeted_text, sampled_pages)

    def _summarize_mock(
        self,
        doc: Dict[str, Any],
        budgeted_text: str,
        sampled_pages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Deterministic, evidence-grounded summary generation without OpenAI calls."""
        pages = doc.get("pages", [])
        overview = ""
        objectives: List[str] = []
        key_points: List[str] = []
        findings: List[Dict[str, Any]] = []
        conclusion = ""

        # 1. Overview: Extract first non-noise narrative sentence from page 1
        if pages:
            first_page_text = pages[0].get("text", "")
            lines = [l.strip() for l in first_page_text.splitlines() if l.strip() and not is_noise_line(l)]
            for line in lines:
                sents = re.split(r"(?<=[.!?])\s+", line)
                for s in sents:
                    clean_s = clean_statement(s)
                    if len(clean_s) >= 20:
                        overview = clean_s if clean_s.endswith((".", "!", "?")) else clean_s + "."
                        break
                if overview:
                    break

        if not overview:
            overview = f"Overview of {doc['filename']} covering {doc['total_pages']} pages."

        # 2. Extract Objectives, Key Points, and Findings with page provenance
        objective_keywords = ["objective", "aim", "goal", "purpose", "intended", "propose", "targets"]
        metric_keywords = ["kW", "MW", "₹", "crore", "lakh", "%", "buildings", "budget", "cost", "schedule", "capacity"]

        for p in pages:
            p_num = p.get("page", 1)
            raw_text = p.get("text", "")
            for line in raw_text.splitlines():
                line_str = line.strip()
                if not line_str or is_noise_line(line_str):
                    continue

                clean_l = clean_statement(line_str)
                lower_l = clean_l.lower()

                # Objectives extraction
                if any(k in lower_l for k in objective_keywords) and len(objectives) < 3:
                    if len(clean_l) > 15:
                        formatted_obj = clean_l if clean_l.endswith((".", "!", "?")) else clean_l + "."
                        if formatted_obj not in objectives:
                            objectives.append(f"{formatted_obj} [Page {p_num}]")

                # Findings with page numbers (sentences with concrete metrics)
                if any(m.lower() in lower_l for m in metric_keywords) and len(findings) < 4:
                    if len(clean_l) > 12:
                        finding_text = clean_l if clean_l.endswith((".", "!", "?")) else clean_l + "."
                        if not any(f["text"] == finding_text for f in findings):
                            findings.append({
                                "text": finding_text,
                                "page": p_num,
                            })

                # Key points
                if len(key_points) < 4 and len(clean_l) > 25 and not any(k in lower_l for k in objective_keywords):
                    formatted_kp = clean_l if clean_l.endswith((".", "!", "?")) else clean_l + "."
                    if formatted_kp not in key_points:
                        key_points.append(formatted_kp)

        # Fallback if no objectives found
        if not objectives and key_points:
            objectives.append(f"Deploy and evaluate specifications documented in {doc['filename']}. [Page 1]")

        # 3. Conclusion: Extract from last page
        if pages:
            last_page_text = pages[-1].get("text", "")
            last_lines = [l.strip() for l in last_page_text.splitlines() if l.strip() and not is_noise_line(l)]
            for l in reversed(last_lines):
                sents = re.split(r"(?<=[.!?])\s+", l)
                for s in reversed(sents):
                    clean_s = clean_statement(s)
                    if len(clean_s) >= 20:
                        conclusion = clean_s if clean_s.endswith((".", "!", "?")) else clean_s + "."
                        break
                if conclusion:
                    break

        if not conclusion:
            conclusion = f"The document provides structured research and operational findings across {doc['total_pages']} pages."

        return {
            "document_id": doc["document_id"],
            "filename": doc["filename"],
            "total_pages": doc["total_pages"],
            "overview": overview,
            "objectives": objectives,
            "key_points": key_points[:4],
            "findings": findings,
            "conclusion": conclusion,
            "llm_provider": "mock",
            "is_simulated": True,
            "model_used": "mock-deterministic-synthesizer",
            "sampled_pages_count": len(sampled_pages),
        }

    def _summarize_openai(
        self,
        doc: Dict[str, Any],
        budgeted_text: str,
        sampled_pages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Summarize document using OpenAI Responses API with structured JSON output."""
        system_prompt = (
            "You are DocuMind, an intelligent document research assistant. "
            "Analyze the provided document pages and output a valid JSON summary adhering strictly to this schema:\n"
            "{\n"
            '  "overview": "Concise high-level overview string",\n'
            '  "objectives": ["Array of key objectives with [Page X] references"],\n'
            '  "key_points": ["Array of 3-5 core takeaways"],\n'
            '  "findings": [{"text": "Finding string", "page": 1}],\n'
            '  "conclusion": "Closing takeaway string"\n'
            "}\n"
            "Rules:\n"
            "1. Ground all statements strictly in the provided pages.\n"
            "2. Preserve source page references for all findings and objectives.\n"
            "3. Output ONLY raw valid JSON, no markdown codeblocks or commentary."
        )

        user_input = f"Document: {doc['filename']}\nTotal Pages: {doc['total_pages']}\n\nContent:\n{budgeted_text}"

        raw_response = openai_service.generate_rag_answer(
            instructions=system_prompt,
            user_input=user_input,
        )

        try:
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            parsed = json.loads(cleaned)
        except Exception as e:
            logger.warning(f"Failed to parse OpenAI JSON summary, falling back to structured representation: {e}")
            parsed = {
                "overview": raw_response[:300],
                "objectives": [],
                "key_points": [],
                "findings": [],
                "conclusion": "Derived from OpenAI synthesis.",
            }

        return {
            "document_id": doc["document_id"],
            "filename": doc["filename"],
            "total_pages": doc["total_pages"],
            "overview": parsed.get("overview", ""),
            "objectives": parsed.get("objectives", []),
            "key_points": parsed.get("key_points", []),
            "findings": parsed.get("findings", []),
            "conclusion": parsed.get("conclusion", ""),
            "llm_provider": "openai",
            "is_simulated": False,
            "model_used": settings.OPENAI_MODEL,
            "sampled_pages_count": len(sampled_pages),
        }


summary_service = DocumentSummaryService()
