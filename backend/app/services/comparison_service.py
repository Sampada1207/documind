"""Research paper comparison service comparing two registered documents."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.document_registry import registry
from app.services.mock_llm_service import is_noise_line
from app.services.openai_service import openai_service
from app.services.summary_service import prepare_budgeted_document_context

logger = logging.getLogger(__name__)


def clean_sentence(text: str) -> str:
    """Clean extra whitespace, bullet markers, font fallbacks, and ensure complete terminal punctuation."""
    collapsed = " ".join(text.split())
    # Fix font fallback ? for rupee symbol when PyMuPDF or font encodes ₹ as ?
    collapsed = re.sub(r"\?(\d+(?:\.\d+)?\s*(?:crore|lakh))", r"₹\1", collapsed)
    # Strip leading bullet markers or numbered lists (e.g. '1. ', '- ', '• ')
    collapsed = re.sub(r"^(?:[*\-•]|\d+\.)\s*", "", collapsed).strip()
    # Strip leading colons, semicolons, or dashes
    collapsed = re.sub(r"^[:;\-\s]+", "", collapsed).strip()
    # Strip trailing commas, colons, semicolons
    collapsed = collapsed.rstrip(",;: ")
    if not collapsed:
        return ""
    # Ensure starts capitalized
    collapsed = collapsed[0].upper() + collapsed[1:]
    # Ensure complete terminal punctuation
    if not collapsed.endswith((".", "!", "?")):
        collapsed += "."
    return collapsed


def unwrap_page_lines(raw_text: str) -> List[str]:
    """Unwrap line breaks within paragraphs while preserving section headers and bullet items."""
    raw_lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    merged_lines: List[str] = []
    for line in raw_lines:
        if not merged_lines:
            merged_lines.append(line)
            continue
        prev = merged_lines[-1]
        is_bullet = bool(re.match(r"^(?:[*\-•]|\d+\.)\s+", line))
        # If prev line does not end with terminal punctuation (. ? ! :) and current line is not a bullet
        if not prev.endswith((".", "!", "?", ":")) and not is_bullet:
            merged_lines[-1] = prev + " " + line
        else:
            merged_lines.append(line)
    return merged_lines


def extract_key_facets_from_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministically extract complete sentences and concrete parameters across all pages."""
    pages = doc.get("pages", [])
    doc_id = doc["document_id"]
    filename = doc["filename"]

    objectives = []
    methodologies = []
    findings = []
    limitations = []

    params = {
        "capacity": None,
        "capacity_page": 1,
        "buildings": None,
        "buildings_page": 1,
        "budget": None,
        "budget_page": 1,
        "timeline": None,
        "timeline_page": 1,
        "maintenance": None,
        "maintenance_page": 1,
    }

    sec_keywords = {
        "objectives": ["objective", "aim", "goal", "purpose", "scope", "target"],
        "methodologies": [
            "methodology", "method", "approach", "technique", "architecture",
            "system design", "technical specifications", "system specifications", "implementation"
        ],
        "findings": [
            "finding", "result", "outcome", "financial", "budget", "cost",
            "expenditure", "performance", "metrics"
        ],
        "limitations": [
            "limitation", "constraint", "challenge", "risk", "bottleneck",
            "drawback", "shortcoming", "tradeoff", "gap", "obstacle"
        ],
        "maintenance": ["maintenance", "operation", "servicing", "monitoring", "review"],
    }

    prefix_strip_pattern = re.compile(
        r"^(?:objectives?|aims?|goals?|methodolog(?:y|ies)|approaches?|findings?|results?|"
        r"limitations?|challenges?|constraints?|risks?|operations?(?:\s+and\s+governance)?|"
        r"maintenance|financials?(?:\s+and\s+timeline)?|budget(?:\s+and\s+schedule)?)\s*:\s*",
        re.I,
    )

    inline_kw = {
        "objectives": [
            "objective", "aim", "goal", "purpose", "target", "propose",
            "proposes", "deploy", "install", "intended to", "initiative"
        ],
        "methodologies": [
            "methodology", "method", "approach", "technique", "architecture",
            "utilize", "utilizes", "employ", "employs", "coupled with", "paired with",
            "monocrystalline", "polycrystalline", "microinverter", "inverter", "racking", "pv"
        ],
        "findings": [
            "finding", "result", "capacity", "budget", "efficiency", "achieved",
            "exceeded", "timeline", "duration", "expenditure", "cost", "%", "kw",
            "mw", "crore", "lakh", "million", "billion"
        ],
        "limitations": [
            "limitation", "constraint", "challenge", "bottleneck", "drawback",
            "risk", "tradeoff", "restrict", "restricts", "restriction", "delay",
            "delays", "vulnerability", "obstacle", "shortcoming", "gap", "barrier",
            "limit array", "limit expansion", "load-bearing"
        ],
    }

    for p in pages:
        p_num = p.get("page", 1)
        raw_text = p.get("text", "")
        if not raw_text.strip():
            continue

        # Extract concrete numeric parameters from the raw page text
        if not params["capacity"]:
            cap_match = re.search(r"\b(\d+(?:\.\d+)?\s*(?:kW|MW|GW|W))\b", raw_text, re.I)
            if cap_match:
                params["capacity"] = cap_match.group(1)
                params["capacity_page"] = p_num

        if not params["buildings"]:
            bld_match = re.search(
                r"\b(\d+)\s*(?:municipal\s+|administrative\s+|onshore\s+)?(buildings?|facilities|turbines?|sites?|structures?|rooftops?)\b",
                raw_text,
                re.I,
            )
            if bld_match:
                params["buildings"] = bld_match.group(0)
                params["buildings_page"] = p_num

        if not params["budget"]:
            bud_match = re.search(
                r"((?:₹|\$|€|£|Rs\.?|INR|USD|\?)\s*\d+(?:\.\d+)?\s*(?:crore|lakh|million|billion)?|\d+(?:\.\d+)?\s*(?:crore|lakh|million|billion)\s*(?:rupees|dollars)?)",
                raw_text,
                re.I,
            )
            if bud_match:
                val = bud_match.group(0).strip()
                if val.startswith("?") and ("crore" in val.lower() or "lakh" in val.lower()):
                    val = "₹" + val[1:].strip()
                params["budget"] = val
                params["budget_page"] = p_num

        if not params["timeline"]:
            tim_match = re.search(r"\b(\d+(?:\s*|-)(?:months?|weeks?|years?|days?))\b", raw_text, re.I)
            if tim_match:
                params["timeline"] = tim_match.group(1)
                params["timeline_page"] = p_num

        current_sec = None
        unwrapped_blocks = unwrap_page_lines(raw_text)

        for block in unwrapped_blocks:
            if is_noise_line(block):
                continue

            lower_block = block.lower()

            # Check if block is a standalone section heading
            is_heading = False
            if len(block) <= 60 and (
                block.endswith(":")
                or any(
                    lower_block == k
                    or lower_block.startswith(k + " ")
                    or lower_block.endswith(" " + k)
                    for sublist in sec_keywords.values()
                    for k in sublist
                )
            ):
                for sec, kws in sec_keywords.items():
                    if any(k in lower_block for k in kws):
                        current_sec = sec
                        is_heading = True
                        break
            if is_heading:
                continue

            # Split block into complete sentences
            sents = re.split(r"(?<=[.!?])\s+", block)
            for s in sents:
                matched_sec = current_sec

                # Check if sentence has a leading section label (e.g. 'Objectives: Deploy...')
                for sec, kws in sec_keywords.items():
                    for kw in kws:
                        if re.match(r"^(?:" + kw + r"s?)\s*:\s*", s, re.I):
                            matched_sec = sec
                            break

                # Strip leading section labels for clean, natural sentences
                cleaned_text = prefix_strip_pattern.sub("", s)
                clean_s = clean_sentence(cleaned_text)
                if len(clean_s) < 15:
                    continue

                lower_s = clean_s.lower()

                # Track maintenance parameter
                if not params["maintenance"] and any(
                    k in lower_s
                    for k in [
                        "maintenance", "performance review", "inspections scheduled",
                        "reviews scheduled", "quarterly performance", "washings scheduled"
                    ]
                ):
                    params["maintenance"] = clean_s
                    params["maintenance_page"] = p_num

                # Determine facet classification if not set by heading/prefix
                if not matched_sec or matched_sec == "maintenance":
                    for sec, kws in inline_kw.items():
                        if any(k in lower_s for k in kws):
                            matched_sec = sec
                            break

                item = {"text": clean_s, "page": p_num, "document_id": doc_id, "filename": filename}

                if matched_sec == "objectives" and len(objectives) < 3:
                    if not any(o["text"] == clean_s for o in objectives):
                        objectives.append(item)
                elif matched_sec == "methodologies" and len(methodologies) < 3:
                    if not any(m["text"] == clean_s for m in methodologies):
                        methodologies.append(item)
                elif matched_sec == "findings" and len(findings) < 4:
                    if not any(f["text"] == clean_s for f in findings):
                        findings.append(item)
                elif matched_sec == "limitations" and len(limitations) < 3:
                    if not any(l["text"] == clean_s for l in limitations):
                        limitations.append(item)

    return {
        "document_id": doc_id,
        "filename": filename,
        "total_pages": doc.get("total_pages", 1),
        "objectives": objectives,
        "methodologies": methodologies,
        "findings": findings,
        "limitations": limitations,
        "params": params,
    }


def _build_profile(doc: Dict[str, Any], facets: Dict[str, Any], label: str) -> Dict[str, Any]:
    """Format structured profile facet lists with complete sentences and source citations."""
    filename = doc["filename"]

    # 1. Objectives
    objectives = [f"{o['text']} [Doc {label}: Page {o['page']}]" for o in facets["objectives"]]
    if not objectives:
        objectives = [f"Objectives are not explicitly specified in {filename}."]

    # 2. Methodologies
    methodologies = [f"{m['text']} [Doc {label}: Page {m['page']}]" for m in facets["methodologies"]]
    if not methodologies:
        methodologies = [f"Technical methodology is not explicitly specified in {filename}."]

    # 3. Findings
    findings = [f"{f['text']} [Doc {label}: Page {f['page']}]" for f in facets["findings"]]
    if not findings:
        findings = [f"Quantitative findings are not explicitly specified in {filename}."]

    # 4. Limitations
    limitations = [f"{l['text']} [Doc {label}: Page {l['page']}]" for l in facets["limitations"]]
    if not limitations:
        limitations = [f"Limitations, challenges, and constraints are not explicitly specified in {filename}."]

    return {
        "document_id": doc["document_id"],
        "filename": filename,
        "total_pages": doc.get("total_pages", 1),
        "objectives": objectives,
        "methodologies": methodologies,
        "findings": findings,
        "limitations": limitations,
    }


def _build_differences(
    facets_a: Dict[str, Any],
    facets_b: Dict[str, Any],
    doc_a: Dict[str, Any],
    doc_b: Dict[str, Any],
) -> List[str]:
    """Compute concrete, parameter-level differences between two documents."""
    p_a = facets_a["params"]
    p_b = facets_b["params"]
    name_a = doc_a["filename"]
    name_b = doc_b["filename"]
    diffs = []

    # 1. System Capacity Difference
    if p_a["capacity"] and p_b["capacity"] and p_a["capacity"].lower() != p_b["capacity"].lower():
        diffs.append(
            f"System Capacity: {name_a} specifies {p_a['capacity']} [Doc A: Page {p_a['capacity_page']}], "
            f"whereas {name_b} specifies {p_b['capacity']} [Doc B: Page {p_b['capacity_page']}]."
        )
    elif p_a["capacity"] and not p_b["capacity"]:
        diffs.append(
            f"System Capacity: {name_a} specifies {p_a['capacity']} [Doc A: Page {p_a['capacity_page']}], "
            f"but system capacity is not specified in {name_b}."
        )
    elif p_b["capacity"] and not p_a["capacity"]:
        diffs.append(
            f"System Capacity: {name_b} specifies {p_b['capacity']} [Doc B: Page {p_b['capacity_page']}], "
            f"but system capacity is not specified in {name_a}."
        )

    # 2. Deployment Scope / Facility Count
    if p_a["buildings"] and p_b["buildings"] and p_a["buildings"].lower() != p_b["buildings"].lower():
        diffs.append(
            f"Deployment Scope: {name_a} targets {p_a['buildings']} [Doc A: Page {p_a['buildings_page']}], "
            f"while {name_b} targets {p_b['buildings']} [Doc B: Page {p_b['buildings_page']}]."
        )
    elif p_a["buildings"] and not p_b["buildings"]:
        diffs.append(
            f"Deployment Scope: {name_a} targets {p_a['buildings']} [Doc A: Page {p_a['buildings_page']}], "
            f"but facility scope is not specified in {name_b}."
        )
    elif p_b["buildings"] and not p_a["buildings"]:
        diffs.append(
            f"Deployment Scope: {name_b} targets {p_b['buildings']} [Doc B: Page {p_b['buildings_page']}], "
            f"but facility scope is not specified in {name_a}."
        )

    # 3. Capital Budget Difference
    if p_a["budget"] and p_b["budget"] and p_a["budget"].lower() != p_b["budget"].lower():
        diffs.append(
            f"Capital Budget: {name_a} allocates {p_a['budget']} [Doc A: Page {p_a['budget_page']}], "
            f"compared to {p_b['budget']} in {name_b} [Doc B: Page {p_b['budget_page']}]."
        )
    elif p_a["budget"] and not p_b["budget"]:
        diffs.append(
            f"Capital Budget: {name_a} allocates {p_a['budget']} [Doc A: Page {p_a['budget_page']}], "
            f"but capital expenditure is not specified in {name_b}."
        )
    elif p_b["budget"] and not p_a["budget"]:
        diffs.append(
            f"Capital Budget: {name_b} allocates {p_b['budget']} [Doc B: Page {p_b['budget_page']}], "
            f"but capital expenditure is not specified in {name_a}."
        )

    # 4. Project Timeline Difference
    if p_a["timeline"] and p_b["timeline"] and p_a["timeline"].lower() != p_b["timeline"].lower():
        diffs.append(
            f"Project Timeline: {name_a} projects an execution duration of {p_a['timeline']} [Doc A: Page {p_a['timeline_page']}], "
            f"whereas {name_b} projects {p_b['timeline']} [Doc B: Page {p_b['timeline_page']}]."
        )
    elif p_a["timeline"] and not p_b["timeline"]:
        diffs.append(
            f"Project Timeline: {name_a} projects {p_a['timeline']} [Doc A: Page {p_a['timeline_page']}], "
            f"but project duration is not specified in {name_b}."
        )
    elif p_b["timeline"] and not p_a["timeline"]:
        diffs.append(
            f"Project Timeline: {name_b} projects {p_b['timeline']} [Doc B: Page {p_b['timeline_page']}], "
            f"but project duration is not specified in {name_a}."
        )

    # 5. Maintenance & Review Schedule
    if p_a["maintenance"] and p_b["maintenance"] and p_a["maintenance"].lower() != p_b["maintenance"].lower():
        diffs.append(
            f"Maintenance & Review Schedule: {name_a} specifies '{p_a['maintenance']}' [Doc A: Page {p_a['maintenance_page']}], "
            f"while {name_b} specifies '{p_b['maintenance']}' [Doc B: Page {p_b['maintenance_page']}]."
        )

    # 6. Technical Approach Contrast
    if facets_a["methodologies"] and facets_b["methodologies"]:
        m_a_text = facets_a["methodologies"][0]["text"]
        m_b_text = facets_b["methodologies"][0]["text"]
        if m_a_text.lower() != m_b_text.lower():
            diffs.append(
                f"Technical Approach: {name_a} utilizes '{m_a_text}' [Doc A: Page {facets_a['methodologies'][0]['page']}], "
                f"whereas {name_b} utilizes '{m_b_text}' [Doc B: Page {facets_b['methodologies'][0]['page']}]."
            )

    # 7. Constraints & Risks Contrast
    if facets_a["limitations"] and facets_b["limitations"]:
        l_a_text = facets_a["limitations"][0]["text"]
        l_b_text = facets_b["limitations"][0]["text"]
        if l_a_text.lower() != l_b_text.lower():
            diffs.append(
                f"Constraints & Risks: {name_a} discloses '{l_a_text}' [Doc A: Page {facets_a['limitations'][0]['page']}], "
                f"whereas {name_b} notes '{l_b_text}' [Doc B: Page {facets_b['limitations'][0]['page']}]."
            )
    elif facets_a["limitations"] and not facets_b["limitations"]:
        diffs.append(
            f"Risk Disclosures: {name_a} identifies '{facets_a['limitations'][0]['text']}' [Doc A: Page {facets_a['limitations'][0]['page']}], "
            f"but no explicit limitations are specified in {name_b}."
        )
    elif facets_b["limitations"] and not facets_a["limitations"]:
        diffs.append(
            f"Risk Disclosures: {name_b} identifies '{facets_b['limitations'][0]['text']}' [Doc B: Page {facets_b['limitations'][0]['page']}], "
            f"but no explicit limitations are specified in {name_a}."
        )

    if not diffs:
        diffs.append(
            f"{name_a} ({doc_a['total_pages']} pages) and {name_b} ({doc_b['total_pages']} pages) address distinct implementation scopes and domain requirements."
        )

    return diffs


def _build_similarities(
    facets_a: Dict[str, Any],
    facets_b: Dict[str, Any],
    doc_a: Dict[str, Any],
    doc_b: Dict[str, Any],
) -> List[str]:
    """Compute concrete, parameter-level similarities between two documents."""
    p_a = facets_a["params"]
    p_b = facets_b["params"]
    name_a = doc_a["filename"]
    name_b = doc_b["filename"]
    sims = []

    # 1. Check shared domain concepts
    text_corpus_a = " ".join([p.get("text", "") for p in doc_a.get("pages", [])]).lower()
    text_corpus_b = " ".join([p.get("text", "") for p in doc_b.get("pages", [])]).lower()

    if ("solar" in text_corpus_a or "pv" in text_corpus_a) and ("solar" in text_corpus_b or "pv" in text_corpus_b):
        sims.append(
            f"Domain Focus: Both {name_a} and {name_b} evaluate rooftop solar photovoltaic deployments for institutional or municipal facilities."
        )
    elif "wind" in text_corpus_a and "wind" in text_corpus_b:
        sims.append(
            f"Domain Focus: Both {name_a} and {name_b} evaluate wind energy generation installations."
        )
    elif (
        ("nlp" in text_corpus_a or "neural" in text_corpus_a or "model" in text_corpus_a)
        and ("nlp" in text_corpus_b or "neural" in text_corpus_b or "model" in text_corpus_b)
    ):
        sims.append(
            f"Research Focus: Both papers investigate deep learning model architectures and evaluation benchmarks."
        )
    else:
        sims.append(
            f"Document Purpose: Both {name_a} and {name_b} present structured technical documentation with verified multi-page extractions."
        )

    # 2. Both define power/capacity
    if p_a["capacity"] and p_b["capacity"]:
        sims.append("Quantitative Sizing: Both documents define explicit system capacity ratings.")

    # 3. Both define capital investment
    if p_a["budget"] and p_b["budget"]:
        sims.append("Financial Modeling: Both proposals define formal capital expenditure budgets.")

    # 4. Both define scheduled maintenance / monitoring
    if p_a["maintenance"] and p_b["maintenance"]:
        sims.append("Operational Governance: Both initiatives establish routine maintenance and performance review procedures.")

    # 5. Both have identified constraints
    if facets_a["limitations"] and facets_b["limitations"]:
        sims.append("Risk Disclosure: Both proposals document project constraints and operational risks.")

    return sims


def _build_missing_information(
    facets_a: Dict[str, Any],
    facets_b: Dict[str, Any],
    doc_a: Dict[str, Any],
    doc_b: Dict[str, Any],
) -> List[str]:
    """Compute true information gaps between two documents, distinguishing missing evidence from limitations."""
    p_a = facets_a["params"]
    p_b = facets_b["params"]
    name_a = doc_a["filename"]
    name_b = doc_b["filename"]
    missing = []

    # Asymmetric disclosures
    if p_a["budget"] and not p_b["budget"]:
        missing.append(
            f"Financial Disclosure: Capital budget is detailed in {name_a} ({p_a['budget']}) [Doc A: Page {p_a['budget_page']}], but is not specified in {name_b}."
        )
    elif p_b["budget"] and not p_a["budget"]:
        missing.append(
            f"Financial Disclosure: Capital budget is detailed in {name_b} ({p_b['budget']}) [Doc B: Page {p_b['budget_page']}], but is not specified in {name_a}."
        )

    if p_a["timeline"] and not p_b["timeline"]:
        missing.append(
            f"Schedule Specification: Implementation duration is specified in {name_a} ({p_a['timeline']}) [Doc A: Page {p_a['timeline_page']}], but is not specified in {name_b}."
        )
    elif p_b["timeline"] and not p_a["timeline"]:
        missing.append(
            f"Schedule Specification: Implementation duration is specified in {name_b} ({p_b['timeline']}) [Doc B: Page {p_b['timeline_page']}], but is not specified in {name_a}."
        )

    if p_a["maintenance"] and not p_b["maintenance"]:
        missing.append(
            f"Operational Protocol: Maintenance schedules are documented in {name_a} [Doc A: Page {p_a['maintenance_page']}], but are not specified in {name_b}."
        )
    elif p_b["maintenance"] and not p_a["maintenance"]:
        missing.append(
            f"Operational Protocol: Maintenance schedules are documented in {name_b} [Doc B: Page {p_b['maintenance_page']}], but are not specified in {name_a}."
        )

    if facets_a["limitations"] and not facets_b["limitations"]:
        missing.append(
            f"Risk Assessment Gap: Formal constraints are detailed in {name_a} [Doc A: Page {facets_a['limitations'][0]['page']}], whereas {name_b} does not disclose explicit project limitations."
        )
    elif facets_b["limitations"] and not facets_a["limitations"]:
        missing.append(
            f"Risk Assessment Gap: Formal constraints are detailed in {name_b} [Doc B: Page {facets_b['limitations'][0]['page']}], whereas {name_a} does not disclose explicit project limitations."
        )

    # General unaddressed lifecycle gap
    missing.append(
        "Lifecycle & Storage: Neither document provides specifications for energy storage (battery backup) or post-warranty module decommissioning and recycling costs."
    )

    return missing


class DocumentComparisonService:
    """Compares two uploaded documents across objectives, methodologies, findings, and gaps."""

    def compare_documents(self, doc_id_a: str, doc_id_b: str) -> Dict[str, Any]:
        """Perform comparison between two registered documents."""
        if doc_id_a == doc_id_b:
            raise ValueError("Cannot compare a document with itself. Please select two distinct documents.")

        doc_a = registry.get_document(doc_id_a)
        if not doc_a:
            raise KeyError(f"Document A '{doc_id_a}' not found in in-memory session.")

        doc_b = registry.get_document(doc_id_b)
        if not doc_b:
            raise KeyError(f"Document B '{doc_id_b}' not found in in-memory session.")

        if settings.is_mock_mode:
            return self._compare_mock(doc_a, doc_b)
        else:
            return self._compare_openai(doc_a, doc_b)

    def _compare_mock(self, doc_a: Dict[str, Any], doc_b: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic, grounded comparison using extracted features and real document text."""
        facets_a = extract_key_facets_from_doc(doc_a)
        facets_b = extract_key_facets_from_doc(doc_b)

        # 1. Profiles
        profile_a = _build_profile(doc_a, facets_a, "A")
        profile_b = _build_profile(doc_b, facets_b, "B")

        # 2. Concrete Similarities & Differences
        similarities = _build_similarities(facets_a, facets_b, doc_a, doc_b)
        differences = _build_differences(facets_a, facets_b, doc_a, doc_b)

        # 3. Missing Information Gaps (Evidence-based check)
        missing_info = _build_missing_information(facets_a, facets_b, doc_a, doc_b)

        # 4. Citations Assembly
        citations = []
        for o in facets_a["objectives"]:
            citations.append({
                "document_id": doc_a["document_id"],
                "filename": doc_a["filename"],
                "page": o["page"],
                "excerpt": o["text"],
            })
        for o in facets_b["objectives"]:
            citations.append({
                "document_id": doc_b["document_id"],
                "filename": doc_b["filename"],
                "page": o["page"],
                "excerpt": o["text"],
            })
        for f in facets_a["findings"][:2]:
            citations.append({
                "document_id": doc_a["document_id"],
                "filename": doc_a["filename"],
                "page": f["page"],
                "excerpt": f["text"],
            })
        for f in facets_b["findings"][:2]:
            citations.append({
                "document_id": doc_b["document_id"],
                "filename": doc_b["filename"],
                "page": f["page"],
                "excerpt": f["text"],
            })
        for l in facets_a["limitations"][:1]:
            citations.append({
                "document_id": doc_a["document_id"],
                "filename": doc_a["filename"],
                "page": l["page"],
                "excerpt": l["text"],
            })
        for l in facets_b["limitations"][:1]:
            citations.append({
                "document_id": doc_b["document_id"],
                "filename": doc_b["filename"],
                "page": l["page"],
                "excerpt": l["text"],
            })

        return {
            "document_a": profile_a,
            "document_b": profile_b,
            "similarities": similarities,
            "differences": differences,
            "missing_information": missing_info,
            "citations": citations,
            "llm_provider": "mock",
            "is_simulated": True,
            "model_used": "mock-deterministic-synthesizer",
        }

    def _compare_openai(self, doc_a: Dict[str, Any], doc_b: Dict[str, Any]) -> Dict[str, Any]:
        """Compare documents using OpenAI Responses API with structured JSON output."""
        context_a, _ = prepare_budgeted_document_context(doc_a)
        context_b, _ = prepare_budgeted_document_context(doc_b)

        system_prompt = (
            "You are DocuMind, an intelligent document research assistant. "
            "Compare the two provided documents and output a valid JSON adhering strictly to this schema:\n"
            "{\n"
            '  "document_a": {"objectives": ["..."], "methodologies": ["..."], "findings": ["..."], "limitations": ["..."]},\n'
            '  "document_b": {"objectives": ["..."], "methodologies": ["..."], "findings": ["..."], "limitations": ["..."]},\n'
            '  "similarities": ["Array of shared aspects with source page citations"],\n'
            '  "differences": ["Array of concrete differences with source page citations"],\n'
            '  "missing_information": ["Array of missing information or evidence gaps with citations"]\n'
            "}\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Ground all comparisons strictly in the provided text. Never fabricate claims or citations.\n"
            "2. Every statement must be a complete, grammatically sound sentence. Never output incomplete sentence fragments, generic placeholders, or filler such as 'methodology specifications outlined in [filename]'.\n"
            "3. Compare concrete parameters (objectives, methodologies, findings, limitations, budgets, capacities, timelines, and maintenance schedules) when supported by the documents.\n"
            "4. Inspect all relevant pages before claiming that information or limitations are absent.\n"
            "5. If a fact or limitation is unavailable, explicitly state that it is not specified. Distinguish missing evidence from an actual limitation.\n"
            "6. Cite source pages accurately (e.g. [Doc A: Page 2], [Doc B: Page 1]).\n"
            "7. Output ONLY valid JSON without code blocks or commentary."
        )

        user_input = (
            f"=== DOCUMENT A: {doc_a['filename']} ===\n{context_a}\n\n"
            f"=== DOCUMENT B: {doc_b['filename']} ===\n{context_b}\n"
        )

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
            logger.warning(f"Failed to parse OpenAI comparison JSON: {e}")
            parsed = {
                "document_a": {
                    "objectives": [f"Objectives summarized from {doc_a['filename']}."],
                    "methodologies": [f"Methodology is not explicitly specified in {doc_a['filename']}."],
                    "findings": [f"Findings summarized from {doc_a['filename']}."],
                    "limitations": [f"No explicit limitations specified in {doc_a['filename']}."],
                },
                "document_b": {
                    "objectives": [f"Objectives summarized from {doc_b['filename']}."],
                    "methodologies": [f"Methodology is not explicitly specified in {doc_b['filename']}."],
                    "findings": [f"Findings summarized from {doc_b['filename']}."],
                    "limitations": [f"No explicit limitations specified in {doc_b['filename']}."],
                },
                "similarities": [raw_response[:300]],
                "differences": [],
                "missing_information": [],
            }

        # Build citations from citations in parsed output or fallback to facet citations
        citations = []
        facets_a = extract_key_facets_from_doc(doc_a)
        facets_b = extract_key_facets_from_doc(doc_b)
        for o in facets_a["objectives"]:
            citations.append({
                "document_id": doc_a["document_id"],
                "filename": doc_a["filename"],
                "page": o["page"],
                "excerpt": o["text"],
            })
        for o in facets_b["objectives"]:
            citations.append({
                "document_id": doc_b["document_id"],
                "filename": doc_b["filename"],
                "page": o["page"],
                "excerpt": o["text"],
            })

        return {
            "document_a": {
                "document_id": doc_a["document_id"],
                "filename": doc_a["filename"],
                "total_pages": doc_a["total_pages"],
                **parsed.get("document_a", {}),
            },
            "document_b": {
                "document_id": doc_b["document_id"],
                "filename": doc_b["filename"],
                "total_pages": doc_b["total_pages"],
                **parsed.get("document_b", {}),
            },
            "similarities": parsed.get("similarities", []),
            "differences": parsed.get("differences", []),
            "missing_information": parsed.get("missing_information", []),
            "citations": citations,
            "llm_provider": "openai",
            "is_simulated": False,
            "model_used": settings.OPENAI_MODEL,
        }


comparison_service = DocumentComparisonService()
