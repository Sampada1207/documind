"""Regression test suite for document comparison quality.

Tests:
1. End-to-end comparison quality on Plan A and Plan B synthetic test PDFs.
2. Complete sentence validation (no sentence fragments or dangling text).
3. Elimination of generic filler (e.g. 'methodology specifications outlined in').
4. Plan B limitations correctly recognized and cited (preventing false claims of absent limitations).
5. Concrete parameter-level differences (capacity, scope, budget, timeline, maintenance, constraints).
6. Authentic citation validity across both documents.
7. Proper distinction between missing evidence and actual limitations.
8. OpenAI mode prompt structure and JSON output validation.
"""

import io
import re
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.comparison_service import comparison_service, clean_sentence, unwrap_page_lines
from app.services.document_registry import registry
from app.services.openai_service import openai_service
from app.services.pdf_service import extract_text_from_pdf
from tests.test_phase6_integration import create_test_pdf_bytes


class TestComparisonQualityRegression(unittest.TestCase):
    """Regression tests verifying high-quality document comparison output."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        registry.clear()
        self.original_provider = settings.LLM_PROVIDER
        settings.LLM_PROVIDER = "mock"

        # Plan A: 6 buildings, 240 kW, ₹1.8 crore, 8 months; maintenance review every 6 months; grid bottleneck limitation
        self.plan_a_pages = [
            "Project Plan A: Municipal Rooftop Solar Deployment. Objectives: Deploy 240 kW rooftop solar across 6 municipal buildings in Riverside.",
            "Methodology utilizes monocrystalline PV panels and string inverters with standard mounting. Total capital budget is ₹1.8 crore with an estimated 8 months execution timeline.",
            "Operations and Governance: Routine maintenance review is scheduled every 6 months. Challenges: Grid interconnection delays remain a bottleneck.",
        ]

        # Plan B: 4 buildings, 180 kW, ₹1.35 crore, 6 months; quarterly performance reviews during the first year; load-bearing constraint
        self.plan_b_pages = [
            "Project Plan B: Distributed Administrative Solar Initiative. Objectives: Install 180 kW decentralized rooftop solar across 4 administrative buildings.",
            "Approach utilizes tier-1 monocrystalline panels coupled with microinverters for individual panel optimization. Total project budget is ₹1.35 crore to be completed within 6 months.",
            "Operations and Governance: Quarterly performance reviews during the first year ensure operational compliance. Constraints: Rooftop load-bearing constraints on older municipal buildings limit array expansion beyond the planned footprint.",
        ]

    def tearDown(self):
        settings.LLM_PROVIDER = self.original_provider
        registry.clear()

    def test_clean_sentence_and_unwrap_page_lines(self):
        """Helper functions ensure complete sentence formation without text mutilation."""
        # Unwrapped lines merge across line breaks inside sentences
        wrapped_sample = "First part of the sentence continues across\nmultiple lines without breaking grammar."
        unwrapped = unwrap_page_lines(wrapped_sample)
        self.assertEqual(len(unwrapped), 1)
        self.assertEqual(unwrapped[0], "First part of the sentence continues across multiple lines without breaking grammar.")

        # Numbers at start of sentence are preserved, not stripped as bullet items
        sentence_with_number = "6 municipal buildings were selected for phase 1."
        cleaned = clean_sentence(sentence_with_number)
        self.assertTrue(cleaned.startswith("6 municipal buildings"))
        self.assertTrue(cleaned.endswith("."))

        # Font fallback question mark for rupee symbol is normalized
        currency_sample = "?1.8 crore capital budget allocated"
        cleaned_currency = clean_sentence(currency_sample)
        self.assertIn("₹1.8 crore", cleaned_currency)

    def test_plan_a_and_plan_b_comparison_quality(self):
        """Verify Plan A vs Plan B comparison eliminates defects, detects limitations, and yields concrete differences."""
        bytes_a = create_test_pdf_bytes(self.plan_a_pages)
        bytes_b = create_test_pdf_bytes(self.plan_b_pages)

        ext_a = extract_text_from_pdf(bytes_a, "Plan_A_Solar.pdf")
        ext_b = extract_text_from_pdf(bytes_b, "Plan_B_Solar.pdf")

        doc_a_id = "doc_test_plan_a"
        doc_b_id = "doc_test_plan_b"

        registry.register_document({
            "document_id": doc_a_id,
            "filename": "Plan_A_Solar.pdf",
            "total_pages": ext_a["total_pages"],
            "total_characters": ext_a["total_characters"],
            "total_words": ext_a["total_words"],
            "pages": ext_a["pages"],
        })

        registry.register_document({
            "document_id": doc_b_id,
            "filename": "Plan_B_Solar.pdf",
            "total_pages": ext_b["total_pages"],
            "total_characters": ext_b["total_characters"],
            "total_words": ext_b["total_words"],
            "pages": ext_b["pages"],
        })

        # Run comparison
        res = comparison_service.compare_documents(doc_a_id, doc_b_id)

        # 1. Verification of No Generic Filler or Incomplete Fragments
        all_text_blocks = (
            res["document_a"]["objectives"]
            + res["document_a"]["methodologies"]
            + res["document_a"]["findings"]
            + res["document_a"]["limitations"]
            + res["document_b"]["objectives"]
            + res["document_b"]["methodologies"]
            + res["document_b"]["findings"]
            + res["document_b"]["limitations"]
            + res["similarities"]
            + res["differences"]
            + res["missing_information"]
        )

        for text in all_text_blocks:
            # Must not contain generic filler
            self.assertNotIn("methodology specifications outlined in", text.lower())
            # Must be grammatically complete and end with terminal punctuation
            self.assertTrue(
                text.endswith((".", "!", "?", "]")),
                f"Sentence does not end with terminal punctuation: {text}",
            )
            # Must not be an empty or truncated fragment
            self.assertGreater(len(text), 15, f"Fragment too short: {text}")

        # 2. Plan B Limitations Accuracy (Prevent false claims of absent limitations)
        plan_b_limits = res["document_b"]["limitations"]
        self.assertGreater(len(plan_b_limits), 0)
        limit_text = " ".join(plan_b_limits).lower()
        self.assertNotIn("no explicit limitations", limit_text)
        self.assertIn("load-bearing", limit_text)
        self.assertIn("limit array expansion", limit_text)

        # 3. Plan A Limitations Accuracy
        plan_a_limits = res["document_a"]["limitations"]
        self.assertGreater(len(plan_a_limits), 0)
        limit_a_text = " ".join(plan_a_limits).lower()
        self.assertIn("interconnection", limit_a_text)
        self.assertIn("bottleneck", limit_a_text)

        # 4. Concrete Differences Extracted
        diff_corpus = "\n".join(res["differences"])
        # Capacity difference: 240 kW vs 180 kW
        self.assertIn("240 kW", diff_corpus)
        self.assertIn("180 kW", diff_corpus)
        # Deployment scope: 6 buildings vs 4 buildings
        self.assertIn("6 municipal buildings", diff_corpus)
        self.assertIn("4 administrative buildings", diff_corpus)
        # Budget: ₹1.8 crore vs ₹1.35 crore
        self.assertTrue("1.8 crore" in diff_corpus or "₹1.8 crore" in diff_corpus)
        self.assertTrue("1.35 crore" in diff_corpus or "₹1.35 crore" in diff_corpus)
        # Timeline: 8 months vs 6 months
        self.assertIn("8 months", diff_corpus)
        self.assertIn("6 months", diff_corpus)
        # Maintenance schedule
        self.assertIn("6 months", diff_corpus)
        self.assertIn("Quarterly", diff_corpus)

        # 5. Verified Citations
        citations = res["citations"]
        self.assertGreater(len(citations), 0)
        doc_ids = {c["document_id"] for c in citations}
        self.assertIn(doc_a_id, doc_ids)
        self.assertIn(doc_b_id, doc_ids)

        for cit in citations:
            self.assertIn("page", cit)
            self.assertIn(cit["page"], [1, 2, 3])
            self.assertIn("excerpt", cit)
            self.assertGreater(len(cit["excerpt"]), 15)
            # Excerpts must not be incomplete fragments
            self.assertTrue(cit["excerpt"].endswith((".", "!", "?")))

    def test_missing_evidence_distinguished_from_limitations(self):
        """Asymmetric disclosures correctly distinguish unaddressed topics from limitations."""
        doc_complete_id = "doc_complete"
        doc_partial_id = "doc_partial"

        # Complete doc defines budget, timeline, maintenance, and limitation
        registry.register_document({
            "document_id": doc_complete_id,
            "filename": "Comprehensive_Project.pdf",
            "total_pages": 1,
            "total_characters": 400,
            "total_words": 60,
            "pages": [{
                "page": 1,
                "text": "Objectives: Deploy 100 kW solar. Capital budget is $500,000 with a 4 months timeline. Routine maintenance is scheduled monthly. Limitation: Supply chain delays may occur.",
                "char_count": 160,
                "word_count": 25,
            }],
        })

        # Partial doc omits budget, timeline, and maintenance, but specifies an objective and methodology
        registry.register_document({
            "document_id": doc_partial_id,
            "filename": "Conceptual_Study.pdf",
            "total_pages": 1,
            "total_characters": 300,
            "total_words": 40,
            "pages": [{
                "page": 1,
                "text": "Objectives: Investigate community microgrid concepts. Approach utilizes decentralized inverters.",
                "char_count": 95,
                "word_count": 12,
            }],
        })

        res = comparison_service.compare_documents(doc_complete_id, doc_partial_id)

        # Conceptual study explicitly states that limitations/budgets are not specified
        partial_profile = res["document_b"]
        self.assertTrue(any("not explicitly specified" in item.lower() for item in partial_profile["limitations"]))
        self.assertTrue(any("not explicitly specified" in item.lower() for item in partial_profile["findings"]))

        # Missing information identifies the asymmetric disclosures explicitly
        missing_corpus = "\n".join(res["missing_information"])
        self.assertIn("Capital budget is detailed in Comprehensive_Project.pdf", missing_corpus)
        self.assertIn("not specified in Conceptual_Study.pdf", missing_corpus)
        self.assertIn("Implementation duration is specified in Comprehensive_Project.pdf", missing_corpus)

    def test_openai_comparison_structured_output_and_prompt(self):
        """OpenAI comparison path enforces prompt rules and parses valid structured JSON."""
        doc_1_id = "openai_doc_1"
        doc_2_id = "openai_doc_2"

        registry.register_document({
            "document_id": doc_1_id,
            "filename": "Doc_One.pdf",
            "total_pages": 1,
            "total_characters": 200,
            "total_words": 30,
            "pages": [{"page": 1, "text": "Objectives: Evaluate model accuracy.", "char_count": 35, "word_count": 5}],
        })

        registry.register_document({
            "document_id": doc_2_id,
            "filename": "Doc_Two.pdf",
            "total_pages": 1,
            "total_characters": 200,
            "total_words": 30,
            "pages": [{"page": 1, "text": "Objectives: Benchmark inference latency.", "char_count": 40, "word_count": 5}],
        })

        mock_llm_json = """
        {
          "document_a": {
            "objectives": ["Evaluate model accuracy across standard datasets. [Doc A: Page 1]"],
            "methodologies": ["Methodology is not explicitly specified in Doc_One.pdf."],
            "findings": ["Findings summarized from Doc_One.pdf. [Doc A: Page 1]"],
            "limitations": ["No explicit limitations specified in Doc_One.pdf."]
          },
          "document_b": {
            "objectives": ["Benchmark inference latency under variable load. [Doc B: Page 1]"],
            "methodologies": ["Methodology is not explicitly specified in Doc_Two.pdf."],
            "findings": ["Findings summarized from Doc_Two.pdf. [Doc B: Page 1]"],
            "limitations": ["No explicit limitations specified in Doc_Two.pdf."]
          },
          "similarities": ["Both documents investigate machine learning system evaluation benchmarks."],
          "differences": ["Doc_One.pdf focuses on accuracy [Doc A: Page 1], whereas Doc_Two.pdf targets latency [Doc B: Page 1]."],
          "missing_information": ["Neither document provides empirical compute hardware specifications."]
        }
        """

        with patch.object(settings, "LLM_PROVIDER", "openai"), \
             patch.object(openai_service, "generate_rag_answer", return_value=mock_llm_json) as mock_generate:
            
            res = comparison_service.compare_documents(doc_1_id, doc_2_id)

        self.assertEqual(res["llm_provider"], "openai")
        self.assertFalse(res["is_simulated"])
        self.assertIn("accuracy", res["differences"][0])
        self.assertIn("latency", res["differences"][0])

        # Verify instructions given to OpenAI enforce complete sentences and parameter comparisons
        call_args = mock_generate.call_args[1]
        instructions = call_args["instructions"]
        self.assertIn("Never output incomplete sentence fragments", instructions)
        self.assertIn("Compare concrete parameters", instructions)
        self.assertIn("Inspect all relevant pages before claiming that information or limitations are absent", instructions)


if __name__ == "__main__":
    unittest.main()
