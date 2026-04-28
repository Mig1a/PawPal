"""
Test suite for PawPal+ AI features.

Tests cover:
  - RAGRetriever: loading, retrieval relevance, species filtering
  - ReliabilitySystem: confidence scoring, urgency detection, validation,
                       logging, review queue
  - PawPalAgent: intent classification, expert modes, workflow (Claude mocked)

Run with:
    python -m pytest tests/test_ai_features.py -v
or standalone:
    python tests/test_ai_features.py
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_retriever import RAGRetriever
from reliability import ReliabilitySystem
from ai_agent import PawPalAgent, classify_intent, EXPERT_MODES


# ---------------------------------------------------------------------------
# Helpers — build a tiny temporary knowledge base for tests
# ---------------------------------------------------------------------------

def _make_kb(tmpdir: str) -> None:
    """Write a minimal knowledge base with 6 entries for testing."""
    data = {
        "metadata": {"version": "1.0", "category": "test"},
        "entries": [
            {
                "id": "t01",
                "title": "Ear Infection in Dogs",
                "species": ["dog"],
                "urgency": "medium",
                "content": (
                    "Ear infections in dogs cause scratching, head shaking, and dark "
                    "discharge. Caused by bacteria, yeast, or ear mites. See a vet if "
                    "symptoms persist for more than 3-4 days."
                ),
                "tags": ["ear", "infection", "scratching", "discharge"],
                "when_to_see_vet": "Within 3-4 days",
            },
            {
                "id": "t02",
                "title": "Puppy Training Basics",
                "species": ["dog"],
                "urgency": "low",
                "content": (
                    "Train puppies using positive reinforcement. Teach sit, stay, come, "
                    "and down. Keep sessions to 5 minutes for young puppies. Reward "
                    "correct behaviour immediately with treats and praise."
                ),
                "tags": ["puppy", "training", "sit", "stay", "positive reinforcement"],
                "when_to_see_vet": "Not applicable",
            },
            {
                "id": "t03",
                "title": "Cat Nutrition — Wet vs Dry Food",
                "species": ["cat"],
                "urgency": "low",
                "content": (
                    "Cats are obligate carnivores and benefit greatly from wet food for "
                    "hydration. Dry food alone increases risk of urinary disease and "
                    "kidney problems. Feed cats two to three measured meals per day."
                ),
                "tags": ["cat", "nutrition", "wet food", "dry food", "hydration", "feeding"],
                "when_to_see_vet": "Annual nutrition review",
            },
            {
                "id": "t04",
                "title": "Seizures in Pets — Emergency Guide",
                "species": ["dog", "cat"],
                "urgency": "emergency",
                "content": (
                    "A seizure is an emergency. Do not put hands in the mouth. Time the "
                    "seizure. If it lasts more than five minutes call the emergency vet "
                    "immediately. Keep the environment calm and safe during the event."
                ),
                "tags": ["seizure", "emergency", "epilepsy", "convulsion"],
                "when_to_see_vet": "EMERGENCY — immediate vet required",
            },
            {
                "id": "t05",
                "title": "Why Cats Scratch Furniture",
                "species": ["cat"],
                "urgency": "low",
                "content": (
                    "Scratching maintains claw health, stretches muscles, and marks "
                    "territory. Provide appropriate scratching posts at least 32 inches "
                    "tall. Use deterrents like double-sided tape on furniture temporarily."
                ),
                "tags": ["scratching", "cat behavior", "scratching post", "furniture", "claws"],
                "when_to_see_vet": "Not medical",
            },
            {
                "id": "t06",
                "title": "Dog Feeding Guidelines for Adults",
                "species": ["dog"],
                "urgency": "low",
                "content": (
                    "Adult dogs should be fed twice daily. Portions vary by size: "
                    "small breeds need 1-1.5 cups, large breeds need 2-4 cups. "
                    "Monitor body condition score and adjust portions accordingly."
                ),
                "tags": ["dog", "feeding", "nutrition", "portions", "adult dog"],
                "when_to_see_vet": "Annual nutrition review",
            },
        ],
    }
    path = os.path.join(tmpdir, "test_kb.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# RAGRetriever tests
# ---------------------------------------------------------------------------

class TestRAGRetriever(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        _make_kb(self.tmpdir)
        self.rag = RAGRetriever(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_loads_documents(self):
        """Knowledge base should load all 6 test entries."""
        self.assertEqual(self.rag.get_document_count(), 6)

    def test_is_available(self):
        """RAG should report available when sklearn is installed and KB loaded."""
        # If sklearn is missing the test is skipped at import — this just
        # checks the flag is consistent with actual availability.
        if self.rag._available:
            self.assertTrue(self.rag.is_available())

    def test_retrieval_returns_results(self):
        """A relevant query should return at least one result."""
        if not self.rag.is_available():
            self.skipTest("scikit-learn not installed")
        results = self.rag.retrieve("dog ear scratching infection", k=3)
        self.assertGreater(len(results), 0)

    def test_retrieval_top_result_is_relevant(self):
        """Ear-related query should surface the ear-infection entry first."""
        if not self.rag.is_available():
            self.skipTest("scikit-learn not installed")
        results = self.rag.retrieve("my dog keeps scratching his ear and shaking his head", k=3)
        self.assertTrue(len(results) > 0)
        top = results[0]
        # The ear infection entry or at least a dog entry should be first
        self.assertIn("ear", top["title"].lower() + " ".join(top["tags"]))

    def test_training_query_returns_training_entry(self):
        """Training query should surface the puppy training entry."""
        if not self.rag.is_available():
            self.skipTest("scikit-learn not installed")
        results = self.rag.retrieve("how do I teach my puppy to sit and stay", k=2)
        titles = [r["title"].lower() for r in results]
        self.assertTrue(any("train" in t or "puppy" in t for t in titles))

    def test_scores_between_zero_and_one(self):
        """All retrieved scores must be in [0, 1]."""
        if not self.rag.is_available():
            self.skipTest("scikit-learn not installed")
        results = self.rag.retrieve("cat food nutrition wet dry", k=4)
        for doc in results:
            self.assertGreaterEqual(doc["score"], 0.0)
            self.assertLessEqual(doc["score"], 1.0)

    def test_species_filter_returns_only_cat_docs(self):
        """retrieve_by_species('cat') should not return dog-only entries."""
        if not self.rag.is_available():
            self.skipTest("scikit-learn not installed")
        results = self.rag.retrieve_by_species("feeding nutrition", "cat", k=4)
        for doc in results:
            species = doc.get("species", [])
            self.assertTrue(
                "cat" in species or "all" in species,
                f"Expected cat species but got {species} in '{doc['title']}'",
            )

    def test_empty_query_returns_empty_or_valid(self):
        """Empty query should not crash."""
        if not self.rag.is_available():
            self.skipTest("scikit-learn not installed")
        results = self.rag.retrieve("", k=3)
        self.assertIsInstance(results, list)

    def test_unknown_directory_returns_zero_docs(self):
        """RAGRetriever on a non-existent dir should have 0 documents."""
        rag = RAGRetriever("/this/does/not/exist")
        self.assertEqual(rag.get_document_count(), 0)


# ---------------------------------------------------------------------------
# ReliabilitySystem tests
# ---------------------------------------------------------------------------

class TestReliabilitySystem(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.tmpdir, "test.log")
        self.rel = ReliabilitySystem(
            log_file=self.log_path,
            confidence_threshold=0.50,
        )

    def tearDown(self):
        # Remove log handlers to avoid file lock issues on Windows
        for handler in self.rel.logger.handlers[:]:
            handler.close()
            self.rel.logger.removeHandler(handler)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_docs(self, score=0.6):
        return [{"score": score, "tags": ["ear", "infection", "scratching"]}]

    def test_confidence_is_in_range(self):
        """Confidence score must always be between 0 and 1."""
        score = self.rel.compute_confidence(
            query="my dog is scratching its ear",
            retrieved_docs=self._make_docs(),
            response_text="Ear infections are common. See your vet within 3-4 days.",
            intent="symptom",
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_higher_retrieval_score_raises_confidence(self):
        """Higher retrieval cosine score should produce higher confidence."""
        low  = self.rel.compute_confidence("query", self._make_docs(score=0.05),
                                           "short response", "general")
        high = self.rel.compute_confidence("query", self._make_docs(score=0.95),
                                           "x " * 200, "symptom")
        self.assertGreater(high, low)

    def test_specific_intent_raises_confidence(self):
        """A specific intent should score higher than 'general'."""
        general  = self.rel.compute_confidence("what", [], "short answer here", "general")
        specific = self.rel.compute_confidence("dog ear scratch", self._make_docs(),
                                               "detailed response " * 10, "symptom")
        self.assertGreater(specific, general)

    def test_review_queue_empty_initially(self):
        """Review queue must be empty before any interactions."""
        self.assertEqual(len(self.rel.get_review_queue()), 0)

    def test_low_confidence_flags_for_review(self):
        """Adding to review queue should appear in get_review_queue()."""
        self.rel.add_to_review_queue(
            {"query": "test", "response": "ok", "confidence": 0.2,
             "intent": "general", "expert_mode": "general"}
        )
        self.assertEqual(len(self.rel.get_review_queue()), 1)

    def test_dismiss_removes_item(self):
        """Dismissing index 0 should shrink the queue."""
        self.rel.add_to_review_queue({"query": "q", "response": "r",
                                      "confidence": 0.1, "intent": "g", "expert_mode": "g"})
        self.rel.dismiss_review_item(0)
        self.assertEqual(len(self.rel.get_review_queue()), 0)

    def test_statistics_total_increments(self):
        """log_interaction should increment total_queries."""
        self.rel.log_interaction("q", "r", 0.7, "general", "nutrition")
        stats = self.rel.get_statistics()
        self.assertEqual(stats["total_queries"], 1)

    def test_statistics_average_confidence(self):
        """Average confidence should be the mean of logged scores."""
        self.rel.log_interaction("q1", "r1", 0.6, "general", "symptom")
        self.rel.log_interaction("q2", "r2", 0.8, "general", "training")
        stats = self.rel.get_statistics()
        self.assertAlmostEqual(stats["average_confidence"], 0.7, places=2)

    def test_urgency_emergency_detected(self):
        """Emergency keywords must trigger emergency urgency level."""
        result = self.rel.detect_urgency(
            "This is an emergency — go to the emergency vet immediately!"
        )
        self.assertEqual(result["level"], "emergency")

    def test_urgency_high_detected(self):
        """High urgency keywords should return high level."""
        result = self.rel.detect_urgency("You should see a vet soon as this is concerning.")
        self.assertEqual(result["level"], "high")

    def test_urgency_none_for_normal_response(self):
        """Ordinary advice text should not trigger any urgency level."""
        result = self.rel.detect_urgency(
            "Feed your dog twice daily with high-quality protein kibble."
        )
        self.assertEqual(result["level"], "none")

    def test_validate_response_too_short(self):
        """A very short response should fail validation."""
        result = self.rel.validate_response("Yes.")
        self.assertFalse(result["valid"])
        self.assertTrue(len(result["issues"]) > 0)

    def test_validate_response_medical_without_vet(self):
        """Medical content without a vet recommendation should flag an issue."""
        result = self.rel.validate_response(
            "You should prescribe this medication at the correct dosage for two weeks."
        )
        self.assertFalse(result["valid"])

    def test_validate_response_good(self):
        """A complete, safe response should pass validation."""
        long_response = (
            "Ear infections in dogs are commonly caused by bacteria or yeast. "
            "Signs include head shaking and scratching. Clean the outer ear and "
            "consult your veterinarian within a few days if symptoms persist. "
            "Treatment depends on the underlying cause and may include antibiotic drops."
        )
        result = self.rel.validate_response(long_response)
        self.assertTrue(result["valid"])

    def test_log_file_created(self):
        """Logging an interaction should create the log file."""
        self.rel.log_interaction("test query", "test response", 0.75, "general", "training")
        self.assertTrue(os.path.exists(self.log_path))

    def test_confidence_label_mapping(self):
        """Confidence labels should map correctly to score ranges."""
        self.assertEqual(self.rel.get_confidence_label(0.85), "High")
        self.assertEqual(self.rel.get_confidence_label(0.65), "Medium")
        self.assertEqual(self.rel.get_confidence_label(0.45), "Low")
        self.assertEqual(self.rel.get_confidence_label(0.20), "Very Low")


# ---------------------------------------------------------------------------
# Intent classification tests (no API call needed)
# ---------------------------------------------------------------------------

class TestIntentClassification(unittest.TestCase):

    def test_symptom_intent(self):
        queries = [
            "My dog is vomiting and not eating",
            "My cat has been scratching her ears",
            "Is my dog sick? He seems lethargic",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(classify_intent(q), "symptom")

    def test_training_intent(self):
        queries = [
            "How do I teach my puppy to sit and stay?",
            "My dog keeps barking at strangers",
            "Crate training guide for a 10-week puppy",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(classify_intent(q), "training")

    def test_nutrition_intent(self):
        queries = [
            "What should I feed my overweight dog?",
            "Can dogs eat grapes or raisins?",
            "How much wet food does my cat need per day?",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(classify_intent(q), "nutrition")

    def test_behavior_intent(self):
        queries = [
            "Why does my cat keep knocking things off the table?",
            "My dog shows anxiety when left alone",
            "Why is my cat hiding under the bed?",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(classify_intent(q), "behavior")

    def test_general_intent_fallback(self):
        """Ambiguous short queries should fall back to general."""
        result = classify_intent("hello")
        self.assertEqual(result, "general")


# ---------------------------------------------------------------------------
# Expert mode tests
# ---------------------------------------------------------------------------

class TestExpertModes(unittest.TestCase):

    def test_all_modes_have_required_keys(self):
        for mode_key, cfg in EXPERT_MODES.items():
            with self.subTest(mode=mode_key):
                self.assertIn("name",          cfg)
                self.assertIn("icon",          cfg)
                self.assertIn("description",   cfg)
                self.assertIn("system_prompt", cfg)

    def test_system_prompts_non_empty(self):
        for mode_key, cfg in EXPERT_MODES.items():
            with self.subTest(mode=mode_key):
                self.assertGreater(len(cfg["system_prompt"]), 50)

    def test_vet_mode_prompt_mentions_vet(self):
        prompt = EXPERT_MODES["veterinary"]["system_prompt"].lower()
        self.assertTrue(
            "veterinar" in prompt or "clinical" in prompt,
            "Veterinary prompt should mention veterinary or clinical context",
        )

    def test_trainer_mode_avoids_punishment(self):
        prompt = EXPERT_MODES["dog_trainer"]["system_prompt"].lower()
        self.assertIn("positive reinforcement", prompt)


# ---------------------------------------------------------------------------
# PawPalAgent tests (OpenAI API is mocked)
# ---------------------------------------------------------------------------

class TestPawPalAgent(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        _make_kb(self.tmpdir)
        self.log_path = os.path.join(self.tmpdir, "test.log")
        self.rag = RAGRetriever(self.tmpdir)
        self.rel = ReliabilitySystem(log_file=self.log_path)

    def tearDown(self):
        for handler in self.rel.logger.handlers[:]:
            handler.close()
            self.rel.logger.removeHandler(handler)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_mock_client(self, response_text: str):
        """Build a mock openai.OpenAI client returning *response_text*."""
        mock_message = MagicMock()
        mock_message.content = response_text
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_agent_returns_required_keys(self):
        """run() must return all required top-level keys."""
        if not self.rag.is_available():
            self.skipTest("scikit-learn not installed")

        agent = PawPalAgent(self.rag, self.rel)
        agent._client = self._make_mock_client(
            "Ear infections in dogs often cause scratching. See your veterinarian soon."
        )

        result = agent.run("my dog keeps scratching its ear", expert_mode="general")

        for key in ("response", "confidence", "confidence_label", "sources",
                    "intent", "urgency", "expert_mode", "steps", "error"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_agent_confidence_between_zero_and_one(self):
        """Confidence returned by agent.run() must be in [0, 1]."""
        if not self.rag.is_available():
            self.skipTest("scikit-learn not installed")

        agent = PawPalAgent(self.rag, self.rel)
        agent._client = self._make_mock_client(
            "Feed your dog twice daily with quality kibble for good health."
        )
        result = agent.run("how much should I feed my dog", expert_mode="general")
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_agent_no_api_key_returns_error(self):
        """Agent without a client should return an error result, not raise."""
        agent = PawPalAgent(self.rag, self.rel)
        agent._client = None   # simulate missing key
        result = agent.run("test query")
        self.assertNotEqual(result["error"], "")
        self.assertEqual(result["confidence"], 0.0)

    def test_emergency_urgency_detected(self):
        """A response with 'emergency' should set urgency level to emergency."""
        if not self.rag.is_available():
            self.skipTest("scikit-learn not installed")

        agent = PawPalAgent(self.rag, self.rel)
        agent._client = self._make_mock_client(
            "This is an emergency — take your pet to the emergency vet immediately!"
        )
        result = agent.run("my dog had a seizure", expert_mode="veterinary")
        self.assertEqual(result["urgency"]["level"], "emergency")

    def test_agentic_steps_populated(self):
        """Workflow steps list should have multiple entries on a successful run."""
        if not self.rag.is_available():
            self.skipTest("scikit-learn not installed")

        agent = PawPalAgent(self.rag, self.rel)
        agent._client = self._make_mock_client(
            "Puppies need positive reinforcement training sessions of 5 minutes."
        )
        result = agent.run("puppy training tips", expert_mode="dog_trainer")
        self.assertGreater(len(result["steps"]), 2)

    def test_low_confidence_adds_to_review_queue(self):
        """A very low-confidence interaction should appear in the review queue."""
        if not self.rag.is_available():
            self.skipTest("scikit-learn not installed")

        agent = PawPalAgent(self.rag, self.rel)
        agent.CONFIDENCE_THRESHOLD = 0.99   # force everything to be flagged
        agent._client = self._make_mock_client("ok")
        agent.run("hello", expert_mode="general")
        self.assertGreater(len(self.rel.get_review_queue()), 0)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    for cls in (
        TestRAGRetriever,
        TestReliabilitySystem,
        TestIntentClassification,
        TestExpertModes,
        TestPawPalAgent,
    ):
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total  = result.testsRun
    passed = total - len(result.failures) - len(result.errors) - len(result.skipped)
    print(f"\n{'='*60}")
    print(f"PawPal+ AI Test Summary")
    print(f"{'='*60}")
    print(f"Total:   {total}")
    print(f"Passed:  {passed}")
    print(f"Failed:  {len(result.failures)}")
    print(f"Errors:  {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"{'='*60}")
