"""
Reliability System for PawPal+
Provides confidence scoring, response validation, structured logging,
and a human review queue for AI-generated pet care advice.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Emergency / urgency keyword sets
# ---------------------------------------------------------------------------
_EMERGENCY_KEYWORDS = frozenset([
    "emergency", "immediately", "urgent", "life-threatening", "life threatening",
    "seizure", "unconscious", "paralyzed", "severe bleeding", "cannot breathe",
    "stop breathing", "pale gums", "white gums", "bloat", "GDV", "collapse",
    "call poison control", "poison control",
])
_HIGH_KEYWORDS = frozenset([
    "see a vet", "see your vet", "veterinarian soon", "within 24 hours",
    "concerning", "worrying", "serious condition", "requires treatment",
])
_MEDIUM_KEYWORDS = frozenset([
    "monitor closely", "if symptoms persist", "within a few days",
    "schedule a vet", "vet check", "keep an eye",
])


class ReliabilitySystem:
    """
    Trust and observability layer for the PawPal+ AI assistant.

    Responsibilities:
    - Compute multi-factor confidence scores (0.0 – 1.0)
    - Validate AI responses for safety and completeness
    - Write structured log lines to a rotating log file
    - Maintain an in-process human-review queue for low-confidence responses
    - Expose aggregate statistics for the System Reports tab
    """

    def __init__(
        self,
        log_file: str = "logs/pawpal_ai.log",
        confidence_threshold: float = 0.50,
    ):
        self.log_file = log_file
        self.confidence_threshold = confidence_threshold

        self._review_queue: List[Dict] = []
        self._stats = {
            "total_queries": 0,
            "confidence_sum": 0.0,
            "flagged_for_review": 0,
            "by_intent": {},
            "by_mode": {},
        }

        self._setup_logger()

    # ------------------------------------------------------------------
    # Logger setup
    # ------------------------------------------------------------------

    def _setup_logger(self) -> None:
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger("pawpal_ai")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        )
        self.logger.addHandler(fh)

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def compute_confidence(
        self,
        query: str,
        retrieved_docs: List[Dict],
        response_text: str,
        intent: str,
    ) -> float:
        """
        Compute a 0.0 – 1.0 confidence score using four weighted factors.

        Factor weights:
          0.35  retrieval quality   (max cosine similarity of retrieved docs)
          0.25  response substance  (word count normalised to 150 words = 1.0)
          0.20  intent clarity      (specific intent > general query)
          0.20  knowledge grounding (fraction of retrieved tags present in response)
        """
        # Factor 1 — retrieval quality
        if retrieved_docs:
            max_score = max(d.get("score", 0.0) for d in retrieved_docs)
            retrieval_factor = min(max_score * 3.0, 1.0)   # scale up cosine scores
        else:
            retrieval_factor = 0.0

        # Factor 2 — response substance
        word_count = len(response_text.split())
        length_factor = min(word_count / 150.0, 1.0)

        # Factor 3 — intent clarity
        intent_factor = 0.9 if intent not in ("general", "unknown") else 0.55

        # Factor 4 — knowledge grounding (tag keyword overlap)
        grounding_factor = 0.35   # baseline
        if retrieved_docs:
            all_tags: set = set()
            for doc in retrieved_docs:
                all_tags.update(t.lower() for t in doc.get("tags", []))
            if all_tags:
                response_lower = response_text.lower()
                matched = sum(1 for tag in all_tags if tag in response_lower)
                grounding_factor = max(matched / len(all_tags), 0.35)
                grounding_factor = min(grounding_factor, 1.0)

        score = (
            0.35 * retrieval_factor
            + 0.25 * length_factor
            + 0.20 * intent_factor
            + 0.20 * grounding_factor
        )

        return round(min(max(score, 0.0), 1.0), 3)

    # ------------------------------------------------------------------
    # Response validation
    # ------------------------------------------------------------------

    def validate_response(self, response_text: str) -> Dict:
        """
        Check AI output for common safety and quality issues.

        Returns a dict with:
          valid   bool   — True when no issues found
          issues  list   — descriptions of any problems found
        """
        issues: List[str] = []

        if len(response_text.split()) < 20:
            issues.append("Response too short (< 20 words)")

        # Medical content without vet recommendation
        medical_terms = {"diagnose", "prescribe", "medication", "dosage", "surgery", "treatment"}
        has_medical = any(term in response_text.lower() for term in medical_terms)
        has_vet_ref = any(w in response_text.lower() for w in ("vet", "veterinarian", "animal hospital"))
        if has_medical and not has_vet_ref:
            issues.append("Medical content present without veterinarian recommendation")

        return {"valid": len(issues) == 0, "issues": issues}

    # ------------------------------------------------------------------
    # Urgency detection
    # ------------------------------------------------------------------

    def detect_urgency(self, response_text: str) -> Dict:
        """
        Scan response text for urgency keywords and return a structured result.

        Returns:
          level    str   — 'none' | 'medium' | 'high' | 'emergency'
          message  str   — human-readable alert string
        """
        lower = response_text.lower()

        if any(kw in lower for kw in _EMERGENCY_KEYWORDS):
            return {
                "level": "emergency",
                "message": "⚠️ EMERGENCY: Contact your vet or emergency animal hospital immediately!",
            }
        if any(kw in lower for kw in _HIGH_KEYWORDS):
            return {
                "level": "high",
                "message": "🔴 Please schedule a veterinary appointment soon.",
            }
        if any(kw in lower for kw in _MEDIUM_KEYWORDS):
            return {
                "level": "medium",
                "message": "🟡 Consider scheduling a vet check within the next few days.",
            }
        return {"level": "none", "message": ""}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_interaction(
        self,
        query: str,
        response: str,
        confidence: float,
        expert_mode: str,
        intent: str,
    ) -> None:
        """Write a structured log line and update aggregate statistics."""
        self._stats["total_queries"] += 1
        self._stats["confidence_sum"] += confidence
        self._stats["by_intent"][intent] = self._stats["by_intent"].get(intent, 0) + 1
        self._stats["by_mode"][expert_mode] = self._stats["by_mode"].get(expert_mode, 0) + 1

        self.logger.info(
            "mode=%-15s | intent=%-10s | confidence=%.3f | "
            "query_len=%d | response_len=%d",
            expert_mode,
            intent,
            confidence,
            len(query),
            len(response),
        )

    # ------------------------------------------------------------------
    # Human review queue
    # ------------------------------------------------------------------

    def add_to_review_queue(self, interaction: Dict) -> None:
        """Flag a low-confidence interaction for human review."""
        interaction["timestamp"] = datetime.now().isoformat()
        self._review_queue.append(interaction)
        self._stats["flagged_for_review"] += 1
        self.logger.warning(
            "FLAGGED FOR REVIEW | confidence=%.3f | query=%s",
            interaction.get("confidence", 0.0),
            interaction.get("query", "")[:80],
        )

    def get_review_queue(self) -> List[Dict]:
        """Return a copy of the current review queue."""
        return list(self._review_queue)

    def dismiss_review_item(self, index: int) -> bool:
        """Remove item at *index* from the queue. Returns True on success."""
        if 0 <= index < len(self._review_queue):
            self._review_queue.pop(index)
            return True
        return False

    def clear_review_queue(self) -> int:
        """Clear all items and return the count cleared."""
        count = len(self._review_queue)
        self._review_queue.clear()
        return count

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        """Return aggregate statistics for the System Reports tab."""
        total = self._stats["total_queries"]
        avg_conf = (
            self._stats["confidence_sum"] / total if total > 0 else 0.0
        )
        return {
            "total_queries": total,
            "average_confidence": round(avg_conf, 3),
            "flagged_for_review": self._stats["flagged_for_review"],
            "review_queue_size": len(self._review_queue),
            "by_intent": dict(self._stats["by_intent"]),
            "by_mode": dict(self._stats["by_mode"]),
        }

    def get_confidence_label(self, score: float) -> str:
        """Map a confidence score to a human-readable label."""
        if score >= 0.80:
            return "High"
        if score >= 0.60:
            return "Medium"
        if score >= 0.40:
            return "Low"
        return "Very Low"
