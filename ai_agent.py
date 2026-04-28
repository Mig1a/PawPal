"""
PawPal+ AI Agent
Multi-step agentic workflow: classify intent → retrieve knowledge → generate
response with OpenAI → score confidence → detect urgency → log.
"""

import os
from typing import Dict, List, Optional

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from rag_retriever import RAGRetriever
from reliability import ReliabilitySystem


# ---------------------------------------------------------------------------
# Expert mode definitions — each has a distinct system prompt and persona
# ---------------------------------------------------------------------------
EXPERT_MODES: Dict[str, Dict] = {
    "general": {
        "name": "General Pet Care Guide",
        "icon": "🐾",
        "description": "Balanced, friendly advice for all pet owners",
        "system_prompt": (
            "You are PawPal, a friendly and knowledgeable pet care assistant. "
            "You give clear, practical, accurate advice about dogs, cats, and other common pets. "
            "You draw on the provided knowledge base context to answer questions. "
            "Always be warm and supportive. For health concerns, always recommend consulting "
            "a veterinarian for anything beyond routine care. "
            "Keep responses concise and well-structured, using bullet points when helpful. "
            "Do not diagnose conditions — only provide general guidance and information."
        ),
    },
    "veterinary": {
        "name": "Veterinary Assistant",
        "icon": "🩺",
        "description": "Clinical, evidence-based medical guidance",
        "system_prompt": (
            "You are a knowledgeable veterinary assistant with deep expertise in small animal "
            "medicine. You provide clinically accurate, evidence-based information about pet "
            "health symptoms, disease processes, and when veterinary care is needed. "
            "Use the provided knowledge base context as your primary reference. "
            "Speak in a professional but approachable tone. Be clear about urgency levels — "
            "distinguish between 'monitor at home', 'schedule an appointment', and 'go to the "
            "emergency vet immediately'. Always remind owners that you are an assistant and "
            "cannot replace an in-person veterinary examination for diagnosis or treatment. "
            "When symptoms suggest a serious condition, state this clearly and directly."
        ),
    },
    "dog_trainer": {
        "name": "Certified Dog Trainer",
        "icon": "🐕",
        "description": "Positive reinforcement training expert",
        "system_prompt": (
            "You are a certified professional dog trainer with expertise in positive reinforcement "
            "and behavioral science. You help dog owners build better relationships with their dogs "
            "through force-free, science-based training methods. "
            "Use the provided knowledge base context to ground your advice. "
            "Structure training advice in clear steps. Explain the 'why' behind your "
            "recommendations — understanding behavior science helps owners be more consistent. "
            "Be encouraging and realistic about timelines — good training takes time and "
            "consistency. Never recommend punishment-based methods, prong collars, shock "
            "collars, or choke chains. If a question is beyond basic training and suggests "
            "a serious behavioral problem, recommend consultation with a CPDT-KA certified "
            "trainer or veterinary behaviorist."
        ),
    },
    "cat_behavior": {
        "name": "Cat Behavior Specialist",
        "icon": "🐈",
        "description": "Feline ethology and behavior expert",
        "system_prompt": (
            "You are a certified cat behavior specialist with deep knowledge of feline ethology, "
            "cognition, and environmental enrichment. You help cat owners understand their cats' "
            "needs, behaviors, and communication — from the cat's perspective, not the owner's. "
            "Use the provided knowledge base context as your primary reference. "
            "Avoid anthropomorphizing cat behaviors — explain them in terms of feline natural "
            "history and instinct. Emphasize environmental enrichment, choice and agency for "
            "the cat, and multi-cat household dynamics where relevant. "
            "Be empathetic toward both the owner's frustration and the cat's needs. "
            "For medical concerns related to behavior changes, always recommend veterinary "
            "evaluation to rule out underlying health causes before assuming behavioral origin."
        ),
    },
}


# ---------------------------------------------------------------------------
# Intent classification — keyword-based, no API call needed
# ---------------------------------------------------------------------------
_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "symptom": [
        "sick", "ill", "symptom", "pain", "hurts", "hurt", "injury", "injured",
        "not eating", "stopped eating", "vomiting", "vomit", "diarrhea", "scratch",
        "scratching", "limp", "limping", "cough", "coughing", "sneeze", "sneezing",
        "lethargy", "lethargic", "tired", "discharge", "swollen", "bleeding",
        "lump", "mass", "trembling", "shaking", "seizure", "pale", "gums",
        "emergency", "urgent", "breathing", "thirsty", "drinking more", "weight loss",
        "losing weight", "blood", "ear", "itching", "itch",
    ],
    "training": [
        "train", "training", "teach", "teaching", "command", "sit", "stay",
        "come", "fetch", "potty", "house train", "crate", "leash", "pulling",
        "bark", "barking", "bite", "biting", "jump", "jumping", "aggressive",
        "obedience", "recall", "heel", "behavior problem", "puppy class",
        "socialization", "socialize", "reward", "treat training", "clicker",
    ],
    "nutrition": [
        "feed", "feeding", "food", "diet", "eat", "eating", "portion", "weight",
        "treat", "treats", "kibble", "raw food", "wet food", "dry food",
        "nutrition", "supplement", "calories", "overweight", "underweight",
        "obese", "obesity", "toxic", "poisonous", "can dogs eat", "can cats eat",
        "how much to feed", "meal", "water", "hydration",
    ],
    "behavior": [
        "why does", "why is", "behavior", "habit", "strange", "weird", "normal",
        "body language", "communication", "social", "play", "anxiety", "fear",
        "stress", "aggression", "aggressive", "separation", "hiding", "knocking",
        "kneading", "scratching furniture", "zoomies", "grass", "biting",
        "growling", "hissing", "resource guarding", "attention seeking",
    ],
}


def classify_intent(query: str) -> str:
    """
    Classify the user query into one of five intent categories using keyword
    frequency scoring. Returns 'general' when no category dominates.
    """
    query_lower = query.lower()
    scores: Dict[str, int] = {}

    for intent, keywords in _INTENT_KEYWORDS.items():
        scores[intent] = sum(1 for kw in keywords if kw in query_lower)

    if not scores or max(scores.values()) == 0:
        return "general"

    return max(scores, key=lambda k: scores[k])


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

class PawPalAgent:
    """
    Multi-step agentic workflow for pet care Q&A.

    Workflow:
      1. Classify intent from query text
      2. Retrieve top-k relevant knowledge base passages (RAG)
      3. Select expert-mode system prompt
      4. Build a context-enriched prompt (retrieved docs + pet info + history)
      5. Call the Claude API
      6. Score confidence via ReliabilitySystem
      7. Detect urgency level from response
      8. Log the interaction
      9. Flag for human review if confidence < threshold
     10. Return a structured result dict
    """

    DEFAULT_MODEL = os.getenv("PAWPAL_AI_MODEL", "gpt-4o-mini")
    CONFIDENCE_THRESHOLD = float(os.getenv("PAWPAL_CONFIDENCE_THRESHOLD", "0.50"))

    def __init__(self, rag: RAGRetriever, reliability: ReliabilitySystem):
        self.rag = rag
        self.reliability = reliability
        self._client: Optional[object] = None

        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key:
                self._client = openai.OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """True when the OpenAI client and RAG index are both available."""
        return self._client is not None and self.rag.is_available()

    def get_status(self) -> Dict:
        return {
            "openai_available": OPENAI_AVAILABLE,
            "api_key_set": bool(os.getenv("OPENAI_API_KEY")),
            "rag_available": self.rag.is_available(),
            "ready": self.is_ready(),
            "model": self.DEFAULT_MODEL,
        }

    def run(
        self,
        query: str,
        pet_info: Optional[Dict] = None,
        expert_mode: str = "general",
        chat_history: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Execute the full agentic workflow and return a structured result.

        Return dict keys:
          response      str    — the AI-generated answer
          confidence    float  — 0.0–1.0 trust score
          confidence_label str — 'High' / 'Medium' / 'Low' / 'Very Low'
          sources       list   — retrieved document metadata
          intent        str    — classified query intent
          urgency       dict   — level + message from urgency detection
          expert_mode   str    — display name of the selected mode
          steps         list   — agentic step trace for transparency
          error         str    — non-empty if something went wrong
        """
        steps: List[str] = []

        # ── Guard: agent not ready ──────────────────────────────────────
        if not self._client:
            reason = (
                "OPENAI_API_KEY is not set."
                if OPENAI_AVAILABLE
                else "openai package is not installed (pip install openai)."
            )
            return self._error_result(f"AI assistant unavailable: {reason}", expert_mode)

        if not self.rag.is_available():
            return self._error_result(
                "Knowledge base not loaded. Check the knowledge_base/ directory.", expert_mode
            )

        # ── Step 1: Classify intent ──────────────────────────────────────
        intent = classify_intent(query)
        steps.append(f"Step 1 — Intent classified as: {intent}")

        # ── Step 2: Retrieve relevant knowledge ──────────────────────────
        species_hint = (pet_info or {}).get("species", "").lower()
        if species_hint in ("dog", "cat"):
            retrieved = self.rag.retrieve_by_species(query, species_hint, k=4)
        else:
            retrieved = self.rag.retrieve(query, k=4)

        steps.append(f"Step 2 — Retrieved {len(retrieved)} knowledge base passage(s)")

        # ── Step 3: Build context ────────────────────────────────────────
        mode_cfg = EXPERT_MODES.get(expert_mode, EXPERT_MODES["general"])
        system_prompt = mode_cfg["system_prompt"]

        context_sections = []
        for i, doc in enumerate(retrieved, 1):
            header = f"[Source {i}: {doc['source']} — {doc['title']}]"
            context_sections.append(f"{header}\n{doc['content']}")
        context_block = (
            "\n\n".join(context_sections)
            if context_sections
            else "No specific knowledge base entries were retrieved for this query."
        )

        pet_block = self._format_pet_info(pet_info)
        steps.append("Step 3 — Context assembled from retrieved passages and pet profile")

        # ── Step 4: Build messages ───────────────────────────────────────
        messages: List[Dict] = []

        # Include recent chat history (last 3 turns = 6 messages)
        if chat_history:
            messages.extend(chat_history[-6:])

        user_content = (
            f"## Relevant Knowledge Base Context\n\n{context_block}\n\n"
            f"{pet_block}"
            f"## User Question\n\n{query}\n\n"
            "Please answer the question using the provided context. "
            "If this involves health concerns, clearly state when a veterinarian should be consulted."
        )
        messages.append({"role": "user", "content": user_content})

        # ── Step 5: Call OpenAI API ──────────────────────────────────────
        try:
            # OpenAI puts the system prompt as the first message in the array
            openai_messages = [{"role": "system", "content": system_prompt}] + messages
            api_response = self._client.chat.completions.create(
                model=self.DEFAULT_MODEL,
                max_tokens=1024,
                messages=openai_messages,
            )
            response_text = api_response.choices[0].message.content
            steps.append(
                f"Step 4 — OpenAI ({self.DEFAULT_MODEL}) generated a "
                f"{len(response_text.split())}-word response"
            )
        except Exception as exc:
            return self._error_result(f"OpenAI API error: {exc}", expert_mode)

        # ── Step 6: Score confidence ──────────────────────────────────────
        confidence = self.reliability.compute_confidence(
            query=query,
            retrieved_docs=retrieved,
            response_text=response_text,
            intent=intent,
        )
        confidence_label = self.reliability.get_confidence_label(confidence)
        steps.append(f"Step 5 — Confidence scored: {confidence:.2f} ({confidence_label})")

        # ── Step 7: Detect urgency ────────────────────────────────────────
        urgency = self.reliability.detect_urgency(response_text)
        steps.append(f"Step 6 — Urgency level: {urgency['level']}")

        # ── Step 8: Log interaction ───────────────────────────────────────
        self.reliability.log_interaction(
            query=query,
            response=response_text,
            confidence=confidence,
            expert_mode=expert_mode,
            intent=intent,
        )

        # ── Step 9: Flag for human review if confidence is low ───────────
        if confidence < self.CONFIDENCE_THRESHOLD:
            self.reliability.add_to_review_queue(
                {
                    "query": query,
                    "response": response_text,
                    "confidence": confidence,
                    "intent": intent,
                    "expert_mode": expert_mode,
                }
            )
            steps.append(
                f"Step 7 — Flagged for human review (confidence {confidence:.2f} < {self.CONFIDENCE_THRESHOLD:.2f})"
            )

        return {
            "response": response_text,
            "confidence": confidence,
            "confidence_label": confidence_label,
            "sources": [
                {
                    "title": d["title"],
                    "source": d["source"],
                    "score": d["score"],
                    "urgency": d.get("urgency", "low"),
                    "when_to_see_vet": d.get("when_to_see_vet", ""),
                }
                for d in retrieved
            ],
            "intent": intent,
            "urgency": urgency,
            "expert_mode": mode_cfg["name"],
            "steps": steps,
            "error": "",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_pet_info(self, pet_info: Optional[Dict]) -> str:
        if not pet_info:
            return ""
        lines = ["## Pet Profile\n"]
        for label, key in [
            ("Name", "name"), ("Species", "species"), ("Breed", "breed"),
            ("Age", "age"), ("Weight", "weight"), ("Notes", "notes"),
        ]:
            val = pet_info.get(key)
            if val:
                suffix = " years" if key == "age" else (" lbs" if key == "weight" else "")
                lines.append(f"- **{label}:** {val}{suffix}")
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _error_result(message: str, expert_mode: str) -> Dict:
        return {
            "response": message,
            "confidence": 0.0,
            "confidence_label": "N/A",
            "sources": [],
            "intent": "unknown",
            "urgency": {"level": "none", "message": ""},
            "expert_mode": EXPERT_MODES.get(expert_mode, EXPERT_MODES["general"])["name"],
            "steps": [],
            "error": message,
        }


# ---------------------------------------------------------------------------
# Convenience accessor for mode metadata
# ---------------------------------------------------------------------------

def get_expert_mode_options() -> List[str]:
    """Return mode keys in display order."""
    return list(EXPERT_MODES.keys())


def get_mode_display_name(mode_key: str) -> str:
    cfg = EXPERT_MODES.get(mode_key, EXPERT_MODES["general"])
    return f"{cfg['icon']} {cfg['name']}"
