# Model Card — PawPal+ AI Pet Assistant

**Author:** Million Aboye
**Course:** AI-110, Spring 2026
**Project:** Module 4 Final — Production AI Application
**Model in use:** GPT-4o-mini (OpenAI) via agentic wrapper
**Date:** April 28, 2026

---

## System Overview

PawPal+ is a Retrieval-Augmented Generation (RAG) application that answers pet care questions using a curated knowledge base of 57 documents across five topic areas (symptoms, nutrition, training, behavior, general care). User queries are routed through a 7-step agentic pipeline: intent classification → knowledge retrieval → context assembly → OpenAI API call → confidence scoring → urgency detection → logging. Four expert personas (General, Veterinary, Dog Trainer, Cat Behavior Specialist) shape the tone and depth of each response.

---

## Intended Use

| Use Case | Supported? |
|----------|-----------|
| General pet care questions (feeding, grooming, vaccines) | Yes |
| Behavioral guidance for dogs and cats | Yes |
| Triage guidance ("should I go to the vet now?") | Partial — urgency flagging only |
| Medical diagnosis or treatment decisions | **No** |
| Exotic or non-domestic animals | No |
| Real-time emergencies | No — always defer to a vet |

---

## Limitations and Biases

### 1. Knowledge Base Scope Is Finite and Manually Curated
The system's retrieval is bounded by 57 hand-written documents. Questions about rare breeds, regional parasites, uncommon diseases (e.g., leptospirosis, blastomycosis), or exotic pets (rabbits, reptiles, birds) will either retrieve tangentially related documents or return low-confidence answers. The system cannot know what it does not have — it will not say "I don't have information on that species" unless the retriever returns nothing at all.

### 2. TF-IDF Has No Semantic Understanding
The RAG retriever uses TF-IDF cosine similarity, a bag-of-words method. It matches on shared vocabulary, not meaning. A query phrased as "my dog won't touch his bowl" may not retrieve the `appetite_loss` document because "bowl" is not in that document's term set, even though the semantic intent is identical. Paraphrased, indirect, or colloquial questions are systematically disadvantaged.

### 3. Keyword-Based Intent Classification Is Brittle
The `classify_intent()` function scores queries against four hardcoded keyword lists. A question like "Why is my dog obsessed with the water bowl?" scores both `nutrition` (water, bowl) and `behavior` (why does, obsessed) — whichever list gets more hits wins, which may not reflect the owner's actual concern. Ambiguous phrasing produces unreliable routing.

### 4. Confidence Scores Are Heuristic, Not Calibrated
The confidence formula (0.35 × retrieval quality + 0.25 × response length + 0.20 × intent clarity + 0.20 × knowledge grounding) is designed to be informative, not statistically calibrated. A score of 0.72 does not mean the answer is correct 72% of the time — it means the system had good retrieval hits and produced a substantive response. Length is not quality. A confident-sounding wrong answer can receive a high score.

### 5. GPT-4o-mini Inherits Training Biases
The underlying language model was trained predominantly on English-language text representing Western veterinary standards and pet ownership norms. Advice about vaccination schedules, diet, and behavioral expectations reflects practices common in North America and Western Europe. Pet owners in other cultural or economic contexts may receive recommendations that are locally inaccessible or culturally misaligned.

### 6. Dogs and Cats Are Privileged
The knowledge base was built specifically for dogs and cats. Multi-pet households with birds, guinea pigs, or fish receive general-mode responses that may be inapplicable. The species filter in `retrieve_by_species()` only branches on `"dog"` or `"cat"` — every other species falls through to the unfiltered retriever.

---

## Could This AI Be Misused? Prevention Strategies

**Yes — the most serious misuse risk is over-reliance in a medical context.**

A pet owner facing a genuine emergency (GDV/bloat, seizure, pale gums, labored breathing) could use PawPal+ as a substitute for calling an emergency vet, especially if the system returns a confident-sounding response that underestimates severity. Delay in these cases can be fatal.

A secondary risk is owners using the Veterinary Assistant persona to make treatment decisions — administering medications, changing doses, or choosing not to seek professional care — based on AI guidance that sounds authoritative but cannot account for the animal's individual history, physical exam findings, or lab results.

**Safeguards currently in place:**

| Risk | Mitigation |
|------|-----------|
| Missing a true emergency | `detect_urgency()` scans every response for emergency-tier language and surfaces a red alert banner in the UI |
| Over-confident wrong answers | Confidence badge displayed on every response; low-confidence responses (<0.50) are flagged for human review |
| Replacing a vet | All four expert mode system prompts explicitly prohibit diagnosis and require recommending professional consultation |
| Normalizing dangerous practices | The Dog Trainer persona explicitly refuses to endorse punishment tools (shock collars, prong collars) regardless of query |

**What additional safeguards would help:**
- Hard-block responses to queries explicitly asking for dosing or medication names
- Rate-limit the review queue so a human actually sees flagged interactions (currently in-memory only, not persisted)
- Add a disclaimer banner that is always visible, not just surfaced when urgency is detected

---

## Reflection: What Surprised Me During Reliability Testing

The confidence scoring system produced a counterintuitive result I did not anticipate: **long, fluent, off-topic responses scored higher than short, accurate, on-topic ones.**

Because 25% of the confidence score is derived from response length (the `response_substance` factor), a response that rambled for 300 words about general cat care when asked about a specific symptom could outscore a crisp 80-word answer that directly addressed the query. The length proxy was designed to catch empty or error responses, but it inadvertently rewarded verbosity. I partially mitigated this by adding the `knowledge_grounding` factor (checking whether the response echoed terminology from the retrieved documents), but the underlying tension between fluency and accuracy remains.

A second surprise: the urgency detector triggered on the phrase "it is important to see a vet" — which appears in almost every response from the Veterinary Assistant persona as routine closing advice. This meant nearly every veterinary response was tagged as `medium` urgency even when the query was benign (e.g., routine vaccination questions). The urgency keyword list had to be tightened to require more explicit alarm language ("immediately," "emergency," "right away," "do not wait") rather than any mention of a vet visit.

---

## AI Collaboration Reflection

This project was built with Claude (Anthropic) as an active collaborator throughout the design and implementation process.

### One Instance Where the AI's Suggestion Was Genuinely Helpful

When designing the confidence scoring system, I initially planned to use a single metric: the cosine similarity score from the top retrieved document. Claude pushed back on this and proposed a weighted multi-factor formula instead — combining retrieval quality, response substance, intent clarity, and knowledge grounding — with specific reasoning for each weight. The argument was that a single retrieval score would punish valid general-knowledge questions (which naturally score lower on a domain-specific retriever) while rewarding narrow queries that happen to have a perfect keyword match. The multi-factor approach produces a more stable and interpretable signal. This was a better design than my original plan, and I adopted it with minor adjustments to the weights.

### One Instance Where the AI's Suggestion Was Flawed

During the migration from Anthropic's API to OpenAI, Claude suggested running `python -m pip install openai scikit-learn numpy` to resolve a binary incompatibility between numpy and pandas. The suggestion was directionally correct but incomplete: the packages were already present in the Python 3.9 environment, so pip reported "Requirement already satisfied" and made no changes — and the error persisted. The actual fix required `--force-reinstall` on both `numpy` and `pandas` together, followed by a force-reinstall of `scikit-learn`, so that all three C-extension packages were recompiled against the same numpy ABI. Claude's initial suggestion did not account for the fact that "already installed" does not mean "built against the current numpy" — a subtle but important distinction in environments where packages were installed across Python version boundaries. I diagnosed the root cause independently and applied the correct fix.

---

## Summary

PawPal+ demonstrates that RAG and agentic workflows can meaningfully improve the relevance and reliability of AI-generated pet care guidance. It also illustrates the core tension in applied AI: a system that is genuinely useful for routine questions is also the system a worried pet owner will consult at 2 a.m. when the stakes are highest. The right response to that tension is not to make the system less capable, but to be disciplined about what it claims, transparent about its uncertainty, and clear about when a human expert is the only appropriate answer.

---

*PawPal+ is an educational project. It is not a licensed veterinary product and should not be used as the sole basis for any medical decision concerning an animal.*
