# PawPal+ — AI-Powered Pet Care Assistant

> RAG · Agentic Workflow · Multi-Expert Personas · Reliability System
>
> AI-110 Spring 2026 — Final Project

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-red)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-green)](https://platform.openai.com/)
[![Tests](https://img.shields.io/badge/tests-54%20passing-brightgreen)](#testing)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)

---

## Original Project (Modules 1–3)

PawPal started as a Streamlit-based pet care task scheduler that allowed owners to register multiple pets and manage feeding, walk, medication, and appointment tasks from a single interface. Its core capabilities included chronological scheduling, same-time conflict detection, daily recurrence, and a priority-based task suggestion engine — all backed by 14 unit tests covering sorting, conflict, and recurrence logic. The system had no AI features; it was a pure scheduling application built to practice object-oriented Python and test-driven development.

---

## What PawPal+ Adds

PawPal+ layers three new systems on top of that preserved scheduling core:

| Layer | What it adds |
|-------|-------------|
| **RAG Knowledge Base** | 57 expert-written entries across 5 categories; TF-IDF retrieval grounds every AI answer in real pet care knowledge |
| **Agentic Workflow** | 7-step pipeline: classify intent → retrieve → build context → call OpenAI → score confidence → detect urgency → log |
| **Reliability System** | Multi-factor confidence scoring, response validation, structured logging, and a human review queue |
| **Expert Modes** | Four specialised personas: General, Veterinary Assistant, Dog Trainer, Cat Behavior Specialist |

---

## Overview

PawPal+ is a full-stack AI pet care assistant built on top of the original PawPal task scheduler. It combines a curated 57-entry knowledge base with Retrieval-Augmented Generation (RAG) and a multi-step agentic workflow to answer pet health, nutrition, training, and behavior questions — grounded in real sources, scored for confidence, and always honest about when a veterinarian is the right answer.

Millions of pet owners turn to the internet for health and care advice, where they encounter outdated, unverified, or outright dangerous information. PawPal+ addresses this by anchoring every AI response to a reviewed knowledge base, making its sources visible, and clearly signalling when professional veterinary care is the only appropriate answer. It is also a complete, reproducible example of how RAG and agentic patterns apply to a real-world care domain.

---

## Features

| Feature | Details |
|---------|---------|
| **RAG Retrieval** | TF-IDF cosine similarity across 57 expert-written knowledge base entries in 5 topic areas |
| **Agentic Workflow** | 7-step pipeline: intent classification → retrieval → context assembly → OpenAI call → confidence scoring → urgency detection → logging |
| **4 Expert Personas** | General Pet Care, Veterinary Assistant, Certified Dog Trainer, Cat Behavior Specialist |
| **Confidence Scoring** | 4-factor weighted formula; responses below threshold auto-queued for human review |
| **Urgency Detection** | Keyword scanning flags emergency, high, and medium urgency with visible UI alerts |
| **Pet Scheduler** | Original multi-pet task scheduler with conflict detection and daily recurrence |
| **System Reports** | Live reliability stats, review queue management, knowledge base status |
| **54 Tests** | Full pytest suite covering scheduling logic, RAG retrieval, confidence scoring, and agent workflow |

---

## Demo

<video src="assets/pawpal-demo.mp4" controls width="100%"></video>

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                    PawPal+ System Architecture                    ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                    ║
║   [User] ──► [Streamlit Web UI  (app.py)]                        ║
║                        │                                          ║
║          ┌─────────────┴──────────────────┐                      ║
║          ▼                                ▼                       ║
║  [PetCareSystem]              [PawPalAgent (ai_agent.py)]        ║
║  (scheduler,                   1. classify_intent()              ║
║   conflicts,                   2. RAGRetriever.retrieve()        ║
║   recurrence)                  3. Build context + expert prompt  ║
║                                4. OpenAI API call                 ║
║                                5. compute_confidence()            ║
║                                6. detect_urgency()                ║
║                                7. log_interaction()               ║
║                                        │                          ║
║              ┌─────────────────────────┼──────────────┐          ║
║              ▼                         ▼               ▼          ║
║  [RAGRetriever]         [OpenAI API]      [ReliabilitySystem]    ║
║  TF-IDF index           gpt-4o-mini       · confidence score     ║
║          │                                · validate response     ║
║          ▼                                · log file              ║
║  [Knowledge Base JSON]                    · human review queue    ║
║  · symptoms_guide.json                                            ║
║  · nutrition_guide.json                                           ║
║  · training_guide.json                                            ║
║  · behavior_guide.json                                            ║
║  · general_care.json                                              ║
╚══════════════════════════════════════════════════════════════════╝

Data Flow:
  User Input
    → Intent Classification (keyword frequency scoring, no API call)
    → Knowledge Retrieval   (TF-IDF cosine similarity, top-k docs)
    → Context Assembly      (retrieved docs + pet profile + history)
    → OpenAI Generation     (expert-mode system prompt + context)
    → Confidence Scoring    (4-factor weighted formula)
    → Urgency Detection     (keyword scan of generated response)
    → Log + Optional Review Queue
    → Structured Response   → Streamlit UI
```

![PawPal+ System Architecture Diagram](assets/PetCareSystem%20User-2026-04-29-053150.png)

### How the architecture works

The Streamlit UI is the single entry point and splits into two independent paths. The left path handles the original pet scheduler — task creation, conflict detection, and recurrence — using the preserved `PetCareSystem` class with no AI involvement. The right path is the agentic AI pipeline: every user question passes through `PawPalAgent`, which first classifies intent locally using keyword scoring (no API call), then queries `RAGRetriever` for the top-4 matching knowledge base passages using TF-IDF cosine similarity, and assembles a context-enriched prompt before calling the OpenAI API. The response comes back to `ReliabilitySystem`, which scores confidence, scans for urgency keywords, and logs the interaction — then everything is returned to the UI as a structured dict so the display layer can show the answer, confidence badge, urgency alert, and source citations independently.

---

## Project Structure

```
pawpal-plus/
├── app.py                     # Streamlit UI — 3 tabs: Scheduler, AI, Reports
├── pawpal_system.py           # Original scheduling core (preserved from Modules 1–3)
├── ai_agent.py                # PawPalAgent — agentic workflow + expert personas
├── rag_retriever.py           # RAGRetriever — TF-IDF knowledge base retrieval
├── reliability.py             # ReliabilitySystem — confidence, logging, review queue
├── main.py                    # Original CLI entry point (preserved)
├── knowledge_base/
│   ├── symptoms_guide.json    # 15 entries — symptoms, urgency levels, vet guidance
│   ├── nutrition_guide.json   # 10 entries — feeding, diet, toxic foods
│   ├── training_guide.json    # 10 entries — positive reinforcement methods
│   ├── behavior_guide.json    # 12 entries — feline/canine behavior explanations
│   └── general_care.json     # 10 entries — vaccines, dental, grooming, wellness
├── tests/
│   ├── test_pawpal.py         # 14 scheduling tests (preserved)
│   └── test_ai_features.py    # 40 AI feature tests
├── logs/
│   └── pawpal_ai.log          # Auto-generated structured interaction log
├── model_card.md              # Model card — limitations, ethics, reflection
├── requirements.txt
├── .env.example               # Environment variable template
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 1. Clone and enter the project

```bash
git clone https://github.com/Mig1a/pawpal-plus.git
cd pawpal-plus
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and set:

```env
OPENAI_API_KEY=sk-...your-key-here...
```

### 5. Run

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`.

---

## Usage

1. **Scheduler tab** — Register a pet (name, species, breed, age, weight) and add tasks (feeding, walk, medication, appointment). Generate today's schedule with conflict warnings.
2. **AI Pet Assistant tab** — Choose an expert mode, optionally link a pet profile, and ask any pet care question. The response shows the answer, confidence score, urgency alert (if any), and the knowledge base sources that informed it.
3. **System Reports tab** — View reliability statistics, the human review queue, and knowledge base status.

### Sample Interactions

**Symptom triage — Veterinary Assistant mode**
```
Q: My dog keeps scratching his ears and shaking his head.
   There's dark brown discharge and a bad smell.

A: These symptoms are classic signs of otitis externa (ear infection).
   Dark brown waxy buildup often indicates yeast involvement.

   What to do now:
   • Gently clean the visible outer ear with a vet-approved solution
   • Do not insert anything into the ear canal
   • Schedule a vet appointment within 3–4 days; sooner if the dog is in pain

   🔴 Please schedule a veterinary appointment.

   Confidence: 78% (High) | Source: Symptoms Guide — Ear Scratching and Head Shaking
```

**Training plan — Dog Trainer mode**
```
Q: Create a training plan for my 8-week-old puppy.

A: Weeks 1–2 (Foundation): name recognition, sit, handling exercises
   Weeks 3–4 (Core Commands): stay, down, leave it, crate intro
   Weeks 5–8 (Generalisation): proof commands, new environments,
   loose-leash walking, socialization outings

   Confidence: 81% (High) | Source: Training Guide — 8-Week Puppy Program
```

**Urgent concern — General mode**
```
Q: My cat stopped eating yesterday and is hiding under the bed.
   She is usually very social.

A: A social cat that suddenly hides and refuses food is showing two
   serious warning signs together. Cats risk hepatic lipidosis
   (fatty liver disease) after 24–48 hours without eating.

   Do not wait more than 24 hours — schedule a vet appointment today.

   🔴 Urgent — please seek veterinary care.

   Confidence: 74% (Medium) | Sources: Symptoms Guide, Behavior Guide
```

---

## Confidence Scoring

Every response is scored using a 4-factor weighted formula:

```
confidence = 0.35 × retrieval_quality    (cosine similarity of top retrieved doc)
           + 0.25 × response_substance   (word count proxy, 150 words = 1.0)
           + 0.20 × intent_clarity       (specific intent scores higher than general)
           + 0.20 × knowledge_grounding  (fraction of retrieved tags appearing in response)
```

Responses below the 0.50 threshold are automatically added to the human review queue and displayed with a "Low Confidence" badge.

---

## Expert Modes

| Mode | Icon | Best for |
|------|------|----------|
| General Pet Care | 🐾 | Broad, everyday questions across species |
| Veterinary Assistant | 🩺 | Symptom triage, when to seek care |
| Certified Dog Trainer | 🐕 | Positive reinforcement training plans |
| Cat Behavior Specialist | 🐈 | Feline ethology, enrichment, multi-cat dynamics |

---

## Testing & Reliability

PawPal+ uses four distinct mechanisms to prove it works, not just seem like it does:

| Mechanism | Implementation | Result |
|-----------|---------------|--------|
| **Automated tests** | 54 pytest tests covering RAG retrieval, confidence scoring, urgency detection, intent classification, expert personas, and the full agent pipeline — all using mocked OpenAI calls so no API key is needed to run them | 54 / 54 pass |
| **Confidence scoring** | Every response is scored 0–1 using a 4-factor weighted formula (retrieval quality, response substance, intent clarity, knowledge grounding). Scores below 0.50 are labelled "Low Confidence" in the UI and automatically queued for human review | Live on every query |
| **Logging & error handling** | Every interaction is appended to `logs/pawpal_ai.log` with timestamp, query, confidence score, expert mode, and intent. The agent catches API failures, missing keys, and unavailable knowledge base with structured error returns rather than crashes | Verified in `test_log_file_created` |
| **Human review queue** | Low-confidence responses are automatically added to an in-app review queue visible in the System Reports tab. A reviewer can inspect the query, response, and confidence score, then dismiss or escalate | Verified in `test_low_confidence_flags_for_review` |

```bash
# Run the full suite
python -m pytest tests/ -v

# Run only AI feature tests
python -m pytest tests/test_ai_features.py -v

# Run only scheduling tests
python -m pytest tests/test_pawpal.py -v
```

### Results — 54 tests, 54 pass

| Suite | Tests | Covers |
|-------|-------|--------|
| **RAGRetriever** | 9 | KB load, top-k retrieval, relevance ranking, species filter, score bounds |
| **ReliabilitySystem** | 15 | Confidence range, urgency levels, validation, logging, review queue |
| **Intent Classification** | 5 | All 4 intent categories + general fallback |
| **Expert Modes** | 4 | Required keys, prompt content, persona-specific guardrails |
| **PawPalAgent** | 6 | Return schema, confidence bounds, no-key error, urgency, step trace, review flagging |
| **Scheduler (original)** | 14 | Task basics, sorting, recurrence, conflict detection |
| **Total** | **54** | |

### What worked

The mock-based approach for testing the AI agent worked exactly as intended — by injecting a fake OpenAI client that returns a controlled response string, the full 7-step pipeline could be exercised without any API calls or costs. The species filter in the retriever also worked better than expected: queries about cats consistently returned cat-specific documents even when symptom vocabulary overlapped with dog entries.

### What didn't work (at first)

The urgency detector initially triggered on the phrase "it is important to see a vet" — which appears as routine closing advice in nearly every Veterinary Assistant response. This caused benign queries (routine vaccination questions) to be tagged `medium` urgency, flooding the UI with unnecessary alerts. The keyword list had to be tightened to require more explicit alarm language ("immediately," "do not wait," "emergency vet") rather than any mention of veterinary care.

A second issue was the confidence formula rewarding long responses regardless of accuracy. A verbose but off-topic response could outscore a short, precise one because 25% of the formula was raw word count. Adding the `knowledge_grounding` factor (checking whether response text echoed retrieved document tags) partially corrected this, though the length proxy remains an imperfect signal.

### What I learned

Testing forced me to make the system's implicit assumptions explicit. Writing the confidence tests first revealed that I had no clear definition of what "high confidence" should actually mean — which led to designing the 4-factor formula deliberately rather than using a single arbitrary metric. The process of making something testable made the design better.

---

## Design Decisions

**TF-IDF over vector embeddings** — Runs entirely locally with no embedding API cost or latency. Handles pet care domain vocabulary well since symptoms, breed names, and procedure terms are highly distinctive tokens. A production system would upgrade to sentence embeddings in FAISS or ChromaDB for better semantic recall.

**Expert modes as system prompt variants** — Four distinct personas are implemented as system prompt swaps, adding no inference cost while meaningfully shaping tone, depth, and guardrails.

**gpt-4o-mini as default** — Fast (under 2 seconds), cost-efficient for an interactive demo. Override with `PAWPAL_AI_MODEL=gpt-4o` in `.env` for more demanding questions.

**Keyword intent classification** — Zero API calls, deterministic, and explainable. Misses paraphrase variations but is sufficient for routing within a well-bounded domain.

**Multi-factor confidence** — Single retrieval score would penalise valid general questions. The 4-factor formula produces a more stable signal across query types.

### Trade-offs

| Decision | Benefit | Trade-off |
|----------|---------|-----------|
| TF-IDF retrieval | Free, local, fast | Lower semantic recall vs. embeddings |
| In-memory review queue | Simple, zero deps | Lost on app restart |
| Keyword intent classifier | Explainable, instant | Brittle to paraphrasing |
| gpt-4o-mini default | Cheap and fast | Less nuanced on complex medical questions |
| JSON knowledge base | Human-editable, version-controlled | Not scalable past ~1000 docs |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI API key |
| `PAWPAL_AI_MODEL` | `gpt-4o-mini` | Model override (`gpt-4o`, `gpt-4-turbo`) |
| `PAWPAL_CONFIDENCE_THRESHOLD` | `0.50` | Below this, responses are flagged for review |
| `PAWPAL_LOG_FILE` | `logs/pawpal_ai.log` | Path for structured interaction log |

---

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t pawpal-plus .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key pawpal-plus
```

---

## Roadmap

- [ ] Sentence-transformer embeddings (FAISS/ChromaDB) for semantic retrieval
- [ ] SQLite persistence for review queue and interaction history
- [ ] Breed-specific health risk profiles in the knowledge base
- [ ] Streamlit Community Cloud one-click deploy
- [ ] Voice input via Whisper
- [ ] Multi-language support

---

## Reflection

Building PawPal+ changed how I think about what an AI system actually is. Before this project, I thought of AI primarily as a model — you send text in, you get text out. After building the full pipeline, I understand that the model is only one component, and often not the most important one. The quality of the knowledge base, the precision of the retrieval step, and the honesty of the confidence signal matter as much as the language model itself. A well-grounded mediocre model beats a powerful model that hallucinates confidently.

The reliability system was the most valuable part to build. Designing the confidence scoring formula forced me to answer a hard question: what does it actually mean for an AI answer to be trustworthy? I could not just say "it feels right" — I had to decompose trust into measurable signals (retrieval quality, response substance, intent clarity, knowledge grounding) and justify the weight of each. That kind of disciplined thinking about AI quality is something I will carry into every future project.

The biggest shift in my problem-solving approach was learning to distrust the happy path. My first instinct was to test that the system works when everything goes right. But the most important tests were the edge cases: what happens when confidence is low, when urgency is high, when the knowledge base has nothing relevant, when the API key is missing? Robust AI is not about making the system succeed under ideal conditions — it is about making it fail gracefully and honestly under real ones.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*PawPal+ · AI-110 Spring 2026 · Built on PawPal project 2*
