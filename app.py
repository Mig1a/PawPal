"""
PawPal+ — AI-Powered Pet Care Assistant
Streamlit web application combining smart task scheduling (original PawPal)
with RAG-backed, expert-mode AI advice (upgraded for GMU AI-110 final project).
"""

import os
import streamlit as st
from datetime import datetime, date, time

# ---------------------------------------------------------------------------
# Optional: load .env for local development
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Try importing AI components — graceful degradation if not installed
# ---------------------------------------------------------------------------
AI_IMPORT_ERROR = ""
try:
    from rag_retriever import RAGRetriever
    from reliability import ReliabilitySystem
    from ai_agent import PawPalAgent, get_expert_mode_options, get_mode_display_name, EXPERT_MODES
    AI_MODULES_AVAILABLE = True
except ImportError as _e:
    AI_MODULES_AVAILABLE = False
    AI_IMPORT_ERROR = str(_e)

# ---------------------------------------------------------------------------
# Core scheduling system (always available)
# ---------------------------------------------------------------------------
from pawpal_system import (
    Pet, PetCareSystem,
    FeedingTask, WalkTask, MedicationTask, AppointmentTask,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PawPal+",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state — initialised once, survives every Streamlit rerun
# ---------------------------------------------------------------------------
if "system" not in st.session_state:
    st.session_state.system = PetCareSystem()
if "active_pet_id" not in st.session_state:
    st.session_state.active_pet_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # Claude message format
if "ai_conversation" not in st.session_state:
    st.session_state.ai_conversation = []       # display format (role + content + metadata)

# Initialise AI components once (expensive: TF-IDF index build)
if AI_MODULES_AVAILABLE:
    if "rag" not in st.session_state:
        kb_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
        try:
            st.session_state.rag = RAGRetriever(kb_dir)
        except Exception as _exc:
            st.session_state.rag = None
            st.session_state._rag_error = str(_exc)

    if "reliability" not in st.session_state:
        log_path = os.getenv("PAWPAL_LOG_FILE", "logs/pawpal_ai.log")
        st.session_state.reliability = ReliabilitySystem(log_file=log_path)

    if "agent" not in st.session_state:
        rag = st.session_state.get("rag")
        rel = st.session_state.get("reliability")
        if rag and rel:
            st.session_state.agent = PawPalAgent(rag, rel)
        else:
            st.session_state.agent = None

system: PetCareSystem = st.session_state.system

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 🐾 PawPal+")
st.caption(
    "AI-Powered Pet Care Assistant — Smart Scheduling + Expert Advice | "
    "GMU AI-110 Final Project"
)
st.divider()

# ---------------------------------------------------------------------------
# Navigation tabs
# ---------------------------------------------------------------------------
tab_scheduler, tab_ai, tab_reports = st.tabs(
    ["📅  Scheduler", "🤖  AI Pet Assistant", "📊  System Reports"]
)


# ============================================================
# TAB 1 — SCHEDULER  (original PawPal functionality, preserved)
# ============================================================
with tab_scheduler:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Pet registration ──────────────────────────────────────
    with col_left:
        st.subheader("🐕 Pet Profile")

        with st.form("pet_form"):
            c1, c2 = st.columns(2)
            with c1:
                owner_name = st.text_input("Owner name", value="Jordan Rivera")
                pet_name   = st.text_input("Pet name",   value="Buddy")
                species    = st.selectbox("Species", ["Dog", "Cat", "Bird", "Other"])
            with c2:
                breed  = st.text_input("Breed",  value="Golden Retriever")
                age    = st.number_input("Age (years)", min_value=0, max_value=30, value=3)
                weight = st.number_input("Weight (lbs)", min_value=0.1, max_value=300.0, value=65.0)
            notes = st.text_area("Notes (allergies, diet, etc.)", value="")
            add_pet_btn = st.form_submit_button("➕ Add Pet", use_container_width=True)

        if add_pet_btn:
            new_id = len(system._pets) + 1
            pet = Pet(
                pet_id=new_id, name=pet_name, species=species, breed=breed,
                age=age, weight=weight, owner_name=owner_name, notes=notes,
            )
            system.add_pet(pet)
            st.session_state.active_pet_id = new_id
            st.success(f"Added **{pet_name}** ({species}) for {owner_name}.")

        if system._pets:
            pet_options = {p._name: p._pet_id for p in system._pets}
            selected_name = st.selectbox("Active pet", list(pet_options.keys()))
            st.session_state.active_pet_id = pet_options[selected_name]

            # Show current pet summary
            active = next((p for p in system._pets if p._pet_id == st.session_state.active_pet_id), None)
            if active:
                with st.container(border=True):
                    st.markdown(f"**{active._name}** · {active._species} · {active._breed}")
                    st.caption(f"Age: {active._age} yr · Weight: {active._weight} lbs · Owner: {active._owner_name}")
                    if active._notes:
                        st.caption(f"Notes: {active._notes}")

    # ── Task creation ─────────────────────────────────────────
    with col_right:
        st.subheader("📝 Add a Task")

        task_category = st.selectbox(
            "Task type", ["Feeding", "Walk", "Medication", "Appointment"]
        )

        with st.form("task_form"):
            title        = st.text_input("Title", value="Morning feeding")
            description  = st.text_area("Description", value="", height=60)
            due_date     = st.date_input("Due date", value=date.today())
            due_time_val = st.time_input("Due time", value=time(8, 0))
            priority     = st.slider("Priority (1 = low, 10 = high)", 1, 10, 5)
            recurring    = st.checkbox("Recurring?")
            recurrence_pattern = (
                st.text_input("Recurrence pattern (daily / weekly)", value="daily")
                if recurring else ""
            )

            if task_category == "Feeding":
                food_type    = st.text_input("Food type",    value="Dry kibble")
                portion_size = st.text_input("Portion size", value="2 cups")
                diet_notes   = st.text_input("Diet notes",   value="")
            elif task_category == "Walk":
                duration      = st.number_input("Duration (min)", min_value=1, max_value=240, value=30)
                distance_goal = st.number_input("Distance goal (miles)", min_value=0.1, max_value=20.0, value=1.0)
                location      = st.text_input("Location", value="Neighbourhood park")
            elif task_category == "Medication":
                medication_name = st.text_input("Medication name", value="")
                dosage          = st.text_input("Dosage",          value="")
                instructions    = st.text_input("Instructions",    value="")
                refill_date     = st.date_input("Refill date", value=date.today())
            elif task_category == "Appointment":
                location         = st.text_input("Location",         value="")
                provider_name    = st.text_input("Provider name",    value="")
                appointment_type = st.text_input("Appointment type", value="Wellness Exam")
                contact_info     = st.text_input("Contact info",     value="")
                reminder_hour    = st.number_input("Reminder hour (0–23)", min_value=0, max_value=23, value=8)

            add_task_btn = st.form_submit_button("➕ Add Task", use_container_width=True)

        if add_task_btn:
            if st.session_state.active_pet_id is None:
                st.warning("Please add a pet first.")
            else:
                due_dt = datetime.combine(due_date, due_time_val)
                kwargs = dict(
                    task_id=0, title=title, category=task_category,
                    description=description, due_time=due_dt,
                    priority=priority, status="pending",
                    recurring=recurring, recurrence_pattern=recurrence_pattern,
                    pet_id=st.session_state.active_pet_id,
                )
                if task_category == "Feeding":
                    task = FeedingTask(**kwargs, food_type=food_type,
                                       portion_size=portion_size, diet_notes=diet_notes)
                elif task_category == "Walk":
                    task = WalkTask(**kwargs, duration=duration,
                                    distance_goal=distance_goal, location=location)
                elif task_category == "Medication":
                    task = MedicationTask(**kwargs, medication_name=medication_name,
                                          dosage=dosage, instructions=instructions,
                                          refill_date=refill_date)
                elif task_category == "Appointment":
                    reminder_dt = datetime.combine(due_date, time(reminder_hour, 0))
                    task = AppointmentTask(**kwargs, location=location,
                                           provider_name=provider_name,
                                           appointment_type=appointment_type,
                                           contact_info=contact_info,
                                           reminder_time=reminder_dt)
                system.add_task_to_pet(st.session_state.active_pet_id, task)
                st.success(f"Task **'{title}'** added.")

    # ── Today's schedule ──────────────────────────────────────
    st.divider()
    st.subheader("📅 Today's Schedule")

    if st.button("🔄 Generate Schedule", use_container_width=False):
        if not system._pets:
            st.warning("No pets in the system yet.")
        else:
            now = datetime.now()
            for pet in system._pets:
                tasks = system.view_pet_schedule(pet._pet_id, date.today())
                tasks_sorted = system._scheduler.sort_by_time(tasks)

                st.markdown(f"#### {pet._name} — {pet._breed}")

                if not tasks_sorted:
                    st.info("No tasks scheduled for today.")
                    continue

                # Summary table
                rows = []
                for t in tasks_sorted:
                    if t._status == "complete":
                        status_lbl = "✅ Complete"
                    elif t.is_overdue(now):
                        status_lbl = "🔴 Overdue"
                    else:
                        status_lbl = "🕐 Pending"
                    recur_lbl = (
                        t._recurrence_pattern.capitalize()
                        if t._recurring and t._recurrence_pattern else "—"
                    )
                    rows.append({
                        "Time":     t._due_time.strftime("%I:%M %p"),
                        "Task":     t._title,
                        "Type":     t._category,
                        "Priority": t._priority,
                        "Recurs":   recur_lbl,
                        "Status":   status_lbl,
                    })
                st.table(rows)

                # Detail cards
                for task in tasks_sorted:
                    is_complete = task._status == "complete"
                    is_overdue  = task.is_overdue(now)
                    with st.container(border=True):
                        ca, cb = st.columns([3, 1])
                        with ca:
                            tlbl = task._due_time.strftime("%I:%M %p")
                            if is_complete:
                                st.success(f"✅ {tlbl} — {task._title}")
                            elif is_overdue:
                                st.error(f"🔴 {tlbl} — {task._title}  *(overdue)*")
                            else:
                                st.markdown(f"**🕐 {tlbl} — {task._title}**")
                            st.caption(
                                f"Category: {task._category} | "
                                f"Priority: {task._priority} | Status: {task._status}"
                            )
                            if task._description:
                                st.write(task._description)
                            if hasattr(task, "_food_type"):
                                st.write(f"Food: {task._food_type}, {task._portion_size}")
                            if hasattr(task, "_distance_goal"):
                                st.write(f"Goal: {task._distance_goal} mi · {task._duration} min · {task._location}")
                            if hasattr(task, "_medication_name"):
                                st.write(f"Medication: {task._medication_name} — {task._dosage}")
                                if task.check_refill_needed():
                                    st.warning("⚠️ Refill needed — contact your vet.")
                            if hasattr(task, "_provider_name"):
                                st.write(f"Provider: {task._provider_name} @ {task._location}")
                        with cb:
                            if not is_complete:
                                if st.button("Mark done", key=f"done_{task._task_id}"):
                                    system.complete_task(task._task_id)
                                    st.rerun()

            # Conflicts + summary
            st.divider()
            warnings = system.get_conflict_warnings(window_minutes=30)
            if warnings:
                st.markdown("#### ⚠️ Scheduling Conflicts")
                for msg in warnings:
                    st.warning(msg)
            else:
                st.success("No scheduling conflicts detected.")

            summary  = system.get_system_summary()
            overdue  = summary["overdue_tasks"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pets",    summary["total_pets"])
            c2.metric("Tasks",   summary["total_tasks"])
            c3.metric("Overdue", overdue,
                      delta=f"{overdue} need attention" if overdue else "None",
                      delta_color="inverse")
            c4.metric("Upcoming", summary["upcoming_tasks"])


# ============================================================
# TAB 2 — AI PET ASSISTANT
# ============================================================
with tab_ai:

    # ── Availability check ────────────────────────────────────
    if not AI_MODULES_AVAILABLE:
        st.error(
            f"AI modules could not be imported: `{AI_IMPORT_ERROR}`\n\n"
            "Run `pip install -r requirements.txt` and restart."
        )
        st.stop()

    agent: PawPalAgent = st.session_state.get("agent")
    agent_status = agent.get_status() if agent else {}

    if not agent_status.get("api_key_set"):
        st.warning(
            "**OPENAI_API_KEY is not set.** Copy `.env.example` to `.env` "
            "and add your key, then restart the app."
        )

    # ── Sidebar-style config in an expander ───────────────────
    with st.expander("⚙️ Assistant Settings", expanded=False):
        mode_keys   = get_expert_mode_options()
        mode_labels = [get_mode_display_name(k) for k in mode_keys]
        selected_idx = st.selectbox(
            "Expert mode", range(len(mode_keys)),
            format_func=lambda i: mode_labels[i],
        )
        selected_mode = mode_keys[selected_idx]
        mode_cfg = EXPERT_MODES[selected_mode]
        st.caption(mode_cfg["description"])

        # Pet context selector
        pet_context = None
        if system._pets:
            pet_names = ["(No specific pet)"] + [p._name for p in system._pets]
            chosen = st.selectbox("Associate with pet", pet_names)
            if chosen != "(No specific pet)":
                pet_obj = next((p for p in system._pets if p._name == chosen), None)
                if pet_obj:
                    pet_context = {
                        "name":    pet_obj._name,
                        "species": pet_obj._species,
                        "breed":   pet_obj._breed,
                        "age":     pet_obj._age,
                        "weight":  pet_obj._weight,
                        "notes":   pet_obj._notes,
                    }

        show_steps   = st.checkbox("Show agentic workflow steps", value=False)
        show_sources = st.checkbox("Show retrieved sources",       value=True)

        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history.clear()
            st.session_state.ai_conversation.clear()
            st.rerun()

    # ── Chat display ──────────────────────────────────────────
    st.subheader(f"{mode_cfg['icon']} AI Pet Assistant — {mode_cfg['name']}")

    if not st.session_state.ai_conversation:
        st.info(
            "Ask anything about your pet — symptoms, training, nutrition, behavior, or general care. "
            "For example: *'My dog keeps scratching his ears'* or *'Create a training plan for an 8-week puppy'*"
        )

    for turn in st.session_state.ai_conversation:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

            if turn["role"] == "assistant" and "meta" in turn:
                meta = turn["meta"]

                # Urgency alert
                urgency = meta.get("urgency", {})
                if urgency.get("level") in ("high", "emergency"):
                    st.error(urgency["message"])
                elif urgency.get("level") == "medium":
                    st.warning(urgency["message"])

                # Confidence badge
                conf  = meta.get("confidence", 0)
                clbl  = meta.get("confidence_label", "")
                color = {"High": "green", "Medium": "blue", "Low": "orange", "Very Low": "red"}.get(clbl, "gray")
                st.markdown(
                    f"<small>Confidence: <span style='color:{color};font-weight:bold'>"
                    f"{conf:.0%} ({clbl})</span> &nbsp;|&nbsp; "
                    f"Mode: {meta.get('expert_mode','')} &nbsp;|&nbsp; "
                    f"Intent: {meta.get('intent','')}</small>",
                    unsafe_allow_html=True,
                )

                # Agentic workflow trace
                if show_steps and meta.get("steps"):
                    with st.expander("🔍 Agentic workflow steps"):
                        for step in meta["steps"]:
                            st.markdown(f"- {step}")

                # Retrieved sources
                if show_sources and meta.get("sources"):
                    with st.expander("📚 Knowledge base sources used"):
                        for src in meta["sources"]:
                            relevance = f"{src['score']:.0%}"
                            st.markdown(
                                f"**{src['title']}**  \n"
                                f"<small>Source: {src['source']} · Relevance: {relevance}</small>",
                                unsafe_allow_html=True,
                            )
                            if src.get("when_to_see_vet"):
                                st.caption(f"🏥 When to see vet: {src['when_to_see_vet']}")

    # ── Input bar ─────────────────────────────────────────────
    user_input = st.chat_input("Ask about your pet's health, training, nutrition, or behavior…")

    if user_input:
        # Show user message immediately
        st.session_state.ai_conversation.append({"role": "user", "content": user_input})

        with st.spinner("Thinking…"):
            if not agent or not agent_status.get("api_key_set"):
                result = {
                    "response": (
                        "The AI assistant is not configured yet. "
                        "Please set your `OPENAI_API_KEY` in the `.env` file and restart."
                    ),
                    "confidence": 0.0,
                    "confidence_label": "N/A",
                    "sources": [],
                    "intent": "unknown",
                    "urgency": {"level": "none", "message": ""},
                    "expert_mode": mode_cfg["name"],
                    "steps": [],
                    "error": "No API key",
                }
            else:
                result = agent.run(
                    query=user_input,
                    pet_info=pet_context,
                    expert_mode=selected_mode,
                    chat_history=st.session_state.chat_history,
                )

        # Update Claude-format history for multi-turn context
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": result["response"]})

        # Update display conversation
        st.session_state.ai_conversation.append(
            {
                "role": "assistant",
                "content": result["response"],
                "meta": {
                    "confidence": result["confidence"],
                    "confidence_label": result["confidence_label"],
                    "expert_mode": result["expert_mode"],
                    "intent": result["intent"],
                    "urgency": result["urgency"],
                    "steps": result["steps"],
                    "sources": result["sources"],
                },
            }
        )
        st.rerun()

    # ── Sample prompts ────────────────────────────────────────
    st.divider()
    st.markdown("**Try these example questions:**")
    examples = [
        "My dog keeps scratching his ears and shaking his head",
        "Create an 8-week training plan for my new puppy",
        "My cat stopped eating yesterday, should I be worried?",
        "What foods are toxic to dogs?",
        "How do I stop my dog from pulling on the leash?",
        "Why does my cat knock things off the table?",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"example_{i}", use_container_width=True):
            st.session_state.ai_conversation.append({"role": "user", "content": ex})
            with st.spinner("Thinking…"):
                if agent and agent_status.get("api_key_set"):
                    res = agent.run(query=ex, pet_info=pet_context,
                                    expert_mode=selected_mode,
                                    chat_history=st.session_state.chat_history)
                else:
                    res = {
                        "response": "Please configure your ANTHROPIC_API_KEY to use the AI assistant.",
                        "confidence": 0.0, "confidence_label": "N/A",
                        "sources": [], "intent": "general",
                        "urgency": {"level": "none", "message": ""},
                        "expert_mode": mode_cfg["name"], "steps": [], "error": "No key",
                    }
            st.session_state.chat_history.append({"role": "user", "content": ex})
            st.session_state.chat_history.append({"role": "assistant", "content": res["response"]})
            st.session_state.ai_conversation.append(
                {"role": "assistant", "content": res["response"],
                 "meta": {"confidence": res["confidence"],
                           "confidence_label": res["confidence_label"],
                           "expert_mode": res["expert_mode"],
                           "intent": res["intent"],
                           "urgency": res["urgency"],
                           "steps": res["steps"],
                           "sources": res["sources"]}}
            )
            st.rerun()


# ============================================================
# TAB 3 — SYSTEM REPORTS
# ============================================================
with tab_reports:
    st.subheader("📊 System Reports")

    col_a, col_b = st.columns(2)

    # ── Scheduler stats ───────────────────────────────────────
    with col_a:
        st.markdown("#### 📅 Scheduler Health")
        summary = system.get_system_summary()
        m1, m2, m3 = st.columns(3)
        m1.metric("Pets",    summary["total_pets"])
        m2.metric("Tasks",   summary["total_tasks"])
        m3.metric("Overdue", summary["overdue_tasks"])

        if system._pets:
            st.markdown("**Registered pets**")
            for p in system._pets:
                task_count = len(p._tasks)
                st.markdown(
                    f"- **{p._name}** ({p._species}, {p._breed}) — "
                    f"{task_count} task{'s' if task_count != 1 else ''}"
                )

    # ── AI reliability stats ──────────────────────────────────
    with col_b:
        st.markdown("#### 🤖 AI Reliability Stats")

        if not AI_MODULES_AVAILABLE:
            st.warning("AI modules not installed.")
        else:
            rel: ReliabilitySystem = st.session_state.get("reliability")
            if rel:
                stats = rel.get_statistics()
                s1, s2, s3 = st.columns(3)
                s1.metric("Total Queries",   stats["total_queries"])
                s2.metric("Avg Confidence",  f"{stats['average_confidence']:.0%}")
                s3.metric("Review Queue",    stats["review_queue_size"])

                if stats["by_intent"]:
                    st.markdown("**Queries by intent**")
                    for intent, count in sorted(stats["by_intent"].items(),
                                                key=lambda x: x[1], reverse=True):
                        bar = "█" * count
                        st.caption(f"{intent}: {bar} ({count})")

                if stats["by_mode"]:
                    st.markdown("**Queries by expert mode**")
                    for mode, count in sorted(stats["by_mode"].items(),
                                              key=lambda x: x[1], reverse=True):
                        st.caption(f"{mode}: {count}")
            else:
                st.info("No AI interactions recorded yet.")

    st.divider()

    # ── Knowledge base info ───────────────────────────────────
    st.markdown("#### 📚 Knowledge Base")
    if AI_MODULES_AVAILABLE:
        rag: RAGRetriever = st.session_state.get("rag")
        if rag:
            kb_status = rag.get_status()
            ka, kb_col, kc = st.columns(3)
            ka.metric("Documents Indexed", kb_status["document_count"])
            kb_col.metric("Categories", len(kb_status["categories"]))
            kc.metric("RAG Ready", "✅ Yes" if kb_status["available"] else "❌ No")
            st.caption("Categories: " + ", ".join(kb_status["categories"]))
        else:
            st.warning("RAG retriever not loaded.")
    else:
        st.warning("AI modules not available.")

    st.divider()

    # ── Human review queue ────────────────────────────────────
    st.markdown("#### 🔎 Human Review Queue")
    st.caption("Low-confidence AI responses flagged for human verification.")

    if AI_MODULES_AVAILABLE:
        rel = st.session_state.get("reliability")
        if rel:
            queue = rel.get_review_queue()
            if not queue:
                st.success("Review queue is empty — all responses met confidence threshold.")
            else:
                st.warning(f"{len(queue)} item(s) awaiting review.")
                for idx, item in enumerate(queue):
                    with st.expander(
                        f"#{idx + 1} · conf={item['confidence']:.2f} · {item.get('timestamp','')[:19]}"
                    ):
                        st.markdown(f"**Query:** {item['query']}")
                        st.markdown(f"**Intent:** {item.get('intent','')} · **Mode:** {item.get('expert_mode','')}")
                        st.markdown("**Response preview:**")
                        st.text(item["response"][:400] + ("…" if len(item["response"]) > 400 else ""))
                        if st.button("✅ Dismiss", key=f"dismiss_{idx}"):
                            rel.dismiss_review_item(idx)
                            st.rerun()

                if st.button("🗑️ Clear all reviewed items"):
                    n = rel.clear_review_queue()
                    st.success(f"Cleared {n} item(s).")
                    st.rerun()
    else:
        st.info("AI modules not available.")

    st.divider()

    # ── Agent status ──────────────────────────────────────────
    st.markdown("#### ⚙️ Component Status")
    if AI_MODULES_AVAILABLE and st.session_state.get("agent"):
        status = st.session_state.agent.get_status()
        rows = [
            {"Component": "OpenAI API (openai)",      "Status": "✅ Installed" if status["openai_available"] else "❌ Missing"},
            {"Component": "OPENAI_API_KEY",           "Status": "✅ Set" if status["api_key_set"] else "❌ Not set"},
            {"Component": "RAG Knowledge Base",       "Status": "✅ Loaded" if status["rag_available"] else "❌ Not loaded"},
            {"Component": "Active AI Model",          "Status": status["model"]},
            {"Component": "Agent Ready",              "Status": "✅ Yes" if status["ready"] else "❌ No"},
        ]
        st.table(rows)
    else:
        if AI_IMPORT_ERROR:
            st.error(f"Import error: {AI_IMPORT_ERROR}")
        st.table([
            {"Component": "AI Modules", "Status": "❌ Not available"},
            {"Component": "Scheduler",  "Status": "✅ Running"},
        ])
