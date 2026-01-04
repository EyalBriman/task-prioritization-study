import streamlit as st
import time, uuid, json, csv, os
from datetime import datetime
from groq import Groq

# -------------------------
# OPTIONAL: Drag & Drop
# -------------------------
HAS_DND = False
try:
    # pip install streamlit-sortables
    from streamlit_sortables import sort_items
    HAS_DND = True
except Exception:
    HAS_DND = False

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Task Prioritization Study", layout="wide")

LOG_PATH = "logs.csv"
GOOGLE_FORM_URL = "https://docs.google.com/forms/d/12DYX8F2_2zYPPO2XnwnpPZfoDi8iSea9InGC5oP7DeE/edit"
GROQ_MODEL = "llama-3.1-8b-instant"

# 1) FIXED TASK LIST across all 3 modes (your Hebrew comment #1)
TASKS_FIXED = [
    "Submit assignment due tomorrow 23:59",
    "Reply to important email from supervisor",
    "Buy groceries for dinner tonight",
    "Book dentist appointment (no deadline)",
    "Call bank about a charge (this week)",
    "Prepare slides for meeting in 2 days",
]

# Modes: same as before, but we will enforce clearer flow differences
MODES = [
    ("advisory", "Advisory: You rank first, then AI advises"),
    ("semi", "Semi-automated: AI ranks first, you may modify"),
    ("full", "Fully automated: AI decides (locked)"),
]

LIKERT_MIN, LIKERT_MAX = 1, 7
MEASURE_ITEMS = [
    ("trust", "Trust (1–7): I trust the AI ranking in this step."),
    ("control", "Control (1–7): I felt in control of the final decision."),
    ("responsibility", "Responsibility (1–7): I feel responsible for the outcome."),
    ("effort", "Effort (1–7): This step required effort from me."),
    ("useful", "Usefulness (1–7): The AI ranking was useful."),
]

LOG_HEADER = [
    "timestamp_utc",
    "participant_id",
    "mode",
    "tasks_json",
    "ai_order_json",
    "final_order_json",
    "time_to_ai_sec",
    "decision_time_sec",
    "time_to_submit_sec",
    "moves_count",
    "kendall_tau_to_ai",
    "accepted_ai_as_is",
    "viewed_reasons",
    "manipulation_check",
    "satisfaction_1to7",
    "trust_1to7",
    "control_1to7",
    "responsibility_1to7",
    "effort_1to7",
    "useful_1to7",
]

def ensure_log_header():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_HEADER)

def append_log(row):
    ensure_log_header()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)
        f.flush()

def safe_extract_json(text: str) -> dict:
    raw = (text or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return JSON. Raw:\n{raw}")
    return json.loads(raw[start:end + 1])

def kendall_tau_distance(order1, order2):
    pos = {item: i for i, item in enumerate(order2)}
    arr = [pos[x] for x in order1]
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv

def groq_rank_tasks(tasks):
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in Streamlit secrets.")

    client = Groq(api_key=api_key)

    prompt = f"""
You are ranking tasks using a fixed rubric:
1) Urgency (time-sensitive, soonest deadline)
2) Importance (high impact, consequences if delayed)
3) Deadline proximity (explicit deadlines beat vague ones)
Resolve ties by choosing the task with clearer negative consequences if delayed.

Return ONLY valid JSON (no markdown, no extra text).
JSON schema:
{{
  "ranking": [int, ...],   // permutation of 0..{len(tasks)-1}, highest priority first
  "reasons": [string, ...] // length {len(tasks)}; reasons[i] max 12 words
}}

Tasks:
{json.dumps(tasks, ensure_ascii=False)}
""".strip()

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Output JSON only. No prose. No markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    text = resp.choices[0].message.content
    data = safe_extract_json(text)

    n = len(tasks)
    ranking = data.get("ranking")
    reasons = data.get("reasons")

    if not isinstance(ranking, list) or sorted(ranking) != list(range(n)):
        raise ValueError(f"Invalid ranking returned: {ranking}")
    if not isinstance(reasons, list) or len(reasons) != n:
        raise ValueError(f"Invalid reasons returned: {reasons}")

    return {"ranking": ranking, "reasons": reasons}

# =========================
# SESSION INIT
# =========================
def init():
    if "participant_id" not in st.session_state:
        st.session_state.participant_id = str(uuid.uuid4())[:8]

    if "mode_order" not in st.session_state:
        # keep your counterbalancing, but only across modes now (tasks are fixed)
        import random
        m = MODES[:]
        random.shuffle(m)
        st.session_state.mode_order = m

    if "step" not in st.session_state:
        st.session_state.step = 0

    # timing
    if "t_step_start" not in st.session_state:
        st.session_state.t_step_start = None
    if "t_ai_shown" not in st.session_state:
        st.session_state.t_ai_shown = None

    # AI + reasons
    if "ai_order" not in st.session_state:
        st.session_state.ai_order = None
    if "reasons" not in st.session_state:
        st.session_state.reasons = None

    # user ranking (list of indices)
    if "user_order" not in st.session_state:
        st.session_state.user_order = None

    # bookkeeping
    if "viewed_reasons" not in st.session_state:
        st.session_state.viewed_reasons = False
    if "accepted_ai_as_is" not in st.session_state:
        st.session_state.accepted_ai_as_is = False

    # advisory-specific: force user to rank BEFORE AI appears
    if "user_ranked_before_ai" not in st.session_state:
        st.session_state.user_ranked_before_ai = False

    if "done" not in st.session_state:
        st.session_state.done = False

init()

# =========================
# UI (HEADER)
# =========================
st.title("Everyday Task Prioritization Study")
st.caption(
    "Same tasks every time. Only the level of human control changes.\n"
    "AI rubric is constant: urgency, importance, deadlines."
)

with st.expander("Rubric used by the AI (same in all modes)", expanded=False):
    st.markdown(
        """
- **Urgency**: time-sensitive tasks first  
- **Importance**: higher impact / clearer consequences if delayed  
- **Deadlines**: explicit and sooner deadlines > vague deadlines  
Tie-break: clearer negative consequences if delayed.
"""
    )

with st.sidebar:
    st.write(f"Participant ID: **{st.session_state.participant_id}**")
    st.write(f"Drag-and-drop enabled: **{HAS_DND}**")
    if st.button("Reset session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# Finished screen
if st.session_state.done:
    st.success("Finished. Please complete the questionnaire.")
    st.write("Participant ID (copy this into the form if needed):")
    st.code(st.session_state.participant_id)
    st.link_button("Open questionnaire (Google Form)", GOOGLE_FORM_URL)
    st.stop()

# Step context
step = st.session_state.step
mode_key, mode_label = st.session_state.mode_order[step]
tasks = TASKS_FIXED
n = len(tasks)

st.progress((step + 1) / 3)

st.subheader(f"Step {step+1}/3 — {mode_label}")

if st.session_state.t_step_start is None:
    st.session_state.t_step_start = time.time()

# Initialize user order (default: identity)
if st.session_state.user_order is None:
    st.session_state.user_order = list(range(n))

# -------------------------
# Layout: side-by-side
# -------------------------
left, right = st.columns([1, 1], gap="large")

# -------------------------
# LEFT: AI recommendation
# -------------------------
with left:
    st.markdown("### AI recommendation")

    # Advisory: AI should not appear until user ranks first (your Hebrew comment #2 + #3)
    advisory_blocked = (mode_key == "advisory" and not st.session_state.user_ranked_before_ai)

    if st.session_state.ai_order is None:
        if advisory_blocked:
            st.info("In Advisory mode, first rank the tasks on the right. Then you can reveal the AI.")
        else:
            if st.button("Generate AI ranking", type="primary"):
                try:
                    out = groq_rank_tasks(tasks)
                    st.session_state.t_ai_shown = time.time()
                    st.session_state.ai_order = out["ranking"]
                    st.session_state.reasons = out["reasons"]
                    # Semi mode: prefill user's ranking with AI ranking (clear distinction)
                    if mode_key == "semi":
                        st.session_state.user_order = out["ranking"][:]
                        st.session_state.accepted_ai_as_is = True
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    else:
        show_reasons = st.toggle("Show AI reasons", value=False, key=f"show_reasons_{step}")
        if show_reasons:
            st.session_state.viewed_reasons = True

        for rank_pos, idx in enumerate(st.session_state.ai_order, start=1):
            if show_reasons:
                st.write(f"{rank_pos}. {tasks[idx]} — {st.session_state.reasons[idx]}")
            else:
                st.write(f"{rank_pos}. {tasks[idx]}")

        # Convenience button (still logs accepted_ai_as_is)
        if mode_key in ("advisory", "semi"):
            if st.button("Copy AI ranking to my ranking"):
                st.session_state.user_order = st.session_state.ai_order[:]
                st.session_state.accepted_ai_as_is = True
                st.rerun()

# -------------------------
# RIGHT: User ranking (drag & drop)
# -------------------------
with right:
    st.markdown("### Your ranking")

    # Explain the control difference plainly (divide-control heuristic)
    if mode_key == "advisory":
        st.caption("You must rank first. Then the AI can advise. You decide the final ranking.")
    elif mode_key == "semi":
        st.caption("AI pre-fills the ranking. You may modify it before submitting.")
    else:
        st.caption("Locked. AI determines the final ranking in this mode.")

    editable = (mode_key != "full")

    # White-card style
    st.markdown(
        """
        <style>
        .card {
          background: white;
          border: 1px solid #e6e6e6;
          border-radius: 10px;
          padding: 10px 12px;
          margin-bottom: 8px;
          box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not editable:
        # Full automation: user's ranking = AI ranking (locked)
        if st.session_state.ai_order is None:
            st.info("Generate AI ranking on the left to proceed.")
        else:
            st.session_state.user_order = st.session_state.ai_order[:]
            for pos, idx in enumerate(st.session_state.user_order, start=1):
                st.markdown(f"<div class='card'>{pos}. {tasks[idx]}</div>", unsafe_allow_html=True)
    else:
        # Drag & Drop if available; else fallback to up/down
        if HAS_DND:
            # sort_items expects list of strings; map back to indices
            label_by_idx = {i: f"{i}. {tasks[i]}" for i in range(n)}
            current_labels = [label_by_idx[i] for i in st.session_state.user_order]

            sorted_labels = sort_items(
                current_labels,
                direction="vertical",
                key=f"dnd_{step}",
                multi_containers=False,
                disabled=False,
            )

            # map labels -> indices
            inv = {v: k for k, v in label_by_idx.items()}
            new_order = [inv[x] for x in sorted_labels]
            st.session_state.user_order = new_order

            # Advisory: as soon as they have interacted, allow AI reveal
            if mode_key == "advisory":
                st.session_state.user_ranked_before_ai = True

            # accepted_ai_as_is flag (only meaningful after AI exists)
            if st.session_state.ai_order is not None:
                st.session_state.accepted_ai_as_is = (st.session_state.user_order == st.session_state.ai_order)

        else:
            # Fallback: your original up/down buttons
            order = st.session_state.user_order
            for pos, idx in enumerate(order):
                row = st.columns([8, 1, 1])
                row[0].write(f"{pos+1}. {tasks[idx]}")
                if row[1].button("↑", key=f"up_{step}_{pos}", disabled=(pos == 0)):
                    order[pos-1], order[pos] = order[pos], order[pos-1]
                    st.session_state.user_order = order
                    if mode_key == "advisory":
                        st.session_state.user_ranked_before_ai = True
                    st.rerun()
                if row[2].button("↓", key=f"down_{step}_{pos}", disabled=(pos == len(order)-1)):
                    order[pos+1], order[pos] = order[pos], order[pos+1]
                    st.session_state.user_order = order
                    if mode_key == "advisory":
                        st.session_state.user_ranked_before_ai = True
                    st.rerun()

# -------------------------
# Compute distances + show small status
# -------------------------
st.divider()

ai_order = st.session_state.ai_order
final_order = st.session_state.user_order

if ai_order is None:
    st.info("Next: Generate AI ranking (left).")
else:
    kdist_preview = kendall_tau_distance(ai_order, final_order)
    st.info(f"Difference from AI: {kdist_preview} (Kendall inversions)")

# -------------------------
# In-app questionnaire
# -------------------------
st.subheader("Quick questions for this step (10–15 seconds)")

satisfaction = st.slider(
    "Satisfaction (1–7): I am satisfied with the final ranking.",
    LIKERT_MIN, LIKERT_MAX, 4, key=f"satisfaction_{step}"
)

answers = {}
for key, label in MEASURE_ITEMS:
    answers[key] = st.slider(label, LIKERT_MIN, LIKERT_MAX, 4, key=f"{key}_{step}")

manipulation = st.radio(
    "In this step, who made the final decision?",
    ["Mostly me", "Mostly the AI", "Both equally"],
    index=0,
    key=f"manip_{step}"
)

# Advisory / Full acknowledgement (minimal)
advisory_ack = True
attention_ack = True
if mode_key == "advisory":
    advisory_ack = st.checkbox("I understand I am responsible for the final decision.", key=f"ack_resp_{step}")
if mode_key == "full":
    attention_ack = st.checkbox("I have reviewed the ranking.", key=f"ack_seen_{step}")

# -------------------------
# Submit
# -------------------------
submit_disabled = False
if mode_key == "advisory" and not advisory_ack:
    submit_disabled = True
if mode_key == "full" and not attention_ack:
    submit_disabled = True

# Also prevent submit if AI not generated yet (because we need ai_order for logging/comparison)
if st.session_state.ai_order is None:
    submit_disabled = True

if st.button("Confirm & Next", type="primary", disabled=submit_disabled):
    now = time.time()
    t0 = st.session_state.t_step_start
    t_ai = st.session_state.t_ai_shown or t0

    time_to_ai = t_ai - t0
    decision_time = now - t_ai
    time_to_submit = now - t0

    # moves_count: how many positions differ from AI (simple, stable)
    moves_count = sum(1 for i in range(n) if final_order[i] != ai_order[i])

    kdist = kendall_tau_distance(ai_order, final_order)
    accepted = int(bool(final_order == ai_order))

    append_log([
        datetime.utcnow().isoformat(),
        st.session_state.participant_id,
        mode_key,
        json.dumps(tasks, ensure_ascii=False),
        json.dumps(ai_order),
        json.dumps(final_order),
        round(time_to_ai, 3),
        round(decision_time, 3),
        round(time_to_submit, 3),
        moves_count,
        kdist,
        accepted,
        int(bool(st.session_state.viewed_reasons)),
        manipulation,
        satisfaction,
        answers["trust"],
        answers["control"],
        answers["responsibility"],
        answers["effort"],
        answers["useful"],
    ])

    # Advance to next mode
    st.session_state.step += 1

    # Reset per-step state
    st.session_state.t_step_start = None
    st.session_state.t_ai_shown = None
    st.session_state.ai_order = None
    st.session_state.reasons = None
    st.session_state.user_order = list(range(n))
    st.session_state.viewed_reasons = False
    st.session_state.accepted_ai_as_is = False
    st.session_state.user_ranked_before_ai = False

    if st.session_state.step >= 3:
        st.session_state.done = True

    st.rerun()
