import streamlit as st
import time, uuid, json, csv, os
from datetime import datetime
from random import shuffle
from groq import Groq

# =========================
# CONFIG
# =========================
LOG_PATH = "logs.csv"
GOOGLE_FORM_URL = "https://docs.google.com/forms/d/12DYX8F2_2zYPPO2XnwnpPZfoDi8iSea9InGC5oP7DeE/edit"

GROQ_MODEL = "llama-3.1-8b-instant"

TASK_SETS = {
    "A": [
        "Submit assignment due tomorrow 23:59",
        "Reply to important email from supervisor",
        "Buy groceries for dinner tonight",
        "Book dentist appointment (no deadline)",
        "Call bank about a charge (this week)",
        "Prepare slides for meeting in 2 days",
    ],
    "B": [
        "Pay electricity bill due in 3 days",
        "Pick up a package before the post office closes today",
        "Schedule car service (this month)",
        "Send RSVP for event by tomorrow",
        "Plan study session for exam in 1 week",
        "Refill prescription (soon)",
    ],
    "C": [
        "Book train tickets for trip (prices rising, no fixed deadline)",
        "Pack essentials for tomorrow morning",
        "Confirm hotel reservation details (this week)",
        "Finish project report draft due in 4 days",
        "Call friend back (no deadline)",
        "Do laundry needed for tomorrow",
    ],
}

MODES = [
    ("advisory", "Advisory: AI proposes, you decide"),
    ("semi", "Semi-automated: AI proposes, you may modify"),
    ("full", "Fully automated: AI decides (locked)"),
]

LIKERT_MIN, LIKERT_MAX = 1, 7

# Core outcomes we want per mode (short, repeatable)
MEASURE_ITEMS = [
    ("trust", "Trust (1–7): I trust the AI ranking in this step."),
    ("control", "Control (1–7): I felt in control of the final decision."),
    ("responsibility", "Responsibility (1–7): I feel responsible for the outcome."),
    ("effort", "Effort (1–7): This step required effort from me."),
    ("useful", "Usefulness (1–7): The AI ranking was useful."),
]

# =========================
# UTIL
# =========================
LOG_HEADER = [
    "timestamp_utc",
    "participant_id",
    "mode",
    "taskset_id",
    "tasks_json",
    "ai_order_json",
    "final_order_json",
    "time_to_ai_sec",
    "decision_time_sec",
    "time_to_submit_sec",
    "moves_count",
    "kendall_tau_to_ai",
    "accepted_ai_as_is",          # NEW: automation bias / reliance
    "viewed_reasons",             # NEW: explanation exposure
    "manipulation_check",         # NEW: who decided?
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

def kendall_tau_distance(order1, order2):
    pos = {item: i for i, item in enumerate(order2)}
    arr = [pos[x] for x in order1]
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv

def move_item(order, idx, direction):
    new_order = order[:]
    j = idx + direction
    if 0 <= idx < len(order) and 0 <= j < len(order):
        new_order[idx], new_order[j] = new_order[j], new_order[idx]
    return new_order

def safe_extract_json(text: str) -> dict:
    raw = (text or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return JSON. Raw:\n{raw}")
    return json.loads(raw[start:end + 1])

# =========================
# LLM CALL (Groq)
# =========================
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
# SESSION STATE INIT
# =========================
def init():
    if "participant_id" not in st.session_state:
        st.session_state.participant_id = str(uuid.uuid4())[:8]

    if "mode_order" not in st.session_state:
        m = MODES[:]
        shuffle(m)
        st.session_state.mode_order = m

    if "taskset_order" not in st.session_state:
        ts = ["A", "B", "C"]
        shuffle(ts)
        st.session_state.taskset_order = ts

    if "step" not in st.session_state:
        st.session_state.step = 0

    if "t_step_start" not in st.session_state:
        st.session_state.t_step_start = None

    if "t_ai_shown" not in st.session_state:
        st.session_state.t_ai_shown = None

    if "ai_order" not in st.session_state:
        st.session_state.ai_order = None

    if "reasons" not in st.session_state:
        st.session_state.reasons = None

    if "final_order" not in st.session_state:
        st.session_state.final_order = None

    if "moves" not in st.session_state:
        st.session_state.moves = 0

    if "accepted_ai_as_is" not in st.session_state:
        st.session_state.accepted_ai_as_is = False

    if "viewed_reasons" not in st.session_state:
        st.session_state.viewed_reasons = False

    if "done" not in st.session_state:
        st.session_state.done = False

init()

# =========================
# UI
# =========================
st.set_page_config(page_title="Task Prioritization Study", layout="centered")
st.title("Everyday Task Prioritization Study")
st.progress((st.session_state.step + 1) / 3 if not st.session_state.done else 1.0)

st.caption(
    "You will see three interaction modes. The AI uses the same rubric each time "
    "(urgency, importance, deadlines). Only the level of human control changes."
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

# Current step context
step = st.session_state.step
mode_key, mode_label = st.session_state.mode_order[step]
taskset_id = st.session_state.taskset_order[step]
tasks = TASK_SETS[taskset_id]

st.subheader(f"Step {step+1}/3 — {mode_label}")
st.caption(f"Task set: {taskset_id}")

if st.session_state.t_step_start is None:
    st.session_state.t_step_start = time.time()

# Tasks
st.write("Tasks:")
for i, t in enumerate(tasks):
    st.write(f"{i}. {t}")

# Generate / Accept
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("Generate AI ranking", type="primary", disabled=st.session_state.ai_order is not None):
        try:
            out = groq_rank_tasks(tasks)
            st.session_state.t_ai_shown = time.time()
            st.session_state.ai_order = out["ranking"]
            st.session_state.reasons = out["reasons"]
            st.session_state.final_order = out["ranking"][:]
            st.session_state.moves = 0
            st.session_state.accepted_ai_as_is = False
            st.session_state.viewed_reasons = False
        except Exception as e:
            st.error(str(e))

with col2:
    if st.button("Clear AI ranking"):
        st.session_state.ai_order = None
        st.session_state.reasons = None
        st.session_state.final_order = None
        st.session_state.t_ai_shown = None
        st.session_state.moves = 0
        st.session_state.accepted_ai_as_is = False
        st.session_state.viewed_reasons = False

with col3:
    # only meaningful after AI exists
    if st.button("Accept AI as-is", disabled=st.session_state.ai_order is None):
        st.session_state.final_order = st.session_state.ai_order[:]
        st.session_state.accepted_ai_as_is = True
        st.rerun()

if st.session_state.ai_order is None:
    st.info('Click "Generate AI ranking" to proceed.')
    st.stop()

# AI output (with optional reasons)
st.divider()
show_reasons = st.toggle("Show AI reasons", value=False)
if show_reasons:
    st.session_state.viewed_reasons = True

st.write("AI ranking:")
for rank_pos, idx in enumerate(st.session_state.ai_order, start=1):
    if show_reasons:
        st.write(f"{rank_pos}. {tasks[idx]} — {st.session_state.reasons[idx]}")
    else:
        st.write(f"{rank_pos}. {tasks[idx]}")

# Editing rules
st.divider()
editable = mode_key in ("advisory", "semi")

if mode_key == "full":
    st.caption("Locked: you cannot change the ranking in this mode.")
else:
    st.caption("You may reorder the ranking if you want, then submit.")

# Final ranking (editable if allowed)
st.write("Final ranking:")
order = st.session_state.final_order

for pos, idx in enumerate(order):
    row = st.columns([8, 1, 1])
    row[0].write(f"{pos+1}. {tasks[idx]}")
    if row[1].button("↑", key=f"up_{step}_{pos}", disabled=(not editable or pos == 0)):
        st.session_state.final_order = move_item(order, pos, -1)
        st.session_state.moves += 1
        st.session_state.accepted_ai_as_is = False
        st.rerun()
    if row[2].button("↓", key=f"down_{step}_{pos}", disabled=(not editable or pos == len(order) - 1)):
        st.session_state.final_order = move_item(order, pos, +1)
        st.session_state.moves += 1
        st.session_state.accepted_ai_as_is = False
        st.rerun()

ai_order = st.session_state.ai_order
final_order = st.session_state.final_order
kdist_preview = kendall_tau_distance(ai_order, final_order)
st.info(f"Edits: {st.session_state.moves} | Difference from AI: {kdist_preview} (Kendall inversions)")

# -------------------------
# In-app mini questionnaire (core outcomes)
# -------------------------
st.divider()
st.subheader("Quick questions for this step (10–15 seconds)")

satisfaction = st.slider(
    "Satisfaction (1–7): I am satisfied with the final ranking.",
    LIKERT_MIN, LIKERT_MAX, 4, key=f"satisfaction_{step}"
)

answers = {}
for key, label in MEASURE_ITEMS:
    answers[key] = st.slider(label, LIKERT_MIN, LIKERT_MAX, 4, key=f"{key}_{step}")

# Manipulation check (important for validity)
manipulation = st.radio(
    "In this step, who made the final decision?",
    ["Mostly me", "Mostly the AI", "Both equally"],
    index=0,
    key=f"manip_{step}"
)

# Optional acknowledgement differences (kept minimal)
advisory_ack = True
attention_ack = True
if mode_key == "advisory":
    advisory_ack = st.checkbox("I understand I am responsible for the final decision.", key=f"ack_resp_{step}")
if mode_key == "full":
    attention_ack = st.checkbox("I have reviewed the ranking.", key=f"ack_seen_{step}")

# Submit
st.divider()
submit_disabled = False
if mode_key == "advisory" and not advisory_ack:
    submit_disabled = True
if mode_key == "full" and not attention_ack:
    submit_disabled = True

submit_label = "Confirm & Next"
if st.button(submit_label, type="primary", disabled=submit_disabled):
    now = time.time()
    t0 = st.session_state.t_step_start
    t_ai = st.session_state.t_ai_shown or t0

    time_to_ai = t_ai - t0
    decision_time = now - t_ai
    time_to_submit = now - t0

    kdist = kendall_tau_distance(ai_order, final_order)

    append_log([
        datetime.utcnow().isoformat(),
        st.session_state.participant_id,
        mode_key,
        taskset_id,
        json.dumps(tasks, ensure_ascii=False),
        json.dumps(ai_order),
        json.dumps(final_order),
        round(time_to_ai, 3),
        round(decision_time, 3),
        round(time_to_submit, 3),
        st.session_state.moves,
        kdist,
        int(bool(st.session_state.accepted_ai_as_is)),
        int(bool(st.session_state.viewed_reasons)),
        manipulation,
        satisfaction,
        answers["trust"],
        answers["control"],
        answers["responsibility"],
        answers["effort"],
        answers["useful"],
    ])

    # Advance
    st.session_state.step += 1
    st.session_state.t_step_start = None
    st.session_state.t_ai_shown = None
    st.session_state.ai_order = None
    st.session_state.reasons = None
    st.session_state.final_order = None
    st.session_state.moves = 0
    st.session_state.accepted_ai_as_is = False
    st.session_state.viewed_reasons = False

    if st.session_state.step >= 3:
        st.session_state.done = True

    st.rerun()
