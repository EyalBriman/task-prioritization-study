import streamlit as st
import time, uuid, json, csv, os
from datetime import datetime
from random import shuffle
from google import genai

# =========================
# CONFIG
# =========================
LOG_PATH = "logs.csv"
GOOGLE_FORM_URL = "https://docs.google.com/forms/d/12DYX8F2_2zYPPO2XnwnpPZfoDi8iSea9InGC5oP7DeE/edit"

# Gemini model (choose one that works for your key/tier)
GEMINI_MODEL = "gemini-2.0-flash"

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


# =========================
# UTIL
# =========================
def ensure_log_header():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
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
                    "satisfaction_1to7",
                    "advisory_responsibility_ack",
                    "attention_ack_full_mode",
                ]
            )


def append_log(row):
    ensure_log_header()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


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
    """
    Extract the first {...} JSON object from a model response.
    Also strips ```json fences if present.
    """
    raw = (text or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return JSON. Raw:\n{raw}")
    return json.loads(raw[start : end + 1])


# =========================
# LLM CALL (Gemini)
# =========================
def gemini_rank_tasks(tasks):
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in Streamlit secrets.")

    client = genai.Client(api_key=api_key)

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

    # Try once; if JSON parsing fails, retry with a stricter reminder
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    try:
        data = safe_extract_json(resp.text)
    except Exception:
        prompt2 = prompt + "\n\nIMPORTANT: Output JSON ONLY. No prose. No code fences."
        resp2 = client.models.generate_content(model=GEMINI_MODEL, contents=prompt2)
        data = safe_extract_json(resp2.text)

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

st.caption("Note: The AI ranking is a heuristic; please use your judgment.")

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

    if "PASTE_YOUR_GOOGLE_FORM_LINK_HERE" not in GOOGLE_FORM_URL:
        st.link_button("Open questionnaire (Google Form)", GOOGLE_FORM_URL)
    else:
        st.warning("Add your Google Form link in GOOGLE_FORM_URL in the code.")
    st.stop()

# Current step context
step = st.session_state.step
mode_key, mode_label = st.session_state.mode_order[step]
taskset_id = st.session_state.taskset_order[step]
tasks = TASK_SETS[taskset_id]

st.subheader(f"Step {step+1}/3 — {mode_label}")
st.caption(f"Task set: {taskset_id}")

# Start timing when the step is first shown
if st.session_state.t_step_start is None:
    st.session_state.t_step_start = time.time()

# Show tasks
st.write("Tasks:")
for i, t in enumerate(tasks):
    st.write(f"{i}. {t}")

# Generate ranking
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Generate AI ranking", type="primary", disabled=st.session_state.ai_order is not None):
        try:
            out = gemini_rank_tasks(tasks)
            st.session_state.t_ai_shown = time.time()
            st.session_state.ai_order = out["ranking"]
            st.session_state.reasons = out["reasons"]
            st.session_state.final_order = out["ranking"][:]
            st.session_state.moves = 0
        except Exception as e:
            st.error(str(e))

with col2:
    if st.button("Clear AI ranking"):
        st.session_state.ai_order = None
        st.session_state.reasons = None
        st.session_state.final_order = None
        st.session_state.t_ai_shown = None
        st.session_state.moves = 0

if st.session_state.ai_order is None:
    st.info('Click "Generate AI ranking" to proceed.')
    st.stop()

# AI output
st.divider()
st.write("AI ranking (with short reasons):")
for rank_pos, idx in enumerate(st.session_state.ai_order, start=1):
    st.write(f"{rank_pos}. {tasks[idx]} — {st.session_state.reasons[idx]}")

# Editing rules
st.divider()
editable = mode_key in ("advisory", "semi")

if mode_key == "full":
    st.caption("Locked: you cannot change the ranking in this mode.")
elif mode_key == "advisory":
    st.caption("You may reorder, then you must explicitly confirm responsibility.")
else:
    st.caption("You may reorder if you want, then submit.")

# Render final ranking (with edit controls where allowed)
st.write("Final ranking:")
order = st.session_state.final_order

for pos, idx in enumerate(order):
    row = st.columns([8, 1, 1])
    row[0].write(f"{pos+1}. {tasks[idx]}")
    if row[1].button("↑", key=f"up_{pos}", disabled=(not editable or pos == 0)):
        st.session_state.final_order = move_item(order, pos, -1)
        st.session_state.moves += 1
        st.rerun()
    if row[2].button("↓", key=f"down_{pos}", disabled=(not editable or pos == len(order) - 1)):
        st.session_state.final_order = move_item(order, pos, +1)
        st.session_state.moves += 1
        st.rerun()

# Edit summary + micro-feedback
ai_order = st.session_state.ai_order
final_order = st.session_state.final_order
kdist_preview = kendall_tau_distance(ai_order, final_order)

st.info(f"Edits: {st.session_state.moves} | Difference from AI: {kdist_preview} (Kendall inversions)")

satisfaction = st.slider(
    "Satisfaction (1–7): I am satisfied with the final ranking.",
    LIKERT_MIN,
    LIKERT_MAX,
    4,
    key=f"satisfaction_{step}",
)

# Mode-specific acknowledgements (distinct control / responsibility)
advisory_ack = False
attention_ack = False

if mode_key == "advisory":
    advisory_ack = st.checkbox("I take responsibility for the final ranking.", key=f"ack_resp_{step}")
elif mode_key == "full":
    attention_ack = st.checkbox("I have reviewed the ranking.", key=f"ack_seen_{step}")

# Submit
st.divider()
submit_label = "Confirm & Next" if editable else "Next"

submit_disabled = False
if mode_key == "advisory" and not advisory_ack:
    submit_disabled = True
if mode_key == "full" and not attention_ack:
    submit_disabled = True

if st.button(submit_label, type="primary", disabled=submit_disabled):
    now = time.time()
    t0 = st.session_state.t_step_start
    t_ai = st.session_state.t_ai_shown or t0

    time_to_ai = t_ai - t0
    decision_time = now - t_ai  # time from AI shown to submit
    time_to_submit = now - t0

    kdist = kendall_tau_distance(ai_order, final_order)

    append_log(
        [
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
            satisfaction,
            int(bool(advisory_ack)),
            int(bool(attention_ack)),
        ]
    )

    # Advance to next step
    st.session_state.step += 1
    st.session_state.t_step_start = None
    st.session_state.t_ai_shown = None
    st.session_state.ai_order = None
    st.session_state.reasons = None
    st.session_state.final_order = None
    st.session_state.moves = 0

    if st.session_state.step >= 3:
        st.session_state.done = True

    st.rerun()
