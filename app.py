import streamlit as st
import time, uuid, json, csv, os
from datetime import datetime
from groq import Groq
import random

# -------------------------
# OPTIONAL: Drag & Drop
# -------------------------
HAS_DND = False
try:
    from streamlit_sortables import sort_items
    HAS_DND = True
except Exception:
    HAS_DND = False

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Task Prioritization Study", layout="wide")

GROQ_MODEL = "llama-3.1-8b-instant"

LIKERT_MIN, LIKERT_MAX = 1, 7
MEASURE_ITEMS = [
    ("trust", "Trust (1–7): I trust the AI ranking in this step."),
    ("control", "Control (1–7): I felt in control of the final decision."),
    ("responsibility", "Responsibility (1–7): I feel responsible for the outcome."),
    ("effort", "Effort (1–7): This step required effort from me."),
    ("useful", "Usefulness (1–7): The AI ranking was useful."),
]

MODES = [
    ("advisory", "Advisory: You rank first, then AI advises"),
    ("semi", "Semi-automated: AI ranks first, you may modify"),
    ("full", "Fully automated: AI decides (locked)"),
]

# Broad, population-relevant task pool
TASK_POOL = [
    "Pay a bill that is due soon",
    "Reply to an important message",
    "Buy groceries for today or tomorrow",
    "Book a medical appointment (no deadline)",
    "Call the bank about a suspicious charge",
    "Prepare for a meeting/class in 2 days",
    "Submit a form/application before a deadline",
    "Pick up a package before closing time today",
    "Do laundry needed for tomorrow",
    "Schedule a repair (car/phone/home) this month",
    "Renew a service/subscription before it expires",
    "Refill an essential medication/prescription",
    "Plan a study/work session for an upcoming exam/project",
    "Confirm travel or hotel details this week",
    "Pack essentials for tomorrow morning",
    "Clean the kitchen / take out trash today",
    "Call a family member back (no deadline)",
    "Send an RSVP by tomorrow",
    "Prepare dinner plan for tonight",
    "Organize documents needed for an appointment",
    "Respond to a request that has consequences if late",
    "Fix a tech issue blocking work (soon)",
    "Buy a household essential needed soon",
    "Return an item before the return window closes",
    "Review terms/contract before making a decision",
    "Prepare materials for a presentation",
    "Handle a parking/municipal issue this week",
    "Follow up on a delayed service ticket",
    "Arrange childcare/pet care for an upcoming day",
    "Confirm an appointment time/location",
    "Send a required message today",
    "Pay rent / transfer money on time",
    "Update a document due in a few days",
    "Finish a small errand before stores close",
]

TASKS_PER_PARTICIPANT = 5
TOTAL_STEPS = 3

# Optional CSV log (JSON download is the main thing)
LOG_PATH = "logs.csv"
LOG_HEADER = [
    "timestamp_utc",
    "participant_id",
    "step",
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

# =========================
# UTIL
# =========================
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

# =========================
# GROQ
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
# SESSION INIT
# =========================
def init():
    if "participant_id" not in st.session_state:
        st.session_state.participant_id = str(uuid.uuid4())[:8]

    if "mode_order" not in st.session_state:
        m = MODES[:]
        random.shuffle(m)
        st.session_state.mode_order = m

    if "step" not in st.session_state:
        st.session_state.step = 0

    # Fixed tasks for the participant (same in all 3 steps)
    if "tasks_for_participant" not in st.session_state:
        st.session_state.tasks_for_participant = None

    # Per-step state
    if "t_step_start" not in st.session_state:
        st.session_state.t_step_start = None
    if "t_ai_shown" not in st.session_state:
        st.session_state.t_ai_shown = None
    if "ai_order" not in st.session_state:
        st.session_state.ai_order = None
    if "reasons" not in st.session_state:
        st.session_state.reasons = None
    if "user_order" not in st.session_state:
        st.session_state.user_order = None

    # Advisory gating
    if "advisory_ready_for_ai" not in st.session_state:
        st.session_state.advisory_ready_for_ai = False

    # Track AI generation per step (for timing/logging)
    if "ai_generated_step_idx" not in st.session_state:
        st.session_state.ai_generated_step_idx = None

    if "viewed_reasons" not in st.session_state:
        st.session_state.viewed_reasons = False
    if "accepted_ai_as_is" not in st.session_state:
        st.session_state.accepted_ai_as_is = False

    # Collect everything for JSON download
    if "records" not in st.session_state:
        st.session_state.records = {
            "participant_id": st.session_state.participant_id,
            "mode_order": None,
            "tasks": None,
            "steps": []
        }

    if "done" not in st.session_state:
        st.session_state.done = False

init()

# Store mode order once
if st.session_state.records["mode_order"] is None:
    st.session_state.records["mode_order"] = [m[0] for m in st.session_state.mode_order]

# =========================
# TASKS (fixed for participant)
# =========================
def get_tasks_for_participant(pid: str):
    if st.session_state.tasks_for_participant is not None:
        return st.session_state.tasks_for_participant

    rng = random.Random(f"{pid}-TASKSET")
    tasks = rng.sample(TASK_POOL, TASKS_PER_PARTICIPANT)
    st.session_state.tasks_for_participant = tasks

    if st.session_state.records.get("tasks") is None:
        st.session_state.records["tasks"] = tasks

    return tasks

# =========================
# AI generation
# =========================
def ensure_ai_generated(tasks, step_idx):
    if st.session_state.ai_generated_step_idx == step_idx and st.session_state.ai_order is not None:
        return
    out = groq_rank_tasks(tasks)
    st.session_state.t_ai_shown = time.time()
    st.session_state.ai_order = out["ranking"]
    st.session_state.reasons = out["reasons"]
    st.session_state.ai_generated_step_idx = step_idx

# =========================
# UI
# =========================
st.title("Everyday Task Prioritization Study")
st.caption("You will complete three steps. The same 5 tasks are used in all steps; only control level changes.")

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

# DONE screen: download data
if st.session_state.done:
    st.success("Finished. Download your data file and send it to the researcher.")
    payload = st.session_state.records
    json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    st.download_button(
        "Download your experiment data (JSON)",
        data=json_bytes,
        file_name=f"task_study_{payload['participant_id']}.json",
        mime="application/json",
    )
    st.write("Your Participant ID:")
    st.code(payload["participant_id"])
    st.stop()

# Step context
step_idx = st.session_state.step
mode_key, mode_label = st.session_state.mode_order[step_idx]
tasks = get_tasks_for_participant(st.session_state.participant_id)
n = len(tasks)

st.progress((step_idx + 1) / TOTAL_STEPS)
st.subheader(f"Step {step_idx+1}/{TOTAL_STEPS} — {mode_label}")

if st.session_state.t_step_start is None:
    st.session_state.t_step_start = time.time()

# Init user order for step
if st.session_state.user_order is None:
    st.session_state.user_order = list(range(n))

# Auto-generate AI for semi + full
if mode_key in ("semi", "full"):
    try:
        ensure_ai_generated(tasks, step_idx)
        # Prefill semi with AI once when step loads
        if mode_key == "semi" and st.session_state.ai_order is not None:
            # only prefill if user hasn't already changed it this step
            if st.session_state.user_order == list(range(n)):
                st.session_state.user_order = st.session_state.ai_order[:]
                st.session_state.accepted_ai_as_is = True
    except Exception as e:
        st.error(str(e))

left, right = st.columns([1, 1], gap="large")

# -------------------------
# LEFT: AI recommendation
# -------------------------
with left:
    st.markdown("### AI recommendation")

    if mode_key == "advisory" and not st.session_state.advisory_ready_for_ai:
        st.info("Advisory: first rank tasks on the right, then click “Reveal AI advice”.")
    else:
        if st.session_state.ai_order is None:
            if st.button("Generate AI ranking", type="primary"):
                try:
                    ensure_ai_generated(tasks, step_idx)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        else:
            show_reasons = st.toggle("Show AI reasons", value=False, key=f"show_reasons_{step_idx}")
            if show_reasons:
                st.session_state.viewed_reasons = True

            for rank_pos, idx in enumerate(st.session_state.ai_order, start=1):
                if show_reasons:
                    st.write(f"{rank_pos}. {tasks[idx]} — {st.session_state.reasons[idx]}")
                else:
                    st.write(f"{rank_pos}. {tasks[idx]}")

            if mode_key in ("advisory", "semi"):
                if st.button("Copy AI ranking to my ranking"):
                    st.session_state.user_order = st.session_state.ai_order[:]
                    st.session_state.accepted_ai_as_is = True
                    st.rerun()

# -------------------------
# RIGHT: User ranking
# -------------------------
with right:
    st.markdown("### Your ranking")

    if mode_key == "advisory":
        st.caption("You rank first. Then you may view AI advice. You decide final ranking.")
    elif mode_key == "semi":
        st.caption("AI ranked first and prefilled. You may modify before submitting.")
    else:
        st.caption("Locked: AI determines final ranking in this mode.")

    editable = (mode_key != "full")

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
        if st.session_state.ai_order is None:
            st.info("Waiting for AI ranking...")
        else:
            st.session_state.user_order = st.session_state.ai_order[:]
            for pos, idx in enumerate(st.session_state.user_order, start=1):
                st.markdown(f"<div class='card'>{pos}. {tasks[idx]}</div>", unsafe_allow_html=True)
    else:
        if HAS_DND:
            label_by_idx = {i: f"{i}. {tasks[i]}" for i in range(n)}
            current_labels = [label_by_idx[i] for i in st.session_state.user_order]

            try:
                sorted_labels = sort_items(
                    current_labels,
                    direction="vertical",
                    key=f"dnd_{step_idx}",
                )
                inv = {v: k for k, v in label_by_idx.items()}
                st.session_state.user_order = [inv[x] for x in sorted_labels]
            except Exception:
                HAS_DND = False

        if not HAS_DND:
            order = st.session_state.user_order
            for pos, idx in enumerate(order):
                row = st.columns([8, 1, 1])
                row[0].write(f"{pos+1}. {tasks[idx]}")
                if row[1].button("↑", key=f"up_{step_idx}_{pos}", disabled=(pos == 0)):
                    order[pos-1], order[pos] = order[pos], order[pos-1]
                    st.session_state.user_order = order
                    st.session_state.accepted_ai_as_is = False
                    st.rerun()
                if row[2].button("↓", key=f"down_{step_idx}_{pos}", disabled=(pos == len(order) - 1)):
                    order[pos+1], order[pos] = order[pos], order[pos+1]
                    st.session_state.user_order = order
                    st.session_state.accepted_ai_as_is = False
                    st.rerun()

        if mode_key == "advisory" and not st.session_state.advisory_ready_for_ai:
            if st.button("Reveal AI advice (I finished my initial ranking)"):
                st.session_state.advisory_ready_for_ai = True
                st.rerun()

# -------------------------
# Status
# -------------------------
st.divider()
ai_order = st.session_state.ai_order
final_order = st.session_state.user_order

if ai_order is None:
    st.info("AI ranking not generated yet.")
else:
    kdist = kendall_tau_distance(ai_order, final_order)
    moves_count = sum(1 for i in range(n) if final_order[i] != ai_order[i])
    st.info(f"Difference from AI: {kdist} (Kendall inversions) | Position mismatches: {moves_count}")
    st.session_state.accepted_ai_as_is = (final_order == ai_order)

# -------------------------
# Questionnaire
# -------------------------
st.subheader("Quick questions for this step (10–15 seconds)")
satisfaction = st.slider(
    "Satisfaction (1–7): I am satisfied with the final ranking.",
    LIKERT_MIN, LIKERT_MAX, 4, key=f"satisfaction_{step_idx}"
)

answers = {}
for key, label in MEASURE_ITEMS:
    answers[key] = st.slider(label, LIKERT_MIN, LIKERT_MAX, 4, key=f"{key}_{step_idx}")

manipulation = st.radio(
    "In this step, who made the final decision?",
    ["Mostly me", "Mostly the AI", "Both equally"],
    index=0,
    key=f"manip_{step_idx}"
)

advisory_ack = True
attention_ack = True
if mode_key == "advisory":
    advisory_ack = st.checkbox("I understand I am responsible for the final decision.", key=f"ack_resp_{step_idx}")
if mode_key == "full":
    attention_ack = st.checkbox("I have reviewed the ranking.", key=f"ack_seen_{step_idx}")

# -------------------------
# Submit
# -------------------------
submit_disabled = False
if mode_key == "advisory" and not advisory_ack:
    submit_disabled = True
if mode_key == "full" and not attention_ack:
    submit_disabled = True
if ai_order is None:
    submit_disabled = True

if st.button("Confirm & Next", type="primary", disabled=submit_disabled):
    now = time.time()
    t0 = st.session_state.t_step_start
    t_ai = st.session_state.t_ai_shown or t0

    time_to_ai = t_ai - t0
    decision_time = now - t_ai
    time_to_submit = now - t0

    moves_count = sum(1 for i in range(n) if final_order[i] != ai_order[i])
    kdist = kendall_tau_distance(ai_order, final_order)
    accepted = int(bool(final_order == ai_order))

    append_log([
        datetime.utcnow().isoformat(),
        st.session_state.participant_id,
        step_idx + 1,
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

    st.session_state.records["steps"].append({
        "timestamp_utc": datetime.utcnow().isoformat(),
        "step": step_idx + 1,
        "mode": mode_key,
        "tasks": tasks,
        "ai_order": ai_order,
        "final_order": final_order,
        "time_to_ai_sec": round(time_to_ai, 3),
        "decision_time_sec": round(decision_time, 3),
        "time_to_submit_sec": round(time_to_submit, 3),
        "moves_count": moves_count,
        "kendall_tau_to_ai": kdist,
        "accepted_ai_as_is": bool(accepted),
        "viewed_reasons": bool(st.session_state.viewed_reasons),
        "manipulation_check": manipulation,
        "satisfaction_1to7": satisfaction,
        "trust_1to7": answers["trust"],
        "control_1to7": answers["control"],
        "responsibility_1to7": answers["responsibility"],
        "effort_1to7": answers["effort"],
        "useful_1to7": answers["useful"],
    })

    st.session_state.step += 1

    # Reset per-step UI state (keep participant tasks!)
    st.session_state.t_step_start = None
    st.session_state.t_ai_shown = None
    st.session_state.ai_order = None
    st.session_state.reasons = None
    st.session_state.user_order = None
    st.session_state.viewed_reasons = False
    st.session_state.accepted_ai_as_is = False
    st.session_state.advisory_ready_for_ai = False
    st.session_state.ai_generated_step_idx = None

    if st.session_state.step >= TOTAL_STEPS:
        st.session_state.done = True

    st.rerun()
