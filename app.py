import streamlit as st
import time, uuid, json, csv, os
from datetime import datetime
from groq import Groq
import random

# -------------------------
# OPTIONAL: Drag & Drop
# -------------------------
# If you want drag-and-drop, add to requirements.txt:
#   streamlit-sortables>=0.3.1
HAS_DND = False
try:
    from streamlit_sortables import sort_items  # type: ignore
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
    ("responsibility", "Responsibility (1–7): I feel responsible for the final ranking."),
    ("effort", "Effort (1–7): This step required effort from me."),
    ("useful", "Usefulness (1–7): The AI ranking was useful."),
]

# Fixed order (swapped): Advisory -> Full -> Semi
MODES = [
    ("advisory", "Advisory: You rank first, then AI advises"),
    ("full", "Fully automated: AI decides (locked)"),
    ("semi", "Semi-automated: AI ranks first, you may modify"),
]

# Broad task pool (population-relevant; no "supervisor")
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
    "Confirm an appointment time and location",
    "Make a decision for something expiring soon",
    "Follow up on a payment that didn’t go through",
    "Handle an urgent household issue (water/electricity) today",
]

TASKS_PER_PARTICIPANT = 5
TOTAL_STEPS = 3

# Optional CSV log (local to Streamlit server; on Streamlit Cloud it is NOT persistent long-term)
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

def _sanitize_ranking(ranking, n: int):
    """
    Robustly convert whatever the model returned into a valid permutation 0..n-1.
    - Remove duplicates
    - Remove non-ints/out-of-range
    - Fill missing in natural order
    """
    if not isinstance(ranking, list):
        return list(range(n)), True

    cleaned = []
    seen = set()
    for x in ranking:
        if isinstance(x, bool):
            continue
        if isinstance(x, int) and 0 <= x < n and x not in seen:
            cleaned.append(x)
            seen.add(x)

    missing = [i for i in range(n) if i not in seen]
    fixed = cleaned + missing

    repaired = (sorted(fixed) != list(range(n)))
    return fixed, repaired

def _tokenize_keywords(task: str):
    stop = {
        "the","a","an","to","for","of","and","or","in","on","by","with","before",
        "is","are","was","were","be","been","this","that","today","tomorrow",
        "soon","needed","no","deadline","due"
    }
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in task)
    words = [w for w in cleaned.split() if len(w) >= 4 and w not in stop]
    uniq = []
    for w in words:
        if w not in uniq:
            uniq.append(w)
    return uniq[:4]

def _reason_matches_task(reason: str, task: str) -> bool:
    r = (reason or "").lower()
    kws = _tokenize_keywords(task)
    if not kws:
        return True
    return any(k in r for k in kws)

# =========================
# GROQ: 2-call approach (ranking then reasons-by-id, with validation/repair)
# =========================
def groq_rank_tasks(tasks):
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in Streamlit secrets.")
    client = Groq(api_key=api_key)

    n = len(tasks)

    # ---- Call 1: ranking only (easier to validate) ----
    prompt_rank = f"""
You are ranking tasks using a fixed rubric:
1) Urgency (time-sensitive, soonest deadline)
2) Importance (high impact, consequences if delayed)
3) Deadline proximity (explicit deadlines beat vague ones)
Tie-break: clearer negative consequences if delayed.

Return ONLY valid JSON.
JSON schema:
{{
  "ranking": [int, ...]  // permutation of 0..{n-1}, highest priority first
}}

Tasks (index = task id):
{json.dumps(tasks, ensure_ascii=False)}
""".strip()

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Output JSON only. No prose. No markdown."},
            {"role": "user", "content": prompt_rank},
        ],
        temperature=0.0,
    )

    data = safe_extract_json(resp.choices[0].message.content)
    ranking_raw = data.get("ranking")
    ranking, repaired = _sanitize_ranking(ranking_raw, n)

    # Retry once if ranking needed repair
    if repaired:
        resp2 = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Output JSON only. No prose. No markdown."},
                {"role": "user", "content": prompt_rank + "\n\nYour previous ranking was invalid. Return a correct permutation."},
            ],
            temperature=0.0,
        )
        data2 = safe_extract_json(resp2.choices[0].message.content)
        ranking2, repaired2 = _sanitize_ranking(data2.get("ranking"), n)
        if not repaired2:
            ranking = ranking2

    # ---- Call 2: reasons per task-id (prevents cross-task leakage) ----
    prompt_reasons = f"""
Write ONE short reason per task, using the SAME rubric.
Rules:
- One reason per task-id.
- Do NOT mention any other task.
- If the task has a time cue (today/tomorrow/closing/due/this week/etc.), include it.
- Keep each reason <= 14 words.

Return ONLY JSON:
{{
  "reasons_by_id": [string, ...]  // length {n}; reasons_by_id[i] explains task i
}}

Tasks:
{json.dumps(tasks, ensure_ascii=False)}
""".strip()

    resp_r = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Output JSON only. No prose. No markdown."},
            {"role": "user", "content": prompt_reasons},
        ],
        temperature=0.0,
    )
    data_r = safe_extract_json(resp_r.choices[0].message.content)
    reasons = data_r.get("reasons_by_id")

    if not isinstance(reasons, list):
        reasons = [""] * n
    reasons = [str(x) for x in reasons[:n]]
    if len(reasons) < n:
        reasons.extend([""] * (n - len(reasons)))

    # ---- Validate reasons match their task; repair only the broken ones ----
    bad_ids = [i for i in range(n) if not _reason_matches_task(reasons[i], tasks[i])]

    if bad_ids:
        subset = [{"id": i, "task": tasks[i]} for i in bad_ids]
        prompt_repair = f"""
For EACH item below, write a short reason for THAT task only.
Rules:
- Do NOT mention any other task.
- Include a time cue if present in the task.
- <= 14 words.

Return ONLY JSON:
{{
  "reasons_patch": {{ "id": "reason", ... }}
}}

Items:
{json.dumps(subset, ensure_ascii=False)}
""".strip()

        resp_fix = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Output JSON only. No prose. No markdown."},
                {"role": "user", "content": prompt_repair},
            ],
            temperature=0.0,
        )
        data_fix = safe_extract_json(resp_fix.choices[0].message.content)
        patch = data_fix.get("reasons_patch", {})

        if isinstance(patch, dict):
            for k, v in patch.items():
                try:
                    i = int(k)
                    if 0 <= i < n:
                        reasons[i] = str(v)
                except Exception:
                    pass

    return {"ranking": ranking, "reasons_by_id": reasons}

# =========================
# SESSION INIT
# =========================
def init():
    if "participant_id" not in st.session_state:
        st.session_state.participant_id = str(uuid.uuid4())[:8]

    if "mode_order" not in st.session_state:
        st.session_state.mode_order = MODES[:]  # fixed order

    if "step" not in st.session_state:
        st.session_state.step = 0

    # Fixed tasks PER participant (same tasks across all 3 stages for that participant)
    if "tasks_fixed" not in st.session_state:
        rng = random.Random(f"{st.session_state.participant_id}-fixed")
        st.session_state.tasks_fixed = rng.sample(TASK_POOL, TASKS_PER_PARTICIPANT)

    # Per-step state
    if "t_step_start" not in st.session_state:
        st.session_state.t_step_start = None
    if "t_ai_shown" not in st.session_state:
        st.session_state.t_ai_shown = None

    if "ai_order" not in st.session_state:
        st.session_state.ai_order = None
    if "reasons_by_id" not in st.session_state:
        st.session_state.reasons_by_id = None

    if "user_order" not in st.session_state:
        st.session_state.user_order = None

    if "advisory_ready_for_ai" not in st.session_state:
        st.session_state.advisory_ready_for_ai = False

    # Track if AI already generated for this step
    if "ai_generated_step_idx" not in st.session_state:
        st.session_state.ai_generated_step_idx = None

    if "viewed_reasons" not in st.session_state:
        st.session_state.viewed_reasons = False
    if "accepted_ai_as_is" not in st.session_state:
        st.session_state.accepted_ai_as_is = False

    # Downloadable JSON records
    if "records" not in st.session_state:
        st.session_state.records = {
            "participant_id": st.session_state.participant_id,
            "mode_order": [m[0] for m in st.session_state.mode_order],
            "tasks_fixed": st.session_state.tasks_fixed,
            "steps": [],
        }

    if "done" not in st.session_state:
        st.session_state.done = False

init()

# =========================
# AI generation (once per step)
# =========================
def ensure_ai_generated(tasks, step_idx):
    if st.session_state.ai_generated_step_idx == step_idx and st.session_state.ai_order is not None:
        return
    out = groq_rank_tasks(tasks)
    st.session_state.t_ai_shown = time.time()
    st.session_state.ai_order = out["ranking"]
    st.session_state.reasons_by_id = out["reasons_by_id"]
    st.session_state.ai_generated_step_idx = step_idx

# =========================
# UI
# =========================
st.title("Everyday Task Prioritization Study")
st.caption("Same 5 tasks for all 3 steps (per participant). Only control level changes. (Order: Advisory → Full → Semi)")

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

# DONE screen
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
tasks = st.session_state.tasks_fixed[:]  # SAME tasks in all stages for this participant
n = len(tasks)

st.progress((step_idx + 1) / TOTAL_STEPS)
st.subheader(f"Step {step_idx+1}/{TOTAL_STEPS} — {mode_label}")

if st.session_state.t_step_start is None:
    st.session_state.t_step_start = time.time()

if st.session_state.user_order is None:
    st.session_state.user_order = list(range(n))

# Auto-generate AI for full + semi
if mode_key in ("full", "semi"):
    try:
        ensure_ai_generated(tasks, step_idx)
        # Semi: prefill user's ranking with AI ranking (only first time on this step)
        if mode_key == "semi" and st.session_state.user_order == list(range(n)):
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

            reasons_by_id = st.session_state.reasons_by_id or [""] * n
            for rank_pos, idx in enumerate(st.session_state.ai_order, start=1):
                if show_reasons:
                    st.write(f"{rank_pos}. {tasks[idx]} — {reasons_by_id[idx]}")
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
    elif mode_key == "full":
        st.caption("Locked: AI determines final ranking in this mode.")
    else:
        st.caption("AI ranked first and prefilled. You may modify before submitting.")

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
        dnd_ok = HAS_DND
        if dnd_ok:
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
                dnd_ok = False

        if not dnd_ok:
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

        # Advisory gating
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
    viewed = int(bool(st.session_state.viewed_reasons))

    # Optional CSV logging (best-effort)
    try:
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
            viewed,
            manipulation,
            satisfaction,
            answers["trust"],
            answers["control"],
            answers["responsibility"],
            answers["effort"],
            answers["useful"],
        ])
    except Exception:
        pass

    # Save into downloadable participant JSON
    st.session_state.records["steps"].append({
        "timestamp_utc": datetime.utcnow().isoformat(),
        "step": step_idx + 1,
        "mode": mode_key,
        "tasks": tasks,
        "ai_order": ai_order,
        "ai_reasons_by_id": (st.session_state.reasons_by_id or [""] * n),
        "final_order": final_order,
        "time_to_ai_sec": round(time_to_ai, 3),
        "decision_time_sec": round(decision_time, 3),
        "time_to_submit_sec": round(time_to_submit, 3),
        "moves_count": moves_count,
        "kendall_tau_to_ai": kdist,
        "accepted_ai_as_is": bool(accepted),
        "viewed_reasons": bool(viewed),
        "manipulation_check": manipulation,
        "satisfaction_1to7": satisfaction,
        "trust_1to7": answers["trust"],
        "control_1to7": answers["control"],
        "responsibility_1to7": answers["responsibility"],
        "effort_1to7": answers["effort"],
        "useful_1to7": answers["useful"],
    })

    # Advance step
    st.session_state.step += 1

    # Reset per-step UI state
    st.session_state.t_step_start = None
    st.session_state.t_ai_shown = None
    st.session_state.ai_order = None
    st.session_state.reasons_by_id = None
    st.session_state.user_order = None
    st.session_state.viewed_reasons = False
    st.session_state.accepted_ai_as_is = False
    st.session_state.advisory_ready_for_ai = False
    st.session_state.ai_generated_step_idx = None

    if st.session_state.step >= TOTAL_STEPS:
        st.session_state.done = True

    st.rerun()
