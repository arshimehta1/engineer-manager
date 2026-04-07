from __future__ import annotations

import time
from typing import Dict, List, Optional

import streamlit as st

from focus_resource_env import (
    DEEP_WORK,
    EMPTY,
    OP_IDLE,
    OP_MUTE_COMMS,
    OP_RESCHEDULE_MEETING,
    FocusResourceEnv,
    Task,
)

BLOCK_TYPES = ["Focus", "Meeting"]
COMPLEXITY_OPTIONS = [1.0, 1.25, 1.5, 1.75]
DEFAULT_TASK_NAMES = ["Architecture", "Review", "Execution"]


def inject_styles() -> None:
    theme_tokens = {
        "__BG0__": "#050816",
        "__BG1__": "#0b1224",
        "__LINE__": "rgba(159, 184, 255, 0.16)",
        "__TEXT__": "#edf3ff",
        "__MUTED__": "#96a7cb",
        "__ACCENT__": "#7ce7ff",
        "__ACCENT2__": "#7cf0c5",
        "__BUTTON_TEXT__": "#06131c",
        "__METRIC_BG__": "linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02))",
        "__SHADOW__": "0 14px 40px rgba(0, 0, 0, 0.22)",
        "__INPUT_BG__": "rgba(255,255,255,0.04)",
        "__HERO_BG__": "linear-gradient(135deg, rgba(16, 26, 52, 0.94), rgba(11, 20, 38, 0.94))",
    }
    with open("styles.css", encoding="utf-8") as css_file:
        css = css_file.read()
    for key, value in theme_tokens.items():
        css = css.replace(key, value)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def friendly_error(exc: Exception) -> str:
    if "end_hour must be after start_hour" in str(exc):
        return "End time must be later than start time."
    return "Those control panel settings do not fit together yet. Try adjusting the workday range and resetting the studio."


def create_env(start_hour: str, end_hour: str, distraction_risk: float, seed: int) -> FocusResourceEnv:
    return FocusResourceEnv(
        start_hour=start_hour,
        end_hour=end_hour,
        distraction_risk=distraction_risk,
        seed=seed,
    )


def default_task_name(index: int) -> str:
    return DEFAULT_TASK_NAMES[index] if index < len(DEFAULT_TASK_NAMES) else f"Task {index + 1}"


def init_state() -> None:
    st.session_state.setdefault("ui_error", "")
    st.session_state.setdefault("ui_error_until", 0.0)
    st.session_state.setdefault("start_hour", "09:00")
    st.session_state.setdefault("end_hour", "17:00")
    st.session_state.setdefault("distraction_risk", 0.15)
    st.session_state.setdefault("seed", 7)
    st.session_state.setdefault("selection_start", None)
    st.session_state.setdefault("selected_range", None)
    st.session_state.setdefault("selected_block_id", None)
    st.session_state.setdefault("move_block_id", None)
    st.session_state.setdefault("armed_task_index", None)
    st.session_state.setdefault("next_block_id", 1)


def build_initial_blocks(env: FocusResourceEnv) -> List[dict]:
    blocks: List[dict] = []
    seen_meetings = set()
    for _, meta in sorted(env.meeting_meta.items()):
        meeting_id = meta["meeting_id"]
        if meeting_id in seen_meetings:
            continue
        seen_meetings.add(meeting_id)
        blocks.append(
            {
                "id": f"meeting-{meeting_id}",
                "start": int(meta["start"]),
                "end": int(meta["start"] + meta["length"] - 1),
                "type": "Meeting",
                "label": "Meeting",
                "priority": int(meta["priority"]),
            }
        )
    return sorted(blocks, key=lambda block: block["start"])


def ensure_task_names(env: FocusResourceEnv) -> None:
    names = st.session_state.get("task_names", [])
    st.session_state.task_names = [names[i] if i < len(names) else default_task_name(i) for i in range(len(env.task_buffer))]


def sync_from_env(env: FocusResourceEnv) -> None:
    st.session_state.env = env
    st.session_state.observation = env._observation()
    st.session_state.done = env.current_slot >= env.timeline_length
    st.session_state.setdefault("last_reward", 0.0)
    st.session_state.setdefault("last_info", {})
    ensure_task_names(env)


def get_env() -> FocusResourceEnv | None:
    if "env" not in st.session_state:
        try:
            env = create_env(
                st.session_state.get("start_hour", "09:00"),
                st.session_state.get("end_hour", "17:00"),
                float(st.session_state.get("distraction_risk", 0.15)),
                int(st.session_state.get("seed", 7)),
            )
            env.reset()
            sync_from_env(env)
            st.session_state.blocks = build_initial_blocks(env)
        except ValueError as exc:
            set_ui_error(friendly_error(exc), seconds=6.0)
            return None
    return st.session_state.env


def set_ui_error(message: str, seconds: float = 5.0) -> None:
    st.session_state.ui_error = message
    st.session_state.ui_error_until = time.time() + seconds


def clear_ui_error() -> None:
    st.session_state.ui_error = ""
    st.session_state.ui_error_until = 0.0


def render_flash_error() -> None:
    message = st.session_state.get("ui_error", "")
    until = float(st.session_state.get("ui_error_until", 0.0))
    if not message or time.time() >= until:
        if message:
            clear_ui_error()
        return
    remaining_ms = max(0, int((until - time.time()) * 1000))
    st.markdown(
        f"""
        <div id="focus-studio-error" style="
            margin: 0.4rem 0 1rem 0;
            padding: 0.9rem 1rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 120, 145, 0.25);
            background: linear-gradient(180deg, rgba(90, 19, 33, 0.88), rgba(60, 15, 25, 0.88));
            color: #ffe8ed;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.22);
        ">
            <strong>Check the plan</strong><br>{message}
        </div>
        <script>
        setTimeout(function() {{
            const el = window.parent.document.getElementById("focus-studio-error");
            if (el) {{
                el.style.transition = "opacity 220ms ease";
                el.style.opacity = "0";
                setTimeout(function() {{ if (el) el.remove(); }}, 240);
            }}
        }}, {remaining_ms});
        </script>
        """,
        unsafe_allow_html=True,
    )


def reset_env(start_hour: str, end_hour: str, distraction_risk: float, seed: int) -> None:
    env = create_env(start_hour, end_hour, distraction_risk, seed)
    env.reset()
    st.session_state.start_hour = start_hour
    st.session_state.end_hour = end_hour
    st.session_state.distraction_risk = float(distraction_risk)
    st.session_state.seed = int(seed)
    st.session_state.last_reward = 0.0
    st.session_state.last_info = {}
    clear_ui_error()
    st.session_state.selection_start = None
    st.session_state.selected_range = None
    st.session_state.selected_block_id = None
    st.session_state.move_block_id = None
    st.session_state.armed_task_index = None
    st.session_state.blocks = build_initial_blocks(env)
    sync_from_env(env)
    st.rerun()


def sync_tasks_from_widgets() -> None:
    env = st.session_state.env
    new_buffer: List[Task] = []
    new_names: List[str] = []
    for i, task in enumerate(env.task_buffer):
        name = str(st.session_state.get(f"task_name_{i}", default_task_name(i))).strip() or default_task_name(i)
        slots = max(1, int(st.session_state.get(f"task_slots_{i}", task.duration)))
        complexity = float(st.session_state.get(f"task_complexity_{i}", task.hidden_complexity))
        new_buffer.append(Task(duration=slots, hidden_complexity=complexity))
        new_names.append(name)
    env.task_buffer = new_buffer
    st.session_state.task_names = new_names


def block_at_slot(slot: int) -> Optional[dict]:
    for block in st.session_state.get("blocks", []):
        if block["start"] <= slot <= block["end"]:
            return block
    return None


def sort_blocks() -> None:
    st.session_state.blocks = sorted(st.session_state.get("blocks", []), key=lambda item: item["start"])


def remove_overlaps(start: int, end: int, ignore_id: Optional[str] = None) -> None:
    kept = []
    for block in st.session_state.get("blocks", []):
        overlaps = not (block["end"] < start or block["start"] > end)
        if overlaps and block["id"] != ignore_id:
            continue
        kept.append(block)
    st.session_state.blocks = kept


def apply_blocks_to_env() -> None:
    env = st.session_state.env
    env.timeline[:] = EMPTY
    env.meeting_meta = {}
    for block in st.session_state.get("blocks", []):
        if block["type"] == "Focus":
            env.timeline[block["start"] : block["end"] + 1] = DEEP_WORK
        else:
            env._place_meeting(
                block["start"],
                block["end"] - block["start"] + 1,
                int(block.get("priority", 5)),
                env._next_meeting_id(),
            )
    sync_from_env(env)


def clear_selection() -> None:
    st.session_state.selection_start = None
    st.session_state.selected_range = None
    st.session_state.selected_block_id = None
    st.session_state.move_block_id = None


def create_block(start: int, end: int, block_type: str, label: str, priority: int) -> None:
    start, end = sorted((start, end))
    remove_overlaps(start, end)
    st.session_state.blocks.append(
        {
            "id": f"user-{st.session_state.get('next_block_id', 1)}",
            "start": start,
            "end": end,
            "type": block_type,
            "label": label.strip() or block_type,
            "priority": int(priority),
        }
    )
    st.session_state.next_block_id = int(st.session_state.get("next_block_id", 1)) + 1
    sort_blocks()
    apply_blocks_to_env()
    clear_selection()
    st.rerun()


def update_block(block_id: str, block_type: str, label: str, priority: int) -> None:
    for block in st.session_state.get("blocks", []):
        if block["id"] == block_id:
            block["type"] = block_type
            block["label"] = label.strip() or block_type
            block["priority"] = int(priority)
            break
    apply_blocks_to_env()
    st.rerun()


def delete_block(block_id: str) -> None:
    st.session_state.blocks = [block for block in st.session_state.get("blocks", []) if block["id"] != block_id]
    apply_blocks_to_env()
    clear_selection()
    st.rerun()


def move_block(block_id: str, new_start: int) -> None:
    block = next((item for item in st.session_state.get("blocks", []) if item["id"] == block_id), None)
    if block is None:
        return
    duration = block["end"] - block["start"]
    new_end = min(st.session_state.env.timeline_length - 1, new_start + duration)
    new_start = max(0, new_end - duration)
    remove_overlaps(new_start, new_end, ignore_id=block_id)
    block["start"] = new_start
    block["end"] = new_end
    sort_blocks()
    apply_blocks_to_env()
    st.session_state.move_block_id = None
    st.session_state.selected_block_id = block_id
    st.rerun()


def arm_task(index: int) -> None:
    sync_tasks_from_widgets()
    st.session_state.armed_task_index = index


def place_armed_task(start: int, end: int) -> None:
    env = st.session_state.env
    sync_tasks_from_widgets()
    task_index = st.session_state.get("armed_task_index")
    if task_index is None or not (0 <= task_index < len(env.task_buffer)):
        set_ui_error("Pick a task from the queue before placing it on the calendar.")
        return
    task = env.task_buffer.pop(task_index)
    task_name = st.session_state.task_names.pop(task_index)
    desired = max(1, int(round(task.duration * task.hidden_complexity)))
    start, end = sorted((start, end))
    end = min(env.timeline_length - 1, max(end, start + desired - 1))
    st.session_state.armed_task_index = None
    create_block(start, end, "Focus", task_name, 5)


def run_simulation_step(operation: int) -> None:
    env = st.session_state.env
    sync_tasks_from_widgets()
    apply_blocks_to_env()
    current_slot = int(env.current_slot)
    target_slot = current_slot
    if operation == OP_RESCHEDULE_MEETING:
        future_meetings = [block for block in st.session_state.get("blocks", []) if block["type"] == "Meeting" and block["start"] >= current_slot]
        if not future_meetings:
            set_ui_error("No future meeting is available to move right now.")
            return
        target_slot = future_meetings[0]["start"]
    obs, reward, done, info = env.step((target_slot, operation))
    st.session_state.observation = obs
    st.session_state.last_reward = reward
    st.session_state.last_info = info
    st.session_state.done = done
    if operation == OP_RESCHEDULE_MEETING and info.get("action_info", {}).get("status") == "meeting_rescheduled":
        moved = next((block for block in st.session_state.get("blocks", []) if block["type"] == "Meeting" and block["start"] == int(info["action_info"]["from_slot"])), None)
        if moved is not None:
            duration = moved["end"] - moved["start"]
            moved["start"] = int(info["action_info"]["to_slot"])
            moved["end"] = moved["start"] + duration
            sort_blocks()
    st.rerun()


def render_header(observation: Dict[str, object]) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Cognitive Resource Simulator</div>
            <h1>engineer-manager</h1>
            <p>Design the day directly on the calendar, lock in meaningful work blocks, and pressure-test the plan against interruption cost with a cleaner single-source workflow.</p>
            <span class="help-chip">One interface, one truth</span>
            <span class="help-chip">Blocks keep their own type and label</span>
            <span class="help-chip">The task queue never rewrites scheduled work</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    a, b, c, d = st.columns(4)
    a.metric("Current time", observation["current_time"])
    b.metric("Current slot", str(observation["current_slot"]))
    c.metric("Focus Fortress", "Active" if observation["mute_comms"] else "Inactive")
    d.metric("Last reward", f"{st.session_state.get('last_reward', 0.0):.2f}")


def render_setup() -> None:
    st.markdown("### Control Panel")
    start_hour = st.text_input("Start", value=st.session_state.get("start_hour", "09:00"), key="setup_start")
    end_hour = st.text_input("End", value=st.session_state.get("end_hour", "17:00"), key="setup_end")
    risk = st.slider(
        "Distraction risk",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("distraction_risk", 0.15)),
        step=0.05,
        help="Higher values create noisier days and lower uninterrupted focus potential.",
        key="setup_risk",
    )
    seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=100000,
        value=int(st.session_state.get("seed", 7)),
        step=1,
        help="? Scenario Consistency: Use a fixed Seed number to regenerate this exact daily challenge for comparison testing.",
        key="setup_seed",
    )
    st.caption("Professional scheduling note: the end time must be later than the start time for the studio to build a valid workday.")
    if st.button("Reset Studio", use_container_width=True):
        try:
            reset_env(start_hour, end_hour, float(risk), int(seed))
        except ValueError as exc:
            set_ui_error(friendly_error(exc), seconds=6.0)


def render_calendar(env: FocusResourceEnv) -> None:
    st.markdown("### Day Plan")
    st.caption("Click one open slot to start a range. Click a second slot to finish it. Click an existing block to edit it or move it.")
    current_slot = int(env.current_slot)
    selection_start = st.session_state.get("selection_start")
    selected_range = st.session_state.get("selected_range")
    selected_block = next((block for block in st.session_state.get("blocks", []) if block["id"] == st.session_state.get("selected_block_id")), None)
    move_block_id = st.session_state.get("move_block_id")

    for slot in range(env.timeline_length):
        block = block_at_slot(slot)
        time_col, body_col, action_col = st.columns([1.0, 4.25, 1.95])
        time_col.markdown(f"**{env._slot_label(slot)}**")
        if block is None:
            label = "Open"
        else:
            label = f"{block['type']} | {block['label']}"
        if slot == current_slot:
            body_col.markdown(f"`NOW` {label}")
        elif selection_start is not None and min(selection_start, slot) <= slot <= max(selection_start, slot):
            body_col.markdown(f"`SELECTED` {label}")
        else:
            body_col.markdown(label)
        if move_block_id:
            if action_col.button("Drop", key=f"slot_drop_{slot}", use_container_width=True, type="secondary"):
                move_block(move_block_id, slot)
        elif block is not None:
            start_col, edit_col = action_col.columns([1, 1])
            start_text = "Finish" if selection_start is not None else "Start"
            if start_col.button(start_text, key=f"slot_start_{slot}", use_container_width=True, type="primary"):
                if selection_start is None:
                    st.session_state.selection_start = slot
                else:
                    st.session_state.selected_range = (selection_start, slot)
                st.rerun()
            if edit_col.button("Edit", key=f"slot_edit_{slot}", use_container_width=True, type="secondary"):
                st.session_state.selected_block_id = block["id"]
                st.session_state.selection_start = None
                st.session_state.selected_range = None
                st.rerun()
        else:
            button_text = "Start" if selection_start is None else "Finish"
            if action_col.button(button_text, key=f"slot_open_{slot}", use_container_width=True, type="primary"):
                if selection_start is None:
                    st.session_state.selection_start = slot
                else:
                    st.session_state.selected_range = (selection_start, slot)
                st.rerun()

    if selected_range is not None:
        start, end = sorted(selected_range)
        st.markdown("#### New block")
        block_type = st.radio("Type", options=BLOCK_TYPES, horizontal=True, key="new_block_type")
        armed_task_index = st.session_state.get("armed_task_index")
        if block_type == "Focus" and armed_task_index is not None and armed_task_index < len(st.session_state.get("task_names", [])):
            default_label = st.session_state["task_names"][armed_task_index]
        else:
            default_label = ""
        label = st.text_input("Label", value=default_label, key="new_block_label")
        priority = st.slider("Meeting priority", 1, 10, 5, key="new_block_priority")
        a, b, c = st.columns(3)
        if a.button("Create block", use_container_width=True):
            if block_type == "Focus" and armed_task_index is not None:
                place_armed_task(start, end)
            else:
                create_block(start, end, block_type, label, int(priority))
        if b.button("Clear to open", use_container_width=True):
            remove_overlaps(start, end)
            apply_blocks_to_env()
            clear_selection()
            st.rerun()
        if c.button("Cancel selection", use_container_width=True):
            clear_selection()
            st.rerun()

    if selected_block is not None:
        st.markdown("#### Edit block")
        block_type = st.radio("Block type", options=BLOCK_TYPES, index=BLOCK_TYPES.index(selected_block["type"]), horizontal=True, key="edit_block_type")
        label = st.text_input("Block label", value=selected_block["label"], key="edit_block_label")
        priority = st.slider("Meeting priority", 1, 10, int(selected_block.get("priority", 5)), key="edit_block_priority")
        a, b, c = st.columns(3)
        if a.button("Save block", use_container_width=True):
            update_block(selected_block["id"], block_type, label, int(priority))
        if b.button("Move block", use_container_width=True):
            st.session_state.move_block_id = selected_block["id"]
            st.rerun()
        if c.button("Delete block", use_container_width=True):
            delete_block(selected_block["id"])
        if st.button("Close inspector", use_container_width=True):
            clear_selection()
            st.rerun()


def render_task_queue(env: FocusResourceEnv) -> None:
    st.markdown("### Task Queue")
    sync_tasks_from_widgets()
    if not env.task_buffer:
        st.info("The queue is empty. Add a task to keep planning.")
    for i, task in enumerate(env.task_buffer):
        cols = st.columns([2.0, 0.78, 0.92, 1.38, 0.52])
        cols[0].text_input("Task", value=st.session_state.get("task_names", [default_task_name(i)])[i], key=f"task_name_{i}", label_visibility="collapsed")
        cols[1].number_input("Slots", min_value=1, max_value=12, value=int(task.duration), key=f"task_slots_{i}", label_visibility="collapsed")
        current_complexity = float(task.hidden_complexity)
        cols[2].selectbox("Complexity", COMPLEXITY_OPTIONS, index=COMPLEXITY_OPTIONS.index(current_complexity) if current_complexity in COMPLEXITY_OPTIONS else 0, key=f"task_complexity_{i}", label_visibility="collapsed")
        armed = st.session_state.get("armed_task_index") == i
        if cols[3].button("Selected" if armed else "Use", key=f"use_task_{i}", use_container_width=True):
            arm_task(i)
            st.rerun()
        if cols[4].button("X", key=f"cancel_task_{i}", use_container_width=True):
            sync_tasks_from_widgets()
            env.task_buffer.pop(i)
            st.session_state.task_names.pop(i)
            st.session_state.armed_task_index = None
            st.rerun()

    st.markdown("#### Add task")
    a, b, c, d = st.columns([2.2, 0.8, 1.0, 0.9])
    task_name = a.text_input("Name", value="", placeholder="Execution block", key="add_task_name")
    slots = b.number_input("Slots", min_value=1, max_value=12, value=2, key="add_task_slots")
    complexity = c.selectbox("Complexity", COMPLEXITY_OPTIONS, index=1, key="add_task_complexity")
    if d.button("Add", use_container_width=True):
        sync_tasks_from_widgets()
        env.task_buffer.append(Task(duration=int(slots), hidden_complexity=float(complexity)))
        st.session_state.task_names.append(task_name.strip() or default_task_name(len(st.session_state.task_names)))
        st.rerun()


def render_status(observation: Dict[str, object]) -> None:
    st.markdown("### Live Scoring")
    scores = st.session_state.get("last_info", {}).get("score_breakdown", {})
    current_block = block_at_slot(int(observation["current_slot"]))
    a, b = st.columns(2)
    a.metric("Flow efficiency", f"{scores.get('flow_score', observation.get('flow_score', 0.0)):.2f}")
    recovery_state = int(observation.get("recovery_state", 0))
    card_class = "status-card-warning" if recovery_state > 0 else "status-card-subtle"
    current_flow_label = current_block["label"] if current_block and current_block["type"] == "Focus" else "Open"
    current_flow_note = "Recovery is active. Focus output is temporarily reduced." if recovery_state > 0 else "Flow is live and ready to compound."
    with b:
        st.markdown(
            f"""
            <div class="status-card {card_class}">
                <div class="status-card-label">Current Flow Block</div>
                <div class="status-card-value">{current_flow_label}</div>
                <div class="status-card-label" style="margin-top:0.4rem;text-transform:none;letter-spacing:0;color:var(--muted);">
                    {current_flow_note}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    c, d = st.columns(2)
    c.metric("Social Debt", f"{scores.get('social_debt', observation.get('social_debt', 0.0)):.2f}")
    d.metric("Calendar Churn", int(scores.get('calendar_churn', observation.get('calendar_churn', 0))))


def render_simulator(observation: Dict[str, object]) -> None:
    st.markdown("### Simulation Command Center")
    quiet_mode = bool(observation.get("mute_comms", False))
    st.caption(
        "?? Focus Fortress: Quiet Mode is active. Notifications are suppressed, boosting your continuous work potential."
        if quiet_mode
        else "?? Focus Fortress is standing by. Activate it when you want maximum uninterrupted focus potential."
    )
    a, b = st.columns(2)
    if a.button("Deactivate Focus Fortress" if quiet_mode else "Activate Focus Fortress", use_container_width=True):
        run_simulation_step(OP_MUTE_COMMS)
    if b.button("Step Simulator", use_container_width=True):
        run_simulation_step(OP_IDLE)
    if st.button("Move next meeting", use_container_width=True):
        run_simulation_step(OP_RESCHEDULE_MEETING)


def main() -> None:
    st.set_page_config(page_title="Focus Studio", layout="wide")
    init_state()
    inject_styles()
    env = get_env()
    if env is None:
        render_flash_error()
        st.stop()
    observation = st.session_state.get("observation", env._observation())
    render_flash_error()
    render_header(observation)
    col1, col2, col3 = st.columns([0.88, 1.5, 1.28], gap="large")
    with col1:
        render_setup()
    with col2:
        render_calendar(env)
    with col3:
        render_task_queue(env)
        render_status(observation)
        render_simulator(observation)


if __name__ == "__main__":
    main()
