from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from focus_resource_env import DEEP_WORK, EMPTY, MEETING, FocusResourceEnv, Task


StepRecord = dict[str, Any]
TaskSetup = Callable[[FocusResourceEnv], None]
TaskGrader = Callable[[list[StepRecord]], float]


def _reset_state(env: FocusResourceEnv) -> None:
    env.timeline[:] = EMPTY
    env.meeting_meta = {}
    env.task_buffer = []
    env.current_slot = 0
    env.current_work_streak_slots = 0
    env.recovery_remaining = 0
    env.mute_comms = False
    env.social_debt = 0.0
    env.calendar_churn = 0
    env.flow_score = 0.0
    env.last_executed_kind = EMPTY
    env.interruptions = 0
    env.invalid_actions = 0


def _set_meeting(
    env: FocusResourceEnv,
    *,
    start: int,
    length: int,
    priority: int,
    meeting_id: int,
) -> None:
    env._place_meeting(start, length, priority, meeting_id)


def _normalized_total_score(env: FocusResourceEnv) -> float:
    max_score = max(1.0, (env.timeline_length * 0.5) ** 2)
    return min(1.0, max(0.0, env._total_score() / max_score))


def setup_quiet_morning(env: FocusResourceEnv) -> None:
    _reset_state(env)
    env.distraction_risk = 0.65
    env.task_buffer = [
        Task(duration=2, hidden_complexity=1.0),
        Task(duration=3, hidden_complexity=1.0),
        Task(duration=2, hidden_complexity=1.25),
    ]
    _set_meeting(env, start=5, length=1, priority=4, meeting_id=1)
    _set_meeting(env, start=7, length=1, priority=3, meeting_id=2)


def setup_meeting_surgery(env: FocusResourceEnv) -> None:
    _reset_state(env)
    env.distraction_risk = 0.10
    env.task_buffer = [
        Task(duration=2, hidden_complexity=1.0),
        Task(duration=2, hidden_complexity=1.25),
        Task(duration=1, hidden_complexity=1.0),
    ]
    _set_meeting(env, start=1, length=1, priority=2, meeting_id=1)
    _set_meeting(env, start=3, length=1, priority=2, meeting_id=2)
    _set_meeting(env, start=6, length=2, priority=8, meeting_id=3)


def setup_delivery_triage(env: FocusResourceEnv) -> None:
    _reset_state(env)
    env.distraction_risk = 0.25
    env.task_buffer = [
        Task(duration=3, hidden_complexity=1.5),
        Task(duration=2, hidden_complexity=1.0),
        Task(duration=1, hidden_complexity=1.0),
    ]
    _set_meeting(env, start=4, length=1, priority=9, meeting_id=1)
    _set_meeting(env, start=8, length=2, priority=7, meeting_id=2)


def grade_quiet_morning(trajectory: list[StepRecord]) -> float:
    if not trajectory:
        return 0.0
    first_action = int(trajectory[0]["action"]["operation"])
    final = trajectory[-1]["observation"]
    final_score = float(final["flow_score"])
    transition_count = sum(1 for step in trajectory if step["info"]["transition_info"]["interrupted"])
    scheduled = sum(1 for slot in final["timeline"] if int(slot) == DEEP_WORK)

    score = 0.0
    score += 0.25 if first_action == 3 else 0.0
    score += min(0.45, final_score / 6.0)
    score += 0.15 if transition_count == 0 else 0.0
    score += min(0.15, scheduled / 6.0)
    return min(1.0, round(score, 4))


def grade_meeting_surgery(trajectory: list[StepRecord]) -> float:
    if not trajectory:
        return 0.0
    final = trajectory[-1]["observation"]
    flow = float(final["flow_score"])
    debt = float(final["social_debt"])
    churn = int(final["calendar_churn"])
    reschedules = sum(
        1
        for step in trajectory
        if step["info"].get("action_info", {}).get("status") == "meeting_rescheduled"
    )

    score = 0.0
    score += min(0.40, flow / 5.0)
    score += 0.20 if reschedules >= 1 else 0.0
    score += 0.20 if 1 <= churn <= 2 else max(0.0, 0.20 - (0.10 * abs(churn - 1)))
    score += max(0.0, 0.20 - (debt / 8.0))
    return min(1.0, round(score, 4))


def grade_delivery_triage(trajectory: list[StepRecord]) -> float:
    if not trajectory:
        return 0.0
    final = trajectory[-1]["observation"]
    total = float(final["flow_score"]) - float(final["social_debt"]) - float(final["calendar_churn"])
    invalid_actions = sum(
        1
        for step in trajectory
        if str(step["info"].get("action_info", {}).get("status", "")).startswith("invalid")
    )
    remaining_tasks = len(final["task_buffer"])
    scheduled = sum(1 for slot in final["timeline"] if int(slot) == DEEP_WORK)

    score = 0.0
    score += min(0.45, max(0.0, total) / 6.0)
    score += min(0.25, scheduled / 8.0)
    score += 0.20 if remaining_tasks <= 1 else 0.10 if remaining_tasks == 2 else 0.0
    score += max(0.0, 0.10 - (0.05 * invalid_actions))
    return min(1.0, round(score, 4))


@dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    setup: TaskSetup
    grader: TaskGrader


TASK_SPECS: dict[str, TaskSpec] = {
    "quiet-morning": TaskSpec(
        name="quiet-morning",
        description="High-noise morning where the agent should mute comms early and protect an uninterrupted work block.",
        setup=setup_quiet_morning,
        grader=grade_quiet_morning,
    ),
    "meeting-surgery": TaskSpec(
        name="meeting-surgery",
        description="A fragmented calendar where the agent should improve flow with limited, selective meeting moves.",
        setup=setup_meeting_surgery,
        grader=grade_meeting_surgery,
    ),
    "delivery-triage": TaskSpec(
        name="delivery-triage",
        description="A constrained day with hidden task complexity where the agent must schedule useful work without spiraling debt.",
        setup=setup_delivery_triage,
        grader=grade_delivery_triage,
    ),
}


DEFAULT_TASK_NAME = "quiet-morning"


def get_task_spec(task_name: str | None) -> TaskSpec:
    normalized = (task_name or os.getenv("TASK_NAME") or DEFAULT_TASK_NAME).strip()
    return TASK_SPECS.get(normalized, TASK_SPECS[DEFAULT_TASK_NAME])


def apply_task(env: FocusResourceEnv, task_name: str | None) -> TaskSpec:
    spec = get_task_spec(task_name)
    spec.setup(env)
    return spec


def grade_trajectory(task_name: str, trajectory: list[StepRecord]) -> float:
    spec = get_task_spec(task_name)
    return spec.grader(trajectory)
