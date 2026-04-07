from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from focus_resource_env import (
    DEEP_WORK,
    EMPTY,
    MEETING,
    OP_RESCHEDULE_MEETING,
    FocusResourceEnv,
    Task,
)


@dataclass
class CaseResult:
    name: str
    passed: bool
    details: dict[str, Any]


def blank_env(length_slots: int, distraction_risk: float = 0.0, seed: int = 0) -> FocusResourceEnv:
    env = FocusResourceEnv(
        start_hour="09:00",
        end_hour=f"{9 + length_slots // 2:02d}:{'30' if length_slots % 2 else '00'}",
        distraction_risk=distraction_risk,
        seed=seed,
    )
    env.reset()
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
    return env


def run_day(env: FocusResourceEnv) -> list[dict[str, Any]]:
    events = []
    while env.current_slot < env.timeline_length:
        _, _, _, info = env.step((env.current_slot, 0))
        events.append(info["transition_info"])
    return events


def case_fragmentation_penalty() -> CaseResult:
    action_a = blank_env(8)
    action_a.timeline[:] = DEEP_WORK
    run_day(action_a)

    action_b = blank_env(11)
    action_b.timeline[:] = EMPTY
    for start in (0, 3, 6, 9):
        action_b.timeline[start : start + 2] = DEEP_WORK
    for slot in (2, 5, 8):
        action_b.timeline[slot] = MEETING
        action_b.meeting_meta[slot] = {
            "meeting_id": slot,
            "start": slot,
            "length": 1,
            "priority": 5,
        }
    events_b = run_day(action_b)

    return CaseResult(
        name="Fragmentation penalty",
        passed=action_a.flow_score > action_b.flow_score,
        details={
            "action_a_flow_score": action_a.flow_score,
            "action_b_flow_score": action_b.flow_score,
            "action_b_productive_steps": [i for i, ev in enumerate(events_b) if ev["productive"]],
            "expected_math_without_switching": {"one_block": 16, "four_blocks": 4},
        },
    )


def case_context_switch_recovery() -> CaseResult:
    env = blank_env(6)
    env.timeline[2] = MEETING
    env.meeting_meta[2] = {"meeting_id": 1, "start": 2, "length": 1, "priority": 5}
    env.timeline[3:6] = DEEP_WORK
    events = run_day(env)

    return CaseResult(
        name="Context switch recovery",
        passed=(
            events[3]["productive"] is False
            and events[4]["productive"] is False
            and events[5]["productive"] is True
        ),
        details={
            "10:30_event": events[3],
            "11:00_event": events[4],
            "11:30_event": events[5],
            "recovery_remaining_end": env.recovery_remaining,
        },
    )


def case_focus_fortress() -> CaseResult:
    env_a = blank_env(5, distraction_risk=0.5, seed=1)
    env_a.timeline[:] = DEEP_WORK
    events_a = run_day(env_a)

    env_b = blank_env(5, distraction_risk=0.5, seed=1)
    env_b.timeline[:] = DEEP_WORK
    env_b.mute_comms = True
    events_b = run_day(env_b)

    return CaseResult(
        name="Focus Fortress",
        passed=(
            any(ev["interrupted"] for ev in events_a)
            and not any(ev["interrupted"] for ev in events_b)
        ),
        details={
            "without_quiet_mode": events_a,
            "with_quiet_mode": events_b,
            "without_quiet_mode_interruptions": env_a.interruptions,
            "with_quiet_mode_interruptions": env_b.interruptions,
        },
    )


def case_hidden_complexity_reveal() -> CaseResult:
    env = blank_env(6)
    env.task_buffer = [Task(duration=2, hidden_complexity=1.5)]
    env.timeline[3] = MEETING
    env.meeting_meta[3] = {"meeting_id": 1, "start": 3, "length": 1, "priority": 7}

    _, _, _, info = env.step((1, 1))

    return CaseResult(
        name="Hidden complexity reveal",
        passed=info["action_info"]["true_slots"] == 3 and info["action_info"]["scheduled_slots"] == 2,
        details={
            "action_info": info["action_info"],
            "timeline_after": env.timeline.astype(int).tolist(),
            "interpretation": "Complexity expands to 3 slots, but the implementation truncates at the meeting instead of overwriting it.",
        },
    )


def case_social_debt_vs_flow() -> CaseResult:
    baseline = blank_env(16)
    baseline.timeline[:] = DEEP_WORK
    for slot in (4, 9, 14):
        baseline.timeline[slot] = MEETING
        baseline.meeting_meta[slot] = {"meeting_id": slot, "start": slot, "length": 1, "priority": 10}
    run_day(baseline)
    baseline_total = baseline._total_score()

    aggressive = blank_env(16)
    aggressive.timeline[:] = DEEP_WORK
    for slot in (4, 9, 14):
        aggressive.timeline[slot] = MEETING
        aggressive.meeting_meta[slot] = {"meeting_id": slot, "start": slot, "length": 1, "priority": 10}

    for slot in (4, 9, 14):
        aggressive._reschedule_meeting(slot)
        aggressive.timeline[slot] = DEEP_WORK

    run_day(aggressive)
    aggressive_total = aggressive._total_score()

    return CaseResult(
        name="Social debt vs flow",
        passed=aggressive_total < baseline_total,
        details={
            "baseline": {
                "flow_score": baseline.flow_score,
                "social_debt": baseline.social_debt,
                "calendar_churn": baseline.calendar_churn,
                "total_score": baseline_total,
            },
            "aggressive": {
                "flow_score": aggressive.flow_score,
                "social_debt": aggressive.social_debt,
                "calendar_churn": aggressive.calendar_churn,
                "total_score": aggressive_total,
            },
            "interpretation": "If aggressive_total stays higher, the current penalty model is too weak to prevent meeting deletion exploits.",
        },
    )


def main() -> None:
    cases = [
        case_fragmentation_penalty(),
        case_context_switch_recovery(),
        case_focus_fortress(),
        case_hidden_complexity_reveal(),
        case_social_debt_vs_flow(),
    ]
    for case in cases:
        print({"name": case.name, "passed": case.passed, "details": case.details})


if __name__ == "__main__":
    main()
