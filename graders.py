from __future__ import annotations

from typing import Any


TASK_NAMES = {
    0: "quiet-morning",
    1: "meeting-surgery",
    2: "delivery-triage",
}


def _normalize_reward(reward: float) -> float:
    return min(max(float(reward), 0.0), 1.0)


def _state_task_id(state: Any) -> int | None:
    if not isinstance(state, dict):
        return None
    task_id = state.get("task_id")
    if isinstance(task_id, int):
        return task_id
    task_name = state.get("task_name")
    if isinstance(task_name, str):
        for index, name in TASK_NAMES.items():
            if name == task_name:
                return index
    metadata = state.get("metadata")
    if isinstance(metadata, dict):
        nested_task_id = metadata.get("task_id")
        if isinstance(nested_task_id, int):
            return nested_task_id
    return None


def grade_task_0(state: dict, reward: float) -> float:
    return _normalize_reward(reward if _state_task_id(state) == 0 else 0.0)


def grade_task_1(state: dict, reward: float) -> float:
    return _normalize_reward(reward if _state_task_id(state) == 1 else 0.0)


def grade_task_2(state: dict, reward: float) -> float:
    return _normalize_reward(reward if _state_task_id(state) == 2 else 0.0)


GRADERS = {
    "engineer_manager_task_0": grade_task_0,
    "engineer_manager_task_1": grade_task_1,
    "engineer_manager_task_2": grade_task_2,
}


TASK_GRADER_PAIRS = [
    ("engineer_manager_task_0", grade_task_0),
    ("engineer_manager_task_1", grade_task_1),
    ("engineer_manager_task_2", grade_task_2),
]


__all__ = [
    "grade_task_0",
    "grade_task_1",
    "grade_task_2",
    "GRADERS",
    "TASK_GRADER_PAIRS",
]
