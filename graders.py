from __future__ import annotations

from typing import Any

from benchmark_tasks import grade_trajectory


def _coerce_trajectory(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(step) for step in payload]
    if isinstance(payload, dict):
        if isinstance(payload.get("trajectory"), list):
            return [dict(step) for step in payload["trajectory"]]
        if isinstance(payload.get("steps"), list):
            return [dict(step) for step in payload["steps"]]
    return []


def _grade(task_name: str, payload: Any) -> dict[str, Any]:
    trajectory = _coerce_trajectory(payload)
    score = grade_trajectory(task_name, trajectory)
    return {
        "task_name": task_name,
        "score": score,
        "passed": score > 0.0,
        "reward": score,
    }


def grade_task_0(payload: Any) -> dict[str, Any]:
    return _grade("quiet-morning", payload)


def grade_task_1(payload: Any) -> dict[str, Any]:
    return _grade("meeting-surgery", payload)


def grade_task_2(payload: Any) -> dict[str, Any]:
    return _grade("delivery-triage", payload)
