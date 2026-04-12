from __future__ import annotations

from benchmark_tasks import TASK_SPECS


TASKS = [
    {
        "id": "engineer_manager_task_0",
        "task_id": "quiet-morning",
        "name": "quiet-morning",
        "difficulty": "easy",
        "description": TASK_SPECS["quiet-morning"].description,
        "max_steps": 32,
        "reset_params": {"task_name": "quiet-morning"},
        "action_schema": {
            "target_slot": "integer slot index within the workday",
            "operation": "0=idle, 1=schedule work, 2=reschedule meeting, 3=toggle mute comms",
        },
        "grader": "graders:grade_task_0",
        "graders": ["graders:grade_task_0"],
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "engineer_manager_task_1",
        "task_id": "meeting-surgery",
        "name": "meeting-surgery",
        "difficulty": "medium",
        "description": TASK_SPECS["meeting-surgery"].description,
        "max_steps": 32,
        "reset_params": {"task_name": "meeting-surgery"},
        "action_schema": {
            "target_slot": "integer slot index within the workday",
            "operation": "0=idle, 1=schedule work, 2=reschedule meeting, 3=toggle mute comms",
        },
        "grader": "graders:grade_task_1",
        "graders": ["graders:grade_task_1"],
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "engineer_manager_task_2",
        "task_id": "delivery-triage",
        "name": "delivery-triage",
        "difficulty": "hard",
        "description": TASK_SPECS["delivery-triage"].description,
        "max_steps": 32,
        "reset_params": {"task_name": "delivery-triage"},
        "action_schema": {
            "target_slot": "integer slot index within the workday",
            "operation": "0=idle, 1=schedule work, 2=reschedule meeting, 3=toggle mute comms",
        },
        "grader": "graders:grade_task_2",
        "graders": ["graders:grade_task_2"],
        "reward_range": [0.0, 1.0],
    },
]


TASK_ID_TO_INDEX = {task["task_id"]: index for index, task in enumerate(TASKS)}


TASK_GRADER_PAIRS = [
    ("engineer_manager_task_0", "graders:grade_task_0"),
    ("engineer_manager_task_1", "graders:grade_task_1"),
    ("engineer_manager_task_2", "graders:grade_task_2"),
]


__all__ = ["TASKS", "TASK_ID_TO_INDEX", "TASK_GRADER_PAIRS"]
