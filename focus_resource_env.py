from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from openenv.env import Env
except ImportError:
    class Env:  # type: ignore[override]
        """Compatibility shim when only openenv-core is installed."""

        def __init__(self, *_: object, **__: object) -> None:
            pass


EMPTY = 0
DEEP_WORK = 1
MEETING = 2

OP_IDLE = 0
OP_SCHEDULE_WORK = 1
OP_RESCHEDULE_MEETING = 2
OP_MUTE_COMMS = 3

RECOVERY_STEPS = 2


@dataclass
class Task:
    duration: int
    hidden_complexity: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "duration": int(self.duration),
            "hidden_complexity": float(self.hidden_complexity),
        }


class FocusResourceEnv(Env):
    def __init__(
        self,
        start_hour: str = "09:00",
        end_hour: str = "17:00",
        distraction_risk: float = 0.15,
        seed: Optional[int] = None,
    ) -> None:
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.distraction_risk = float(distraction_risk)
        self.rng = np.random.default_rng(seed)
        self.slot_minutes = 30

        self.timeline_length = self._compute_timeline_length(start_hour, end_hour)
        if self.timeline_length <= 0:
            raise ValueError("end_hour must be after start_hour")

        super().__init__(
            name="FocusResourceEnv",
            state_space={
                "timeline": self.timeline_length,
                "task_buffer": 3,
                "distraction_risk": (0.0, 1.0),
            },
            action_space={
                "target_slot": (0, self.timeline_length - 1),
                "operation": {
                    OP_IDLE: "Idle",
                    OP_SCHEDULE_WORK: "Schedule Work",
                    OP_RESCHEDULE_MEETING: "Reschedule Meeting",
                    OP_MUTE_COMMS: "Mute Comms",
                },
            },
            episode_max_length=self.timeline_length,
        )

        self.reset()

    def reset(self) -> Dict[str, Any]:
        self.current_slot = 0
        self.timeline = np.zeros(self.timeline_length, dtype=np.int8)
        self.meeting_meta: Dict[int, Dict[str, int]] = {}
        self._meeting_id_counter = 0
        self.task_buffer = self._generate_task_buffer()
        self.current_work_streak_slots = 0
        self.recovery_remaining = 0
        self.mute_comms = False
        self.social_debt = 0.0
        self.calendar_churn = 0
        self.flow_score = 0.0
        self.last_executed_kind = EMPTY
        self.interruptions = 0
        self.invalid_actions = 0
        self._scatter_initial_meetings()
        return self._observation()

    def step(self, action: Tuple[int, int]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        target_slot, operation = self._normalize_action(action)
        action_info = self._apply_action(target_slot, operation)
        previous_score = self._total_score()
        transition_info = self._advance_execution()
        done = self.current_slot >= self.timeline_length
        reward = self._total_score() - previous_score

        info = {
            "slot_executed": self.current_slot - 1,
            "action": {"target_slot": target_slot, "operation": operation},
            "action_info": action_info,
            "transition_info": transition_info,
            "score_breakdown": {
                "flow_score": self.flow_score,
                "social_debt": self.social_debt,
                "calendar_churn": self.calendar_churn,
                "total_score": self._total_score(),
            },
        }
        return self._observation(), reward, done, info

    def render_text(self) -> str:
        symbols = {EMPTY: ".", DEEP_WORK: "W", MEETING: "M"}
        timeline = "".join(symbols[int(slot)] for slot in self.timeline)
        return (
            f"time={self._slot_label(self.current_slot)} "
            f"muted={self.mute_comms} recovery={self.recovery_remaining} "
            f"flow={self.flow_score:.2f} debt={self.social_debt:.2f} churn={self.calendar_churn} "
            f"timeline={timeline}"
        )

    def _normalize_action(self, action: Tuple[int, int]) -> Tuple[int, int]:
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            raise ValueError("action must be a (target_slot, operation) pair")

        target_slot = int(action[0])
        operation = int(action[1])
        if target_slot < 0 or target_slot >= self.timeline_length:
            raise ValueError("target_slot is outside the work day")
        if operation not in {OP_IDLE, OP_SCHEDULE_WORK, OP_RESCHEDULE_MEETING, OP_MUTE_COMMS}:
            raise ValueError("operation is invalid")
        return target_slot, operation

    def _apply_action(self, target_slot: int, operation: int) -> Dict[str, Any]:
        if target_slot < self.current_slot:
            self.invalid_actions += 1
            self.social_debt += 0.25
            return {"status": "invalid_past_slot"}

        if operation == OP_IDLE:
            return {"status": "idle"}
        if operation == OP_MUTE_COMMS:
            self.mute_comms = not self.mute_comms
            return {"status": "mute_toggled", "muted": self.mute_comms}
        if operation == OP_SCHEDULE_WORK:
            return self._schedule_work(target_slot)
        if operation == OP_RESCHEDULE_MEETING:
            return self._reschedule_meeting(target_slot)
        return {"status": "noop"}

    def _schedule_work(self, target_slot: int) -> Dict[str, Any]:
        if not self.task_buffer:
            self.invalid_actions += 1
            return {"status": "no_tasks_available"}
        if self.timeline[target_slot] == MEETING:
            self.invalid_actions += 1
            self.calendar_churn += 1
            return {"status": "meeting_blocks_target"}

        task = self.task_buffer.pop(0)
        true_slots = int(np.ceil(task.duration * task.hidden_complexity))
        contiguous = self._contiguous_empty_slots_from(target_slot)
        scheduled_slots = min(true_slots, contiguous)
        if scheduled_slots == 0:
            self.invalid_actions += 1
            self.task_buffer.insert(0, task)
            return {"status": "no_capacity"}

        self.timeline[target_slot : target_slot + scheduled_slots] = DEEP_WORK
        overflow = true_slots - scheduled_slots

        if overflow > 0:
            self.social_debt += 0.5
            self.invalid_actions += 1

        return {
            "status": "work_scheduled",
            "estimated_slots": task.duration,
            "true_slots": true_slots,
            "scheduled_slots": scheduled_slots,
            "overflow_slots": overflow,
        }

    def _reschedule_meeting(self, target_slot: int) -> Dict[str, Any]:
        if self.timeline[target_slot] != MEETING:
            self.invalid_actions += 1
            return {"status": "no_meeting_at_target"}

        meeting = self.meeting_meta.get(target_slot)
        if meeting is None:
            self.invalid_actions += 1
            return {"status": "missing_meeting_metadata"}

        length = meeting["length"]
        priority = meeting["priority"]
        meeting_id = meeting["meeting_id"]
        start = meeting["start"]

        candidate = self._find_latest_empty_block(length, exclude=(start, start + length))
        self._clear_meeting(start, length)
        self.calendar_churn += 1
        # High-priority meeting churn needs to outweigh the quadratic flow upside
        # from deleting collaboration entirely, while still charging a smaller
        # cost when a meeting is merely moved instead of cancelled.
        reschedule_penalty = priority / 2.0
        cancellation_penalty = (priority**2) / 5.0

        if candidate is None:
            self.social_debt += cancellation_penalty
            return {"status": "meeting_cancelled", "meeting_id": meeting_id, "priority": priority}

        self.social_debt += reschedule_penalty
        self._place_meeting(candidate, length, priority, meeting_id)
        return {
            "status": "meeting_rescheduled",
            "meeting_id": meeting_id,
            "from_slot": start,
            "to_slot": candidate,
            "priority": priority,
        }

    def _advance_execution(self) -> Dict[str, Any]:
        if self.current_slot >= self.timeline_length:
            return {"status": "episode_complete"}

        slot_kind = int(self.timeline[self.current_slot])
        event = {
            "slot_kind": slot_kind,
            "recovery_triggered": False,
            "interrupted": False,
            "productive": False,
        }

        if self._is_context_switch(self.last_executed_kind, slot_kind):
            self.recovery_remaining = RECOVERY_STEPS
            self.current_work_streak_slots = 0
            event["recovery_triggered"] = True

        if self.recovery_remaining > 0:
            self.recovery_remaining -= 1
            self.current_work_streak_slots = 0
        elif slot_kind == DEEP_WORK:
            if not self.mute_comms and self.rng.random() < self.distraction_risk:
                self.current_work_streak_slots = 0
                self.interruptions += 1
                event["interrupted"] = True
            else:
                previous_streak = self.current_work_streak_slots
                self.current_work_streak_slots += 1
                self.flow_score += self._power_law_delta(previous_streak, self.current_work_streak_slots)
                event["productive"] = True
        else:
            self.current_work_streak_slots = 0

        self.last_executed_kind = slot_kind
        self.current_slot += 1
        return event

    def _observation(self) -> Dict[str, Any]:
        return {
            "timeline": self.timeline.astype(int).tolist(),
            "task_buffer": [task.to_dict() for task in self.task_buffer],
            "distraction_risk": float(self.distraction_risk),
            "current_slot": int(self.current_slot),
            "current_time": self._slot_label(self.current_slot),
            "recovery_state": int(self.recovery_remaining),
            "mute_comms": bool(self.mute_comms),
            "social_debt": float(self.social_debt),
            "calendar_churn": int(self.calendar_churn),
            "flow_score": float(self.flow_score),
        }

    def _generate_task_buffer(self) -> List[Task]:
        return [self._make_task() for _ in range(3)]

    def _make_task(self) -> Task:
        return Task(
            duration=int(self.rng.integers(1, 5)),
            hidden_complexity=float(self.rng.choice([1.0, 1.25, 1.5, 1.75])),
        )

    def _scatter_initial_meetings(self) -> None:
        meeting_count = int(self.rng.integers(3, 6))
        attempts = 0
        while meeting_count > 0 and attempts < 100:
            attempts += 1
            length = int(self.rng.integers(1, 3))
            latest_start = self.timeline_length - length
            if latest_start < 0:
                break

            start = int(self.rng.integers(0, latest_start + 1))
            if np.any(self.timeline[start : start + length] != EMPTY):
                continue

            priority = int(self.rng.integers(1, 11))
            meeting_id = self._next_meeting_id()
            self._place_meeting(start, length, priority, meeting_id)
            meeting_count -= 1

    def _place_meeting(self, start: int, length: int, priority: int, meeting_id: int) -> None:
        self.timeline[start : start + length] = MEETING
        for slot in range(start, start + length):
            self.meeting_meta[slot] = {
                "meeting_id": meeting_id,
                "start": start,
                "length": length,
                "priority": priority,
            }

    def _clear_meeting(self, start: int, length: int) -> None:
        self.timeline[start : start + length] = EMPTY
        for slot in range(start, start + length):
            self.meeting_meta.pop(slot, None)

    def _find_latest_empty_block(
        self,
        length: int,
        exclude: Optional[Tuple[int, int]] = None,
    ) -> Optional[int]:
        for start in range(self.timeline_length - length, -1, -1):
            end = start + length
            if exclude is not None and not (end <= exclude[0] or start >= exclude[1]):
                continue
            if np.all(self.timeline[start:end] == EMPTY):
                return start
        return None

    def _contiguous_empty_slots_from(self, start: int) -> int:
        count = 0
        for slot in range(start, self.timeline_length):
            if self.timeline[slot] != EMPTY:
                break
            count += 1
        return count

    def _compute_timeline_length(self, start_hour: str, end_hour: str) -> int:
        return int((self._to_minutes(end_hour) - self._to_minutes(start_hour)) / self.slot_minutes)

    def _slot_label(self, slot_index: int) -> str:
        minute_value = self._to_minutes(self.start_hour) + slot_index * self.slot_minutes
        hours = (minute_value // 60) % 24
        minutes = minute_value % 60
        return f"{hours:02d}:{minutes:02d}"

    def _to_minutes(self, hhmm: str) -> int:
        hours, minutes = hhmm.split(":")
        return int(hours) * 60 + int(minutes)

    def _power_law_delta(self, previous_streak_slots: int, current_streak_slots: int) -> float:
        prev_hours = previous_streak_slots * 0.5
        curr_hours = current_streak_slots * 0.5
        return curr_hours ** 2 - prev_hours ** 2

    def _is_context_switch(self, previous_kind: int, current_kind: int) -> bool:
        work_meeting = {DEEP_WORK, MEETING}
        return previous_kind in work_meeting and current_kind in work_meeting and previous_kind != current_kind

    def _total_score(self) -> float:
        return self.flow_score - self.social_debt - self.calendar_churn

    def _next_meeting_id(self) -> int:
        self._meeting_id_counter += 1
        return self._meeting_id_counter
