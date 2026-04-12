"""OpenEnv server wrapper for the focus scheduling simulator."""

from __future__ import annotations

from uuid import uuid4
import os

from openenv.core.env_server.interfaces import Environment, EnvironmentMetadata
from openenv.core.env_server.types import State

from benchmark_tasks import TASK_SPECS, apply_task
from focus_resource_env import FocusResourceEnv

try:
    from ..models import EngineerManagerAction, EngineerManagerObservation
except ImportError:
    from models import EngineerManagerAction, EngineerManagerObservation


class EngineerManagerEnvironment(
    Environment[EngineerManagerAction, EngineerManagerObservation, State]
):
    """Expose the scheduling simulator through the OpenEnv HTTP contract."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        start_hour: str = "09:00",
        end_hour: str = "17:00",
        distraction_risk: float = 0.15,
        seed: int | None = 7,
        task_name: str | None = None,
    ) -> None:
        super().__init__()
        self._start_hour = start_hour
        self._end_hour = end_hour
        self._distraction_risk = distraction_risk
        self._seed = seed
        self._task_name = task_name or os.getenv("TASK_NAME")
        self._task_id = 0
        self._step_count = 0
        self._episode_id = str(uuid4())
        self._trajectory: list[dict[str, object]] = []
        self._env = FocusResourceEnv(
            start_hour=start_hour,
            end_hour=end_hour,
            distraction_risk=distraction_risk,
            seed=seed,
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_name: str | None = None,
        task_id: int | None = None,
        **_: object,
    ) -> EngineerManagerObservation:
        self._seed = self._seed if seed is None else seed
        task_names = ["quiet-morning", "meeting-surgery", "delivery-triage"]
        if task_id is not None and 0 <= int(task_id) < len(task_names):
            self._task_id = int(task_id)
            self._task_name = task_names[self._task_id]
        else:
            self._task_name = task_name or self._task_name or os.getenv("TASK_NAME")
            self._task_id = task_names.index(self._task_name) if self._task_name in task_names else 0
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0
        self._trajectory = []
        self._env = FocusResourceEnv(
            start_hour=self._start_hour,
            end_hour=self._end_hour,
            distraction_risk=self._distraction_risk,
            seed=self._seed,
        )
        self._env.reset()
        apply_task(self._env, self._task_name)
        return self._to_observation(self._env._observation(), reward=0.0, done=False)

    def step(
        self,
        action: EngineerManagerAction,
        timeout_s: float | None = None,
        **_: object,
    ) -> EngineerManagerObservation:
        del timeout_s
        observation, reward, done, info = self._env.step(
            (action.target_slot, action.operation)
        )
        self._step_count += 1
        self._trajectory.append(
            {
                "action": {"target_slot": int(action.target_slot), "operation": int(action.operation)},
                "observation": observation,
                "reward": float(reward),
                "done": bool(done),
                "info": info,
            }
        )
        return self._to_observation(observation, reward=reward, done=done, info=info)

    @property
    def state(self) -> State:
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
            current_slot=self._env.current_slot,
            done=self._env.current_slot >= self._env.timeline_length,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Engineer Manager",
            description=(
                "Manage a workday by scheduling deep work, rescheduling meetings, "
                "and controlling communication noise. "
                f"Available tasks: {', '.join(sorted(TASK_SPECS))}."
            ),
            version="0.1.0",
        )

    def _to_observation(
        self,
        observation: dict[str, object],
        *,
        reward: float | None,
        done: bool,
        info: dict[str, object] | None = None,
    ) -> EngineerManagerObservation:
        payload = dict(observation)
        payload["reward"] = reward
        payload["done"] = done
        metadata = dict(info or {})
        metadata["task_name"] = self._task_name
        metadata["task_id"] = self._task_id
        metadata["episode_metrics"] = {
            "interruptions": int(self._env.interruptions),
            "invalid_actions": int(self._env.invalid_actions),
            "remaining_tasks": len(self._env.task_buffer),
            "scheduled_work_slots": sum(1 for slot in self._env.timeline if int(slot) == 1),
            "successful_reschedules": sum(
                1
                for step in self._trajectory
                if step["info"].get("action_info", {}).get("status") == "meeting_rescheduled"
            ),
            "total_score": float(self._env._total_score()),
            "grader_score": min(max(float(reward or 0.0), 0.0), 1.0),
        }
        payload["metadata"] = metadata
        return EngineerManagerObservation.model_validate(payload)
