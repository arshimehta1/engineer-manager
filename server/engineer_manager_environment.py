"""OpenEnv server wrapper for the focus scheduling simulator."""

from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment, EnvironmentMetadata
from openenv.core.env_server.types import State

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
    ) -> None:
        super().__init__()
        self._start_hour = start_hour
        self._end_hour = end_hour
        self._distraction_risk = distraction_risk
        self._seed = seed
        self._step_count = 0
        self._episode_id = str(uuid4())
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
        **_: object,
    ) -> EngineerManagerObservation:
        self._seed = self._seed if seed is None else seed
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0
        self._env = FocusResourceEnv(
            start_hour=self._start_hour,
            end_hour=self._end_hour,
            distraction_risk=self._distraction_risk,
            seed=self._seed,
        )
        return self._to_observation(self._env.reset(), reward=0.0, done=False)

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
                "and controlling communication noise."
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
        payload["metadata"] = info or {}
        return EngineerManagerObservation.model_validate(payload)
