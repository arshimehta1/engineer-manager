"""OpenEnv client for the Engineer Manager environment."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import EngineerManagerAction, EngineerManagerObservation


class EngineerManagerEnv(
    EnvClient[EngineerManagerAction, EngineerManagerObservation, State]
):
    """Persistent client for a running Engineer Manager OpenEnv server."""

    def _step_payload(self, action: EngineerManagerAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(
        self, payload: dict[str, Any]
    ) -> StepResult[EngineerManagerObservation]:
        observation = EngineerManagerObservation.model_validate(
            payload.get("observation", {})
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        return State.model_validate(payload)
