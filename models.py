"""Pydantic models for the Engineer Manager environment."""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class EngineerManagerAction(Action):
    """Scheduling action applied to the focus-planning environment."""

    target_slot: int = Field(..., ge=0, description="Target half-hour slot index.")
    operation: int = Field(
        ...,
        ge=0,
        le=3,
        description="Operation id: 0 idle, 1 schedule work, 2 reschedule meeting, 3 mute comms.",
    )


class EngineerManagerObservation(Observation):
    """Serializable observation returned by the environment server."""

    timeline: list[int] = Field(default_factory=list)
    task_buffer: list[dict[str, Any]] = Field(default_factory=list)
    distraction_risk: float = Field(default=0.15)
    current_slot: int = Field(default=0)
    current_time: str = Field(default="09:00")
    recovery_state: int = Field(default=0)
    mute_comms: bool = Field(default=False)
    social_debt: float = Field(default=0.0)
    calendar_churn: int = Field(default=0)
    flow_score: float = Field(default=0.0)
