import asyncio
import json
import math
import os
import textwrap
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from openenv.core.generic_client import GenericEnvClient

try:
    from server.engineer_manager_environment import EngineerManagerEnvironment
except ImportError:
    EngineerManagerEnvironment = None  # type: ignore[assignment]


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL")
TASK_NAME = os.getenv("TASK_NAME", "engineer-manager")
BENCHMARK = os.getenv("BENCHMARK", "openenv")
MAX_STEPS = int(os.getenv("MAX_STEPS", "32"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "120"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are selecting actions for an environment that simulates an engineer-manager workday.
    Return exactly one compact JSON object with integer keys:
    {"target_slot": <int>, "operation": <int>}

    Operations:
    0 = idle
    1 = schedule work at target_slot
    2 = reschedule a meeting at target_slot
    3 = toggle mute comms

    Goals:
    - Maximize sustained deep work flow_score.
    - Avoid unnecessary social_debt and calendar_churn.
    - Prefer scheduling work into future empty slots.
    - Use reschedule_meeting only when it clearly helps.
    - Toggle mute comms early if distractions are high and it is currently off.

    Rules:
    - target_slot must be within the timeline bounds.
    - Return JSON only. No markdown. No explanation.
    """
).strip()


@dataclass
class _EnvResult:
    observation: dict[str, Any]
    reward: float | None
    done: bool


class _InProcessEnvClient:
    def __init__(self) -> None:
        if EngineerManagerEnvironment is None:
            raise RuntimeError("Bundled EngineerManagerEnvironment is unavailable")
        self._env = EngineerManagerEnvironment()

    async def connect(self) -> None:
        return None

    async def reset(self) -> _EnvResult:
        observation = self._env.reset().model_dump()
        return _EnvResult(
            observation=dict(observation),
            reward=float(observation.get("reward") or 0.0),
            done=bool(observation.get("done")),
        )

    async def step(self, action: dict[str, int]) -> _EnvResult:
        observation = self._env.step(type("Action", (), action)()).model_dump()
        return _EnvResult(
            observation=dict(observation),
            reward=float(observation.get("reward") or 0.0),
            done=bool(observation.get("done")),
        )

    async def close(self) -> None:
        return None


def _sanitize_field(value: Any) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def _format_error(error: str | None) -> str:
    return "null" if error in (None, "") else _sanitize_field(error)


def _action_to_text(action: dict[str, int]) -> str:
    return f'{{"target_slot":{int(action["target_slot"])},"operation":{int(action["operation"])}}}'


def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={_sanitize_field(task)} env={_sanitize_field(env)} model={_sanitize_field(model)}",
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={_sanitize_field(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={_format_error(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_text}",
        flush=True,
    )


def estimate_max_flow_score(timeline: list[int]) -> float:
    slot_count = len(timeline)
    if slot_count <= 0:
        return 1.0
    return max(1.0, (slot_count * 0.5) ** 2)


def normalize_score(total_reward: float, observation: dict[str, Any]) -> float:
    max_score = estimate_max_flow_score(observation.get("timeline") or [])
    return min(1.0, max(0.0, total_reward / max_score))


def first_future_slot(observation: dict[str, Any], kind: int) -> int | None:
    timeline = observation.get("timeline") or []
    current_slot = int(observation.get("current_slot", 0))
    for index in range(current_slot, len(timeline)):
        if int(timeline[index]) == kind:
            return index
    return None


def build_user_prompt(
    step: int,
    observation: dict[str, Any],
    rewards: list[float],
    history: list[str],
) -> str:
    metadata = observation.get("metadata") or {}
    return textwrap.dedent(
        f"""
        step={step}
        current_slot={int(observation.get("current_slot", 0))}
        current_time={observation.get("current_time", "unknown")}
        mute_comms={bool(observation.get("mute_comms", False))}
        distraction_risk={float(observation.get("distraction_risk", 0.0))}
        flow_score={float(observation.get("flow_score", 0.0)):.2f}
        social_debt={float(observation.get("social_debt", 0.0)):.2f}
        calendar_churn={int(observation.get("calendar_churn", 0))}
        recovery_state={int(observation.get("recovery_state", 0))}
        timeline={json.dumps(observation.get("timeline", []), separators=(",", ":"))}
        task_buffer={json.dumps(observation.get("task_buffer", []), separators=(",", ":"))}
        last_rewards={",".join(f"{reward:.2f}" for reward in rewards[-5:]) or "none"}
        recent_history={json.dumps(history[-5:], separators=(",", ":"))}
        last_metadata={json.dumps(metadata, separators=(",", ":"))}
        Choose the single next action.
        """
    ).strip()


def choose_fallback_action(observation: dict[str, Any]) -> dict[str, int]:
    current_slot = int(observation.get("current_slot", 0))
    distraction_risk = float(observation.get("distraction_risk", 0.0))
    mute_comms = bool(observation.get("mute_comms", False))
    recovery_state = int(observation.get("recovery_state", 0))
    timeline = observation.get("timeline") or []

    if current_slot == 0 and distraction_risk > 0.0 and not mute_comms:
        return {"target_slot": current_slot, "operation": 3}

    if recovery_state > 0:
        return {"target_slot": current_slot, "operation": 0}

    if current_slot < len(timeline) and int(timeline[current_slot]) == 0 and observation.get("task_buffer"):
        return {"target_slot": current_slot, "operation": 1}

    empty_slot = first_future_slot(observation, 0)
    if empty_slot is not None and observation.get("task_buffer"):
        return {"target_slot": empty_slot, "operation": 1}

    meeting_slot = first_future_slot(observation, 2)
    if meeting_slot is not None:
        return {"target_slot": meeting_slot, "operation": 2}

    return {"target_slot": current_slot, "operation": 0}


def coerce_action(raw_text: str, observation: dict[str, Any]) -> dict[str, int]:
    timeline = observation.get("timeline") or []
    fallback = choose_fallback_action(observation)
    max_slot = max(0, len(timeline) - 1)
    try:
        data = json.loads(raw_text)
        target_slot = int(data["target_slot"])
        operation = int(data["operation"])
    except Exception:
        return fallback

    if operation not in {0, 1, 2, 3}:
        return fallback
    return {"target_slot": min(max(target_slot, 0), max_slot), "operation": operation}


def get_model_action(
    client: OpenAI,
    step: int,
    observation: dict[str, Any],
    rewards: list[float],
    history: list[str],
) -> dict[str, int]:
    user_prompt = build_user_prompt(step, observation, rewards, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = (completion.choices[0].message.content or "").strip()
        return coerce_action(content, observation)
    except Exception:
        return choose_fallback_action(observation)


async def create_env() -> Any:
    if OPENENV_BASE_URL:
        env = GenericEnvClient(base_url=OPENENV_BASE_URL)
        await env.connect()
        return env

    if LOCAL_IMAGE_NAME:
        try:
            return await GenericEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
        except Exception:
            pass

    env = _InProcessEnvClient()
    await env.connect()
    return env


async def main() -> None:
    env = None
    rewards: list[float] = []
    history: list[str] = []
    steps_taken = 0
    success = False
    score = 0.0
    observation: dict[str, Any] = {}
    completed = False

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
        env = await create_env()
        result = await env.reset()
        observation = dict(result.observation)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                completed = True
                break

            if client is None:
                action = choose_fallback_action(observation)
            else:
                action = get_model_action(client, step, observation, rewards, history)
            action_text = _action_to_text(action)
            step_error: str | None = None

            try:
                result = await env.step(action)
                observation = dict(result.observation)
                reward = float(result.reward or 0.0)
                done = bool(result.done)
                metadata = observation.get("metadata") or {}
                step_error = metadata.get("last_action_error")
            except Exception as exc:
                reward = 0.0
                done = True
                step_error = str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step, action_text, reward, done, step_error)
            history.append(
                f"step={step} action={action_text} reward={reward:.2f} error={_format_error(step_error)}"
            )

            if done:
                completed = True
                break

        score = round(normalize_score(math.fsum(rewards), observation), 2)
        success = completed and score >= 0.0
    except Exception:
        success = False
        score = 0.0
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
