import asyncio
import json
import math
import os
import socket
import subprocess
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from openenv.core.generic_client import GenericEnvClient

try:
    from server.engineer_manager_environment import EngineerManagerEnvironment
except ImportError:
    EngineerManagerEnvironment = None  # type: ignore[assignment]


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL")
TASK_NAME = os.getenv("TASK_NAME", "engineer-manager")
BENCHMARK = os.getenv("BENCHMARK", "openenv")
MAX_STEPS = int(os.getenv("MAX_STEPS", "32"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "120"))
LOCAL_SERVER_HOST = os.getenv("LOCAL_SERVER_HOST", "127.0.0.1")
LOCAL_SERVER_STARTUP_TIMEOUT = float(os.getenv("LOCAL_SERVER_STARTUP_TIMEOUT", "15"))

_LOCAL_SERVER_PROCESS: subprocess.Popen[str] | None = None

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


def _require_env(name: str, value: str | None) -> str:
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


def _sanitize_field(value: Any) -> str:
    text = str(value)
    return text.replace("\r", " ").replace("\n", " ").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={_sanitize_field(task)} env={_sanitize_field(env)} model={_sanitize_field(model)}",
        flush=True,
    )


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None,
) -> None:
    error_text = "null" if error in (None, "") else _sanitize_field(error)
    print(
        f"[STEP] step={step} action={_sanitize_field(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_text}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_text}",
        flush=True,
    )


def log_error(stage: str, error: Exception) -> None:
    print(
        f"[ERROR] stage={_sanitize_field(stage)} error={_sanitize_field(error)}",
        flush=True,
    )


def log_traceback(stage: str, error: BaseException) -> None:
    traceback_text = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    ).rstrip()
    print(f"[TRACEBACK] stage={_sanitize_field(stage)}", flush=True)
    print(traceback_text, flush=True)


def log_info(stage: str, message: str) -> None:
    print(
        f"[INFO] stage={_sanitize_field(stage)} message={_sanitize_field(message)}",
        flush=True,
    )


def log_env_status() -> None:
    env_fields = {
        "API_BASE_URL": API_BASE_URL,
        "MODEL_NAME": MODEL_NAME,
        "HF_TOKEN": "<set>" if HF_TOKEN else "<missing>",
        "LOCAL_IMAGE_NAME": LOCAL_IMAGE_NAME or "<missing>",
        "OPENENV_BASE_URL": OPENENV_BASE_URL or "<missing>",
        "TASK_NAME": TASK_NAME,
        "BENCHMARK": BENCHMARK,
        "MAX_STEPS": MAX_STEPS,
        "TEMPERATURE": TEMPERATURE,
        "MAX_TOKENS": MAX_TOKENS,
        "LOCAL_SERVER_HOST": LOCAL_SERVER_HOST,
        "LOCAL_SERVER_STARTUP_TIMEOUT": LOCAL_SERVER_STARTUP_TIMEOUT,
    }
    formatted = ", ".join(
        f"{name}={_sanitize_field(value)}" for name, value in env_fields.items()
    )
    log_info("env", formatted)


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


def estimate_max_flow_score(timeline: list[int]) -> float:
    slot_count = len(timeline)
    if slot_count <= 0:
        return 1.0
    hours = slot_count * 0.5
    return max(1.0, hours * hours)


def normalize_score(total_reward: float, observation: dict[str, Any]) -> float:
    timeline = observation.get("timeline") or []
    max_score = estimate_max_flow_score(timeline)
    normalized = total_reward / max_score
    return min(1.0, max(0.0, normalized))


def first_future_slot(observation: dict[str, Any], kind: int) -> int | None:
    timeline = observation.get("timeline") or []
    current_slot = int(observation.get("current_slot", 0))
    for index in range(current_slot, len(timeline)):
        if int(timeline[index]) == kind:
            return index
    return None


def first_future_empty_slot(observation: dict[str, Any]) -> int | None:
    return first_future_slot(observation, 0)


def build_user_prompt(
    step: int,
    observation: dict[str, Any],
    rewards: list[float],
    history: list[str],
) -> str:
    timeline = observation.get("timeline") or []
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
        timeline={timeline}
        task_buffer={json.dumps(observation.get("task_buffer", []), separators=(",", ":"))}
        last_rewards={",".join(f"{reward:.2f}" for reward in rewards[-5:]) or "none"}
        recent_history={json.dumps(history[-5:])}
        last_metadata={json.dumps(metadata, separators=(",", ":"))}
        Choose the single next action.
        """
    ).strip()


def choose_fallback_action(observation: dict[str, Any]) -> dict[str, int]:
    current_slot = int(observation.get("current_slot", 0))
    distraction_risk = float(observation.get("distraction_risk", 0.0))
    mute_comms = bool(observation.get("mute_comms", False))
    if distraction_risk >= 0.2 and not mute_comms:
        return {"target_slot": current_slot, "operation": 3}

    empty_slot = first_future_empty_slot(observation)
    if empty_slot is not None and observation.get("task_buffer"):
        return {"target_slot": empty_slot, "operation": 1}

    meeting_slot = first_future_slot(observation, 2)
    if meeting_slot is not None and current_slot <= meeting_slot:
        return {"target_slot": meeting_slot, "operation": 2}

    return {"target_slot": current_slot, "operation": 0}


def coerce_action(raw_text: str, observation: dict[str, Any]) -> dict[str, int]:
    timeline = observation.get("timeline") or []
    max_slot = max(0, len(timeline) - 1)
    fallback = choose_fallback_action(observation)
    try:
        data = json.loads(raw_text)
        target_slot = int(data["target_slot"])
        operation = int(data["operation"])
    except Exception:
        return fallback

    if operation not in {0, 1, 2, 3}:
        return fallback
    target_slot = min(max(target_slot, 0), max_slot)
    return {"target_slot": target_slot, "operation": operation}


def get_model_action(
    client: OpenAI | None,
    step: int,
    observation: dict[str, Any],
    rewards: list[float],
    history: list[str],
) -> dict[str, int]:
    if client is None:
        return choose_fallback_action(observation)

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


def _reserve_local_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _server_script_path() -> Path:
    return Path(__file__).resolve().parent / "server" / "app.py"


async def _connect_env(base_url: str) -> GenericEnvClient:
    env = GenericEnvClient(base_url=base_url)
    await env.connect()
    return env


def _start_local_server() -> str:
    global _LOCAL_SERVER_PROCESS

    if _LOCAL_SERVER_PROCESS is not None:
        raise RuntimeError("Local server process is already running")

    host = LOCAL_SERVER_HOST
    port = _reserve_local_port(host)
    script_path = _server_script_path()
    process = subprocess.Popen(
        [sys.executable, str(script_path), "--host", host, "--port", str(port)],
        cwd=str(Path(__file__).resolve().parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    _LOCAL_SERVER_PROCESS = process

    deadline = time.monotonic() + LOCAL_SERVER_STARTUP_TIMEOUT
    health_url = f"http://{host}:{port}/health"
    base_url = f"http://{host}:{port}"

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Local server process exited before becoming healthy")
        try:
            import urllib.request

            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if response.status == 200:
                    return base_url
        except Exception:
            time.sleep(0.25)

    raise RuntimeError("Timed out waiting for the local server to become healthy")


def stop_local_server() -> None:
    global _LOCAL_SERVER_PROCESS

    process = _LOCAL_SERVER_PROCESS
    _LOCAL_SERVER_PROCESS = None
    if process is None:
        return

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


async def create_env() -> tuple[Any, str]:
    if OPENENV_BASE_URL:
        return await _connect_env(OPENENV_BASE_URL), "remote"

    if LOCAL_IMAGE_NAME:
        try:
            return await GenericEnvClient.from_docker_image(LOCAL_IMAGE_NAME), "docker"
        except Exception as error:
            log_error("docker", error)
            log_info("docker", "Falling back to bundled local server")

    else:
        log_info("startup", "LOCAL_IMAGE_NAME not set; using in-process bundled environment")

    local_env = _InProcessEnvClient()
    await local_env.connect()
    log_info("env", "Using in-process bundled environment")
    return local_env, "in-process"


async def main() -> None:
    client: OpenAI | None = None
    env = None
    env_mode = "unknown"
    rewards: list[float] = []
    history: list[str] = []
    steps_taken = 0
    success = False
    score = 0.0
    observation: dict[str, Any] = {}

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)
    log_env_status()

    try:
        if HF_TOKEN:
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        else:
            log_error("startup", RuntimeError("Missing HF_TOKEN; using fallback policy"))

        env, env_mode = await create_env()
        log_info("env", f"Connected via {env_mode}")
        result = await env.reset()
        observation = dict(result.observation)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(client, step, observation, rewards, history)
            result = await env.step(action)
            observation = dict(result.observation)

            reward = float(result.reward or 0.0)
            done = bool(result.done)
            metadata = observation.get("metadata") or {}
            error = metadata.get("last_action_error")

            rewards.append(reward)
            steps_taken = step

            action_text = (
                f"target_slot={int(action['target_slot'])},operation={int(action['operation'])}"
            )
            log_step(step, action_text, reward, done, error)

            history.append(
                f"step={step} action={action_text} reward={reward:.2f} "
                f"flow={float(observation.get('flow_score', 0.0)):.2f} "
                f"debt={float(observation.get('social_debt', 0.0)):.2f}"
            )

            if done:
                break

        total_reward = math.fsum(rewards)
        score = normalize_score(total_reward, observation)
        score = round(score, 2)
        success = score > 0.0
    except Exception as error:
        log_error("runtime", error)
        log_traceback("runtime", error)
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        stop_local_server()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BaseException as error:
        log_error("fatal", error)
        log_traceback("fatal", error)
        log_end(success=False, steps=0, score=0.0, rewards=[])
